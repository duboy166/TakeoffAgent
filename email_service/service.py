#!/usr/bin/env python3
"""
AutoWork Email Service - Main Entry Point

Polls Gmail for construction plans, processes them, and sends back results.
"""

import time
import logging
import signal
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from .config import load_config, EmailServiceConfig
from .fetcher import EmailFetcher, IncomingEmail
from .processor import AutoWorkProcessor, ProcessingResult, cleanup_output
from .sender import EmailSender
from .security import is_sender_allowed, RateLimiter, validate_pdf, extract_email_address

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


class EmailService:
    """Main email service that orchestrates everything."""
    
    def __init__(self, config: Optional[EmailServiceConfig] = None):
        """
        Initialize the email service.
        
        Args:
            config: Configuration object. If None, loads from default location.
        """
        self.config = config or load_config()
        
        self.fetcher = EmailFetcher(self.config.imap)
        self.sender = EmailSender(self.config.smtp)
        self.processor = AutoWorkProcessor(
            parallel=self.config.processing.parallel_ocr,
        )
        self.rate_limiter = RateLimiter(
            hourly_limit=self.config.security.rate_limit_hourly,
            daily_limit=self.config.security.rate_limit_daily,
        )
        
        self.running = False
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def process_email(self, email: IncomingEmail) -> bool:
        """
        Process a single incoming email.
        
        Returns:
            True if processed successfully
        """
        sender = extract_email_address(email.from_addr)
        subject = email.subject or "(no subject)"
        
        logger.info(f"Processing email from {sender}: {subject}")
        
        # Security check: sender allowed?
        if not is_sender_allowed(email.from_addr, self.config.security):
            logger.warning(f"Unauthorized sender: {sender}")
            self.sender.send_unauthorized(sender, subject)
            return False
        
        # Rate limit check
        allowed, reason = self.rate_limiter.check(sender)
        if not allowed:
            logger.warning(f"Rate limited: {sender} - {reason}")
            self.sender.send_error(sender, subject, f"Rate limit exceeded: {reason}")
            return False
        
        # Check for PDFs
        if not email.attachments and not email.cloud_links:
            self.sender.send_error(
                sender, subject,
                "No PDF attachments or cloud storage links found. "
                "Please attach your construction plans or include a Dropbox/Google Drive link."
            )
            return False
        
        # Validate attachments
        valid_pdfs = []
        for attachment in email.attachments:
            is_valid, reason = validate_pdf(
                attachment.path,
                max_mb=self.config.security.max_attachment_mb
            )
            if is_valid:
                valid_pdfs.append(attachment.path)
            else:
                logger.warning(f"Invalid PDF {attachment.filename}: {reason}")
        
        # Handle cloud links (future: download and process)
        if email.cloud_links and not valid_pdfs:
            self.sender.send_error(
                sender, subject,
                "Cloud storage links detected but not yet supported. "
                "Please attach PDFs directly (up to 25MB) or split larger files."
            )
            return False
        
        if not valid_pdfs:
            self.sender.send_error(
                sender, subject,
                "No valid PDF attachments found. Please check your files and try again."
            )
            return False
        
        # Process PDFs
        results = []
        output_files = []
        
        for pdf_path in valid_pdfs:
            result = self.processor.process(pdf_path)
            results.append({
                'filename': result.filename,
                'success': result.success,
                'items': result.items,
                'matched_items': result.matched_items,
                'estimate': result.estimate,
                'error': result.error,
            })
            
            if result.success and result.csv_path:
                output_files.append(result.csv_path)
            
            # Cleanup after collecting outputs
            # (don't cleanup yet - need the files for sending)
        
        # Record rate limit
        self.rate_limiter.record(sender)
        
        # Send results
        if output_files:
            success = self.sender.send_results(
                to_addr=sender,
                original_subject=subject,
                results=results,
                attachments=output_files,
                reply_to_message_id=email.message_id,
            )
        else:
            # All failed
            error_details = "\n".join([
                f"• {r['filename']}: {r['error']}"
                for r in results if not r['success']
            ])
            success = self.sender.send_error(
                sender, subject,
                f"Processing failed for all files:\n\n{error_details}"
            )
        
        # Send Telegram notification if enabled
        if self.config.notifications.telegram_enabled:
            self._notify_telegram(email, results)
        
        return success
    
    def _notify_telegram(self, email: IncomingEmail, results: list):
        """Send Telegram notification about completed job."""
        try:
            import subprocess
            
            total_items = sum(r.get('items', 0) for r in results)
            matched_items = sum(r.get('matched_items', 0) for r in results)
            total_estimate = sum(r.get('estimate', 0) for r in results)
            success_count = sum(1 for r in results if r.get('success'))
            fail_count = len(results) - success_count
            
            # Build message
            if success_count == len(results):
                status = "✅"
            elif success_count > 0:
                status = "⚠️"
            else:
                status = "❌"
            
            msg = f"{status} AutoWork Job Complete\n"
            msg += f"From: {extract_email_address(email.from_addr)}\n"
            msg += f"Files: {success_count}/{len(results)} successful"
            if fail_count > 0:
                msg += f" ({fail_count} failed)"
            msg += f"\nMaterials: {total_items}"
            if matched_items > 0:
                msg += f" ({matched_items} priced)"
            if total_estimate > 0:
                msg += f"\nEstimate: ${total_estimate:,.2f}"
            
            # Send via OpenClaw CLI
            result = subprocess.run(
                ['openclaw', 'message', 'send', '--channel', 'telegram', '--to', '7570826467', msg],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                logger.info("Telegram notification sent")
            else:
                logger.warning(f"Telegram notification failed: {result.stderr}")
            
        except FileNotFoundError:
            logger.debug("OpenClaw CLI not available for Telegram notifications")
        except Exception as e:
            logger.warning(f"Could not send Telegram notification: {e}")
    
    def poll_once(self) -> int:
        """
        Poll for new emails and process them.
        
        Returns:
            Number of emails processed
        """
        try:
            with EmailFetcher(self.config.imap) as fetcher:
                emails = fetcher.fetch_unread()
                
                if not emails:
                    return 0
                
                logger.info(f"Found {len(emails)} new email(s) to process")
                
                processed = 0
                for email in emails:
                    try:
                        if self.process_email(email):
                            processed += 1
                        fetcher.mark_as_processed(email.imap_id)
                    except Exception as e:
                        logger.exception(f"Error processing email: {e}")
                        # Always respond - never silent failures
                        try:
                            sender = extract_email_address(email.from_addr)
                            subject = email.subject or "(no subject)"
                            self.sender.send_error(
                                sender, subject,
                                "An unexpected error occurred while processing your request. "
                                "Our team has been notified. Please try again or reply to this email for assistance."
                            )
                        except Exception as send_err:
                            logger.error(f"Could not send error response: {send_err}")
                
                return processed
                
        except Exception as e:
            logger.exception(f"Error polling emails: {e}")
            return 0
    
    def run(self, poll_interval: int = 60):
        """
        Run the email service (polling loop).
        
        Args:
            poll_interval: Seconds between polls
        """
        logger.info("=" * 60)
        logger.info("AutoWork Email Service Starting")
        logger.info(f"Email: {self.config.imap.username}")
        logger.info(f"Poll interval: {poll_interval}s")
        logger.info("=" * 60)
        
        self.running = True
        
        while self.running:
            try:
                processed = self.poll_once()
                if processed:
                    logger.info(f"Processed {processed} email(s)")
            except Exception as e:
                logger.exception(f"Error in polling loop: {e}")
            
            # Sleep with interrupt check
            for _ in range(poll_interval):
                if not self.running:
                    break
                time.sleep(1)
        
        logger.info("Email service stopped")
    
    def stop(self):
        """Stop the service."""
        self.running = False


def main():
    """Entry point for email service."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoWork Email Service')
    parser.add_argument(
        '--poll-interval', type=int, default=60,
        help='Seconds between email checks (default: 60)'
    )
    parser.add_argument(
        '--once', action='store_true',
        help='Poll once and exit (for testing)'
    )
    parser.add_argument(
        '--config', type=str,
        help='Path to config file'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = load_config(args.config) if args.config else None
    service = EmailService(config)
    
    if args.once:
        processed = service.poll_once()
        print(f"Processed {processed} email(s)")
        sys.exit(0 if processed >= 0 else 1)
    else:
        service.run(poll_interval=args.poll_interval)


if __name__ == '__main__':
    main()
