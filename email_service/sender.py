"""SMTP email sender for the email service."""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.utils import formataddr
from pathlib import Path
import logging
from typing import List, Optional

from .config import SMTPConfig

logger = logging.getLogger(__name__)


class EmailSender:
    """Sends emails via SMTP."""
    
    def __init__(self, config: SMTPConfig):
        self.config = config
        self.from_name = "AutoWork Takeoff"
    
    def send(
        self,
        to_addr: str,
        subject: str,
        body: str,
        attachments: Optional[List[Path]] = None,
        reply_to_message_id: Optional[str] = None,
    ) -> bool:
        """
        Send an email with optional attachments.
        
        Args:
            to_addr: Recipient email address
            subject: Email subject
            body: Plain text body
            attachments: List of file paths to attach
            reply_to_message_id: Original message ID for threading
            
        Returns:
            True if sent successfully
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = formataddr((self.from_name, self.config.username))
            msg['To'] = to_addr
            msg['Subject'] = subject
            
            # Threading headers for reply
            if reply_to_message_id:
                msg['In-Reply-To'] = reply_to_message_id
                msg['References'] = reply_to_message_id
            
            # Body
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # Attachments
            for filepath in (attachments or []):
                path = Path(filepath)
                if not path.exists():
                    logger.warning(f"Attachment not found: {path}")
                    continue
                
                with open(path, 'rb') as f:
                    part = MIMEApplication(f.read(), Name=path.name)
                    part['Content-Disposition'] = f'attachment; filename="{path.name}"'
                    msg.attach(part)
                    logger.debug(f"Attached: {path.name}")
            
            # Send
            with smtplib.SMTP(self.config.host, self.config.port) as server:
                server.starttls()
                server.login(self.config.username, self.config.password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_addr}: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_addr}: {e}")
            return False
    
    def send_results(
        self,
        to_addr: str,
        original_subject: str,
        results: List[dict],
        attachments: List[Path],
        reply_to_message_id: Optional[str] = None,
    ) -> bool:
        """
        Send processing results email.
        
        Args:
            to_addr: Recipient
            original_subject: Original email subject
            results: List of processing results
            attachments: Output files to attach
            reply_to_message_id: For threading
        """
        subject = f"Re: {original_subject} — Takeoff Complete"
        
        # Build body
        lines = [
            "AutoWork Takeoff Results",
            "=" * 40,
            "",
        ]
        
        total_items = 0
        total_estimate = 0.0
        
        for r in results:
            lines.append(f"📄 {r.get('filename', 'Unknown')}")
            
            if r.get('success'):
                items = r.get('items', 0)
                estimate = r.get('estimate', 0)
                total_items += items
                total_estimate += estimate
                
                lines.append(f"   ✅ Success")
                lines.append(f"   Materials Detected: {items}")
                if estimate > 0:
                    lines.append(f"   Estimated Cost: ${estimate:,.2f}")
                else:
                    lines.append(f"   Cost: Review CSV for details (some items may need prices or quantities)")
            else:
                lines.append(f"   ❌ Error: {r.get('error', 'Unknown error')}")
            
            lines.append("")
        
        lines.extend([
            "-" * 40,
            f"Total Materials Found: {total_items}",
        ])
        
        if total_estimate > 0:
            lines.append(f"Total Estimate: ${total_estimate:,.2f}")
        else:
            lines.append("Note: No items matched our price catalog. The CSV lists all")
            lines.append("detected materials — prices can be added manually.")
        
        lines.extend([
            "",
            "Attached: CSV report(s) with full details",
            "",
            "---",
            "AutoWork Takeoff Service",
            "Questions? Reply to this email.",
        ])
        
        body = "\n".join(lines)
        
        return self.send(
            to_addr=to_addr,
            subject=subject,
            body=body,
            attachments=attachments,
            reply_to_message_id=reply_to_message_id,
        )
    
    def send_error(
        self,
        to_addr: str,
        original_subject: str,
        error_message: str,
        reply_to_message_id: Optional[str] = None,
    ) -> bool:
        """Send an error notification email."""
        subject = f"Re: {original_subject} — Processing Error"
        
        body = f"""Hi,

We encountered an error processing your submission:

{error_message}

Please check your files and try again. If the issue persists, reply to this email for assistance.

Common issues:
• File too large (max 25MB per attachment, use Dropbox/Drive links for larger files)
• Invalid PDF format
• Scanned documents may take longer to process

---
AutoWork Takeoff Service
"""
        
        return self.send(
            to_addr=to_addr,
            subject=subject,
            body=body,
            reply_to_message_id=reply_to_message_id,
        )
    
    def send_unauthorized(
        self,
        to_addr: str,
        original_subject: str,
    ) -> bool:
        """Send unauthorized sender notification."""
        subject = f"Re: {original_subject} — Access Required"
        
        body = """Hi,

Your email address is not yet authorized to use the AutoWork Takeoff Service.

To request access, please contact the service administrator.

---
AutoWork Takeoff Service
"""
        
        return self.send(
            to_addr=to_addr,
            subject=subject,
            body=body,
        )
