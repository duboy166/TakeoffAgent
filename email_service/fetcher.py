"""IMAP email fetcher for the email service."""

import imaplib
import email
from email.message import Message as EmailMessage
from email.header import decode_header
from email.utils import parseaddr
from pathlib import Path
import tempfile
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .config import IMAPConfig

logger = logging.getLogger(__name__)


@dataclass
class EmailAttachment:
    """Represents an email attachment."""
    filename: str
    path: Path
    size_bytes: int
    content_type: str


@dataclass 
class IncomingEmail:
    """Represents an incoming email to process."""
    message_id: str
    imap_id: str  # IMAP sequence number for marking as read
    from_addr: str
    from_name: str
    subject: str
    body_text: str
    attachments: List[EmailAttachment] = field(default_factory=list)
    cloud_links: List[str] = field(default_factory=list)
    raw_message: Any = None


# Patterns for cloud storage links
CLOUD_LINK_PATTERNS = [
    r'https?://(?:www\.)?dropbox\.com/\S+',
    r'https?://(?:www\.)?dl\.dropboxusercontent\.com/\S+',
    r'https?://drive\.google\.com/\S+',
    r'https?://docs\.google\.com/\S+',
    r'https?://wetransfer\.com/downloads/\S+',
    r'https?://we\.tl/\S+',
    r'https?://(?:www\.)?onedrive\.live\.com/\S+',
    r'https?://1drv\.ms/\S+',
]


def extract_cloud_links(text: str) -> List[str]:
    """Extract cloud storage links from email body."""
    links = []
    for pattern in CLOUD_LINK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        links.extend(matches)
    return list(set(links))  # Dedupe


def decode_header_value(value: str) -> str:
    """Decode an email header value."""
    if not value:
        return ""
    
    decoded_parts = []
    for part, charset in decode_header(value):
        if isinstance(part, bytes):
            charset = charset or 'utf-8'
            try:
                decoded_parts.append(part.decode(charset, errors='replace'))
            except (LookupError, UnicodeDecodeError):
                decoded_parts.append(part.decode('utf-8', errors='replace'))
        else:
            decoded_parts.append(part)
    
    return ''.join(decoded_parts)


class EmailFetcher:
    """Fetches and parses incoming emails via IMAP."""
    
    def __init__(self, config: IMAPConfig):
        self.config = config
        self.conn: Optional[imaplib.IMAP4_SSL] = None
        self._temp_dirs: List[Path] = []
    
    def connect(self):
        """Connect to IMAP server."""
        logger.info(f"Connecting to IMAP: {self.config.host}")
        self.conn = imaplib.IMAP4_SSL(self.config.host, self.config.port)
        self.conn.login(self.config.username, self.config.password)
        logger.info("IMAP connection established")
    
    def disconnect(self):
        """Disconnect from IMAP server."""
        if self.conn:
            try:
                self.conn.logout()
            except Exception:
                pass
            self.conn = None
        
        # Cleanup temp directories
        for temp_dir in self._temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        self._temp_dirs = []
    
    def fetch_unread(self) -> List[IncomingEmail]:
        """
        Fetch unread emails with PDF attachments or cloud links.
        
        Returns:
            List of IncomingEmail objects
        """
        if not self.conn:
            self.connect()
        
        self.conn.select('INBOX')
        
        # Search for unread emails
        status, message_ids = self.conn.search(None, 'UNSEEN')
        if status != 'OK':
            logger.warning(f"IMAP search failed: {status}")
            return []
        
        emails = []
        for msg_id in message_ids[0].split():
            try:
                incoming = self._fetch_single(msg_id)
                if incoming and (incoming.attachments or incoming.cloud_links):
                    emails.append(incoming)
                elif incoming:
                    logger.info(f"Skipping email with no PDFs/links: {incoming.subject}")
            except Exception as e:
                logger.error(f"Error fetching email {msg_id}: {e}")
        
        return emails
    
    def _fetch_single(self, msg_id: bytes) -> Optional[IncomingEmail]:
        """Fetch and parse a single email."""
        status, msg_data = self.conn.fetch(msg_id, '(RFC822)')
        if status != 'OK':
            return None
        
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)
        
        # Parse headers
        subject = decode_header_value(msg.get('Subject', ''))
        from_header = msg.get('From', '')
        from_name, from_addr = parseaddr(from_header)
        from_name = decode_header_value(from_name)
        message_id = msg.get('Message-ID', str(msg_id))
        
        # Extract body text
        body_text = self._extract_body(msg)
        
        # Extract attachments
        attachments = self._extract_attachments(msg)
        
        # Extract cloud links from body
        cloud_links = extract_cloud_links(body_text)
        
        return IncomingEmail(
            message_id=message_id,
            imap_id=msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id),
            from_addr=from_addr,
            from_name=from_name,
            subject=subject,
            body_text=body_text,
            attachments=attachments,
            cloud_links=cloud_links,
            raw_message=msg,
        )
    
    def _extract_body(self, msg: EmailMessage) -> str:
        """Extract plain text body from email."""
        body_parts = []
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        try:
                            body_parts.append(payload.decode(charset, errors='replace'))
                        except (LookupError, UnicodeDecodeError):
                            body_parts.append(payload.decode('utf-8', errors='replace'))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                try:
                    body_parts.append(payload.decode(charset, errors='replace'))
                except (LookupError, UnicodeDecodeError):
                    body_parts.append(payload.decode('utf-8', errors='replace'))
        
        return '\n'.join(body_parts)
    
    def _extract_attachments(self, msg: EmailMessage) -> List[EmailAttachment]:
        """Extract PDF attachments to temporary files."""
        attachments = []
        
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = part.get('Content-Disposition', '')
            
            # Check if it's a PDF attachment
            is_pdf = (
                content_type == 'application/pdf' or 
                (content_disposition and '.pdf' in content_disposition.lower())
            )
            
            if not is_pdf:
                continue
            
            filename = part.get_filename()
            if filename:
                filename = decode_header_value(filename)
            else:
                filename = 'attachment.pdf'
            
            # Ensure .pdf extension
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'
            
            # Get payload
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            
            # Save to temp file
            temp_dir = Path(tempfile.mkdtemp(prefix='autowork_'))
            self._temp_dirs.append(temp_dir)
            
            # Sanitize filename
            safe_filename = re.sub(r'[^\w\-\.]', '_', filename)
            pdf_path = temp_dir / safe_filename
            pdf_path.write_bytes(payload)
            
            attachments.append(EmailAttachment(
                filename=filename,
                path=pdf_path,
                size_bytes=len(payload),
                content_type=content_type,
            ))
            
            logger.info(f"Extracted attachment: {filename} ({len(payload)/1024:.1f}KB)")
        
        return attachments
    
    def mark_as_read(self, msg_id: str):
        """Mark an email as read."""
        if self.conn:
            # msg_id from fetch is bytes like b'1', need to decode for store
            if isinstance(msg_id, bytes):
                msg_id = msg_id.decode()
            # Remove angle brackets if it's a Message-ID header value
            if msg_id.startswith('<'):
                # Can't use Message-ID for STORE, would need to search first
                # For now, just skip marking - email will stay unread
                return
            self.conn.store(msg_id, '+FLAGS', '\\Seen')
    
    def mark_as_processed(self, msg_id: str, label: str = 'Processed'):
        """Mark email as processed (add label if supported)."""
        self.mark_as_read(msg_id)
        # Gmail supports labels via IMAP, but basic implementation just marks as read
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
