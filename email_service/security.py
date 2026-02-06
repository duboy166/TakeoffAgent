"""Security utilities for email service."""

import re
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Tuple, Optional
from .config import SecurityConfig

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for email processing."""
    
    def __init__(self, hourly_limit: int = 20, daily_limit: int = 100):
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self.hourly_counts = defaultdict(list)  # sender -> [timestamps]
        self.daily_counts = defaultdict(list)
    
    def check(self, sender: str) -> Tuple[bool, Optional[str]]:
        """
        Check if sender is within rate limits.
        
        Returns:
            Tuple of (allowed, reason_if_denied)
        """
        now = datetime.now()
        sender = sender.lower()
        
        # Clean old entries
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        self.hourly_counts[sender] = [
            t for t in self.hourly_counts[sender] if t > hour_ago
        ]
        self.daily_counts[sender] = [
            t for t in self.daily_counts[sender] if t > day_ago
        ]
        
        # Check limits
        if len(self.hourly_counts[sender]) >= self.hourly_limit:
            return False, f"Hourly limit ({self.hourly_limit}) exceeded"
        
        if len(self.daily_counts[sender]) >= self.daily_limit:
            return False, f"Daily limit ({self.daily_limit}) exceeded"
        
        return True, None
    
    def record(self, sender: str):
        """Record a processed email from sender."""
        now = datetime.now()
        sender = sender.lower()
        self.hourly_counts[sender].append(now)
        self.daily_counts[sender].append(now)


def extract_email_address(from_header: str) -> str:
    """Extract email address from From header."""
    # Handle formats like "Name <email@domain.com>" or just "email@domain.com"
    match = re.search(r'<([^>]+)>', from_header)
    if match:
        return match.group(1).lower()
    
    # Try to find bare email
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', from_header)
    if match:
        return match.group(0).lower()
    
    return from_header.lower()


def is_sender_allowed(from_header: str, config: SecurityConfig) -> bool:
    """
    Check if sender is allowed to use the service.
    
    Supports:
    - Exact email matches
    - Domain wildcards (e.g., "*@domain.com" or just "domain.com" in allowed_domains)
    - Prefix wildcards (e.g., "joe@*" matches joe@anything.com)
    """
    email = extract_email_address(from_header)
    local_part, domain = email.split('@') if '@' in email else (email, '')
    
    # Check exact sender matches
    for pattern in config.allowed_senders:
        pattern = pattern.lower()
        
        # Wildcard patterns
        if pattern.endswith('@*'):
            # Match local part (e.g., "joe@*" matches joe@anything.com)
            if local_part == pattern[:-2]:
                return True
        elif pattern.startswith('*@'):
            # Match domain (e.g., "*@company.com")
            if domain == pattern[2:]:
                return True
        elif pattern == email:
            # Exact match
            return True
    
    # Check allowed domains
    for allowed_domain in config.allowed_domains:
        allowed_domain = allowed_domain.lower()
        if domain == allowed_domain:
            return True
    
    return False


def validate_pdf(pdf_path: Path, max_mb: int = 25) -> Tuple[bool, Optional[str]]:
    """
    Validate a PDF file for safety.
    
    Checks:
    1. File size within limit
    2. Actually a PDF (magic bytes)
    3. Optional: virus scan if ClamAV available
    
    Returns:
        Tuple of (is_safe, reason_if_not)
    """
    # Check file exists
    if not pdf_path.exists():
        return False, "File not found"
    
    # Check size
    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        return False, f"File too large ({size_mb:.1f}MB > {max_mb}MB limit)"
    
    # Check magic bytes (PDF starts with %PDF-)
    try:
        with open(pdf_path, 'rb') as f:
            header = f.read(8)
            if not header.startswith(b'%PDF-'):
                return False, "Not a valid PDF file"
    except Exception as e:
        return False, f"Could not read file: {e}"
    
    # Optional: ClamAV scan
    try:
        result = subprocess.run(
            ['clamscan', '--no-summary', str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 1:  # Virus found
            return False, "Security scan failed"
    except FileNotFoundError:
        # ClamAV not installed, skip
        pass
    except subprocess.TimeoutExpired:
        logger.warning(f"ClamAV scan timed out for {pdf_path}")
    except Exception as e:
        logger.debug(f"ClamAV scan error: {e}")
    
    return True, None
