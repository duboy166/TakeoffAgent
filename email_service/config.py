"""Configuration loader for email service."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class IMAPConfig:
    host: str
    port: int
    username: str
    password: str


@dataclass
class SMTPConfig:
    host: str
    port: int
    username: str
    password: str


@dataclass
class SecurityConfig:
    allowed_senders: list = field(default_factory=list)
    allowed_domains: list = field(default_factory=list)
    max_attachment_mb: int = 25
    rate_limit_hourly: int = 20
    rate_limit_daily: int = 100


@dataclass
class NotificationConfig:
    telegram_enabled: bool = False
    telegram_chat_id: str = ""


@dataclass
class ProcessingConfig:
    sla_minutes: int = 15
    output_format: str = "csv"
    parallel_ocr: bool = False


@dataclass
class EmailServiceConfig:
    imap: IMAPConfig
    smtp: SMTPConfig
    security: SecurityConfig
    notifications: NotificationConfig
    processing: ProcessingConfig


def load_config(config_path: Optional[str] = None) -> EmailServiceConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks in default locations.
        
    Returns:
        EmailServiceConfig object
    """
    if config_path is None:
        # Look in default locations
        search_paths = [
            Path(__file__).parent.parent / 'config' / 'email_config.yaml',
            Path.home() / '.autowork' / 'email_config.yaml',
            Path('/etc/autowork/email_config.yaml'),
        ]
        
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break
        else:
            raise FileNotFoundError(
                f"Config file not found. Searched: {[str(p) for p in search_paths]}"
            )
    
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    
    return EmailServiceConfig(
        imap=IMAPConfig(**raw.get('imap', {})),
        smtp=SMTPConfig(**raw.get('smtp', {})),
        security=SecurityConfig(**raw.get('security', {})),
        notifications=NotificationConfig(**raw.get('notifications', {})),
        processing=ProcessingConfig(**raw.get('processing', {})),
    )


def get_config() -> EmailServiceConfig:
    """Get the default configuration."""
    return load_config()
