"""
Integrations package for Ambient Email Agent
Contains external service integrations
"""

from .gmail import GmailService

__all__ = [
    "GmailService",
]
