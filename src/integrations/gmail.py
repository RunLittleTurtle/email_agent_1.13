"""
Gmail Integration Service
Handles Gmail API interactions for email fetching and sending
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle

logger = structlog.get_logger()


class GmailService:
    """
    Service for interacting with Gmail API
    """
    
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/gmail.modify'
    ]
    
    def __init__(self):
        """Initialize Gmail service"""
        self.service = None
        self.creds = None
        logger.info("Initializing Gmail service")
    
    async def authenticate(self):
        """
        Authenticate with Gmail API using existing OAuth credentials
        """
        try:
            # Try different token pickle files that might exist
            token_files = ['fresh_token.pickle', 'token.pickle']
            
            for token_file in token_files:
                if os.path.exists(token_file):
                    try:
                        with open(token_file, 'rb') as token:
                            self.creds = pickle.load(token)
                            logger.info(f"📁 Loaded Gmail credentials from {token_file}")
                            break
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load {token_file}: {e}")
                        continue
                    
            # If no valid credentials, try to refresh
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    logger.info("🔄 Refreshing expired Gmail credentials")
                    self.creds.refresh(Request())
                else:
                    logger.error("❌ No valid Gmail credentials found for sending")
                    return False
                        
            self.service = build('gmail', 'v1', credentials=self.creds)
            logger.info("✅ Gmail service authenticated successfully for sending")
            return True
            
        except Exception as e:
            logger.error(f"❌ Gmail authentication failed: {e}")
            return False
    
    async def fetch_emails(self, query: str = "is:unread") -> List[Dict[str, Any]]:
        """
        Fetch emails from Gmail
        
        Args:
            query: Gmail search query
            
        Returns:
            List of email dictionaries
        """
        logger.info(f"Fetching emails with query: {query}")
        
        # For testing, return mock data
        # TODO: Implement real Gmail fetching
        return []
    
    async def send_email(self, to: str, subject: str, body: str, reply_to: str = None) -> bool:
        """
        Send an email via Gmail API
        
        Args:
            to: Recipient email
            subject: Email subject
            body: Email body
            reply_to: Optional reply-to email ID
            
        Returns:
            Success status
        """
        logger.info(f"📧 Sending email via Gmail API to {to} with subject: {subject}")
        
        try:
            if not self.service:
                logger.error("Gmail service not authenticated")
                return False
                
            import base64
            from email.message import EmailMessage
            
            # Create email message (following official Gmail API documentation)
            message = EmailMessage()
            message.set_content(body)
            message['To'] = to
            message['Subject'] = subject
            message['From'] = 'info@800m.ca'  # Your Gmail account
            
            # If replying to an email, set In-Reply-To header for threading (RFC 2822)
            if reply_to:
                message['In-Reply-To'] = reply_to
                message['References'] = reply_to
            
            # Encode message (following official Gmail API pattern)
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Send via Gmail API
            send_result = self.service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            
            logger.info(f"✅ Email sent successfully! Message ID: {send_result.get('id')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email: {e}")
            return False
    
    async def mark_as_read(self, email_id: str) -> bool:
        """
        Mark an email as read
        
        Args:
            email_id: Gmail message ID
            
        Returns:
            Success status
        """
        logger.info(f"Marking email {email_id} as read")
        
        # For testing, just return success
        # TODO: Implement real marking as read
        return True
