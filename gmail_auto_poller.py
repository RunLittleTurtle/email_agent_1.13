#!/usr/bin/env python3
"""
Gmail Auto Poller - Automated Email Processing
Checks Gmail every few minutes for new emails and triggers the workflow automatically.
Designed to run as a cronjob or background service.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Set, Dict, Any, List

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.google_auth import GoogleAuthHelper
    from googleapiclient.discovery import build
    import httpx
    import base64
except ImportError as e:
    print(f"ERROR: Missing dependencies: {e}")
    print("Make sure to activate virtual environment and install requirements")
    sys.exit(1)

# Configuration
PROCESSED_EMAILS_FILE = project_root / "processed_emails.json"
LOG_FILE = project_root / "gmail_poller.log"
LANGGRAPH_API = "http://127.0.0.1:2024"
MAX_EMAILS_TO_CHECK = 10

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GmailAutoPoller:
    """Automatically polls Gmail and triggers workflow for new emails"""
    
    def __init__(self):
        self.processed_emails: Set[str] = self._load_processed_emails()
        self.gmail_service = None
        
    def _load_processed_emails(self) -> Set[str]:
        """Load list of already processed email IDs"""
        if PROCESSED_EMAILS_FILE.exists():
            try:
                with open(PROCESSED_EMAILS_FILE, 'r') as f:
                    data = json.load(f)
                    return set(data.get('processed_email_ids', []))
            except Exception as e:
                logger.warning(f"Could not load processed emails: {e}")
        return set()
    
    def _save_processed_emails(self):
        """Save list of processed email IDs"""
        try:
            data = {
                'processed_email_ids': list(self.processed_emails),
                'last_updated': datetime.now().isoformat()
            }
            with open(PROCESSED_EMAILS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save processed emails: {e}")
    
    def _authenticate_gmail(self) -> bool:
        """Authenticate with Gmail API"""
        try:
            scopes = [
                'https://www.googleapis.com/auth/gmail.readonly',
                'https://www.googleapis.com/auth/gmail.send'
            ]
            
            # Try different token files
            token_files = ['fresh_token.pickle', 'token.pickle']
            
            for token_file in token_files:
                if os.path.exists(token_file):
                    creds = GoogleAuthHelper.get_credentials(scopes, token_file)
                    if creds:
                        self.gmail_service = build('gmail', 'v1', credentials=creds)
                        logger.info(f"✅ Authenticated using {token_file}")
                        return True
            
            logger.error("❌ Could not authenticate with Gmail")
            return False
            
        except Exception as e:
            logger.error(f"Gmail authentication failed: {e}")
            return False
    
    def _extract_email_data(self, message_id: str) -> Dict[str, Any]:
        """Extract email data from Gmail API message"""
        try:
            msg = self.gmail_service.users().messages().get(
                userId='me', 
                id=message_id,
                format='full'
            ).execute()
            
            # Extract headers
            headers = {h['name']: h['value'] for h in msg['payload']['headers']}
            
            sender = headers.get('From', 'Unknown')
            subject = headers.get('Subject', 'No Subject')
            date_str = headers.get('Date', '')
            
            # Extract recipients
            recipients = []
            for header_name in ['To', 'Cc', 'Bcc']:
                header_value = headers.get(header_name, '')
                if header_value:
                    recipients.extend([addr.strip() for addr in header_value.split(',')])
            
            if not recipients:
                recipients = ['info@800m.ca']  # Default recipient
            
            # Extract body
            body = ""
            if 'parts' in msg['payload']:
                for part in msg['payload']['parts']:
                    if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                        body_data = part['body']['data']
                        body = base64.urlsafe_b64decode(body_data).decode('utf-8')
                        break
            elif msg['payload']['mimeType'] == 'text/plain' and 'data' in msg['payload']['body']:
                body_data = msg['payload']['body']['data']
                body = base64.urlsafe_b64decode(body_data).decode('utf-8')
            
            return {
                'id': message_id,
                'sender': sender,
                'subject': subject,
                'body': body,
                'recipients': recipients,
                'timestamp': datetime.now().isoformat(),
                'attachments': []
            }
            
        except Exception as e:
            logger.error(f"Error extracting email data for {message_id}: {e}")
            return None
    
    async def _send_to_workflow(self, email_data: Dict[str, Any]) -> bool:
        """Send email to LangGraph workflow"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Create thread
                thread_response = await client.post(
                    f"{LANGGRAPH_API}/threads",
                    json={"metadata": {"source": "gmail_auto_poller"}}
                )
                
                if thread_response.status_code != 200:
                    logger.error(f"Failed to create thread: {thread_response.text}")
                    return False
                
                thread_data = thread_response.json()
                thread_id = thread_data["thread_id"]
                
                # Start workflow
                run_response = await client.post(
                    f"{LANGGRAPH_API}/threads/{thread_id}/runs",
                    json={
                        "assistant_id": "email_agent",
                        "input": {"email": email_data},
                        "stream_mode": "values"
                    }
                )
                
                if run_response.status_code != 200:
                    logger.error(f"Failed to start workflow: {run_response.text}")
                    return False
                
                run_data = run_response.json()
                logger.info(f"✅ Email sent to workflow - Thread: {thread_id}, Run: {run_data['run_id']}")
                return True
                
        except Exception as e:
            logger.error(f"Error sending email to workflow: {e}")
            return False
    
    async def check_and_process_new_emails(self):
        """Main method to check for new emails and process them"""
        logger.info("🔍 Checking for new emails...")
        
        if not self._authenticate_gmail():
            return
        
        try:
            # Fetch recent emails
            results = self.gmail_service.users().messages().list(
                userId='me',
                maxResults=MAX_EMAILS_TO_CHECK,
                q='in:inbox'
            ).execute()
            
            messages = results.get('messages', [])
            logger.info(f"📧 Found {len(messages)} emails in inbox")
            
            # Find new emails
            new_emails = [msg for msg in messages if msg['id'] not in self.processed_emails]
            
            if not new_emails:
                logger.info("✅ No new emails to process")
                return
            
            logger.info(f"🆕 Found {len(new_emails)} new email(s) to process")
            
            # Process each new email
            for message in new_emails:
                email_id = message['id']
                logger.info(f"📬 Processing email {email_id}")
                
                # Extract email data
                email_data = self._extract_email_data(email_id)
                if not email_data:
                    logger.error(f"❌ Could not extract data for email {email_id}")
                    continue
                
                logger.info(f"📧 Email from: {email_data['sender']}, Subject: {email_data['subject']}")
                
                # Send to workflow
                if await self._send_to_workflow(email_data):
                    # Mark as processed
                    self.processed_emails.add(email_id)
                    logger.info(f"✅ Email {email_id} processed successfully")
                else:
                    logger.error(f"❌ Failed to process email {email_id}")
            
            # Save processed emails list
            self._save_processed_emails()
            logger.info(f"💾 Processed {len(new_emails)} new emails")
            
        except Exception as e:
            logger.error(f"Error checking emails: {e}")


async def main():
    """Main entry point"""
    logger.info("🚀 Gmail Auto Poller starting...")
    
    # Check if LangGraph server is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(LANGGRAPH_API)
            if response.status_code not in [200, 404]:
                logger.error(f"❌ LangGraph server not responding at {LANGGRAPH_API}")
                return
    except Exception as e:
        logger.error(f"❌ Cannot connect to LangGraph server: {e}")
        logger.error("Make sure to start: python cli.py langgraph")
        return
    
    # Initialize and run poller
    poller = GmailAutoPoller()
    await poller.check_and_process_new_emails()
    logger.info("🏁 Gmail Auto Poller finished")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️  Gmail Auto Poller stopped by user")
    except Exception as e:
        logger.error(f"💥 Gmail Auto Poller crashed: {e}")
        sys.exit(1)
