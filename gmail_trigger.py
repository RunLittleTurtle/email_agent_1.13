"""
Gmail Trigger Script
Monitors Gmail for new emails and triggers the multi-agent workflow
"""

import os
import asyncio
import base64
import email
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import structlog
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.models.state import EmailMessage, AgentState
from src.graph.workflow import create_workflow

logger = structlog.get_logger()

class GmailMonitor:
    """
    Gmail monitoring service that fetches emails and triggers workflows
    """
    
    def __init__(self):
        """Initialize Gmail monitor with authentication"""
        self.service = None
        self.creds = None
        self.target_email = "info@800m.ca"
        
    async def authenticate(self) -> bool:
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
                            logger.info(f"📁 Loaded credentials from {token_file}")
                            break
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load {token_file}: {e}")
                        continue
                    
            # If no valid credentials, try to use the GoogleAuthHelper approach
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    logger.info("🔄 Refreshing expired credentials")
                    self.creds.refresh(Request())
                else:
                    logger.error("❌ No valid Google credentials found.")
                    logger.info("💡 Try running: python simple_oauth_setup.py")
                    return False
                        
            self.service = build('gmail', 'v1', credentials=self.creds)
            logger.info("✅ Gmail API authenticated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Gmail authentication failed: {e}")
            return False
    
    async def get_recent_emails(self, hours_back: int = 1) -> List[Dict[str, Any]]:
        """
        Get recent emails from the past specified hours
        
        Args:
            hours_back: How many hours back to search for emails
            
        Returns:
            List of email data dictionaries
        """
        try:
            if not self.service:
                logger.error("Gmail service not authenticated")
                return []
                
            # Calculate time threshold 
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Gmail query to get recent emails to our target address
            query = f"to:{self.target_email} after:{cutoff_time.strftime('%Y/%m/%d')}"
            
            logger.info(f"🔍 Searching for emails with query: {query}")
            
            # Get message list
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=10
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                logger.info("📭 No recent emails found")
                return []
                
            logger.info(f"📧 Found {len(messages)} recent emails")
            
            # Get full message details
            emails = []
            for message in messages:
                try:
                    msg = self.service.users().messages().get(
                        userId='me',
                        id=message['id'],
                        format='full'
                    ).execute()
                    
                    emails.append(self._parse_gmail_message(msg))
                    
                except HttpError as e:
                    logger.warning(f"⚠️ Could not fetch message {message['id']}: {e}")
                    continue
                    
            return emails
            
        except HttpError as e:
            logger.error(f"❌ Gmail API error: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching emails: {e}")
            return []
    
    def _parse_gmail_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a Gmail API message into our EmailMessage format
        
        Args:
            msg: Gmail API message object
            
        Returns:
            Email data dictionary
        """
        headers = {h['name'].lower(): h['value'] for h in msg['payload'].get('headers', [])}
        
        # Extract body
        body = ""
        if 'parts' in msg['payload']:
            for part in msg['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
        elif msg['payload']['body'].get('data'):
            body = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8')
        
        # Convert timestamp
        timestamp = datetime.fromtimestamp(int(msg['internalDate']) / 1000)
        
        return {
            'id': msg['id'],
            'sender': headers.get('from', 'Unknown'),
            'recipients': [headers.get('to', self.target_email)],
            'subject': headers.get('subject', 'No Subject'),
            'body': body.strip(),
            'timestamp': timestamp,
            'thread_id': msg.get('threadId'),
            'labels': msg.get('labelIds', [])
        }
    
    async def process_email_with_workflow(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an email through the LangGraph dev server API so it appears in Agent Inbox
        
        Args:
            email_data: Email data dictionary
            
        Returns:
            API response with thread and run info
        """
        logger.info(f"🚀 Processing email via LangGraph API: {email_data['subject']}")
        
        import httpx
        
        # LangGraph dev server endpoint
        LANGGRAPH_API = "http://127.0.0.1:2024"
        
        try:
            async with httpx.AsyncClient() as client:
                # Create thread
                thread_response = await client.post(
                    f"{LANGGRAPH_API}/threads",
                    json={"metadata": {"source": "gmail_trigger"}}
                )
                
                if thread_response.status_code != 200:
                    logger.error(f"Failed to create thread: {thread_response.status_code}")
                    return {"error": f"Thread creation failed: {thread_response.text}"}
                
                thread_data = thread_response.json()
                thread_id = thread_data["thread_id"]
                logger.info(f"📋 Created thread: {thread_id}")
                
                # Convert email data to proper format for workflow
                email_input = {
                    "id": email_data['id'],
                    "subject": email_data['subject'],
                    "body": email_data['body'],
                    "sender": email_data['sender'],
                    "recipients": email_data['recipients'],
                    "timestamp": email_data['timestamp'].isoformat(),
                    "attachments": [],
                    "thread_id": email_data.get('thread_id')
                }
                
                # Start workflow run
                run_response = await client.post(
                    f"{LANGGRAPH_API}/threads/{thread_id}/runs",
                    json={
                        "assistant_id": "email_agent",
                        "input": {
                            "email": email_input,
                            "messages": []
                        }
                    }
                )
                
                if run_response.status_code != 200:
                    logger.error(f"Failed to start workflow: {run_response.status_code}")
                    return {"error": f"Workflow start failed: {run_response.text}"}
                
                run_data = run_response.json()
                run_id = run_data["run_id"]
                
                logger.info(f"✅ Workflow started via LangGraph API")
                logger.info(f"   Thread ID: {thread_id}")
                logger.info(f"   Run ID: {run_id}")
                logger.info(f"   🌐 Check Agent Inbox at: http://localhost:3000")
                
                return {
                    "success": True,
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "agent_inbox_url": "http://localhost:3000"
                }
                
        except httpx.RequestError as e:
            logger.error(f"LangGraph API connection error: {e}")
            logger.error("💡 Make sure LangGraph dev server is running: python cli.py langgraph")
            return {"error": f"API connection failed: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": f"Unexpected error: {e}"}
    
    async def mark_email_processed(self, email_id: str) -> bool:
        """
        Mark an email as processed (add a custom label)
        
        Args:
            email_id: Gmail message ID
            
        Returns:
            Success status
        """
        try:
            if not self.service:
                return False
                
            # Add a custom label to mark as processed
            # Note: You might need to create this label first in Gmail
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={
                    'addLabelIds': [],
                    'removeLabelIds': ['UNREAD']  # Mark as read
                }
            ).execute()
            
            logger.info(f"✅ Marked email {email_id} as processed")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Could not mark email as processed: {e}")
            return False

async def main():
    """
    Main function to monitor Gmail and trigger workflows
    """
    print("🚀 Starting Gmail Monitor...")
    
    monitor = GmailMonitor()
    
    # Authenticate with Gmail
    if not await monitor.authenticate():
        print("❌ Failed to authenticate with Gmail")
        return
    
    print(f"📧 Monitoring Gmail for emails to: {monitor.target_email}")
    
    # Get recent emails
    emails = await monitor.get_recent_emails(hours_back=24)  # Look back 24 hours
    
    if not emails:
        print("📭 No recent emails found to process")
        return
    
    print(f"📨 Found {len(emails)} emails to process:")
    
    for email_data in emails:
        print(f"\n📧 Email: {email_data['subject']}")
        print(f"   From: {email_data['sender']}")
        print(f"   Time: {email_data['timestamp']}")
        print(f"   Preview: {email_data['body'][:100]}...")
        
        try:
            # Process through LangGraph API
            result = await monitor.process_email_with_workflow(email_data)
            
            if result.get('success'):
                print(f"✅ Processed successfully via LangGraph API")
                print(f"   Thread ID: {result.get('thread_id', 'N/A')}")
                print(f"   Run ID: {result.get('run_id', 'N/A')}")
                print(f"   🌐 Check Agent Inbox: {result.get('agent_inbox_url', 'http://localhost:3000')}")
                
                # Mark as processed
                await monitor.mark_email_processed(email_data['id'])
            else:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ Error processing email: {e}")
            logger.error(f"Workflow error for email {email_data['id']}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
