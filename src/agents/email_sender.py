"""
Email Sender Agent
Dedicated agent for sending email responses via Gmail API
Handles authentication, formatting, and sending with robust error handling and logging
"""

import os
import base64
import asyncio
from email.message import EmailMessage
from typing import Dict, Any, Optional
from datetime import datetime

from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from langsmith import traceable
from langgraph.runtime import Runtime
from src.agents.base_agent import BaseAgent
from src.models.state import AgentState
from src.models.context import RuntimeContext
from src.utils.google_auth import GoogleAuthHelper


class EmailSenderAgent(BaseAgent):
    """Dedicated agent for sending email responses via Gmail API"""

    def __init__(self):
        super().__init__(
            name="email_sender",
            model="gpt-4o",  # We'll use this for any future email formatting logic
            temperature=0.0
        )
        self.gmail_service = None
        self._initialize_gmail_service()

    def _initialize_gmail_service(self):
        """Initialize Gmail service with authentication"""
        try:
            # Gmail API scopes for sending emails
            scopes = [
                'https://www.googleapis.com/auth/gmail.send',
                'https://www.googleapis.com/auth/gmail.readonly'
            ]

            # Try different token files
            token_files = ['fresh_token.pickle', 'token.pickle']

            for token_file in token_files:
                if os.path.exists(token_file):
                    creds = GoogleAuthHelper.get_credentials(scopes, token_file)
                    if creds:
                        self.gmail_service = build('gmail', 'v1', credentials=creds)
                        self.logger.info(f"Gmail service initialized successfully using {token_file}")
                        return

            self.logger.error("Could not initialize Gmail service - no valid credentials found")

        except Exception as e:
            self.logger.error(f"Failed to initialize Gmail service: {e}")
            self.gmail_service = None

    @traceable(name="email_sender_process", tags=["agent", "email_sender"])
    async def process(self, state: AgentState, runtime: Optional[Runtime[RuntimeContext]] = None) -> Dict[str, Any]:
        """
        Send the approved email response via Gmail API

        Args:
            state: Current workflow state with email and draft_response

        Returns:
            Updated state with sending status
        """

        self.logger.info("📮 Email Sender Agent - Starting email send process")

        try:
            # Validate required data
            if not state.draft_response:
                error_msg = "No draft response available for sending"
                self.logger.error(error_msg)
                return {
                    "error_messages": [error_msg]
                }

            if not state.email:
                error_msg = "No original email context for reply"
                self.logger.error(error_msg)
                return {
                    "error_messages": [error_msg]
                }

            # Check Gmail service
            if not self.gmail_service:
                error_msg = "Gmail service not initialized"
                self.logger.error(error_msg)
                return {
                    "error_messages": [error_msg]
                }

            # Prepare email data
            email_data = self._prepare_email_data(state)
            self.logger.info(f"📧 Prepared email data: To={email_data['to']}, Subject={email_data['subject']}")

            # Send email (wrap in asyncio.to_thread for blocking Gmail API)
            success = await asyncio.to_thread(self._send_via_gmail_api, email_data)

            if success:
                success_msg = f"✅ Email sent successfully to {email_data['to']}"
                self.logger.info(success_msg)
                
                # Create AI message for success
                success_message = self.create_ai_message(success_msg)
                
                # Return state updates
                return {
                    "messages": [success_message],
                    "status": "completed",
                    "response_metadata": {
                        "email_sent": {
                            "to": email_data['to'],
                            "subject": email_data['subject'],
                            "sent_at": datetime.now().isoformat(),
                            "status": "success"
                        }
                    }
                }
            else:
                error_msg = f"❌ Failed to send email to {email_data['to']}"
                self.logger.error(error_msg)
                return {
                    "error_messages": [error_msg]
                }

        except Exception as e:
            error_msg = f"Email sending failed with exception: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "error_messages": [error_msg]
            }

    def _prepare_email_data(self, state: AgentState) -> Dict[str, Any]:
        """
        Prepare email data in the correct format for Gmail API

        Args:
            state: Current workflow state

        Returns:
            Dictionary with email data ready for Gmail API
        """

        # Determine reply subject (add "Re:" if not already present)
        original_subject = state.email.subject
        if original_subject.startswith("Re:"):
            reply_subject = original_subject
        else:
            reply_subject = f"Re: {original_subject}"

        # Use message_id for proper Gmail threading (e.g., <CAG41pbv...@mail.gmail.com>)
        # If message_id is not available, fall back to email.id
        reply_to_id = state.email.message_id or state.email.id
        
        self.logger.info(f"📧 Reply threading: Using message_id='{state.email.message_id}' or fallback id='{state.email.id}'")

        return {
            "to": state.email.sender,  # Reply to original sender
            "subject": reply_subject,
            "body": state.draft_response,
            "reply_to_id": reply_to_id,  # Gmail Message-ID for proper threading
            "thread_id": state.email.thread_id,  # Gmail thread ID for proper threading
            "from_address": "info@800m.ca"  # Your Gmail account
        }

    def _send_via_gmail_api(self, email_data: Dict[str, Any]) -> bool:
        """
        Send email using Gmail API with proper error handling

        Args:
            email_data: Prepared email data

        Returns:
            True if sent successfully, False otherwise
        """

        try:
            self.logger.info(f"🚀 Sending email via Gmail API...")
            self.logger.info(f"   To: {email_data['to']}")
            self.logger.info(f"   Subject: {email_data['subject']}")
            self.logger.info(f"   Body length: {len(email_data['body'])} chars")
            self.logger.info(f"   From: {email_data['from_address']}")
            self.logger.info(f"   Reply-To ID: {email_data.get('reply_to_id', 'None')}")

            # Verify Gmail service is initialized
            if not self.gmail_service:
                self.logger.error("❌ Gmail service is None - not initialized!")
                return False

            # Create email message following official Gmail API documentation
            message = EmailMessage()
            message.set_content(email_data['body'])
            message['To'] = email_data['to']
            message['Subject'] = email_data['subject']
            message['From'] = email_data['from_address']

            # Set reply headers for email threading (RFC 2822)
            if email_data.get('reply_to_id'):
                message['In-Reply-To'] = email_data['reply_to_id']
                message['References'] = email_data['reply_to_id']

            # Encode message (following official Gmail API pattern)
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

            self.logger.info(f"📦 Message encoded successfully, size: {len(encoded_message)} chars")

            # Create the message body for Gmail API with thread ID for proper threading
            create_message = {'raw': encoded_message}
            
            # CRITICAL: Add threadId to message metadata for proper Gmail threading
            if email_data.get('thread_id'):
                create_message['threadId'] = email_data['thread_id']
                self.logger.info(f"🧵 Using Gmail thread ID: {email_data['thread_id']}")
            else:
                self.logger.warning("⚠️ No thread_id available - email will start new thread")
                
            self.logger.info(f"📤 Calling Gmail API send with body: {list(create_message.keys())}")

            # Send via Gmail API
            send_result = self.gmail_service.users().messages().send(
                userId='me',
                body=create_message
            ).execute()

            message_id = send_result.get('id')
            self.logger.info(f"✅ Email sent successfully! Gmail Message ID: {message_id}")
            self.logger.info(f"   Full send result: {send_result}")

            return True

        except HttpError as e:
            self.logger.error(f"❌ Gmail API HTTP error: {e}")
            self.logger.error(f"   Status code: {e.resp.status if hasattr(e, 'resp') else 'Unknown'}")
            self.logger.error(f"   Error details: {e.content if hasattr(e, 'content') else 'No details'}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Gmail API send failed with exception: {type(e).__name__}: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return False
