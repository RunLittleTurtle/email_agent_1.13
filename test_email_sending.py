"""
Test script to verify Gmail email sending functionality
"""

import asyncio
import os
from src.integrations.gmail import GmailService

async def test_email_sending():
    """Test the Gmail email sending functionality"""
    print("🧪 Testing Gmail Email Sending...")
    
    # Initialize Gmail service
    gmail_service = GmailService()
    
    # Test authentication
    print("🔐 Testing Gmail authentication...")
    auth_success = await gmail_service.authenticate()
    
    if not auth_success:
        print("❌ Gmail authentication failed!")
        return
    
    print("✅ Gmail authentication successful!")
    
    # Test email sending
    print("📧 Testing email sending...")
    
    test_recipient = "samuel.audette1@gmail.com"  # Your email for testing
    test_subject = "Test Email from Agent Inbox"
    test_body = """Hello!

This is a test email sent from the Agent Inbox Gmail integration to verify that email sending is working properly.

If you receive this email, the Gmail API sending functionality is working correctly!

Best regards,
Agent Inbox System
"""

    success = await gmail_service.send_email(
        to=test_recipient,
        subject=test_subject,
        body=test_body
    )
    
    if success:
        print(f"✅ Test email sent successfully to {test_recipient}")
        print("📬 Check your inbox for the test email!")
    else:
        print(f"❌ Failed to send test email to {test_recipient}")

if __name__ == "__main__":
    asyncio.run(test_email_sending())
