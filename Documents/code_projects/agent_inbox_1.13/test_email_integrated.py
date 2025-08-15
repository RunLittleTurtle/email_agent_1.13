#!/usr/bin/env python3
"""
Integrated test for Gmail sending using existing project modules
Tests email sending using the EmailSenderAgent in isolation
"""

import asyncio
import os
from datetime import datetime
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.email_sender import EmailSenderAgent
from src.models.state import AgentState, EmailMessage


async def test_email_sending():
    """Test email sending using the EmailSenderAgent directly"""
    print("=" * 60)
    print("🧪 Testing EmailSenderAgent Direct Email Send")
    print("=" * 60)
    
    # Create a test email message
    test_email = EmailMessage(
        id="test-message-001",
        sender="test@example.com",
        recipients=["samuel.audette1@gmail.com"],
        subject="Test Email from Agent Inbox",
        body="This is a test email to verify Gmail API sending.",
        timestamp=datetime.now()
    )
    
    # Create a test state with approved draft
    test_state = AgentState()
    test_state.email = test_email
    test_state.draft_response = f"""Hello Samuel,

This is a test email sent directly through the EmailSenderAgent to verify that the Gmail API integration is working correctly.

Test Details:
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Test ID: test-message-001
- Sender: EmailSenderAgent
- Purpose: Verify Gmail API sending functionality

If you receive this email, it confirms:
✅ Gmail API authentication is working
✅ The email sending flow is properly configured
✅ OAuth scopes include gmail.send permission

Best regards,
Agent Inbox Test System"""
    
    print("\n📧 Test Email Details:")
    print(f"   To: {test_email.recipients[0]}")
    print(f"   Subject: {test_email.subject}")
    print(f"   Draft length: {len(test_state.draft_response)} chars")
    
    # Initialize and test the EmailSenderAgent
    print("\n🚀 Initializing EmailSenderAgent...")
    try:
        email_sender = EmailSenderAgent()
        
        # Check if Gmail service was initialized
        if email_sender.gmail_service:
            print("✅ Gmail service initialized successfully")
        else:
            print("❌ Gmail service initialization failed")
            print("💡 Make sure you have valid OAuth tokens (fresh_token.pickle or token.pickle)")
            return False
        
        # Process the email (send it)
        print("\n📤 Sending email...")
        result_state = await email_sender.process(test_state)
        
        # Check results
        print("\n📊 Results:")
        if result_state.status == "completed":
            print("✅ Email sent successfully!")
            if "email_sent" in result_state.response_metadata:
                sent_info = result_state.response_metadata["email_sent"]
                print(f"   To: {sent_info.get('to', 'N/A')}")
                print(f"   Subject: {sent_info.get('subject', 'N/A')}")
                print(f"   Status: {sent_info.get('status', 'N/A')}")
        else:
            print("❌ Email sending failed")
            
        # Check for errors
        if result_state.errors:
            print("\n⚠️ Errors encountered:")
            for error in result_state.errors:
                print(f"   - {error}")
                
        # Check messages
        if result_state.messages:
            print("\n💬 Messages:")
            for msg in result_state.messages:
                if msg.role == "system":
                    print(f"   - {msg.content}")
                    
        return result_state.status == "completed"
        
    except Exception as e:
        print(f"\n❌ Exception occurred: {type(e).__name__}: {e}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


async def main():
    """Main test runner"""
    success = await test_email_sending()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Test completed successfully!")
        print("📬 Check samuel.audette1@gmail.com for the test email")
    else:
        print("❌ Test failed")
        print("💡 Troubleshooting tips:")
        print("   1. Ensure OAuth tokens exist (fresh_token.pickle or token.pickle)")
        print("   2. Run: python3 simple_oauth_setup.py to re-authenticate")
        print("   3. Check that gmail.send scope is authorized")
        print("   4. Verify the 'From' email matches the authenticated account")
    print("=" * 60)


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
