#!/usr/bin/env python3
"""
Direct Gmail API test to send an email
Tests the Gmail sending functionality in isolation
"""

import base64
import os
import pickle
from email.message import EmailMessage
from datetime import datetime

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def test_send_email():
    """Test sending an email directly via Gmail API"""
    print("🧪 Testing Direct Gmail API Email Sending...")
    print("=" * 60)
    
    # Step 1: Load credentials
    print("\n1️⃣ Loading Gmail credentials...")
    
    creds = None
    token_files = ['fresh_token.pickle', 'token.pickle']
    
    for token_file in token_files:
        if os.path.exists(token_file):
            try:
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)
                    print(f"✅ Loaded credentials from {token_file}")
                    break
            except Exception as e:
                print(f"⚠️ Could not load {token_file}: {e}")
                continue
    
    if not creds:
        print("❌ No valid credentials found!")
        print("💡 Run: python3 simple_oauth_setup.py")
        return False
    
    # Step 2: Refresh if needed
    if creds and creds.expired and creds.refresh_token:
        print("🔄 Refreshing expired credentials...")
        try:
            creds.refresh(Request())
            print("✅ Credentials refreshed successfully")
        except Exception as e:
            print(f"❌ Failed to refresh credentials: {e}")
            return False
    
    # Step 3: Build Gmail service
    print("\n2️⃣ Building Gmail service...")
    try:
        service = build('gmail', 'v1', credentials=creds)
        print("✅ Gmail service built successfully")
    except Exception as e:
        print(f"❌ Failed to build Gmail service: {e}")
        return False
    
    # Step 4: Create test email
    print("\n3️⃣ Creating test email...")
    
    # Email details
    to_email = "samuel.audette1@gmail.com"
    from_email = "info@800m.ca"  # This should be the authenticated Gmail account
    subject = f"Test Email from Agent Inbox - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    body = """Hello Samuel,

This is a test email sent directly via the Gmail API to verify the email sending functionality.

If you receive this email, it means:
✅ Gmail API authentication is working
✅ Email sending permissions are correctly configured
✅ The Gmail send scope is properly authorized

Test Details:
- Sent at: {timestamp}
- From: {from_email}
- To: {to_email}
- Using Gmail API v1

Best regards,
Agent Inbox Test System
""".format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        from_email=from_email,
        to_email=to_email
    )
    
    # Create the email message
    message = EmailMessage()
    message.set_content(body)
    message['To'] = to_email
    message['From'] = from_email
    message['Subject'] = subject
    
    print(f"📧 Email details:")
    print(f"   From: {from_email}")
    print(f"   To: {to_email}")
    print(f"   Subject: {subject}")
    print(f"   Body length: {len(body)} characters")
    
    # Step 5: Encode the message
    print("\n4️⃣ Encoding email message...")
    try:
        # Encode to base64url as required by Gmail API
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        print(f"✅ Message encoded successfully (size: {len(raw_message)} chars)")
    except Exception as e:
        print(f"❌ Failed to encode message: {e}")
        return False
    
    # Step 6: Send the email
    print("\n5️⃣ Sending email via Gmail API...")
    try:
        # Create the message body for Gmail API
        create_message = {'raw': raw_message}
        
        # Send the message
        send_result = service.users().messages().send(
            userId='me',
            body=create_message
        ).execute()
        
        message_id = send_result.get('id')
        thread_id = send_result.get('threadId')
        
        print(f"\n✅ EMAIL SENT SUCCESSFULLY!")
        print(f"   Message ID: {message_id}")
        print(f"   Thread ID: {thread_id}")
        print(f"\n📬 Check the inbox of {to_email} for the test email!")
        
        return True
        
    except HttpError as e:
        print(f"\n❌ Gmail API HTTP Error: {e}")
        print(f"   Status code: {e.resp.status if hasattr(e, 'resp') else 'Unknown'}")
        print(f"   Error details: {e.content if hasattr(e, 'content') else 'No details'}")
        
        # Common error explanations
        if hasattr(e, 'resp') and e.resp.status == 403:
            print("\n💡 Error 403 typically means:")
            print("   - Missing gmail.send scope")
            print("   - Need to re-authenticate with: python3 simple_oauth_setup.py")
        elif hasattr(e, 'resp') and e.resp.status == 400:
            print("\n💡 Error 400 typically means:")
            print("   - Invalid email format")
            print("   - Check the email addresses and message format")
            
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


def check_token_scopes():
    """Check what scopes are available in the token"""
    print("\n📋 Checking token scopes...")
    
    token_files = ['fresh_token.pickle', 'token.pickle']
    
    for token_file in token_files:
        if os.path.exists(token_file):
            try:
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)
                    if hasattr(creds, 'scopes') and creds.scopes:
                        print(f"\n✅ Scopes in {token_file}:")
                        for scope in creds.scopes:
                            print(f"   - {scope}")
                            if 'gmail.send' in scope:
                                print("     ✅ Has send permission!")
                    else:
                        print(f"⚠️ No scopes found in {token_file}")
            except Exception as e:
                print(f"⚠️ Could not read {token_file}: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Gmail API Direct Send Test")
    print("=" * 60)
    
    # First check what scopes we have
    check_token_scopes()
    
    print("\n" + "=" * 60)
    
    # Then try to send the email
    success = test_send_email()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("❌ Test failed. Please check the errors above.")
    print("=" * 60)
