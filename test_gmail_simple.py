#!/usr/bin/env python3
"""
Simple Gmail API test - minimal dependencies
Tests just the core Gmail sending functionality
"""

import base64
import os
import pickle
from email.message import EmailMessage
from datetime import datetime


def test_gmail_send():
    """Simple test to send email via Gmail API"""
    print("=" * 60)
    print("🧪 Simple Gmail API Test")
    print("=" * 60)
    
    # Step 1: Try to load credentials from pickle files
    print("\n1️⃣ Loading credentials...")
    creds = None
    token_files = ['fresh_token.pickle', 'token.pickle']
    
    for token_file in token_files:
        if os.path.exists(token_file):
            print(f"   Found {token_file}")
            try:
                with open(token_file, 'rb') as f:
                    creds = pickle.load(f)
                print(f"   ✅ Loaded credentials from {token_file}")
                break
            except Exception as e:
                print(f"   ❌ Error loading {token_file}: {e}")
    
    if not creds:
        print("\n❌ No valid credentials found!")
        print("💡 Please run: python3 simple_oauth_setup.py")
        return
    
    # Step 2: Try to import Google API libraries
    print("\n2️⃣ Importing Google API libraries...")
    try:
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        print("   ✅ Google API libraries imported successfully")
    except ImportError as e:
        print(f"   ❌ Missing Google API libraries: {e}")
        print("\n💡 To fix this, you need to install Google API dependencies:")
        print("   Option 1: Add to requirements.txt and install:")
        print("      google-auth")
        print("      google-auth-oauthlib")
        print("      google-auth-httplib2")
        print("      google-api-python-client")
        print("\n   Option 2: Create a virtual environment:")
        print("      python3 -m venv venv")
        print("      source venv/bin/activate")
        print("      pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        return
    
    # Step 3: Refresh credentials if needed
    if creds.expired and creds.refresh_token:
        print("\n3️⃣ Refreshing expired credentials...")
        try:
            creds.refresh(Request())
            print("   ✅ Credentials refreshed")
        except Exception as e:
            print(f"   ❌ Failed to refresh: {e}")
            return
    
    # Step 4: Build Gmail service
    print("\n4️⃣ Building Gmail service...")
    try:
        service = build('gmail', 'v1', credentials=creds)
        print("   ✅ Gmail service created")
    except Exception as e:
        print(f"   ❌ Failed to build service: {e}")
        return
    
    # Step 5: Create test email
    print("\n5️⃣ Creating test email...")
    
    to_email = "samuel.audette1@gmail.com"
    from_email = "info@800m.ca"
    subject = f"Test from Agent Inbox - {datetime.now().strftime('%H:%M:%S')}"
    body = f"""Hello Samuel,

This is a test email sent via Gmail API at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

If you receive this email, the Gmail API integration is working correctly!

Test performed by: test_gmail_simple.py
Python environment: {os.environ.get('VIRTUAL_ENV', 'System Python')}

Best regards,
Agent Inbox Test"""
    
    # Create email message
    msg = EmailMessage()
    msg.set_content(body)
    msg['To'] = to_email
    msg['From'] = from_email
    msg['Subject'] = subject
    
    print(f"   To: {to_email}")
    print(f"   From: {from_email}")
    print(f"   Subject: {subject}")
    
    # Step 6: Send email
    print("\n6️⃣ Sending email...")
    try:
        # Encode the message
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        
        # Send it
        result = service.users().messages().send(
            userId='me',
            body={'raw': raw}
        ).execute()
        
        print(f"\n✅ SUCCESS! Email sent!")
        print(f"   Message ID: {result['id']}")
        print(f"   Thread ID: {result.get('threadId', 'N/A')}")
        print(f"\n📬 Check {to_email} inbox!")
        
    except HttpError as e:
        print(f"\n❌ HTTP Error {e.resp.status}: {e}")
        if e.resp.status == 403:
            print("   💡 Error 403: Missing gmail.send scope")
            print("   Run: python3 simple_oauth_setup.py")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_gmail_send()
