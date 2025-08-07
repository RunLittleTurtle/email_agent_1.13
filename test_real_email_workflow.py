#!/usr/bin/env python3
"""
Test the complete email workflow with REAL email addresses
This will actually send an email to verify the full flow works
"""

import asyncio
import httpx
import json
import time
from datetime import datetime


async def test_real_email_workflow():
    """Test workflow with real email addresses that Gmail can actually send to"""
    print("=" * 60)
    print("🧪 Testing Agent Inbox Email Workflow with REAL Email")
    print("=" * 60)
    
    # Configuration
    api_base_url = "http://127.0.0.1:2024"
    assistant_id = "email_agent"
    
    # Use REAL email addresses
    real_recipient = "samuel.audette1@gmail.com"  # Where the reply will be sent
    
    # Test email data - simulating an email FROM a real address
    test_email = {
        "email": {
            "id": f"real-test-{int(time.time())}",
            "sender": real_recipient,  # Use real email as sender
            "recipients": ["info@800m.ca"],  # Your inbox
            "subject": f"Real Test Email - {datetime.now().strftime('%H:%M:%S')}",
            "body": "Hi, I need help setting up a meeting for next week. Can you check my calendar and suggest some times?",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    print(f"\n📧 Test Configuration:")
    print(f"   From (simulated): {real_recipient}")
    print(f"   To (your inbox): info@800m.ca")
    print(f"   Reply will go to: {real_recipient}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Create thread
        print("\n1️⃣ Creating new thread...")
        thread_response = await client.post(
            f"{api_base_url}/threads",
            json={"metadata": {"test": "real_email_test"}}
        )
        
        if thread_response.status_code != 200:
            print(f"❌ Failed to create thread: {thread_response.text}")
            return False
            
        thread_data = thread_response.json()
        thread_id = thread_data["thread_id"]
        print(f"✅ Thread created: {thread_id}")
        
        # Step 2: Start workflow
        print("\n2️⃣ Starting email workflow...")
        run_response = await client.post(
            f"{api_base_url}/threads/{thread_id}/runs",
            json={
                "assistant_id": assistant_id,
                "input": test_email,
                "stream_mode": "values"
            }
        )
        
        if run_response.status_code != 200:
            print(f"❌ Failed to start workflow: {run_response.text}")
            return False
            
        run_data = run_response.json()
        run_id = run_data["run_id"]
        print(f"✅ Workflow started: {run_id}")
        
        # Step 3: Monitor workflow status
        print("\n3️⃣ Monitoring workflow progress...")
        print("⏳ The workflow should now appear in Agent Inbox for review")
        print("👉 Please go to Agent Inbox and ACCEPT the draft response")
        print("   You have about 30 seconds to review and accept\n")
        
        # Poll for status changes
        last_status = None
        email_sent = False
        max_wait = 120  # 2 minutes total
        
        for i in range(max_wait):
            # Check thread state
            state_response = await client.get(
                f"{api_base_url}/threads/{thread_id}/state"
            )
            
            if state_response.status_code == 200:
                state_data = state_response.json()
                values = state_data.get("values", {})
                
                # Check current status
                status = values.get("status", "unknown")
                if status != last_status:
                    print(f"   Status changed: {last_status} → {status}")
                    last_status = status
                
                # Check for human review state
                if "messages" in values:
                    for msg in values["messages"]:
                        if msg.get("role") == "human_review" and "human_review" not in str(last_status):
                            print("   🔔 HUMAN REVIEW ACTIVE - Please accept in Agent Inbox!")
                
                # Check if email was sent
                response_metadata = values.get("response_metadata", {})
                if "email_sent" in response_metadata:
                    email_sent = True
                    sent_info = response_metadata["email_sent"]
                    print("\n✅ EMAIL SENT SUCCESSFULLY!")
                    print(f"   To: {sent_info.get('to', 'N/A')}")
                    print(f"   Subject: {sent_info.get('subject', 'N/A')}")
                    print(f"   Message ID: {sent_info.get('message_id', 'N/A')}")
                    break
                
                # Check for errors
                errors = values.get("errors", [])
                if errors:
                    print(f"\n⚠️ Errors detected:")
                    for error in errors:
                        print(f"   - {error}")
                
                if status == "completed":
                    print(f"\n   Workflow completed")
                    break
            
            # Show progress every 5 seconds
            if i % 5 == 0 and i > 0:
                print(f"   Still waiting... ({i}/{max_wait}s)")
            
            await asyncio.sleep(1)
        
        # Final results
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS:")
        print(f"   Thread ID: {thread_id}")
        print(f"   Run ID: {run_id}")
        print(f"   Final Status: {last_status}")
        print(f"   Email Sent: {'✅ Yes' if email_sent else '❌ No'}")
        
        if email_sent:
            print(f"\n🎉 SUCCESS! Email was sent to {real_recipient}")
            print(f"📬 Check {real_recipient} inbox for the reply!")
        else:
            print("\n❌ Email was not sent")
            print("💡 Possible reasons:")
            print("   - Draft was not accepted in Agent Inbox")
            print("   - Gmail authentication issue")
            print("   - Check server logs for details")
        
        print("=" * 60)
        
        return email_sent


async def main():
    """Main test runner"""
    print("⚠️  IMPORTANT: This test uses REAL email addresses!")
    print("   The workflow will generate a real email reply")
    print("   Make sure to ACCEPT the draft in Agent Inbox when prompted\n")
    
    # Give server a moment to be ready
    await asyncio.sleep(2)
    
    success = await test_real_email_workflow()
    
    if success:
        print("\n🎉 Full workflow test passed!")
        print("   - Agent Inbox review ✅")
        print("   - Human approval ✅")
        print("   - Email sending ✅")
    else:
        print("\n📋 Next steps:")
        print("   1. Check if you accepted the draft in Agent Inbox")
        print("   2. Review server logs for any errors")
        print("   3. Verify Gmail OAuth token has send permissions")


if __name__ == "__main__":
    asyncio.run(main())
