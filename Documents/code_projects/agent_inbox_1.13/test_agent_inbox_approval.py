#!/usr/bin/env python3
"""
Test end-to-end email sending after human approval in Agent Inbox
This script:
1. Sends a test email to the workflow via API
2. Waits for human review interrupt
3. Simulates human approval
4. Verifies email was sent
"""

import asyncio
import httpx
import json
import time
from datetime import datetime
from typing import Dict, Any


async def test_email_workflow_with_approval():
    """Test the complete email workflow with human approval"""
    print("=" * 60)
    print("🧪 Testing Agent Inbox Email Workflow with Human Approval")
    print("=" * 60)
    
    # Configuration
    api_base_url = "http://127.0.0.1:2024"
    assistant_id = "email_agent"
    
    # Test email data
    test_email = {
        "email": {
            "id": f"test-{int(time.time())}",
            "sender": "test.sender@example.com",
            "recipients": ["samuel.audette1@gmail.com"],
            "subject": f"Test Email for Agent Inbox - {datetime.now().strftime('%H:%M:%S')}",
            "body": "This is a test email to verify the Agent Inbox workflow. Please generate a professional response.",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Create a new thread
        print("\n1️⃣ Creating new thread...")
        thread_response = await client.post(
            f"{api_base_url}/threads",
            json={"metadata": {"test": "true"}}
        )
        
        if thread_response.status_code != 200:
            print(f"❌ Failed to create thread: {thread_response.text}")
            return False
            
        thread_data = thread_response.json()
        thread_id = thread_data["thread_id"]
        print(f"✅ Thread created: {thread_id}")
        
        # Step 2: Start the workflow
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
        
        # Step 3: Wait for human review interrupt
        print("\n3️⃣ Waiting for human review interrupt...")
        max_attempts = 30
        interrupt_found = False
        
        for attempt in range(max_attempts):
            # Check thread state
            state_response = await client.get(
                f"{api_base_url}/threads/{thread_id}/state"
            )
            
            if state_response.status_code == 200:
                state_data = state_response.json()
                values = state_data.get("values", {})
                
                # Check if we're at human review
                if "messages" in values:
                    messages = values["messages"]
                    for msg in messages:
                        if msg.get("role") == "human_review":
                            print("✅ Human review interrupt detected!")
                            print(f"   Draft response preview: {values.get('draft_response', 'N/A')[:100]}...")
                            interrupt_found = True
                            break
                
                if interrupt_found:
                    break
            
            await asyncio.sleep(1)
            print(f"   Waiting... ({attempt + 1}/{max_attempts})")
        
        if not interrupt_found:
            print("❌ Human review interrupt not found within timeout")
            return False
        
        # Step 4: Simulate human approval
        print("\n4️⃣ Simulating human approval...")
        
        # Update the thread with approval action
        update_response = await client.post(
            f"{api_base_url}/threads/{thread_id}/runs",
            json={
                "assistant_id": assistant_id,
                "input": {
                    "action": "accept",
                    "feedback": "Looks good, send it!"
                },
                "stream_mode": "values"
            }
        )
        
        if update_response.status_code != 200:
            print(f"❌ Failed to approve draft: {update_response.text}")
            return False
            
        print("✅ Draft approved!")
        
        # Step 5: Wait for email to be sent
        print("\n5️⃣ Waiting for email to be sent...")
        email_sent = False
        
        for attempt in range(20):
            # Check thread state again
            state_response = await client.get(
                f"{api_base_url}/threads/{thread_id}/state"
            )
            
            if state_response.status_code == 200:
                state_data = state_response.json()
                values = state_data.get("values", {})
                
                # Check status
                status = values.get("status", "unknown")
                print(f"   Current status: {status}")
                
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
                    print(f"⚠️ Errors detected: {errors}")
                
                if status == "completed":
                    break
            
            await asyncio.sleep(1)
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS:")
        print(f"   Thread ID: {thread_id}")
        print(f"   Run ID: {run_id}")
        print(f"   Human Review: {'✅ Found' if interrupt_found else '❌ Not found'}")
        print(f"   Email Sent: {'✅ Yes' if email_sent else '❌ No'}")
        
        if email_sent:
            print("\n🎉 SUCCESS! Email was sent after human approval!")
            print("📬 Check samuel.audette1@gmail.com for the email")
        else:
            print("\n❌ FAILED! Email was not sent after approval")
            print("💡 Check the workflow logs for errors")
        
        print("=" * 60)
        
        return email_sent


async def main():
    """Main test runner"""
    # Give server a moment to be ready
    print("⏳ Waiting for server to be fully ready...")
    await asyncio.sleep(2)
    
    success = await test_email_workflow_with_approval()
    
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("   1. Check the LangGraph server logs for errors")
        print("   2. Verify OAuth tokens have gmail.send scope")
        print("   3. Check Agent Inbox UI for the thread")
        print("   4. Look for 'email_sender' agent logs in the server output")


if __name__ == "__main__":
    asyncio.run(main())
