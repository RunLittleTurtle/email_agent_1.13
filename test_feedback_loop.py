#!/usr/bin/env python3
"""
Test script for Agent Inbox feedback/refinement loop.
Creates a thread, waits for human interrupt, and validates feedback handling.
"""

import asyncio
import httpx
import json
from datetime import datetime

# Test configuration
API_URL = "http://127.0.0.1:2024"
AGENT_INBOX_URL = "http://localhost:3000"

# Simple test email
TEST_EMAIL = {
    "id": f"test_email_{int(datetime.now().timestamp())}",
    "subject": "Test Feedback Loop",
    "body": "Hi there, I need help writing a professional email response. Please make it formal and include specific details about our meeting schedule.",
    "sender": "test@example.com",
    "recipients": ["me@company.com"],
    "timestamp": datetime.now().isoformat(),
    "attachments": [],
    "thread_id": None
}


async def create_test_thread():
    """Create a new test thread and start the workflow."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            print("🧪 Testing Feedback/Refinement Loop")
            print("=" * 50)
            
            print(f"📧 Test Email:")
            print(f"   From: {TEST_EMAIL['sender']}")
            print(f"   Subject: {TEST_EMAIL['subject']}")
            print(f"   Body: {TEST_EMAIL['body'][:100]}...")
            print()
            
            # Create thread
            print(f"🚀 Creating thread via API...")
            print(f"   API URL: {API_URL}")
            
            thread_response = await client.post(f"{API_URL}/threads", json={})
            
            if thread_response.status_code != 200:
                print(f"❌ Failed to create thread: {thread_response.status_code}")
                print(f"   Response: {thread_response.text}")
                return None
                
            thread_data = thread_response.json()
            thread_id = thread_data["thread_id"]
            print(f"   ✅ Created Thread ID: {thread_id}")
            
            # Start workflow
            print(f"🔄 Starting workflow...")
            run_response = await client.post(
                f"{API_URL}/threads/{thread_id}/runs",
                json={
                    "assistant_id": "email_agent",
                    "input": {
                        "email": TEST_EMAIL,
                        "messages": []
                    }
                }
            )
            
            if run_response.status_code != 200:
                print(f"❌ Failed to start workflow: {run_response.status_code}")
                print(f"   Response: {run_response.text}")
                return None
                
            run_data = run_response.json()
            run_id = run_data["run_id"]
            print(f"   ✅ Started Run ID: {run_id}")
            print(f"   Status: {run_data.get('status', 'unknown')}")
            
            # Wait for workflow to reach interrupt
            print(f"⏳ Waiting for workflow to reach human review interrupt...")
            
            for attempt in range(10):  # Wait up to 30 seconds
                await asyncio.sleep(3)
                
                # Check thread status
                status_response = await client.get(f"{API_URL}/threads/{thread_id}")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    thread_status = status_data.get("status", "unknown")
                    
                    print(f"   📊 Attempt {attempt + 1}: Status = {thread_status}")
                    
                    if thread_status == "interrupted":
                        print(f"   ✅ Thread interrupted! Ready for human review.")
                        break
                        
                    if thread_status in ["success", "error"]:
                        print(f"   ⚠️  Workflow completed without interrupt: {thread_status}")
                        break
                else:
                    print(f"   ❌ Failed to check status: {status_response.status_code}")
            
            print()
            print("🎯 NEXT STEPS FOR MANUAL TESTING:")
            print("=" * 50)
            print(f"1. Open Agent Inbox: {AGENT_INBOX_URL}")
            print(f"2. Look for Thread ID: {thread_id}")
            print(f"3. You should see an interrupted thread with:")
            print(f"   - Clean email context (no JSON)")
            print(f"   - Accept button")
            print(f"   - Respond to assistant button")
            print(f"4. Click 'Respond to assistant' and provide feedback like:")
            print(f"   'Make the response more casual and friendly'")
            print(f"5. The workflow should:")
            print(f"   - Route to supervisor (handles feedback)")
            print(f"   - Go to adaptive_writer (processes feedback)")
            print(f"   - Return to human_review (new interrupt)")
            print(f"6. Check the new draft incorporates your feedback")
            print()
            print("🔍 VALIDATION CHECKLIST:")
            print("✓ Thread appears in Agent Inbox")
            print("✓ Email context is readable (not JSON)")
            print("✓ 'Respond to assistant' works without errors")
            print("✓ Feedback creates new interrupt with updated draft")
            print("✓ Feedback history is preserved in workflow state")
            
            return thread_id
            
        except httpx.TimeoutException:
            print("❌ Request timed out - check if LangGraph dev server is running")
            return None
        except Exception as e:
            print(f"❌ Error creating test thread: {e}")
            return None


async def check_thread_state(thread_id: str):
    """Check the current state of a thread."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            print(f"🔍 Checking Thread State: {thread_id}")
            print("=" * 40)
            
            response = await client.get(f"{API_URL}/threads/{thread_id}")
            
            if response.status_code != 200:
                print(f"❌ Failed to get thread: {response.status_code}")
                return
                
            thread_data = response.json()
            
            print(f"Status: {thread_data.get('status', 'unknown')}")
            print(f"Created: {thread_data.get('created_at', 'unknown')}")
            print(f"Updated: {thread_data.get('updated_at', 'unknown')}")
            
            # Check for feedback history
            values = thread_data.get("values", {})
            response_metadata = values.get("response_metadata", {})
            
            if "feedback_context" in response_metadata:
                feedback_ctx = response_metadata["feedback_context"]
                print(f"Feedback Iterations: {feedback_ctx.get('refinement_iteration', 0)}")
                print(f"Feedback Count: {feedback_ctx.get('feedback_count', 0)}")
                print(f"All Feedback: {feedback_ctx.get('all_feedback', [])}")
            
            if "human_feedback" in response_metadata:
                print(f"Human Feedback: {response_metadata['human_feedback']}")
                
            if values.get("draft_response"):
                print(f"Current Draft: {values['draft_response'][:100]}...")
                
        except Exception as e:
            print(f"❌ Error checking thread: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        if len(sys.argv) > 2:
            thread_id = sys.argv[2]
            asyncio.run(check_thread_state(thread_id))
        else:
            print("Usage: python test_feedback_loop.py check <thread_id>")
    else:
        # Create new test
        thread_id = asyncio.run(create_test_thread())
        if thread_id:
            print(f"\n💡 To check this thread later, run:")
            print(f"   python test_feedback_loop.py check {thread_id}")
