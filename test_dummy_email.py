#!/usr/bin/env python3
"""
Test script to send dummy email through the workflow and verify human interrupt
"""

import asyncio
import os
import httpx
import json
from datetime import datetime
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_dummy_email_workflow():
    """
    Test the workflow with a dummy email via LangGraph API
    """
    print("🧪 Testing MVP Workflow with Dummy Email")
    print("=" * 50)
    
    # Create simple dummy email data that will trigger SIMPLE_DIRECT intent
    # This avoids routing to calendar/CRM/RAG agents and focuses on human interrupt testing
    dummy_email = {
        "id": str(uuid.uuid4()),
        "subject": "Thank you for your help",
        "body": "Hi there,\n\nI just wanted to say thank you for all your help with the project last week. Your assistance made a big difference and I really appreciate it.\n\nHave a great day!\n\nBest regards,\nSarah",
        "sender": "sarah.johnson@example.com",
        "recipients": ["support@company.com"],
        "timestamp": datetime.now().isoformat(),
        "attachments": [],
        "thread_id": None
    }
    
    # Create initial state for the workflow
    initial_state = {
        "email": dummy_email,
        "workflow_id": str(uuid.uuid4()),
        "messages": [],
        "extracted_context": None,
        "intent": None,
        "calendar_data": None,
        "document_data": None,
        "contact_data": None,
        "decomposed_tasks": None,
        "draft_response": None,
        "response_metadata": {},
        "current_agent": None,
        "status": "processing",
        "human_feedback": None,
        "error_messages": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    print(f"📧 Dummy Email:")
    print(f"   From: {dummy_email['sender']}")
    print(f"   Subject: {dummy_email['subject']}")
    print(f"   Body: {dummy_email['body'][:100]}...")
    print()
    
    # Create thread and invoke workflow
    api_url = "http://127.0.0.1:2024"
    thread_id = str(uuid.uuid4())
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            print(f"🚀 Starting workflow via API...")
            print(f"   API URL: {api_url}")
            print(f"   Thread ID: {thread_id}")
            print()
            
            # Create the thread first
            thread_response = await client.post(f"{api_url}/threads", json={})
            if thread_response.status_code != 200:
                print(f"❌ Failed to create thread: {thread_response.status_code}")
                print(thread_response.text)
                return
                
            thread_data = thread_response.json()
            actual_thread_id = thread_data["thread_id"]
            print(f"   Created Thread ID: {actual_thread_id}")
            
            # Invoke the workflow using the correct endpoint format
            response = await client.post(
                f"{api_url}/threads/{actual_thread_id}/runs",
                json={
                    "assistant_id": "email_agent",
                    "input": initial_state,
                    "config": {"configurable": {"thread_id": actual_thread_id}},
                    "stream_mode": "values"
                }
            )
            
            if response.status_code == 200:
                run_data = response.json()
                run_id = run_data.get("run_id")
                print(f"✅ Workflow started successfully!")
                print(f"   Run ID: {run_id}")
                print(f"   Status: {run_data.get('status')}")
                print()
                
                # Wait for the workflow to reach the interrupt
                print("⏳ Waiting for workflow execution...")
                await asyncio.sleep(3)
                
                # Check the run status
                status_response = await client.get(f"{api_url}/threads/{actual_thread_id}/runs/{run_id}")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"📊 Workflow Status: {status_data.get('status')}")
                    
                    if status_data.get('status') == 'interrupted':
                        print("🎯 SUCCESS! Workflow is interrupted after adaptive_writer!")
                        print("   This means the workflow should appear in Agent Inbox for human review.")
                        
                        # Get the thread state to see where it stopped
                        state_response = await client.get(f"{api_url}/threads/{actual_thread_id}/state")
                        if state_response.status_code == 200:
                            state_data = state_response.json()
                            print(f"   Current Agent: {state_data.get('values', {}).get('current_agent', 'N/A')}")
                            print(f"   Draft Response Available: {'draft_response' in state_data.get('values', {})}")
                            
                            if state_data.get('values', {}).get('draft_response'):
                                draft = state_data['values']['draft_response'][:200]
                                print(f"   Draft Preview: {draft}...")
                        
                        print()
                        print("🎉 AGENT INBOX TEST SUCCESS!")
                        print("   1. Check Agent Inbox - the workflow should appear there")
                        print("   2. Check LangSmith - trace should show interrupt at 'human_review'")
                        print(f"   3. Thread ID for reference: {actual_thread_id}")
                        
                    else:
                        print(f"⚠️  Workflow status: {status_data.get('status')}")
                        print("   Expected 'interrupted' status after adaptive_writer node")
                
                else:
                    print(f"❌ Failed to check workflow status: {status_response.status_code}")
                    print(status_response.text)
                    
            else:
                print(f"❌ Failed to start workflow: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Error testing workflow: {str(e)}")
            raise

if __name__ == "__main__":
    asyncio.run(test_dummy_email_workflow())
