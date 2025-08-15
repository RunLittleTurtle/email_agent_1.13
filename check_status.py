#!/usr/bin/env python3
"""
Quick script to check the status of the running workflow
"""

import asyncio
import httpx
import json

async def check_workflow_status():
    """Check the current status of the workflow"""
    print("🔍 Checking Workflow Status")
    print("=" * 30)
    
    # Use the thread ID from the latest dynamic interrupt test
    api_url = "http://127.0.0.1:2024"
    thread_id = "91342d37-8fd4-4866-a608-65799b4b2046"  # From the latest dynamic interrupt test
    run_id = "1f072fac-e654-6b6c-854e-143e323d7e1b"     # From the latest dynamic interrupt test
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            print(f"📡 Checking thread: {thread_id}")
            print(f"📡 Checking run: {run_id}")
            print()
            
            # Check run status
            status_response = await client.get(f"{api_url}/threads/{thread_id}/runs/{run_id}")
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get('status', 'unknown')
                
                print(f"📊 Current Status: {status}")
                
                if status == 'interrupted':
                    print("🎯 SUCCESS! Workflow is interrupted!")
                    print("   ✅ The workflow should now appear in Agent Inbox")
                    print("   ✅ Check LangSmith for the trace")
                    
                elif status == 'running':
                    print("⏳ Still running... processing through agents")
                    print("   The workflow is working through: email_processor → supervisor → adaptive_writer")
                    
                elif status == 'completed':
                    print("✅ Workflow completed (might have bypassed interrupt)")
                    
                elif status == 'failed':
                    print("❌ Workflow failed")
                    print(f"   Error: {status_data.get('error', 'No error details')}")
                    
                else:
                    print(f"❓ Unknown status: {status}")
                
                # Get thread state
                print("\n🧠 Checking Thread State...")
                state_response = await client.get(f"{api_url}/threads/{thread_id}/state")
                
                if state_response.status_code == 200:
                    state_data = state_response.json()
                    values = state_data.get('values', {})
                    
                    print(f"   Current Agent: {values.get('current_agent', 'N/A')}")
                    print(f"   Workflow Status: {values.get('status', 'N/A')}")
                    print(f"   Has Draft Response: {'draft_response' in values}")
                    print(f"   Message Count: {len(values.get('messages', []))}")
                    
                    if values.get('draft_response'):
                        draft = values['draft_response'][:150]
                        print(f"   Draft Preview: {draft}...")
                        
                    if values.get('error_messages'):
                        print(f"   Errors: {values['error_messages']}")
                        
                else:
                    print(f"❌ Failed to get thread state: {state_response.status_code}")
                    
            else:
                print(f"❌ Failed to check status: {status_response.status_code}")
                print(status_response.text)
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(check_workflow_status())
