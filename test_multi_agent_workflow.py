#!/usr/bin/env python3
"""
Test script for multi-agent workflow with Google Workspace integration
"""

import asyncio
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure LangSmith tracing is enabled
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agent-inbox-phase2-test"

from src.models.state import AgentState, EmailMessage
from src.graph.workflow import create_workflow


async def test_calendar_request():
    """Test email that should trigger calendar agent"""
    print("\n📅 Testing Calendar Request Email...")
    
    email = EmailMessage(
        id="test-cal-001",
        sender="john.doe@example.com",
        recipients=["assistant@example.com"],
        subject="Meeting Request: Project Review",
        body="""Hi,

I'd like to schedule a meeting to review the project progress. 
Could we meet next Tuesday at 2 PM for about an hour? 
If that doesn't work, please suggest some alternative times.

Best regards,
John""",
        timestamp=datetime.now()
    )
    
    state = AgentState(email=email)
    workflow = create_workflow()
    
    # Run workflow
    result = await workflow.ainvoke(state)
    
    print(f"\n✅ Calendar test completed")
    print(f"Intent: {result.get('intent', 'N/A')}")
    print(f"Draft Response Preview: {result.get('draft_response', 'N/A')[:200]}...")
    return result


async def test_document_search():
    """Test email that should trigger RAG agent"""
    print("\n📄 Testing Document Search Email...")
    
    email = EmailMessage(
        id="test-rag-001",
        sender="sarah.johnson@example.com",
        recipients=["assistant@example.com"],
        subject="Need Q3 Report",
        body="""Hello,

Could you please send me the Q3 financial report? 
I also need the marketing strategy document we discussed last week.

Thanks,
Sarah""",
        timestamp=datetime.now()
    )
    
    state = AgentState(email=email)
    workflow = create_workflow()
    
    # Run workflow
    result = await workflow.ainvoke(state)
    
    print(f"\n✅ Document search test completed")
    print(f"Intent: {result.get('intent', 'N/A')}")
    print(f"Draft Response Preview: {result.get('draft_response', 'N/A')[:200]}...")
    return result


async def test_crm_delegation():
    """Test email that should trigger CRM agent"""
    print("\n👥 Testing CRM/Delegation Email...")
    
    email = EmailMessage(
        id="test-crm-001",
        sender="mike.wilson@example.com",
        recipients=["assistant@example.com"],
        subject="Task Assignment",
        body="""Hi,

Please assign the new client onboarding task to Lisa Chen from the sales team.
Also, can you provide me with the contact details for our legal advisor?

Thanks,
Mike""",
        timestamp=datetime.now()
    )
    
    state = AgentState(email=email)
    workflow = create_workflow()
    
    # Run workflow
    result = await workflow.ainvoke(state)
    
    print(f"\n✅ CRM test completed")
    print(f"Intent: {result.get('intent', 'N/A')}")
    print(f"Draft Response Preview: {result.get('draft_response', 'N/A')[:200]}...")
    return result


async def test_multi_agent_email():
    """Test email that should trigger multiple agents"""
    print("\n🔄 Testing Multi-Agent Email...")
    
    email = EmailMessage(
        id="test-multi-001",
        sender="alex.morgan@example.com",
        recipients=["assistant@example.com"],
        subject="Prep for Client Meeting",
        body="""Hi,

I need help preparing for tomorrow's client meeting:

1. Schedule a prep meeting with the team for today at 3 PM
2. Send me the latest project proposal document
3. Get contact info for the client's technical lead

Let me know if you need anything else.

Best,
Alex""",
        timestamp=datetime.now()
    )
    
    state = AgentState(email=email)
    workflow = create_workflow()
    
    # Run workflow
    result = await workflow.ainvoke(state)
    
    print(f"\n✅ Multi-agent test completed")
    print(f"Intent: {result.get('intent', 'N/A')}")
    print(f"Draft Response Preview: {result.get('draft_response', 'N/A')[:200]}...")
    
    # Check which agents were involved
    routing = result.get('response_metadata', {}).get("routing", {})
    print(f"\nAgents involved:")
    print(f"  Required: {routing.get('required_agents', [])}")
    print(f"  Completed: {routing.get('completed_agents', [])}")
    print(f"  Failed: {routing.get('failed_agents', [])}")
    
    return result


async def test_error_handling():
    """Test workflow error handling with problematic email"""
    print("\n⚠️ Testing Error Handling...")
    
    email = EmailMessage(
        id="test-error-001",
        sender="test@example.com",
        recipients=["assistant@example.com"],
        subject="",
        body="",  # Empty email to test error handling
        timestamp=datetime.now()
    )
    
    state = AgentState(email=email)
    workflow = create_workflow()
    
    # Run workflow
    result = await workflow.ainvoke(state)
    
    print(f"\n✅ Error handling test completed")
    print(f"Errors: {result.get('error_messages', [])}")
    print(f"Draft Response: {result.get('draft_response', 'No draft generated')}")
    return result


async def main():
    """Run all tests"""
    print("🚀 Starting Multi-Agent Workflow Tests...")
    print(f"LangSmith Project: {os.getenv('LANGCHAIN_PROJECT')}")
    print(f"LangSmith Tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
    
    try:
        # Run individual agent tests
        await test_calendar_request()
        await asyncio.sleep(2)  # Brief pause between tests
        
        await test_document_search()
        await asyncio.sleep(2)
        
        await test_crm_delegation()
        await asyncio.sleep(2)
        
        # Run multi-agent test
        await test_multi_agent_email()
        await asyncio.sleep(2)
        
        # Test error handling
        await test_error_handling()
        
        print("\n✨ All tests completed successfully!")
        print("\n📊 Check LangSmith for detailed traces:")
        print("https://smith.langchain.com/")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
