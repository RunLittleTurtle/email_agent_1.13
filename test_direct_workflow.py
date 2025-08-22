#!/usr/bin/env python3
"""
Test workflow directly with real email data to isolate migration issues
Bypasses LangGraph API server to test core functionality
"""

import os
import sys
import asyncio
import uuid
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.state import AgentState, EmailMessage
from src.graph.workflow import create_workflow
import structlog

# Set up environment
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-compilation")

logger = structlog.get_logger()

async def test_real_email_workflow():
    """Test the workflow directly with real email data from CLI"""
    print("🧪 Testing workflow directly with real email data...")
    
    # Create the exact email data that CLI fetched
    real_email_data = {
        'id': '198d25a3693e262b',
        'sender': 'Samuel <samuel.audette1@gmail.com>',
        'subject': 'motocross event',
        'body': 'Hi 800m,\n\nI would like to invite you at the motocross supershow the 28th at 4pm for 3h. are you available?\n\nThanks,\n\nSam',
        'recipients': ['info@800m.ca'],
        'timestamp': datetime.now().isoformat(),
        'attachments': [],
        'message_id': None,
        'thread_id': None
    }
    
    print(f"📧 Testing with email: {real_email_data['subject']}")
    print(f"📤 From: {real_email_data['sender']}")
    print(f"📝 Body preview: {real_email_data['body'][:80]}...")
    
    try:
        # Create EmailMessage object
        email_obj = EmailMessage(
            id=real_email_data['id'],
            subject=real_email_data['subject'],
            body=real_email_data['body'],
            sender=real_email_data['sender'],
            recipients=real_email_data['recipients'],
            timestamp=datetime.fromisoformat(real_email_data['timestamp'].replace('Z', '+00:00')) if 'Z' in real_email_data['timestamp'] else datetime.now(),
            attachments=real_email_data.get('attachments', [])
        )
        
        print("✅ EmailMessage created successfully")
        
        # Create AgentState with proper structure
        workflow_id = str(uuid.uuid4())
        initial_state = AgentState(
            email=email_obj,
            workflow_id=workflow_id
        )
        
        print("✅ AgentState created successfully")
        print(f"   Workflow ID: {workflow_id}")
        print(f"   Email ID: {initial_state.email.id}")
        print(f"   Messages: {len(initial_state.messages)}")
        
        # Create workflow
        print("🔄 Creating workflow...")
        workflow = create_workflow()
        print("✅ Workflow created successfully")
        
        # Test configuration
        config = {
            "configurable": {
                "thread_id": workflow_id
            }
        }
        
        print("🚀 Invoking workflow...")
        
        # Use model_dump() for Pydantic v2 compatibility
        result = await workflow.ainvoke(initial_state.model_dump(), config)
        
        print("✅ Workflow completed successfully!")
        print(f"📄 Result keys: {list(result.keys())}")
        
        if 'draft_response' in result and result['draft_response']:
            print(f"📝 Draft response: {result['draft_response'][:100]}...")
        
        if 'status' in result:
            print(f"📊 Final status: {result['status']}")
            
        if 'errors' in result and result['errors']:
            print(f"⚠️  Errors encountered: {result['errors']}")
            
        return True
        
    except Exception as e:
        print(f"❌ Direct workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_state_compatibility():
    """Test AgentState compatibility with workflow"""
    print("\n🧪 Testing AgentState compatibility...")
    
    try:
        # Test creating state with minimal data
        minimal_email = EmailMessage(
            id="test_123",
            subject="Test Subject",
            body="Test body content",
            sender="test@example.com",
            recipients=["recipient@example.com"]
        )
        
        # Test state creation
        state = AgentState(email=minimal_email)
        print("✅ Basic AgentState creation works")
        
        # Test model_dump
        state_dict = state.model_dump()
        print("✅ model_dump() works")
        print(f"   Keys: {list(state_dict.keys())}")
        
        # Test required fields
        required_fields = ['email', 'workflow_id', 'messages', 'status']
        missing_fields = [field for field in required_fields if field not in state_dict]
        
        if missing_fields:
            print(f"⚠️  Missing fields: {missing_fields}")
        else:
            print("✅ All required fields present")
            
        return True
        
    except Exception as e:
        print(f"❌ State compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_direct_tests():
    """Run all direct workflow tests"""
    print("🚀 Direct Workflow Migration Test Suite")
    print("=" * 50)
    
    results = {
        "state_compatibility": await test_state_compatibility(),
        "real_email_workflow": await test_real_email_workflow()
    }
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 Direct workflow tests successful!")
        print("🔍 Issue is likely in LangGraph API integration")
    else:
        print("⚠️  Core workflow has issues - fix these first")
    
    return results


if __name__ == "__main__":
    # Run direct tests
    results = asyncio.run(run_direct_tests())
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)
