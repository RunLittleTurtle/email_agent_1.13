#!/usr/bin/env python3
"""
Quick test to isolate workflow hanging issue
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

async def test_workflow_components():
    """Test workflow components step by step to isolate hanging issue"""

    print("🧪 Testing Workflow Components to Isolate Hanging...")
    print("=" * 60)

    # Test 1: Basic imports
    print("\n1️⃣ Testing basic imports...")
    try:
        from src.models.state import AgentState, EmailMessage
        from src.agents.email_processor import EmailProcessorAgent
        from src.agents.supervisor import SupervisorAgent
        from src.agents.adaptive_writer import AdaptiveWriterAgent
        print("✅ Basic imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return

    # Test 2: Create simple state
    print("\n2️⃣ Testing state creation...")
    try:
        test_email = EmailMessage(
            id="test_123",
            subject="pickle ball game",
            body="Hi 800m,\n\nI would like to invite you for the pickleball game on september 4 at 14h?\n\nThanks,\n\nSam",
            sender="Samuel <samuel.audette1@gmail.com>",
            recipients=["info@800m.ca"]
        )

        state = AgentState(email=test_email)
        print(f"✅ State created with email: {state.email.subject}")
    except Exception as e:
        print(f"❌ State creation failed: {e}")
        return

    # Test 3: Email processor agent
    print("\n3️⃣ Testing Email Processor Agent...")
    try:
        agent = EmailProcessorAgent()
        result = await agent.process(state)
        print(f"✅ Email processor returned: {type(result)} with keys: {list(result.keys())}")
    except Exception as e:
        print(f"❌ Email processor failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 4: Update state and test supervisor
    print("\n4️⃣ Testing Supervisor Agent...")
    try:
        # Apply email processor results to state
        for key, value in result.items():
            if hasattr(state, key):
                setattr(state, key, value)

        supervisor = SupervisorAgent()
        supervisor_result = await supervisor.process(state)
        print(f"✅ Supervisor returned: {type(supervisor_result)} with keys: {list(supervisor_result.keys())}")
    except Exception as e:
        print(f"❌ Supervisor failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 5: Test workflow creation (potential hanging point)
    print("\n5️⃣ Testing Workflow Creation...")
    try:
        from src.graph.workflow import create_workflow
        print("📝 Creating workflow...")
        workflow = create_workflow()
        print("✅ Workflow created successfully")
    except Exception as e:
        print(f"❌ Workflow creation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 6: Simple workflow invocation (most likely hanging point)
    print("\n6️⃣ Testing Simple Workflow Invocation...")
    try:
        print("📤 Invoking workflow with test email...")

        # Create fresh state for workflow
        workflow_state = {
            "email": test_email.dict(),
            "messages": [],
            "status": "processing",
            "error_messages": [],
            "output": [],
            "response_metadata": {},
            "dynamic_context": {
                "execution_step": 0,
                "current_phase": "initialization",
                "accumulated_insights": [],
                "execution_metadata": {},
                "performance_metrics": {}
            }
        }

        # Try to get first step only
        async for chunk in workflow.astream(workflow_state, {"recursion_limit": 3}):
            print(f"📊 Workflow chunk: {chunk}")
            break  # Just get first chunk to test

        print("✅ Workflow invocation successful (at least first step)")

    except Exception as e:
        print(f"❌ Workflow invocation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ Component testing completed!")

if __name__ == "__main__":
    asyncio.run(test_workflow_components())
