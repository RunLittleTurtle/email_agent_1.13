#!/usr/bin/env python3
"""
Test the refactored LangChain MCP Calendar Agent
"""

import asyncio
import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.calendar_agent import CalendarAgent
from src.models.state import AgentState

async def test_calendar_agent():
    """Test the new LangChain MCP calendar agent"""
    
    print("🧪 Testing LangChain MCP Calendar Agent...")
    print("=" * 50)
    
    # Initialize agent
    try:
        agent = CalendarAgent()
        print("✅ Calendar agent initialized")
        
        # Check available tools
        tools = agent.list_available_tools()
        print(f"📋 Available tools: {len(tools)}")
        for tool in tools:
            print(f"  - {tool}")
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Test calendar request processing
    print("\n🗓️ Testing calendar request processing...")
    
    # Create test state with meeting request
    test_state = AgentState()
    test_state.extracted_context = {
        "calendar_requirements": {
            "meeting_request": True,
            "subject": "Test Meeting with LangChain MCP",
            "requested_datetime": "2024-08-19T13:00:00-04:00",  # Today at 1 PM
            "duration_minutes": 60,
            "attendees": ["test@example.com"],
            "description": "Testing the new LangChain MCP calendar integration",
            "location": "Video Call"
        }
    }
    
    try:
        # Process the calendar request
        result_state = await agent.process(test_state)
        
        if result_state.calendar_data:
            print("✅ Calendar processing successful")
            calendar_data = result_state.calendar_data
            
            print(f"📊 Action taken: {calendar_data.action_taken}")
            print(f"📅 Availability status: {calendar_data.availability_status}")
            print(f"💬 Message: {calendar_data.message}")
            
            if hasattr(calendar_data, 'agent_response'):
                print(f"🤖 Agent response preview: {str(calendar_data.agent_response)[:200]}...")
            
            if hasattr(calendar_data, 'tools_used'):
                print(f"🔧 Tools used: {calendar_data.tools_used}")
            
        else:
            print("⚠️ No calendar data returned")
            if result_state.error_messages:
                print(f"❌ Errors: {result_state.error_messages}")
            
    except Exception as e:
        print(f"❌ Calendar processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_calendar_agent())
