#!/usr/bin/env python3
"""
Test script to validate calendar modification detection in email processor and supervisor.
"""

import asyncio
import json
from src.models.state import AgentState, EmailMessage
from src.agents.email_processor import EmailProcessorAgent
from src.agents.supervisor import SupervisorAgent
from langchain_core.messages import HumanMessage

async def test_calendar_modification_detection():
    """Test that 'change the event' language is properly detected"""
    
    # Create test email that should trigger calendar modification
    from datetime import datetime
    test_email = EmailMessage(
        id="test_change_event",
        subject="Re: motocross event",
        sender="Samuel <samuel.audette1@gmail.com>",
        recipients=["info@800m.ca"],
        body="ok change the event in the calendar please Le ven. 22 août 2025 at 2pm instead of 28th",
        timestamp=datetime.now(),
        thread_id="test_thread",
        message_id="test_message"
    )
    
    # Create initial state
    state = AgentState(
        email=test_email,
        messages=[HumanMessage(content="Previous booking for motocross event on 28th")]
    )
    
    print("Testing Email Processor...")
    print(f"Email body: {test_email.body}")
    print()
    
    # Test email processor
    email_processor = EmailProcessorAgent()
    state = await email_processor.process(state)
    
    # Extract parsing results
    parsing = state.response_metadata.get("parsing", {})
    context = state.response_metadata.get("context", {})
    
    print("EMAIL PROCESSOR RESULTS:")
    print(f"- Conversation Type: {parsing.get('conversation_type', 'unknown')}")
    print(f"- Main Request: {parsing.get('main_request', 'N/A')}")
    print(f"- Requested Actions: {context.get('requested_actions', [])}")
    print(f"- Dates Mentioned: {context.get('dates_mentioned', [])}")
    print()
    
    # Test supervisor
    print("Testing Supervisor...")
    supervisor = SupervisorAgent()
    state = await supervisor.process(state)
    
    # Check supervisor routing decision
    routing = state.response_metadata.get("routing", {})
    
    print("SUPERVISOR ROUTING RESULTS:")
    print(f"- Next Agent: {routing.get('next_agent', 'unknown')}")
    print(f"- Calendar Agent Needed: {routing.get('agent_assignments', {}).get('calendar_agent', {}).get('needed', False)}")
    print(f"- Calendar Task: {routing.get('agent_assignments', {}).get('calendar_agent', {}).get('task', 'N/A')}")
    print(f"- Routing Rationale: {routing.get('routing_rationale', 'N/A')}")
    print()
    
    # Validate results
    success = True
    issues = []
    
    # Check if conversation_type indicates modification
    conv_type = parsing.get('conversation_type', '').lower()
    if 'modification' not in conv_type and 'follow_up' not in conv_type:
        success = False
        issues.append(f"Email processor should detect 'modification' or 'follow_up', got: {conv_type}")
    
    # Check if calendar agent is needed
    calendar_needed = routing.get('agent_assignments', {}).get('calendar_agent', {}).get('needed', False)
    if not calendar_needed:
        success = False
        issues.append("Supervisor should route to calendar_agent for calendar modification")
    
    # Check if task mentions modification/update
    calendar_task = routing.get('agent_assignments', {}).get('calendar_agent', {}).get('task', '').lower()
    if calendar_task and not any(word in calendar_task for word in ['modify', 'update', 'change', 'reschedule']):
        success = False
        issues.append(f"Calendar task should mention modification action, got: {calendar_task}")
    
    print("=" * 60)
    if success:
        print("✅ SUCCESS: Calendar modification detection working correctly!")
    else:
        print("❌ FAILURE: Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    asyncio.run(test_calendar_modification_detection())
