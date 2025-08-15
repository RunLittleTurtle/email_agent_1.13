"""
Calendar Agent using Pipedream MCP Server with proper session management
Handles calendar operations with Google Calendar via MCP tools
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage

from .base_agent import BaseAgent
from ..models.state import AgentState, CalendarData

load_dotenv()

class CalendarAgent(BaseAgent):
    """
    Calendar agent using Pipedream MCP server with proper session management.
    """

    def __init__(self):
        super().__init__(
            name="calendar_agent",
            model="gpt-4o",
            temperature=0.3
        )

    async def _get_mcp_tools(self):
        """Get MCP tools using direct client approach for v0.1.0"""
        pipedream_url = os.getenv("PIPEDREAM_MCP_SERVER")
        if not pipedream_url:
            raise ValueError("PIPEDREAM_MCP_SERVER environment variable not set")
        
        self.logger.info(f"Connecting to MCP server: {pipedream_url}")
        
        client = MultiServerMCPClient({
            "pipedream_calendar": {
                "url": pipedream_url,
                "transport": "streamable_http"
            }
        })
        
        # Direct client usage as per v0.1.0 documentation
        tools = await client.get_tools()
        self.logger.info(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")
        return tools

    @traceable(name="calendar_agent_process", tags=["agent", "calendar", "mcp"])
    async def process(self, state: AgentState) -> AgentState:
        """Process calendar requests with proper MCP session management"""
        try:
            if not state.extracted_context:
                state.add_error("No extracted context for calendar processing")
                return state

            self.logger.info("Processing calendar request")

            # Extract calendar requirements first
            requirements = await self._extract_calendar_requirements(state)
            
            if not requirements or not requirements.get("is_meeting_request"):
                state.add_error("No meeting request found in email")
                return state

            # Get MCP tools using direct client approach
            tools = await self._get_mcp_tools()
            
            if not tools:
                state.add_error("No MCP tools available")
                return state
            
            # Create agent with tools
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            agent = create_react_agent(
                llm,
                tools
            )
            
            # Format task for agent
            task = self._format_task(requirements)
            self.logger.info(f"Executing agent task: {task[:200]}...")
            
            # Create messages with system message included
            messages = [
                SystemMessage(content=self._get_calendar_system_message()),
                HumanMessage(content=task)
            ]
            
            # Execute agent with messages including system message
            result = await agent.ainvoke({
                "messages": messages
            })
            
            self.logger.info(f"Agent execution completed: {result}")
            
            # Parse and store results
            parsed_result = self._parse_agent_result(result, requirements)
            
            calendar_data = CalendarData(
                events_checked=parsed_result.get("events_checked", []),
                availability_status=parsed_result.get("availability_status", "unknown"),
                suggested_times=parsed_result.get("suggested_times", []),
                booked_event=parsed_result.get("booked_event"),
                action_taken=parsed_result.get("action_taken", ""),
                attendees_notified=parsed_result.get("attendees_notified", [])
            )
            
            state.calendar_data = calendar_data
            
            self._add_message(
                state,
                f"Calendar: {parsed_result.get('action_taken', 'processed')}",
                metadata=parsed_result
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Calendar processing failed: {e}", exc_info=True)
            state.add_error(f"Calendar error: {str(e)}")
            return state

    def _get_calendar_system_message(self) -> str:
        """Get system message that ensures MCP tool usage"""
        return """You are a calendar assistant with Google Calendar access through MCP tools.

CRITICAL INSTRUCTIONS:
1. You MUST use the provided MCP tools for ALL calendar operations
2. NEVER simulate or fake calendar operations
3. ALWAYS call the appropriate tool function
4. If a tool fails, report the actual error

Available tools allow you to:
- List calendar events
- Create new events
- Check availability
- Update existing events

For meeting requests:
1. First check availability using the list events tool
2. Create the event using the create event tool
3. Report the actual result from the tools

Timezone: America/New_York

Remember: You must use tools, not just describe what you would do."""

    async def _extract_calendar_requirements(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Extract calendar requirements from email"""
        email = state.email
        if not email:
            return None
        
        prompt = f"""Extract calendar info from this email:

SUBJECT: {email.subject}
FROM: {email.sender}  
BODY: {email.body}

Return JSON:
{{
    "is_meeting_request": true/false,
    "requested_datetime": "2025-08-19T13:00:00",
    "duration_minutes": 60,
    "attendees": ["email@example.com"],
    "subject": "Meeting title",
    "description": "Meeting description"
}}"""

        try:
            response = await self._call_llm(prompt, "Extract calendar info as JSON.")
            requirements = json.loads(response)
            
            # Add sender as attendee
            sender_email = self._extract_email_from_sender(email.sender)
            if requirements.get("is_meeting_request") and sender_email:
                attendees = requirements.get("attendees", [])
                if sender_email not in attendees:
                    attendees.append(sender_email)
                requirements["attendees"] = attendees
                    
            return requirements
            
        except Exception as e:
            self.logger.error(f"Failed to extract calendar requirements: {e}")
            return None

    def _format_task(self, requirements: Dict[str, Any]) -> str:
        """Format calendar requirements into task"""
        requested_datetime = requirements.get("requested_datetime")
        attendees = requirements.get("attendees", [])
        subject = requirements.get("subject", "Meeting")
        duration = requirements.get("duration_minutes", 60)
        description = requirements.get("description", "")
        
        if requested_datetime:
            try:
                dt = datetime.fromisoformat(requested_datetime)
                formatted_time = dt.strftime("%A, %B %d, %Y at %I:%M %p")
            except ValueError:
                formatted_time = requested_datetime
            
            return f"""Schedule meeting:
- Title: {subject}
- Date/Time: {formatted_time}
- Duration: {duration} minutes
- Attendees: {', '.join(attendees)}
- Description: {description}

First check availability, then create if available."""
        else:
            return f"""Find available times for:
- Title: {subject}
- Duration: {duration} minutes
- Attendees: {', '.join(attendees)}

Suggest 3 business day options."""

    def _parse_agent_result(self, result: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Parse agent result into structured format"""
        
        # Get the last message content
        messages = result.get("messages", [])
        if not messages:
            return {"action_taken": "no_response", "availability_status": "unknown"}
        
        last_message = messages[-1]
        output = last_message.content.lower() if hasattr(last_message, 'content') else str(last_message).lower()
        
        if "created" in output or "scheduled" in output:
            return {
                "action_taken": "meeting_booked",
                "availability_status": "available",
                "booked_event": {"summary": requirements.get("subject", "Meeting")},
                "attendees_notified": requirements.get("attendees", [])
            }
        elif "conflict" in output or "not available" in output:
            return {
                "action_taken": "alternatives_suggested",
                "availability_status": "conflict",
                "suggested_times": []
            }
        elif "available" in output:
            return {
                "action_taken": "availability_checked",
                "availability_status": "available",
                "suggested_times": []
            }
        else:
            return {
                "action_taken": "processed",
                "availability_status": "unknown",
                "message": output[:200]
            }

    def _extract_email_from_sender(self, sender: str) -> Optional[str]:
        """Extract email address from sender string"""
        import re
        
        email_match = re.search(r'<([^>]+)>', sender)
        if email_match:
            return email_match.group(1)
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, sender.strip()):
            return sender.strip()
        
        return None