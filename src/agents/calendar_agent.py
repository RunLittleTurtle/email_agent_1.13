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

            # Add meeting request info for adaptive writer
            calendar_data.meeting_request = {
                "title": requirements.get("subject", "Meeting"),
                "requested_datetime": requirements.get("requested_datetime"),
                "duration_minutes": requirements.get("duration_minutes", 60),
                "attendees": requirements.get("attendees", []),
                "description": requirements.get("description", "")
            }

            state.calendar_data = calendar_data

            # Enhanced message with conflict details for downstream agents
            if parsed_result.get("action_taken") == "conflict_detected":
                conflict_message = f"Calendar: CONFLICT DETECTED - Alternative times suggested: {parsed_result.get('suggested_times', [])}"
            else:
                conflict_message = f"Calendar: {parsed_result.get('action_taken', 'processed')}"

            self._add_message(
                state,
                conflict_message,
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

CONFLICT DETECTION WORKFLOW:
1. MANDATORY: First check availability using the list events tool for the EXACT requested time slot
2. CAREFULLY examine the tool response for ANY existing events that overlap with the requested time
3. If there's ANY overlap or conflict (even partial):
   - Do NOT create the event
   - IMPORTANT: YOU MUST Search, Find and Propose 2 alternative available time slots (same day or next business days or later)
   - Return the alternatives in this format: "CONFLICT_DETECTED: Alternative times: [time1, time2]"
4. ONLY if there are NO conflicts whatsoever:
   - Create the event using the create event tool
   - Confirm creation

CRITICAL: You MUST examine the list events tool response thoroughly. Even if there are existing events close to the requested time, you must check for any time overlap. DO NOT create events if there are ANY conflicts detected in the tool response.

For meeting requests:
1. Always check availability first with list events tool
2. Parse tool response carefully for conflicts
3. If ANY conflict exists → suggest alternatives (do not book)
4. If completely clear → create event
5. Always report actual tool results

Timezone: America/New_York

Remember: You must use tools, not just describe what you would do."""

    async def _extract_calendar_requirements(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Extract calendar requirements from email"""
        email = state.email
        if not email:
            return None

        # Get current date/time for context
        current_date = datetime.now()
        current_year = current_date.year
        current_date_str = current_date.strftime("%Y-%m-%d")

        prompt = f"""Extract calendar info from this email:

SUBJECT: {email.subject}
FROM: {email.sender}
BODY: {email.body}

IMPORTANT: Current date is {current_date_str} and current year is {current_year}.
When parsing dates, always use the current year ({current_year}) unless explicitly specified otherwise.

Return JSON:
{{
    "is_meeting_request": true/false,
    "requested_datetime": "{current_year}-08-19T13:00:00-04:00",
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

            # Validate and correct the datetime to ensure current year
            if requirements.get("requested_datetime"):
                requirements["requested_datetime"] = self._validate_and_correct_datetime(
                    requirements["requested_datetime"],
                    current_year
                )

            return requirements

        except Exception as e:
            self.logger.error(f"Failed to extract calendar requirements: {e}")
            return None

    def _validate_and_correct_datetime(self, datetime_str: str, current_year: int) -> str:
        """Validate and correct datetime to use current year if needed, with timezone"""
        try:
            from zoneinfo import ZoneInfo
            
            # Parse the datetime string
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))

            # If year is not current year, update it
            if dt.year != current_year:
                dt = dt.replace(year=current_year)
                self.logger.info(f"Corrected year from {datetime_str} to {dt.isoformat()}")

            # Ensure timezone is set - default to America/Toronto if naive
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("America/Toronto"))
                self.logger.info(f"Added timezone America/Toronto to datetime: {dt.isoformat()}")

            return dt.isoformat()
        except Exception as e:
            self.logger.error(f"Error validating datetime {datetime_str}: {e}")
            # Fallback: add timezone to original string if it doesn't have one
            if 'T' in datetime_str and '+' not in datetime_str and 'Z' not in datetime_str:
                return datetime_str + "-04:00"  # EDT timezone offset
            return datetime_str

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

                # Add current context for better understanding
                current_time = datetime.now()
                current_time_str = current_time.strftime("%A, %B %d, %Y at %I:%M %p")

            except ValueError:
                formatted_time = requested_datetime
                current_time_str = "current time unavailable"

            return f"""USE YOUR CALENDAR TOOLS to schedule this meeting:
- Title: {subject}
- Date/Time: {formatted_time}
- Duration: {duration} minutes
- Attendees: {', '.join(attendees)}
- Description: {description}
- Timezone: America/Toronto (EDT/EST)

CONTEXT: Current time is {current_time_str}. The meeting should be scheduled for the correct year ({datetime.now().year}).

IMPORTANT TIMEZONE REQUIREMENT: When creating events, use the exact datetime from this request: {requested_datetime}
This datetime already includes proper timezone information. Do NOT modify the datetime format.

WORKFLOW:
1. First, check availability for {formatted_time} using list events tool
2. If there's a conflict with existing events:
   - Find 2 alternative available time slots (prefer same day if possible, otherwise next business days)
   - Return: "CONFLICT_DETECTED: Alternative times: [time1, time2]"
   - Do NOT create the event
3. If no conflict exists:
   - Create the event using create event tool with the exact datetime: {requested_datetime}
   - Confirm successful creation

IMPORTANT: Always check for conflicts before creating events. Use timezone-aware datetime strings."""
        else:
            return f"""Find available times for:
- Title: {subject}
- Duration: {duration} minutes
- Attendees: {', '.join(attendees)}

Suggest 3 business day options."""

    def _parse_agent_result(self, result: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Parse agent result into structured format with conflict detection"""

        # Get the last message content
        messages = result.get("messages", [])
        if not messages:
            return {"action_taken": "no_response", "availability_status": "unknown"}

        last_message = messages[-1]

        # Log the actual message for debugging
        self.logger.info(f"Last message type: {type(last_message)}")
        self.logger.info(f"Last message content: {last_message}")

        output = last_message.content if hasattr(last_message, 'content') else str(last_message)
        output_lower = output.lower()

        # Check for tool calls in the message
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            self.logger.info(f"Tool calls detected: {last_message.tool_calls}")

        # Priority 1: Check for conflict detection with alternatives
        if "conflict_detected" in output_lower:
            suggested_times = self._extract_alternative_times(output)
            return {
                "action_taken": "conflict_detected",
                "availability_status": "conflict",
                "suggested_times": suggested_times,
                "message": "Meeting time conflicts with existing event. Alternative times suggested."
            }

        # Priority 2: Check for successful event creation
        if "created" in output_lower or "scheduled" in output_lower or "successfully" in output_lower:
            # Extract meeting link if available
            meeting_link = self._extract_meeting_link(output)

            booked_event = {
                "summary": requirements.get("subject", "Meeting"),
                "datetime": requirements.get("requested_datetime"),
                "attendees": requirements.get("attendees", [])
            }

            if meeting_link:
                booked_event["meeting_link"] = meeting_link

            return {
                "action_taken": "meeting_booked",
                "availability_status": "available",
                "booked_event": booked_event,
                "attendees_notified": requirements.get("attendees", [])
            }

        # Priority 3: Check for general conflicts without alternatives
        elif "conflict" in output_lower or "not available" in output_lower or "busy" in output_lower:
            return {
                "action_taken": "alternatives_needed",
                "availability_status": "conflict",
                "suggested_times": [],
                "message": "Meeting time conflicts detected but no alternatives provided."
            }

        # Priority 4: Check for availability confirmation
        elif "available" in output_lower and "no conflict" in output_lower:
            return {
                "action_taken": "availability_confirmed",
                "availability_status": "available",
                "suggested_times": []
            }

        # Priority 5: Check for tool usage
        elif hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return {
                "action_taken": "tools_called",
                "availability_status": "processing",
                "tool_calls": [tc.get('name') for tc in last_message.tool_calls] if hasattr(last_message, 'tool_calls') else []
            }

        # Default: General processing
        else:
            return {
                "action_taken": "processed",
                "availability_status": "unknown",
                "message": output[:200]
            }

    def _extract_alternative_times(self, output: str) -> List[str]:
        """Extract alternative time slots from agent output"""
        import re

        # Look for the specific format: "Alternative times: [time1, time2]"
        pattern = r"Alternative times?\s*:\s*\[(.*?)\]"
        match = re.search(pattern, output, re.IGNORECASE)

        if match:
            times_str = match.group(1)
            # Split by comma and clean up
            times = [time.strip().strip('"\'') for time in times_str.split(',')]
            self.logger.info(f"Extracted alternative times: {times}")
            return times

        # Fallback: look for time patterns in the output
        time_patterns = [
            r'\d{1,2}:\d{2}\s*(?:AM|PM)',
            r'\d{1,2}\s*(?:AM|PM)',
            r'\w+,?\s+\w+\s+\d{1,2},?\s+\d{4}\s+at\s+\d{1,2}:\d{2}\s*(?:AM|PM)'
        ]

        suggested_times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            suggested_times.extend(matches)

        # Remove duplicates and limit to 2
        unique_times = list(dict.fromkeys(suggested_times))[:2]
        self.logger.info(f"Fallback extracted times: {unique_times}")
        return unique_times

    def _extract_meeting_link(self, output: str) -> Optional[str]:
        """Extract meeting link from agent output"""
        import re

        # Look for various meeting link patterns
        link_patterns = [
            r'https://meet\.google\.com/[a-z0-9-]+',
            r'https://zoom\.us/j/\d+',
            r'https://teams\.microsoft\.com/[^\s]+',
            r'https://[^\s]*meet[^\s]*',
            r'Meeting link:\s*(https?://[^\s]+)',
            r'Join at:\s*(https?://[^\s]+)',
            r'Link:\s*(https?://[^\s]+)'
        ]

        for pattern in link_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                link = match.group(1) if match.groups() else match.group(0)
                self.logger.info(f"Extracted meeting link: {link}")
                return link

        return None

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
