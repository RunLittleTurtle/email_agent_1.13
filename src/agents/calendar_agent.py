"""
Calendar Agent using Pipedream MCP Server
Core business logic for calendar operations with Google Calendar via MCP tools
"""

import json
import os
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
    Calendar agent using Pipedream MCP server for Google Calendar operations.
    Handles availability checking and event creation.
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

        tools = await client.get_tools()
        self.logger.info(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")
        return tools

    @traceable(name="calendar_analyze", tags=["calendar", "analysis"])
    async def analyze_availability(self, state: AgentState) -> AgentState:
        """
        Analyze calendar request and check availability.
        Does NOT create any events - only checks and reports.
        """
        try:
            if not state.extracted_context:
                state.add_error("No extracted context for calendar processing")
                return state

            self.logger.info("Analyzing calendar request for availability")

            # Extract calendar requirements
            requirements = await self._extract_calendar_requirements(state)

            if not requirements or not requirements.get("is_meeting_request"):
                state.add_error("No meeting request found in email")
                return state

            # Get MCP tools
            tools = await self._get_mcp_tools()
            if not tools:
                state.add_error("No MCP tools available")
                return state

            # Execute availability check
            result = await self._check_availability(requirements, tools)

            # Parse and store results
            parsed_result = self._parse_agent_result(result, requirements)

            # LOG THE PARSED RESULT FOR DEBUGGING
            self.logger.info(f"Parsed result - Action: {parsed_result.get('action_taken')}, Status: {parsed_result.get('availability_status')}")

            # Create calendar data
            calendar_data = CalendarData(
                events_checked=parsed_result.get("events_checked", []),
                availability_status=parsed_result.get("availability_status", "unknown"),
                suggested_times=parsed_result.get("suggested_times", []),
                action_taken=parsed_result.get("action_taken", ""),
                meeting_request={
                    "title": requirements.get("subject", "Meeting"),
                    "requested_datetime": requirements.get("requested_datetime"),
                    "duration_minutes": requirements.get("duration_minutes", 60),
                    "attendees": requirements.get("attendees", []),
                    "description": requirements.get("description", "")
                }
            )

            state.calendar_data = calendar_data

            # Determine if this is ready to book based on analysis
            is_available = parsed_result.get("availability_status") not in ["conflict", "unknown"]
            has_alternatives = len(parsed_result.get("suggested_times", [])) > 0
            is_meeting_request = requirements.get("is_meeting_request", False)

            # Set booking intent flags for LLM router
            state.response_metadata["booking_intent"] = {
                "requirements": requirements,
                "ready_to_book": is_available and is_meeting_request and not has_alternatives,
                "slot_available": is_available,
                "alternatives_suggested": has_alternatives,
                "analysis_complete": True
            }

            self.logger.info(f"Calendar analysis complete - ready_to_book: {state.response_metadata['booking_intent']['ready_to_book']}, slot_available: {is_available}")
            self.logger.info("LLM router will make final routing decision")

            # Add AI response to state
            self._add_analysis_to_state(state, result, parsed_result)

            return state

        except Exception as e:
            self.logger.error(f"Calendar analysis failed: {e}", exc_info=True)
            state.add_error(f"Calendar analysis error: {str(e)}")
            return state

    @traceable(name="calendar_book", tags=["calendar", "booking"])
    async def create_event(self, state: AgentState) -> AgentState:
        """
        Create calendar event after human approval.
        This method assumes availability has been confirmed and approved.
        """
        try:
            self.logger.info("Creating calendar event after approval")

            booking_intent = state.response_metadata.get("booking_intent", {})
            requirements = booking_intent.get("requirements", {})

            if not requirements:
                state.add_error("No booking requirements found")
                return state

            # Get MCP tools
            tools = await self._get_mcp_tools()
            if not tools:
                state.add_error("No MCP tools available for booking")
                return state

            # Execute booking
            result = await self._book_event(requirements, tools)

            # CRITICAL: Extract and store the AI response IMMEDIATELY
            messages_list = result.get("messages", [])
            booking_response = None

            # Find the last AI message with actual content
            for msg in reversed(messages_list):
                if hasattr(msg, 'content') and msg.content:
                    booking_response = msg.content
                    self.logger.info(f"Found booking response: {booking_response[:200]}")
                    break

            # Parse booking result
            parsed_result = self._parse_agent_result(result, requirements)

            # Update calendar data with booking info
            if state.calendar_data:
                state.calendar_data.booked_event = parsed_result.get("booked_event")
                state.calendar_data.action_taken = "meeting_booked"
                state.calendar_data.attendees_notified = parsed_result.get("attendees_notified", [])

            # CRITICAL: Add the full booking response to output and messages
            if booking_response:
                # Add to messages for history
                self._add_message(
                    state,
                    f"Calendar Agent: {booking_response}",
                    metadata=parsed_result
                )

                # CRITICAL: Ensure output list exists and add the booking confirmation
                if not hasattr(state, 'output') or state.output is None:
                    state.output = []

                # Add the FULL booking response with all details
                state.output.append({
                    "agent": "CALENDAR AGENT",
                    "message": booking_response  # This contains meeting link, attendees, etc.
                })

                self.logger.info(f"✅ Added booking response to output: {booking_response[:100]}...")
                self.logger.info(f"Output now has {len(state.output)} entries")

            else:
                # Fallback if we couldn't extract the response
                self.logger.warning("Could not extract booking response from result")
                if parsed_result.get("booked_event"):
                    fallback_msg = f"Successfully created calendar event: {requirements.get('subject')} at {requirements.get('requested_datetime')}"

                    self._add_message(state, f"Calendar Agent: {fallback_msg}", metadata=parsed_result)

                    if not hasattr(state, 'output') or state.output is None:
                        state.output = []

                    state.output.append({
                        "agent": "CALENDAR AGENT",
                        "message": fallback_msg
                    })
                else:
                    state.add_error("Failed to create calendar event - no confirmation received")
                    self.logger.error("Failed to create calendar event")

            return state

        except Exception as e:
            self.logger.error(f"Calendar booking failed: {e}", exc_info=True)
            state.add_error(f"Booking error: {str(e)}")
            return state

    async def _check_availability(self, requirements: Dict[str, Any], tools: List) -> Dict:
        """Check calendar availability without booking"""
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        agent = create_react_agent(llm, tools)

        task = self._format_availability_check_task(requirements)
        messages = [
            SystemMessage(content=self._get_availability_check_system_message()),
            HumanMessage(content=task)
        ]

        result = await agent.ainvoke({"messages": messages})
        self.logger.info("Availability check completed")
        return result

    async def _book_event(self, requirements: Dict[str, Any], tools: List) -> Dict:
        """Book calendar event"""
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        agent = create_react_agent(llm, tools)

        task = self._format_booking_task(requirements)
        messages = [
            SystemMessage(content=self._get_booking_system_message()),
            HumanMessage(content=task)
        ]

        result = await agent.ainvoke({"messages": messages})
        self.logger.info("Event booking completed")
        return result

    def _get_availability_check_system_message(self) -> str:
        """System message for availability checking only"""
        return """You are a calendar assistant with Google Calendar access through MCP tools.

CRITICAL: This is an AVAILABILITY CHECK ONLY. DO NOT create any events.

Your job is to:
1. Use the list-events tool to check the requested time slot
2. Look for any conflicts with existing events
3. If conflicts exist, suggest 2-3 alternative times and days
4. Report your findings clearly and in bullet points

IMPORTANT: Only check and report. Never create events during availability checking.

Timezone: America/New_York"""

    def _get_booking_system_message(self) -> str:
        """System message for direct booking"""
        return """You are a calendar assistant with Google Calendar access through MCP tools.

This is a BOOKING task. Availability has been confirmed and approved.

Your job is to:
1. Use the create-event tool to book the meeting immediately
2. Include all attendees and details provided
3. Confirm successful creation
4. Return the event details and meeting link if available

Proceed directly to booking.

Timezone: America/New_York"""

    def _format_availability_check_task(self, requirements: Dict[str, Any]) -> str:
        """Format task for availability check only"""
        requested_datetime = requirements.get("requested_datetime")
        duration = requirements.get("duration_minutes", 30)

        if requested_datetime:
            try:
                dt = datetime.fromisoformat(requested_datetime)
                formatted_time = dt.strftime("%A, %B %d, %Y at %I:%M %p")
            except ValueError:
                formatted_time = requested_datetime

            return f"""CHECK AVAILABILITY (DO NOT BOOK):

Requested Time: {formatted_time}
Duration: {duration} minutes

Instructions:
1. Use list-events tool to check for conflicts at this time
2. If there's a conflict, find 2-3 alternative available slots
3. At least one of the alternatives slots MUST be another day
4. For the time slots, you must respect the working hours, between 9h00 and 17h00
5. Report what you found

Remember: This is only checking - do not create any events."""

        return "Check calendar availability for the requested meeting time"

    def _format_booking_task(self, requirements: Dict[str, Any]) -> str:
        """Format task for direct booking"""
        return f"""CREATE CALENDAR EVENT NOW:

Event Details:
- Title: {requirements.get('subject', 'Meeting')}
- DateTime: {requirements.get('requested_datetime')}
- Duration: {requirements.get('duration_minutes', 30)} minutes
- Attendees: {', '.join(requirements.get('attendees', []))}
- Description: {requirements.get('description', '')}

Use the create-event tool to book this meeting immediately.
Include the meeting link in your response."""

    async def _extract_calendar_requirements(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Extract calendar requirements from email"""
        email = state.email
        if not email:
            return None

        current_date = datetime.now()
        current_year = current_date.year
        current_date_str = current_date.strftime("%Y-%m-%d")

        prompt = f"""Extract calendar info from this email:

SUBJECT: {email.subject}
FROM: {email.sender}
BODY: {email.body}

Current date: {current_date_str} (year {current_year})

Return JSON:
{{
    "is_meeting_request": true/false,
    "requested_datetime": "ISO format with timezone",
    "duration_minutes": 30,
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

            # Validate datetime
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
        """Validate and correct datetime to use current year if needed"""
        try:
            from zoneinfo import ZoneInfo

            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))

            if dt.year != current_year:
                dt = dt.replace(year=current_year)
                self.logger.info(f"Corrected year to {current_year}")

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("America/Toronto"))

            return dt.isoformat()
        except Exception as e:
            self.logger.error(f"Error validating datetime: {e}")
            if 'T' in datetime_str and '+' not in datetime_str and 'Z' not in datetime_str:
                return datetime_str + "-04:00"
            return datetime_str

    def _parse_agent_result(self, result: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Parse agent result into structured format - let LLM router handle routing logic"""
        messages = result.get("messages", [])
        if not messages:
            return {"action_taken": "no_response", "availability_status": "unknown"}

        last_message = messages[-1]
        output = last_message.content if hasattr(last_message, 'content') else str(last_message)

        self.logger.info(f"Parsing agent output: {output[:200]}")

        # Just extract basic info, let LLM router make routing decisions
        output_lower = output.lower()

        # Check for successful event creation (for booking phase)
        if "created" in output_lower or "scheduled" in output_lower or "successfully" in output_lower:
            self.logger.info("📅 Detected: Event CREATED")
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
                "availability_status": "booked",
                "booked_event": booked_event,
                "attendees_notified": requirements.get("attendees", []),
                "message": output,
                "full_response": output
            }

        # For analysis phase, just return raw info
        suggested_times = self._extract_alternative_times(output)

        return {
            "action_taken": "analyzed",
            "availability_status": "pending_routing",  # Let LLM router decide
            "suggested_times": suggested_times,
            "message": output,
            "full_response": output
        }

    def _extract_alternative_times(self, output: str) -> List[Dict[str, Any]]:
        """Extract alternative time slots from output"""
        import re

        pattern = r"Alternative times?\s*:\s*\[(.*?)\]"
        match = re.search(pattern, output, re.IGNORECASE)

        if match:
            times_str = match.group(1)
            times = [time.strip().strip('"\'') for time in times_str.split(',')]
            return [{"time_slot": time, "status": "suggested"} for time in times if time]

        # Fallback patterns
        time_patterns = [
            r'\d{1,2}:\d{2}\s*(?:AM|PM)',
            r'\d{1,2}\s*(?:AM|PM)',
            r'\w+,?\s+\w+\s+\d{1,2},?\s+\d{4}\s+at\s+\d{1,2}:\d{2}\s*(?:AM|PM)'
        ]

        suggested_times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            suggested_times.extend(matches)

        unique_times = list(dict.fromkeys(suggested_times))[:3]
        return [{"time_slot": time, "status": "suggested"} for time in unique_times if time]

    def _extract_meeting_link(self, output: str) -> Optional[str]:
        """Extract meeting link from output"""
        import re

        link_patterns = [
            r'https://meet\.google\.com/[a-z0-9-]+',
            r'https://zoom\.us/j/\d+',
            r'https://teams\.microsoft\.com/[^\s]+',
            r'Meeting link:\s*(https?://[^\s]+)'
        ]

        for pattern in link_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                link = match.group(1) if match.groups() else match.group(0)
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

    def _add_analysis_to_state(self, state: AgentState, result: Dict, parsed_result: Dict):
        """Add analysis results to state messages and output"""
        messages_list = result.get("messages", [])
        if messages_list and hasattr(messages_list[-1], 'content'):
            full_ai_response = messages_list[-1].content
            self._add_message(
                state,
                f"Calendar Agent: {full_ai_response}",
                metadata=parsed_result
            )

            if not hasattr(state, 'output'):
                state.output = []
            state.output.append({
                "agent": "CALENDAR AGENT",
                "message": full_ai_response
            })

    # Backward compatibility method
    @traceable(name="calendar_agent_process", tags=["agent", "calendar"])
    async def process(self, state: AgentState) -> AgentState:
        """Legacy method for backward compatibility - redirects to analyze_availability"""
        return await self.analyze_availability(state)
