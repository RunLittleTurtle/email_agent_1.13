"""
calendar_agent.py
=================
Google Calendar MCP Agent - Actually uses MCP tools for real conflict detection.
Simplified and under 400 lines.
"""

import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

class CalendarMCPAgent:
    """Calendar Agent using Gumloop's Google Calendar MCP with REAL MCP tool calls."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._initialized = False
        self._client = None

    async def initialize(self):
        """Initialize MCP connection."""
        if self._initialized:
            return

        url = os.environ.get("MCP_SERVER_GOOGLE_AGENDA")
        api_key = os.environ.get("GUMCP_API_KEY")
        
        if not url:
            raise RuntimeError("Missing MCP_SERVER_GOOGLE_AGENDA in .env")
        if not api_key:
            raise RuntimeError("Missing GUMCP_API_KEY in .env")

        logger.info("Connecting to Gumloop Calendar MCP with API key...")

        try:
            servers = {
                "google-calendar": {
                    "url": url,
                    "transport": "sse",
                    "headers": {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                }
            }

            self._client = MultiServerMCPClient(servers)
            tools = await self._client.get_tools()

            # Index tools by name
            for tool in tools:
                self._tools[tool.name] = tool
                if "/" in tool.name:
                    _, short_name = tool.name.split("/", 1)
                    self._tools[short_name] = tool

            logger.info(f"✓ Loaded {len(tools)} calendar tools")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize calendar MCP: {e}")
            raise RuntimeError(f"Calendar MCP initialization failed: {e}")

    async def list_events_mcp(self, days_ahead: int = 7) -> Dict[str, Any]:
        """ACTUALLY call MCP list_events tool."""
        if not self._initialized:
            await self.initialize()

        tool = self._tools.get("list_events")
        if not tool:
            return {"error": f"list_events tool not found. Available: {list(self._tools.keys())}"}

        try:
            # Simplify MCP call - let Gumloop handle default time ranges
            logger.info(f"🔍 MCP Call: list_events (simplified parameters)")
            result = await tool.ainvoke({
                "calendar_id": "primary",
                "max_results": 20
            })

            # Parse MCP result
            if isinstance(result, str):
                import json
                try:
                    result = json.loads(result)
                except:
                    logger.error(f"Failed to parse MCP result: {result}")
                    return {"error": "Failed to parse MCP response", "raw_result": result}

            logger.info(f"✅ MCP returned: {len(result.get('items', []))} events")
            return result

        except Exception as e:
            logger.error(f"MCP list_events failed: {e}")
            return {"error": str(e)}

    async def create_event_mcp(self, summary: str, start_datetime: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """ACTUALLY call MCP create_event tool."""
        if not self._initialized:
            await self.initialize()

        tool = self._tools.get("create_event")
        if not tool:
            return {"error": f"create_event tool not found. Available: {list(self._tools.keys())}"}

        try:
            # Calculate end time
            start_dt = datetime.fromisoformat(start_datetime.replace("Z", "+00:00"))
            end_dt = start_dt + timedelta(minutes=duration_minutes)

            # Format for Gumloop MCP (flat structure)
            event_data = {
                "calendar_id": "primary",
                "summary": summary,
                "start_datetime": start_dt.strftime("%Y-%m-%d %H:%M"),
                "end_datetime": end_dt.strftime("%Y-%m-%d %H:%M"),
                "time_zone": "America/New_York",
                "description": f"Meeting created from email request"
            }

            logger.info(f"🔧 MCP Call: create_event - {summary} at {event_data['start_datetime']}")
            result = await tool.ainvoke(event_data)

            logger.info(f"✅ MCP Event Created: {result}")
            return {"status": "created", "summary": summary, "start": start_datetime, "result": result}

        except Exception as e:
            logger.error(f"MCP create_event failed: {e}")
            return {"error": str(e), "summary": summary}

    def _parse_datetime_from_event(self, event_time: Dict) -> Optional[datetime]:
        """Parse datetime from MCP event response."""
        try:
            dt_str = event_time.get("dateTime") or event_time.get("date")
            if not dt_str:
                return None

            # Handle different formats
            if "T" in dt_str:
                return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            else:
                # All-day event
                return datetime.fromisoformat(dt_str + "T00:00:00")
        except:
            return None

    async def check_conflicts_with_mcp(self, requested_start: datetime, duration_minutes: int = 60) -> Dict[str, Any]:
        """ACTUALLY check conflicts using MCP list_events."""
        logger.info(f"🔍 Checking conflicts for {requested_start.strftime('%Y-%m-%d %H:%M')}")

        # Get events from MCP for the requested day
        start_of_day = requested_start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = requested_start.replace(hour=23, minute=59, second=59, microsecond=0)

        # Call MCP to get actual events
        events_response = await self.list_events_mcp(days_ahead=30)

        if "error" in events_response:
            logger.error(f"Failed to get events from MCP: {events_response['error']}")
            return {"status": "error", "message": "Could not check calendar", "error": events_response["error"]}

        events = events_response.get("items", [])
        requested_end = requested_start + timedelta(minutes=duration_minutes)

        # Check for actual conflicts
        for event in events:
            event_start = self._parse_datetime_from_event(event.get("start", {}))
            event_end = self._parse_datetime_from_event(event.get("end", {}))

            if event_start and event_end:
                # Check for overlap
                if requested_start < event_end and requested_end > event_start:
                    logger.info(f"⚠️ CONFLICT DETECTED: {event.get('summary', 'Untitled')} ({event_start} - {event_end})")

                    # Generate alternative times
                    option1 = event_end  # Right after conflicting event
                    option2 = requested_start + timedelta(hours=2)  # 2 hours later

                    # Ensure alternatives are in business hours
                    if option1.hour > 17:
                        option1 = (option1 + timedelta(days=1)).replace(hour=9)
                    if option2.hour > 17:
                        option2 = (option2 + timedelta(days=1)).replace(hour=9)

                    return {
                        "status": "conflict",
                        "conflicting_event": event.get("summary", "Untitled"),
                        "conflict_time": f"{event_start.strftime('%I:%M %p')} - {event_end.strftime('%I:%M %p')}",
                        "suggested_alternatives": [
                            {
                                "datetime": option1.isoformat(),
                                "formatted": option1.strftime('%A, %B %d, %Y at %I:%M %p')
                            },
                            {
                                "datetime": option2.isoformat(),
                                "formatted": option2.strftime('%A, %B %d, %Y at %I:%M %p')
                            }
                        ]
                    }

        logger.info("✅ No conflicts found")
        return {"status": "available"}

    async def ainvoke(self, state) -> "AgentState":
        """LangGraph interface - ACTUALLY uses MCP tools."""
        from src.models.state import CalendarData

        if not self._initialized:
            await self.initialize()

        # Extract meeting request from email
        email_body = state.email.body.lower() if state.email else ""
        summary = f"Meeting: {state.email.subject}" if state.email else "Meeting"

        # Extract date from context
        if not (state.extracted_context and state.extracted_context.dates_mentioned):
            # No date found - suggest next business day
            now = datetime.now()
            next_day = now + timedelta(days=1)
            while next_day.weekday() >= 5:  # Skip weekends
                next_day += timedelta(days=1)

            calendar_data = CalendarData(
                meeting_request={
                    "action": "no_date_found",
                    "error": "No date found in email",
                    "suggested_dates": [
                        next_day.replace(hour=14, minute=0).strftime('%A, %B %d, %Y at %I:%M %p'),
                        (next_day + timedelta(days=1)).replace(hour=10, minute=0).strftime('%A, %B %d, %Y at %I:%M %p')
                    ]
                }
            )
            state.calendar_data = calendar_data
            state.add_message("assistant", "No date found in email. Suggested alternatives provided.")
            return state

        extracted_date = state.extracted_context.dates_mentioned[0]
        current_year = datetime.now().year

        # Adjust year if needed
        if extracted_date.year != current_year:
            extracted_date = extracted_date.replace(year=current_year)

        # Check if date is in past
        if extracted_date.date() < datetime.now().date():
            # Generate future alternatives
            days_ahead = (7 - (datetime.now().weekday() - extracted_date.weekday())) % 7
            if days_ahead == 0:
                days_ahead = 7

            option1 = datetime.now() + timedelta(days=days_ahead)
            option1 = option1.replace(hour=extracted_date.hour, minute=extracted_date.minute)
            option2 = option1 + timedelta(days=7)

            calendar_data = CalendarData(
                meeting_request={
                    "action": "date_in_past",
                    "error": f"Requested date {extracted_date.strftime('%B %d')} is in the past",
                    "suggested_dates": [
                        option1.strftime('%A, %B %d, %Y at %I:%M %p'),
                        option2.strftime('%A, %B %d, %Y at %I:%M %p')
                    ]
                }
            )
            state.calendar_data = calendar_data
            state.add_message("assistant", f"Date is in past. Suggested: {option1.strftime('%A, %B %d at %I:%M %p')} or {option2.strftime('%A, %B %d at %I:%M %p')}")
            return state

        # ACTUALLY check for conflicts using MCP
        logger.info("🔍 Using MCP to check for real calendar conflicts...")
        conflict_result = await self.check_conflicts_with_mcp(extracted_date)

        if conflict_result["status"] == "conflict":
            # Conflict found - return alternatives
            calendar_data = CalendarData(
                meeting_request={
                    "action": "conflict_detected",
                    "error": f"Conflict with {conflict_result['conflicting_event']} at {conflict_result['conflict_time']}",
                    "suggested_dates": [alt["formatted"] for alt in conflict_result["suggested_alternatives"]]
                }
            )
            state.calendar_data = calendar_data
            state.add_message("assistant", f"Time conflict detected with '{conflict_result['conflicting_event']}'. Suggested alternatives provided.")
            return state

        elif conflict_result["status"] == "available":
            # No conflicts - create the event using MCP
            logger.info("✅ No conflicts found - creating event with MCP")
            create_result = await self.create_event_mcp(
                summary=summary,
                start_datetime=extracted_date.isoformat(),
                duration_minutes=60
            )

            if "error" in create_result:
                calendar_data = CalendarData(
                    meeting_request={
                        "action": "creation_failed",
                        "error": f"Failed to create event: {create_result['error']}"
                    }
                )
                state.calendar_data = calendar_data
                state.add_message("assistant", f"Failed to create meeting: {create_result['error']}")
            else:
                calendar_data = CalendarData(
                    meeting_request={
                        "action": "created",
                        "summary": summary,
                        "start_datetime": extracted_date.isoformat()
                    }
                )
                state.calendar_data = calendar_data
                state.add_message("assistant", f"✅ Successfully created meeting '{summary}' for {extracted_date.strftime('%A, %B %d at %I:%M %p')}")

            return state

        else:
            # Error checking conflicts
            calendar_data = CalendarData(
                meeting_request={
                    "action": "check_failed",
                    "error": conflict_result.get("message", "Failed to check calendar")
                }
            )
            state.calendar_data = calendar_data
            state.add_message("assistant", "Could not check calendar for conflicts")
            return state


# Direct MCP tools for LangGraph
async def get_calendar_tools_for_langgraph():
    """Get calendar MCP tools directly for LangGraph ToolNode."""
    url = os.environ.get("MCP_SERVER_GOOGLE_AGENDA")
    if not url:
        raise RuntimeError("Missing MCP_SERVER_GOOGLE_AGENDA in .env")

    servers = {
        "google-calendar": {
            "url": url,
            "transport": "sse"
        }
    }

    client = MultiServerMCPClient(servers)
    return await client.get_tools()
