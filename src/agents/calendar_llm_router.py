"""
LLM-based Router for Calendar Decisions
Uses GPT-4o to make intelligent routing decisions based on context
"""

import json
import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langsmith import traceable
import structlog

logger = structlog.get_logger()


class CalendarLLMRouter:
    """
    LLM-based router for calendar workflow decisions.
    Analyzes context to determine if slot is available or has conflicts.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1):
        """Initialize the LLM router with low temperature for consistency"""
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.logger = logger

    @traceable(name="llm_routing_decision", tags=["routing", "llm"])
    async def decide_availability_route(
        self,
        calendar_data: Dict[str, Any],
        analysis_output: str,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to determine if the calendar slot is available or has conflicts.

        Returns:
            Dict with:
                - route: "review" (available) or "exit" (conflict/other)
                - reason: explanation for the decision
                - confidence: confidence score
                - slot_available: boolean
                - ready_to_book: boolean
        """

        prompt = self._build_routing_prompt(calendar_data, analysis_output, requirements)

        try:
            response = await self.llm.ainvoke(prompt)
            decision = self._parse_routing_response(response.content)

            self.logger.info(
                f"LLM Routing Decision: {decision['route']} "
                f"(confidence: {decision['confidence']}, reason: {decision['reason'][:100]}...)"
            )

            return decision

        except Exception as e:
            self.logger.error(f"LLM routing failed: {e}", exc_info=True)
            # Fallback to safe default
            return {
                "route": "exit",
                "reason": f"Routing error: {str(e)}",
                "confidence": 0.0,
                "slot_available": False,
                "ready_to_book": False,
                "detected_conflicts": [],
                "alternatives_suggested": False
            }

    def _build_routing_prompt(
        self,
        calendar_data: Dict[str, Any],
        analysis_output: str,
        requirements: Dict[str, Any]
    ) -> str:
        """Build the prompt for routing decision"""

        return f"""You are a calendar routing assistant. Analyze the calendar check results and determine the correct route.

CALENDAR ANALYSIS OUTPUT:
{analysis_output}

EXTRACTED CALENDAR DATA:
- Availability Status: {calendar_data.get('availability_status', 'unknown')}
- Events Checked: {calendar_data.get('events_checked', [])}
- Suggested Times: {calendar_data.get('suggested_times', [])}
- Action Taken: {calendar_data.get('action_taken', '')}

MEETING REQUIREMENTS:
- Title: {requirements.get('subject', 'Unknown')}
- Requested Time: {requirements.get('requested_datetime', 'Unknown')}
- Duration: {requirements.get('duration_minutes', 60)} minutes
- Attendees: {requirements.get('attendees', [])}
- Is Meeting Request: {requirements.get('is_meeting_request', False)}

ROUTING RULES:
1. Route to "review" if ALL of these are true:
   - The requested time slot is AVAILABLE (no conflicts)
   - This is a valid meeting request (is_meeting_request = true)
   - The system successfully checked the calendar
   - NO alternative times were suggested

2. Route to "exit" if ANY of these are true:
   - There is a CONFLICT at the requested time
   - This is NOT a meeting request
   - The calendar check failed
   - Alternative times were suggested (indicating conflict)
   - The availability status is "conflict" or "unknown"

IMPORTANT: Look for key phrases in the analysis output:
- Phrases indicating AVAILABLE: "available", "no conflicts", "free", "can schedule", "slot is open", "proceed with booking"
- Phrases indicating CONFLICT: "conflict", "busy", "already booked", "not available", "alternative times", "overlapping"

Return a JSON object with:
{{
    "route": "review" or "exit",
    "reason": "Clear explanation of the routing decision",
    "confidence": 0.0 to 1.0,
    "slot_available": true or false,
    "ready_to_book": true or false,
    "detected_conflicts": ["list of any conflicts found"],
    "alternatives_suggested": true or false
}}

Analyze carefully and return ONLY the JSON object:"""

    def _parse_routing_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a routing decision"""

        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                # Fallback parsing if no valid JSON
                decision = self._fallback_parse(response)

            # Validate required fields
            if "route" not in decision:
                decision["route"] = "exit"
            if decision["route"] not in ["review", "exit"]:
                decision["route"] = "exit"

            # Ensure all fields exist
            decision.setdefault("reason", "Parsed from LLM response")
            decision.setdefault("confidence", 0.5)
            decision.setdefault("slot_available", decision["route"] == "review")
            decision.setdefault("ready_to_book", decision["route"] == "review")
            decision.setdefault("detected_conflicts", [])
            decision.setdefault("alternatives_suggested", False)

            return decision

        except Exception as e:
            self.logger.error(f"Failed to parse routing response: {e}")
            return {
                "route": "exit",
                "reason": f"Parse error: {str(e)}",
                "confidence": 0.0,
                "slot_available": False,
                "ready_to_book": False,
                "detected_conflicts": [],
                "alternatives_suggested": False
            }

    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing if JSON extraction fails"""
        response_lower = response.lower()

        # Simple heuristic based on keywords
        if "review" in response_lower and "available" in response_lower:
            return {
                "route": "review",
                "reason": "Detected availability keywords",
                "confidence": 0.6,
                "slot_available": True,
                "ready_to_book": True,
                "detected_conflicts": [],
                "alternatives_suggested": False
            }
        else:
            return {
                "route": "exit",
                "reason": "Default to exit for safety",
                "confidence": 0.3,
                "slot_available": False,
                "ready_to_book": False,
                "detected_conflicts": [],
                "alternatives_suggested": False
            }
