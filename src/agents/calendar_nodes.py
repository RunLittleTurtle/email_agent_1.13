"""
Calendar Subgraph Nodes
Node functions for the calendar subgraph workflow
"""

from datetime import datetime
from typing import Dict, Any
from langgraph.types import interrupt
from langsmith import traceable
import structlog

from ..models.state import AgentState
from .calendar_agent import CalendarAgent

logger = structlog.get_logger()


@traceable(name="calendar_analysis_node", tags=["calendar", "analysis", "node"])
async def calendar_analysis_node(state: AgentState) -> AgentState:
    """
    Node that analyzes calendar availability.
    Uses CalendarAgent to check for conflicts and suggest alternatives.
    """
    logger.info("📅 Calendar Analysis Node - Checking availability")

    try:
        agent = CalendarAgent()
        state = await agent.analyze_availability(state)

        # Log the booking intent for debugging
        booking_intent = state.response_metadata.get("booking_intent", {})
        if booking_intent.get("ready_to_book"):
            logger.info("✅ Analysis complete: Slot available, ready for booking approval")
        elif booking_intent.get("slot_available") is False:
            logger.info("⚠️ Analysis complete: Conflict detected, alternatives suggested")
        else:
            logger.info("ℹ️ Analysis complete: Not a booking request")

        return state

    except Exception as e:
        logger.error(f"Calendar analysis failed: {e}", exc_info=True)
        state.add_error(f"Calendar analysis error: {str(e)}")
        return state


@traceable(name="human_booking_review_node", tags=["calendar", "human_review", "node"])
async def human_booking_review_node(state: AgentState) -> AgentState:
    """
    Node that interrupts for human approval before booking.
    Presents booking details and waits for human decision.
    """
    logger.info("🔔 Human Booking Review Node - Requesting approval")

    booking_intent = state.response_metadata.get("booking_intent", {})
    requirements = booking_intent.get("requirements", {})

    # Check if booking is actually needed
    if not booking_intent.get("ready_to_book", False):
        logger.info("No booking approval needed - skipping human review")
        state.response_metadata["booking_approved"] = False
        return state

    # Format the datetime nicely
    try:
        dt = datetime.fromisoformat(requirements.get("requested_datetime", ""))
        formatted_time = dt.strftime("%A, %B %d, %Y at %I:%M %p")
        short_time = dt.strftime("%b %d at %I:%M %p")
    except:
        formatted_time = requirements.get("requested_datetime", "Unknown time")
        short_time = "Unknown time"

    # Build detailed booking information
    booking_details = f"""📅 **Calendar Booking Approval Required**

**Meeting Details:**
• **Title:** {requirements.get('subject', 'Meeting')}
• **Date/Time:** {formatted_time}
• **Duration:** {requirements.get('duration_minutes', 60)} minutes
• **Attendees:** {', '.join(requirements.get('attendees', [])) or 'No attendees specified'}

**Description:**
{requirements.get('description', 'No description provided')}

**Status:** ✅ Time slot is AVAILABLE and ready to book

Would you like to proceed with creating this calendar event?"""

    # Create the interrupt for human review
    human_response = interrupt({
        "action_request": {
            "action": f"📅 Book: {requirements.get('subject', 'Meeting')} - {short_time}",
            "args": {
                "booking_details": booking_details,
                "message": "Please approve or reject this calendar booking",
                "requirements": requirements  # Include raw data for potential UI use
            }
        },
        "config": {
            "allow_accept": True,   # Creates the booking
            "allow_ignore": True,   # Cancels the booking
            "allow_respond": True,  # Future: Allow modifications
            "timeout": 300  # 5 minute timeout
        },
        "description": f"Approve calendar booking for {formatted_time}"
    })

    # Process the human response
    booking_approved = _process_human_booking_response(human_response, state)
    state.response_metadata["booking_approved"] = booking_approved

    if booking_approved:
        logger.info("✅ Human APPROVED the booking")
        state.add_message("system", f"Booking approved for: {requirements.get('subject')}")
    else:
        logger.info("❌ Human REJECTED the booking")
        state.add_message("system", "Calendar booking cancelled by user")

    return state


@traceable(name="calendar_booking_node", tags=["calendar", "booking", "node"])
async def calendar_booking_node(state: AgentState) -> AgentState:
    """
    Node that creates the actual calendar event.
    Only executed after human approval.
    """
    logger.info("📅 Calendar Booking Node - Creating event")

    # Double-check approval
    if not state.response_metadata.get("booking_approved", False):
        logger.warning("Booking node called without approval - skipping")
        state.add_error("Cannot book without approval")
        return state

    try:
        agent = CalendarAgent()
        state = await agent.create_event(state)

        # Log success
        if state.calendar_data and state.calendar_data.booked_event:
            logger.info("✅ Calendar event created successfully")

            # Add user-friendly message
            booking_intent = state.response_metadata.get("booking_intent", {})
            requirements = booking_intent.get("requirements", {})
            state.add_message(
                "system",
                f"✅ Meeting '{requirements.get('subject', 'Meeting')}' has been scheduled"
            )
        else:
            logger.error("Failed to create calendar event")

        return state

    except Exception as e:
        logger.error(f"Calendar booking failed: {e}", exc_info=True)
        state.add_error(f"Booking error: {str(e)}")
        return state


def _process_human_booking_response(human_response: Any, state: AgentState) -> bool:
    """
    Process the human response from the interrupt.
    Returns True if booking is approved, False otherwise.
    """
    # Handle list responses (Agent Inbox can return updates as a list)
    if isinstance(human_response, list):
        human_response = human_response[-1] if human_response else None

    if not human_response:
        logger.warning("No human response received - defaulting to reject")
        return False

    if isinstance(human_response, dict):
        response_type = human_response.get("type", "ignore")

        if response_type == "accept":
            return True
        elif response_type == "ignore":
            return False
        elif response_type in ["response", "edit"]:
            # Future enhancement: Handle modifications
            response_args = human_response.get("args", {})
            if response_args:
                state.human_feedback = (
                    response_args if isinstance(response_args, str)
                    else response_args.get("feedback", "")
                )
                logger.info(f"Human provided modifications: {state.human_feedback}")
            return False  # For now, treat modifications as rejection
        else:
            logger.warning(f"Unknown response type: {response_type}")
            return False
    else:
        logger.warning(f"Unexpected response format: {type(human_response)}")
        return False
