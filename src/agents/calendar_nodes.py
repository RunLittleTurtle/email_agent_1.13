"""
Calendar Subgraph Nodes
Node functions for the calendar subgraph workflow
"""

from datetime import datetime
from typing import Dict, Any
from langgraph.types import interrupt
from langsmith import traceable
import structlog

from .human_feedback_processor import format_feedback_for_processing, human_feedback_processor_node

from ..models.state import AgentState
from .calendar_agent import CalendarAgent

logger = structlog.get_logger()


@traceable(name="calendar_analysis_node", tags=["calendar", "analysis", "node"])
async def calendar_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    Node that analyzes calendar availability.
    Uses CalendarAgent to check for conflicts and suggest alternatives.
    """
    logger.info("📅 Calendar Analysis Node - Checking availability")

    try:
        agent = CalendarAgent()
        result = await agent.analyze_availability(state)

        # Log the booking intent for debugging
        response_metadata = result.get("response_metadata", state.response_metadata)
        booking_intent = response_metadata.get("booking_intent", {})
        if booking_intent.get("ready_to_book"):
            logger.info("✅ Analysis complete: Slot available, ready for booking approval")
        elif booking_intent.get("slot_available") is False:
            logger.info("⚠️ Analysis complete: Conflict detected, alternatives suggested")
        else:
            logger.info("ℹ️ Analysis complete: Not a booking request")

        return result

    except Exception as e:
        logger.error(f"Calendar analysis failed: {e}", exc_info=True)
        return {
            "error_messages": [f"Calendar analysis error: {str(e)}"],
            "status": "error"
        }


@traceable(name="human_booking_review_node", tags=["calendar", "human_review", "node"])
async def human_booking_review_node(state: AgentState) -> Dict[str, Any]:
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
        return {
            "response_metadata": {**state.response_metadata, "booking_approved": False}
        }

    # Format the datetime nicely
    try:
        dt = datetime.fromisoformat(requirements.get("requested_datetime", ""))
        formatted_time = dt.strftime("%A, %B %d, %Y at %I:%M %p")
        short_time = dt.strftime("%b %d at %I:%M %p")
    except:
        formatted_time = requirements.get("requested_datetime", "Unknown time")
        short_time = "Unknown time"

    # Build detailed booking information
    booking_details = f"""📅 Calendar Booking Approval Required

Meeting Details:
• Title: {requirements.get('subject', 'Meeting')}
• Date/Time: {formatted_time}
• Duration: {requirements.get('duration_minutes', 60)} minutes
• Attendees: {', '.join(requirements.get('attendees', [])) or 'No attendees specified'}

Description:
{requirements.get('description', 'No description provided')}

Status: ✅ Time slot is AVAILABLE and ready to book

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

    # Format human response for the feedback processor
    pending_feedback = format_feedback_for_processing(
        human_response,
        source_node="calendar_booking_review",
        action_context=f"Calendar booking approval for {requirements.get('subject', 'meeting')} on {short_time}"
    )

    # Process the feedback with LLM
    feedback_result = await human_feedback_processor_node({
        "pending_human_feedback": pending_feedback,
        "email": state.email.dict() if state.email else {}
    })

    # Extract decision from processed feedback
    feedback_metadata = feedback_result.get("response_metadata", {}).get("human_feedback_processed", {})
    decision = feedback_metadata.get("decision", "unclear")
    booking_approved = decision == "approved"

    # Update response metadata
    updated_response_metadata = {**state.response_metadata, "booking_approved": booking_approved}
    updated_response_metadata.update(feedback_result.get("response_metadata", {}))

    # Prepare return with processed feedback messages and agent output
    result_updates = {
        "response_metadata": updated_response_metadata
    }

    # Add the LLM-processed messages to the result
    if "messages" in feedback_result:
        result_updates["messages"] = feedback_result["messages"]

    # Add agent output based on decision
    from ..models.state import AgentOutput
    if booking_approved:
        logger.info("✅ Human APPROVED the booking")
        agent_output = AgentOutput(
            agent="human_booking_review",
            message=f"Booking approved for: {requirements.get('subject')}",
            confidence=1.0,
            data=feedback_metadata
        )
    else:
        logger.info("❌ Human REJECTED/MODIFIED the booking")
        agent_output = AgentOutput(
            agent="human_booking_review",
            message=f"Calendar booking decision: {decision}",
            confidence=1.0,
            data=feedback_metadata
        )

    result_updates["output"] = [agent_output]

    return result_updates




@traceable(name="calendar_booking_node", tags=["calendar", "booking", "node"])
async def calendar_booking_node(state: AgentState) -> Dict[str, Any]:
    """
    Node that creates the actual calendar event.
    Only executed after human approval.
    """
    logger.info("📅 Calendar Booking Node - Creating event")

    # Double-check approval
    response_metadata = state.response_metadata
    if not response_metadata.get("booking_approved", False):
        logger.warning("Booking node called without approval - skipping")
        return {
            "error_messages": ["Cannot book without approval"],
            "status": "error"
        }

    try:
        agent = CalendarAgent()
        result = await agent.create_event(state)

        # Log success
        calendar_data = result.get("calendar_data")
        if calendar_data and hasattr(calendar_data, 'booked_event') and calendar_data.booked_event:
            logger.info("✅ Calendar event created successfully")

            # Add user-friendly message
            booking_intent = response_metadata.get("booking_intent", {})
            requirements = booking_intent.get("requirements", {})

            from ..models.state import AgentOutput
            output_update = {
                "output": [AgentOutput(
                    agent="calendar_booking",
                    message=f"✅ Meeting '{requirements.get('subject', 'Meeting')}' has been scheduled",
                    confidence=1.0
                )]
            }
            result.update(output_update)
        else:
            logger.error("Failed to create calendar event")

        return result

    except Exception as e:
        logger.error(f"Calendar booking failed: {e}", exc_info=True)
        return {
            "error_messages": [f"Booking error: {str(e)}"],
            "status": "error"
        }


def _process_human_booking_response(human_response: Any, state: AgentState) -> bool:
    """
    Legacy function - now just determines approval status for backward compatibility
    The detailed feedback processing is handled by human_feedback_processor
    """
    if isinstance(human_response, list):
        human_response = human_response[-1] if human_response else None

    if not human_response:
        return False

    if isinstance(human_response, dict):
        response_type = human_response.get("type", "ignore")
        return response_type == "accept"

    return False
