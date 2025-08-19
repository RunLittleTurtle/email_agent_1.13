"""
Calendar Subgraph
Orchestrates the calendar workflow with human approval for bookings
"""

from langgraph.graph import StateGraph, END
from langsmith import traceable
import structlog

from ..models.state import AgentState
from .calendar_nodes import (
    calendar_analysis_node,
    human_booking_review_node,
    calendar_booking_node
)

logger = structlog.get_logger()


def create_calendar_subgraph() -> StateGraph:
    """
    Create the calendar subgraph with human interrupt before booking.

    Flow:
    1. calendar_analysis: Check availability and detect conflicts
    2. If available → human_booking_review: Get human approval for booking
       If conflict → EXIT to supervisor with alternatives
    3. calendar_booking: If approved, create the event

    Returns:
        Compiled StateGraph for calendar operations
    """
    logger.info("🔧 Creating calendar subgraph with booking approval workflow")

    # Create the subgraph using the main AgentState
    subgraph = StateGraph(AgentState)

    # Add all nodes
    subgraph.add_node("calendar_analysis", calendar_analysis_node)
    subgraph.add_node("human_booking_review", human_booking_review_node)
    subgraph.add_node("calendar_booking", calendar_booking_node)

    # Set entry point
    subgraph.set_entry_point("calendar_analysis")

    # Add routing from analysis node
    subgraph.add_conditional_edges(
        "calendar_analysis",
        route_after_analysis,
        {
            "review": "human_booking_review",  # Slot available → get approval
            "exit": END  # Conflict or not a meeting request → back to supervisor
        }
    )

    # Add routing from human review node
    subgraph.add_conditional_edges(
        "human_booking_review",
        route_after_human_review,
        {
            "book": "calendar_booking",
            "exit": END
        }
    )

    # Booking node always exits
    subgraph.add_edge("calendar_booking", END)

    # Compile the subgraph
    compiled = subgraph.compile()
    logger.info("✅ Calendar subgraph compiled successfully")

    return compiled


@traceable(name="route_after_analysis", tags=["calendar", "routing"])
def route_after_analysis(state: AgentState) -> str:
    """
    Determine next step after calendar analysis.

    Returns:
        - "review": If a booking is ready and needs approval (NO CONFLICT)
        - "exit": If there's a conflict or not a meeting request (goes back to supervisor)
    """
    booking_intent = state.response_metadata.get("booking_intent", {})

    # Check if slot is available and ready to book
    if booking_intent.get("ready_to_book", False) and booking_intent.get("slot_available", True):
        logger.info("✅ No conflict detected - routing directly to human review for booking approval")
        return "review"
    else:
        # Exit cases (will go back to supervisor)
        if booking_intent.get("slot_available") is False:
            logger.info("⚠️ Conflict detected - exiting to supervisor with alternatives")
            # Ensure the supervisor knows about the conflict
            state.response_metadata["calendar_conflict"] = True
            state.response_metadata["calendar_alternatives"] = booking_intent.get("alternatives", [])
        else:
            logger.info("ℹ️ Not a booking request - exiting to supervisor")
        return "exit"


@traceable(name="route_after_human_review", tags=["calendar", "routing"])
def route_after_human_review(state: AgentState) -> str:
    """
    Determine next step after human review.

    Returns:
        - "book": If human approved the booking
        - "exit": If human rejected or modified the request
    """
    if state.response_metadata.get("booking_approved", False):
        logger.info("✅ Booking approved - routing to create event")
        return "book"
    else:
        logger.info("❌ Booking not approved - exiting to supervisor")
        # Mark that booking was rejected for supervisor awareness
        state.response_metadata["booking_rejected"] = True
        return "exit"


# Optional: Export a convenience function for backward compatibility
async def calendar_node(state: AgentState) -> AgentState:
    """
    Legacy node function that uses the subgraph.
    This allows gradual migration from the old single-node approach.
    """
    logger.info("📅 Calendar Node (using subgraph)")
    subgraph = create_calendar_subgraph()
    return await subgraph.ainvoke(state)
