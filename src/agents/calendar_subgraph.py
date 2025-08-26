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
from .calendar_llm_router import CalendarLLMRouter

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
    subgraph.add_node("llm_routing", llm_routing_node)
    subgraph.add_node("human_booking_review", human_booking_review_node)
    subgraph.add_node("calendar_booking", calendar_booking_node)

    # Set entry point
    subgraph.set_entry_point("calendar_analysis")

    # Analysis always goes to LLM routing
    subgraph.add_edge("calendar_analysis", "llm_routing")

    # Add routing from LLM routing node
    subgraph.add_conditional_edges(
        "llm_routing",
        route_after_llm_decision,
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


@traceable(name="llm_routing_node", tags=["calendar", "llm_routing", "node"])
async def llm_routing_node(state: AgentState) -> AgentState:
    """
    Node that uses LLM to determine routing after calendar analysis.
    Makes the intelligent routing decision and stores it in state.
    """
    logger.info("🤖 LLM Routing Node - Starting LLM-based routing decision")
    
    try:
        # Initialize LLM router
        logger.info("📝 Initializing CalendarLLMRouter...")
        llm_router = CalendarLLMRouter()
        logger.info("✅ CalendarLLMRouter initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize CalendarLLMRouter: {e}", exc_info=True)
        # Fallback to exit route
        state.response_metadata["llm_routing_decision"] = "exit"
        return state
    
    # Get data needed for routing decision
    logger.info("📊 Extracting data from state...")
    booking_intent = state.response_metadata.get("booking_intent", {})
    requirements = booking_intent.get("requirements", {})
    logger.info(f"📋 Requirements: {requirements}")
    
    # Get calendar data from analysis - clean it for LLM router
    calendar_data = {}
    if state.calendar_data:
        # Only include suggested_times if there are actual conflicts
        booking_intent = state.response_metadata.get("booking_intent", {})
        is_available = booking_intent.get("slot_available", False)
        
        calendar_data = {
            "availability_status": state.calendar_data.availability_status,
            "action_taken": state.calendar_data.action_taken,
            "conflicts": state.calendar_data.conflicts,
            "availability": state.calendar_data.availability
        }
        
        # Only include suggested_times if slot is NOT available (indicating real conflicts)
        if not is_available:
            calendar_data["suggested_times"] = state.calendar_data.suggested_times
        else:
            calendar_data["suggested_times"] = []  # Clear suggested times for available slots
            
        logger.info(f"📅 Calendar data (cleaned for LLM): {calendar_data}")
    else:
        logger.warning("⚠️ No calendar_data found in state")
    
    # Get the analysis output from the last output message
    analysis_output = ""
    if state.output and len(state.output) > 0:
        # AgentOutput is a Pydantic model, access message attribute directly
        analysis_output = state.output[-1].message
        logger.info(f"💬 Analysis output: {analysis_output[:200]}...")
    else:
        logger.warning("⚠️ No output messages found in state")
    
    # Use LLM to make routing decision
    logger.info("🧠 Calling LLM router for decision...")
    decision = await llm_router.decide_availability_route(
        calendar_data=calendar_data,
        analysis_output=analysis_output,
        requirements=requirements
    )
    logger.info(f"✅ LLM decision received: {decision}")
    
    # Update booking intent with LLM decision
    state.response_metadata["booking_intent"].update({
        "ready_to_book": decision.get("ready_to_book", False),
        "slot_available": decision.get("slot_available", False),
        "alternatives": decision.get("detected_conflicts", []),
        "llm_confidence": decision.get("confidence", 0.0),
        "llm_reason": decision.get("reason", "")
    })
    
    route = decision.get("route", "exit")
    
    # Store the routing decision for the conditional edge to use
    state.response_metadata["llm_routing_decision"] = route
    
    if route == "review":
        logger.info(f"✅ LLM Router: No conflict detected - will route to human review (confidence: {decision.get('confidence', 0.0)})")
    else:
        # Exit cases (will go back to supervisor)
        if not decision.get("slot_available", True):
            logger.info(f"⚠️ LLM Router: Conflict detected - will exit to supervisor (reason: {decision.get('reason', '')[:100]})")
            # Ensure the supervisor knows about the conflict
            state.response_metadata["calendar_conflict"] = True
            state.response_metadata["calendar_alternatives"] = decision.get("detected_conflicts", [])
        else:
            logger.info(f"ℹ️ LLM Router: Not a booking request - will exit to supervisor (reason: {decision.get('reason', '')[:100]})")
    
    return state


@traceable(name="route_after_llm_decision", tags=["calendar", "routing"])
def route_after_llm_decision(state: AgentState) -> str:
    """
    Simple synchronous router that reads the LLM decision from state.
    This is called by the conditional edge after the LLM routing node.
    """
    decision = state.response_metadata.get("llm_routing_decision", "exit")
    logger.info(f"📍 Routing based on LLM decision: {decision}")
    return decision


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
