"""
Calendar Subgraph
Orchestrates the calendar workflow with human approval for bookings
"""

import os
import json
from langgraph.graph import StateGraph, END
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
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


async def _llm_router_decision(ai_response: str, original_request: str) -> str:
    """
    Use GPT-4o to determine routing based on AI response context.
    More reliable than keyword matching.
    
    Returns:
        - "review": Route to human booking approval 
        - "exit": Route back to supervisor
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    system_prompt = """You are a routing decision maker for a calendar agent workflow.

Your job: Analyze the AI's calendar response and decide the next routing step.

ROUTING OPTIONS:
1. "review" - Route to human booking approval (when slot is AVAILABLE and ready to book)
2. "exit" - Route back to supervisor (when there are CONFLICTS or it's not a booking request)

DECISION CRITERIA:
- If the AI found the requested time slot is AVAILABLE with NO conflicts → "review"  
- If the AI found CONFLICTS or scheduling issues → "exit"
- If the AI suggests alternative times due to conflicts → "exit"
- If it's not a meeting/booking request → "exit"

Respond with ONLY the routing decision: "review" or "exit"."""

    human_prompt = f"""ORIGINAL REQUEST: {original_request}

AI CALENDAR RESPONSE: {ai_response}

Based on the AI's response, what should be the next routing step?
Respond with ONLY: "review" or "exit\""""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        decision = response.content.strip().lower()
        if decision not in ["review", "exit"]:
            logger.warning(f"Invalid LLM router decision: {decision}, defaulting to exit")
            return "exit"
            
        return decision
        
    except Exception as e:
        logger.error(f"LLM router failed: {e}, defaulting to exit")
        return "exit"


@traceable(name="route_after_analysis", tags=["calendar", "routing"])
async def route_after_analysis(state: AgentState) -> str:
    """
    Determine next step after calendar analysis using LLM-based routing.
    Much more reliable than keyword matching.

    Returns:
        - "review": If a booking is ready and needs approval (NO CONFLICT)
        - "exit": If there's a conflict or not a meeting request (goes back to supervisor)
    """
    # Get the AI response from calendar analysis
    ai_response = ""
    if state.output and len(state.output) > 0:
        # Get the last calendar agent response
        for output_entry in reversed(state.output):
            if output_entry.get("agent") == "CALENDAR AGENT":
                ai_response = output_entry.get("message", "")
                break
    
    if not ai_response:
        logger.warning("No AI response found for routing decision, defaulting to exit")
        return "exit"
    
    # Get original email context
    original_request = ""
    if state.email:
        original_request = f"Subject: {state.email.subject}\nBody: {state.email.body}"
    
    logger.info(f"🤖 LLM ROUTER - Analyzing AI response: {ai_response[:100]}...")
    
    # Use LLM to make routing decision
    decision = await _llm_router_decision(ai_response, original_request)
    
    if decision == "review":
        logger.info("✅ LLM ROUTER → human_booking_review (slot available for booking)")
        # Ensure booking intent is set for downstream nodes
        if "booking_intent" not in state.response_metadata:
            state.response_metadata["booking_intent"] = {
                "ready_to_book": True,
                "slot_available": True,
                "requirements": {}  # Will be extracted by booking node if needed
            }
        return "review"
    else:
        logger.info("⚠️ LLM ROUTER → supervisor (conflicts or not a booking request)")
        # Mark potential conflict for supervisor awareness
        state.response_metadata["calendar_analyzed"] = True
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
