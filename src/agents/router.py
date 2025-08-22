"""
Router Node
Handles human-in-the-loop decisions after draft review.
Sends control either to send_email, supervisor (for feedback),
or ends the workflow.
"""

from langsmith import traceable
import structlog

from src.models.state import AgentState

logger = structlog.get_logger().bind(agent="router")


@traceable(name="router_node", tags=["control"])
async def router_node(state: AgentState) -> AgentState:
    """
    Route workflow based on human decision from the Agent Inbox.

    Expected values in state:
    - state.response_metadata["decision"]: "accept" | "instruction" | "ignore"
    - state.human_feedback (optional): Any text/structured feedback from human

    Returns:
    - state with updated response_metadata
    """
    decision = state.response_metadata.get("decision", "ignore")
    feedback = state.human_feedback

    logger.info("Routing human decision", decision=decision, feedback=feedback)

    if decision == "accept":
        # Human approved the draft → send it
        logger.info("✅ Draft accepted, will route to send_email")
        state.response_metadata["router_decision"] = "send_email"

    elif decision == "instruction":
        # Human provided instructions/feedback → check if work is already complete
        if feedback:
            if "human_feedback" not in state.response_metadata:
                state.response_metadata["human_feedback"] = []
            state.response_metadata["human_feedback"].append(feedback)
            logger.info("📥 Stored human feedback in state.response_metadata")

        # CRITICAL: Check if specialized agents already completed their work
        routing = state.response_metadata.get("routing", {})
        completed_agents = routing.get("completed_agents", [])
        
        # If calendar agent completed successfully, don't loop back to supervisor
        if ("calendar_agent" in completed_agents and 
            state.calendar_data and 
            state.calendar_data.action_taken and 
            "successfully" in state.calendar_data.action_taken.lower()):
            
            logger.info("🛑 Calendar work complete - ending workflow to prevent loop")
            state.response_metadata["router_decision"] = "END"
        else:
            logger.info("✏️ Human provided instructions, will route back to supervisor")
            state.response_metadata["router_decision"] = "supervisor"

    else:
        # Default case: ignore → end workflow
        logger.info("🛑 Draft ignored, will end workflow")
        state.response_metadata["router_decision"] = "END"
        
    return state
