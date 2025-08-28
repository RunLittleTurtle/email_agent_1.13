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
    - state.response_metadata["human_feedback_processed"]: Contains analyzed feedback
    - state.human_feedback (optional): Any text/structured feedback from human

    Returns:
    - state with updated response_metadata
    """
    decision = state.response_metadata.get("decision", "ignore")
    feedback = state.human_feedback

    # Check if human feedback processor found modifications
    feedback_processed = state.response_metadata.get("human_feedback_processed", {})
    feedback_decision = feedback_processed.get("decision", "")

    logger.info("Routing human decision",
                decision=decision,
                feedback_decision=feedback_decision,
                feedback=feedback)

    if decision == "accept":
        # Human approved the draft → send it
        logger.info("✅ Draft accepted, will route to send_email")
        state.response_metadata["router_decision"] = "send_email"

    elif decision == "instruction" or feedback_decision == "modified":
        # Human provided instructions/feedback → back to Supervisor
        if feedback:
            if "human_feedback" not in state.response_metadata:
                state.response_metadata["human_feedback"] = []
            state.response_metadata["human_feedback"].append(feedback)
            logger.info("📥 Stored human feedback in state.response_metadata")

        # Also store the processed feedback for supervisor to use
        if feedback_decision == "modified":
            modifications = feedback_processed.get("modifications_requested", "")
            if modifications:
                if "human_feedback" not in state.response_metadata:
                    state.response_metadata["human_feedback"] = []
                state.response_metadata["human_feedback"].append(modifications)

        logger.info("✏️ Human provided modifications/instructions, will route back to supervisor")
        state.response_metadata["router_decision"] = "supervisor"

    else:
        # Default case: ignore → end workflow
        logger.info("🛑 Draft ignored, will end workflow")
        state.response_metadata["router_decision"] = "END"

    return state
