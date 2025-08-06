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
async def router_node(state: AgentState) -> str:
    """
    Route workflow based on human decision from the Agent Inbox.

    Expected values in state:
    - state.decision: "accept" | "edit" | "ignore"
    - state.human_feedback (optional): Any text/structured feedback from human

    Returns:
        The name of the next node to run.
    """
    decision = getattr(state, "decision", "ignore")
    feedback = getattr(state, "human_feedback", None)

    logger.info("Routing human decision", decision=decision, feedback=feedback)

    if decision == "accept":
        # Human approved the draft → send it
        logger.info("✅ Draft accepted, routing to send_email")
        return "send_email"

    elif decision == "edit":
        # Human requested edits/feedback → back to Supervisor
        if feedback:
            state.response_metadata["human_feedback"] = feedback
            logger.info("📥 Stored human feedback in state.response_metadata")

        logger.info("✏️ Draft needs modification, routing back to supervisor")
        return "supervisor"

    else:
        # Ignore = stop workflow
        logger.info("🚫 Draft ignored, ending workflow")
        return "__end__"  # END node
