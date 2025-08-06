"""
Agent Inbox MVP Workflow
Simplified workflow using only implemented agents: email_processor -> supervisor -> adaptive_writer
With interrupt_before for human-in-the-loop via Agent Inbox
"""

import os
from typing import Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END, START
# MemorySaver not needed in API mode - persistence handled automatically
from langsmith import traceable

from src.models.state import AgentState
from src.agents.email_processor import EmailProcessorAgent
from src.agents.supervisor import SupervisorAgent
from src.agents.adaptive_writer import AdaptiveWriterAgent

import structlog

logger = structlog.get_logger()


# Global agent variables (will be initialized in create_workflow)
email_processor_agent = None
supervisor_agent = None
adaptive_writer_agent = None


@traceable
async def email_processor_node(state: AgentState) -> AgentState:
    """Process incoming email and extract context"""
    logger.info("📧 Email Processor Node")
    return await email_processor_agent.ainvoke(state)


@traceable
async def supervisor_node(state: AgentState) -> AgentState:
    """Route email based on intent classification (simplified for MVP)"""
    logger.info("🧭 Supervisor Node (MVP - Direct to Writer)")

    # For MVP: Always route to adaptive_writer since other agents aren't implemented
    # Process with supervisor to get intent classification
    updated_state = await supervisor_agent.ainvoke(state)

    # Override routing to always go to adaptive_writer for MVP
    if "routing" not in updated_state.response_metadata:
        updated_state.response_metadata["routing"] = {
            "required_agents": ["adaptive_writer"],
            "completed_agents": [],
            "next": "adaptive_writer",
            "mvp_mode": True
        }
    else:
        # Force routing to adaptive_writer for MVP
        updated_state.response_metadata["routing"]["next"] = "adaptive_writer"
        updated_state.response_metadata["routing"]["mvp_mode"] = True

    return updated_state


@traceable
async def adaptive_writer_node(state: AgentState) -> AgentState:
    """Generate draft response (this will be interrupted for human review)"""
    logger.info("✍️ Adaptive Writer Node")
    return await adaptive_writer_agent.ainvoke(state)


@traceable
async def human_review_node(state: AgentState) -> AgentState:
    """Human-in-the-loop node (will be interrupted by Agent Inbox)"""
    logger.info("⏸️ Human Review Node (Interrupt Point)")

    # This node is where the human will review the draft
    # Agent Inbox will interrupt here and allow human interaction
    state.add_message(
        "system",
        "Draft response ready for human review. Please review and approve, edit, or reject."
    )

    return state


def create_workflow() -> StateGraph:
    """
    Create MVP workflow with only implemented agents + human interrupt
    Flow: email_processor -> supervisor -> adaptive_writer -> human_review -> END
    """
    global email_processor_agent, supervisor_agent, adaptive_writer_agent

    logger.info("🚀 Creating Agent Inbox MVP Workflow...")

    # Initialize agents here (after environment variables are loaded)
    email_processor_agent = EmailProcessorAgent()
    supervisor_agent = SupervisorAgent()
    adaptive_writer_agent = AdaptiveWriterAgent()

    # Create workflow with AgentState
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("email_processor", email_processor_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("adaptive_writer", adaptive_writer_node)
    workflow.add_node("human_review", human_review_node)

    # Define the flow
    workflow.set_entry_point("email_processor")
    workflow.add_edge("email_processor", "supervisor")
    workflow.add_edge("supervisor", "adaptive_writer")
    workflow.add_edge("adaptive_writer", "human_review")
    workflow.add_edge("human_review", END)

    # CRITICAL: Compile with interrupt_before for Agent Inbox
    # Note: In langgraph API mode, persistence is handled automatically
    logger.info("🔧 Compiling with interrupt_after=['adaptive_writer'] for Agent Inbox")

    app = workflow.compile(
        interrupt_after=["adaptive_writer"]  # Human will review the draft here
    )

    logger.info("✅ MVP Workflow compiled successfully with human interrupt")

    return app
