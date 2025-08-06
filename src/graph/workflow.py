"""
Agent Inbox MVP Workflow
Simplified workflow using only implemented agents: email_processor -> supervisor -> adaptive_writer
With interrupt_before for human-in-the-loop via Agent Inbox
"""

import os
from typing import Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langsmith import traceable

from src.models.state import AgentState
from src.agents.email_processor import EmailProcessorAgent
from src.agents.supervisor import SupervisorAgent
from src.agents.adaptive_writer import AdaptiveWriterAgent
from src.agents.router import router_node

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
    """Human-in-the-loop node using dynamic interrupt for Agent Inbox"""
    logger.info("⏸️ Human Review Node (Dynamic Interrupt for Agent Inbox)")
    
    # Build email context for the interrupt - formatted for readability
    if state.email:
        # Create a clean, formatted email context string
        email_context_str = f"From: {state.email.sender}\nSubject: {state.email.subject}\n\n{state.email.body[:200]}{'...' if len(state.email.body) > 200 else ''}"
    else:
        email_context_str = "No email context available"
    
    # Dynamic interrupt using Agent Inbox expected structure
    human_response = interrupt({
        "action_request": {
            "action": "review_email_draft",
            "args": {
                "draft_response": state.draft_response,
                "email_context": email_context_str,  # Use formatted string instead of dict
                "message": "Please review the draft response and choose an action"
            }
        },
        "config": {
            "allow_accept": True,
            "allow_ignore": True,
            "allow_respond": True,  # This allows providing instructions
            "allow_edit": True      # This allows editing the draft directly
        },
        "description": f"Review draft response for email from {state.email.sender if state.email else 'Unknown'}"
    })
    
    # Process human response when workflow is resumed
    if human_response:
        # Handle case where response might be a list (Agent Inbox can return updates as a list)
        if isinstance(human_response, list):
            # Take the last response if it's a list of updates
            human_response = human_response[-1] if human_response else None
            
        if human_response and isinstance(human_response, dict):
            # Agent Inbox returns response in format: {"type": "accept|ignore|response|edit", "args": ...}
            response_type = human_response.get("type", "ignore")
            response_args = human_response.get("args")
            
            # Map Agent Inbox response types to our workflow decisions
            if response_type == "accept":
                state.response_metadata["decision"] = "accept"
                logger.info("✅ Human accepted the draft")
            elif response_type == "ignore":
                state.response_metadata["decision"] = "ignore"
                logger.info("🚫 Human ignored the draft")
            elif response_type in ["response", "edit"]:
                state.response_metadata["decision"] = "instruction"
                # Extract feedback from args
                if response_args:
                    if isinstance(response_args, str):
                        state.human_feedback = response_args
                    elif isinstance(response_args, dict):
                        # For edit type, args might contain the edited draft
                        state.human_feedback = response_args.get("args", {}).get("draft_response", "")
                        state.draft_response = state.human_feedback  # Update draft with edited version
                logger.info(f"✏️ Human provided instructions: {response_type}")
            
            state.add_message("human", f"Decision: {state.response_metadata.get('decision', 'unknown')}")
        else:
            # Default to ignore if response format is unexpected
            state.response_metadata["decision"] = "ignore"
            logger.warning(f"Unexpected human response format: {type(human_response)}")
    else:
        # Default to ignore if no response
        state.response_metadata["decision"] = "ignore"
        logger.info("⚠️ No human response received, defaulting to ignore")
    
    return state


def create_workflow() -> StateGraph:
    """
    Create MVP workflow with only implemented agents + human interrupt + router
    Flow: email_processor -> supervisor -> adaptive_writer -> human_review -> router -> (send_email/supervisor/END)
    """
    global email_processor_agent, supervisor_agent, adaptive_writer_agent

    logger.info("🚀 Creating Agent Inbox MVP Workflow with Action-Based Human Review...")

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
    workflow.add_node("router", router_node)
    
    # Add placeholder send_email node for now
    async def send_email_node(state: AgentState) -> AgentState:
        logger.info("📮 Send Email Node (Placeholder)")
        state.add_message("system", f"Email would be sent: {state.draft_response[:100]}...")
        return state
    
    workflow.add_node("send_email", send_email_node)

    # Define the flow
    workflow.set_entry_point("email_processor")
    workflow.add_edge("email_processor", "supervisor")
    workflow.add_edge("supervisor", "adaptive_writer")
    workflow.add_edge("adaptive_writer", "human_review")
    workflow.add_edge("human_review", "router")
    
    # Router conditional edges
    workflow.add_conditional_edges(
        "router",
        lambda state: state.response_metadata.get("router_decision", "END"),
        {
            "send_email": "send_email",
            "supervisor": "supervisor",  # Go back to supervisor with feedback
            "END": END
        }
    )
    
    # send_email always ends the workflow
    workflow.add_edge("send_email", END)

    # CRITICAL: Using dynamic interrupts in human_review_node for Agent Inbox
    # Note: In langgraph API mode, persistence is handled automatically
    logger.info("🔧 Compiling workflow with action-based human review")

    app = workflow.compile()

    logger.info("✅ MVP Workflow compiled successfully with action-based human review")

    return app
