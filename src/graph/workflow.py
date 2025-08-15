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
from src.agents.calendar_agent import CalendarAgent
from src.agents.rag_agent import RAGAgent
from src.agents.crm_agent import CRMAgent
from src.agents.email_sender import EmailSenderAgent
from src.agents.router import router_node

import structlog

logger = structlog.get_logger()


# Global agent variables (will be initialized in create_workflow)
email_processor_agent = None
supervisor_agent = None
adaptive_writer_agent = None
calendar_agent = None
rag_agent = None
crm_agent = None
email_sender_agent = None


@traceable
async def email_processor_node(state: AgentState) -> AgentState:
    """Process incoming email and extract context"""
    logger.info("📧 Email Processor Node")
    return await email_processor_agent.ainvoke(state)


@traceable
async def supervisor_node(state: AgentState) -> AgentState:
    """Route email based on intent classification with multi-agent support"""
    logger.info("🧭 Supervisor Node")
    return await supervisor_agent.ainvoke(state)


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
        # Format the received date nicely
        try:
            if hasattr(state.email, 'timestamp') and state.email.timestamp:
                from datetime import datetime
                if isinstance(state.email.timestamp, str):
                    date_received = datetime.fromisoformat(state.email.timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    date_received = state.email.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                date_received = "Date not available"
        except Exception:
            date_received = "Date parsing error"
        
        # Create a clean, formatted email context string with FULL body and date
        email_context_str = f"From: {state.email.sender}\nSubject: {state.email.subject}\nReceived: {date_received}\n\n{state.email.body}"
    else:
        email_context_str = "No email context available"
    
    # Create descriptive action name with sender and subject for better UI display
    if state.email:
        # Truncate subject if too long for UI display
        subject_short = state.email.subject[:50] + "..." if len(state.email.subject) > 50 else state.email.subject
        sender_name = state.email.sender.split('<')[0].strip() if '<' in state.email.sender else state.email.sender
        action_name = f"📧 {sender_name}: {subject_short}"
        description_text = f"Review draft response for email from {sender_name} - {subject_short}"
    else:
        action_name = "📧 Email review (Unknown sender)"
        description_text = "Review draft response for email from Unknown sender"
    
    # Dynamic interrupt using Agent Inbox expected structure
    human_response = interrupt({
        "action_request": {
            "action": action_name,  # Now shows sender and subject instead of generic "review_email_draft"
            "args": {
                "draft_response": state.draft_response,
                "email_context": email_context_str,  # Now contains full email body and received date
                "message": "Please review the draft response and choose an action"
            }
        },
        "config": {
            "allow_accept": True,
            "allow_ignore": True,
            "allow_respond": True,  # This allows providing instructions
            "allow_edit": True      # This allows editing the draft directly
        },
        "description": description_text  # More descriptive description
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
    Create full workflow with all agents + human interrupt + router
    Flow: email_processor -> supervisor -> (calendar/rag/crm) -> adaptive_writer -> human_review -> router
    """
    global email_processor_agent, supervisor_agent, adaptive_writer_agent
    global calendar_agent, rag_agent, crm_agent, email_sender_agent

    logger.info("🚀 Creating Agent Inbox Phase 2 Workflow with Multi-Agent Support...")

    # Initialize all agents
    email_processor_agent = EmailProcessorAgent()
    supervisor_agent = SupervisorAgent()
    adaptive_writer_agent = AdaptiveWriterAgent()
    email_sender_agent = EmailSenderAgent()
    calendar_agent = CalendarAgent()
    rag_agent = RAGAgent()
    crm_agent = CRMAgent()

    # Create workflow with AgentState
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("email_processor", email_processor_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("adaptive_writer", adaptive_writer_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("router", router_node)
    
    # Add specialized agent nodes with traceable wrappers
    @traceable
    async def calendar_node(state: AgentState) -> AgentState:
        logger.info("📅 Calendar Agent Node")
        return await calendar_agent.ainvoke(state)
    
    @traceable
    async def rag_node(state: AgentState) -> AgentState:
        logger.info("📄 RAG Agent Node")
        return await rag_agent.ainvoke(state)
    
    @traceable
    async def crm_node(state: AgentState) -> AgentState:
        logger.info("👥 CRM Agent Node")
        return await crm_agent.ainvoke(state)
    
    workflow.add_node("calendar_agent", calendar_node)
    workflow.add_node("rag_agent", rag_node)
    workflow.add_node("crm_agent", crm_node)
    
    # Add dedicated EmailSenderAgent node for robust email sending
    @traceable
    async def send_email_node(state: AgentState) -> AgentState:
        """Send the approved email response using dedicated EmailSenderAgent"""
        logger.info("📮 Send Email Node (Using EmailSenderAgent)")
        return await email_sender_agent.ainvoke(state)
        
    workflow.add_node("send_email", send_email_node)

    # Define the flow
    workflow.set_entry_point("email_processor")
    workflow.add_edge("email_processor", "supervisor")
    
    # Add conditional routing from supervisor to specialized agents or adaptive_writer
    def route_from_supervisor(state: AgentState) -> str:
        """Determine next node based on supervisor's routing decision"""
        routing = state.response_metadata.get("routing", {})
        next_agent = routing.get("next", "adaptive_writer")
        
        # Track which agent we're routing to for completion detection
        if next_agent in ["calendar_agent", "rag_agent", "crm_agent"]:
            routing["last_routed_to"] = next_agent
            state.response_metadata["routing"] = routing
        
        logger.info(f"Routing from supervisor to: {next_agent}")
        return next_agent
    
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "calendar_agent": "calendar_agent",
            "rag_agent": "rag_agent",
            "crm_agent": "crm_agent",
            "adaptive_writer": "adaptive_writer",
            "END": END  # In case of critical failure
        }
    )
    
    # Route specialized agents back to supervisor for progress check
    workflow.add_edge("calendar_agent", "supervisor")
    workflow.add_edge("rag_agent", "supervisor")
    workflow.add_edge("crm_agent", "supervisor")
    
    # Continue with adaptive_writer flow
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

    logger.info("✅ Phase 2 Workflow compiled successfully with multi-agent support")

    return app
