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
# Removed MemorySaver import - using API-managed persistence
from langsmith import traceable

from src.models.state import AgentState
from src.agents.email_processor import EmailProcessorAgent
from src.agents.supervisor import SupervisorAgent
from src.agents.adaptive_writer import AdaptiveWriterAgent
from src.agents.calendar_subgraph import create_calendar_subgraph
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
                    feedback_text = ""
                    if isinstance(response_args, str):
                        feedback_text = response_args
                    elif isinstance(response_args, dict):
                        # For edit type, args might contain the edited draft
                        feedback_text = response_args.get("args", {}).get("draft_response", "")
                        state.draft_response = feedback_text  # Update draft with edited version
                    
                    # Store in both locations for compatibility
                    state.human_feedback = feedback_text
                    if "human_feedback" not in state.response_metadata:
                        state.response_metadata["human_feedback"] = []
                    state.response_metadata["human_feedback"].append(feedback_text)
                logger.info(f"✏️ Human provided instructions: {response_type}")

            # Add human decision as message using LangGraph pattern
            from langchain_core.messages import HumanMessage
            decision_message = HumanMessage(
                content=f"Decision: {state.response_metadata.get('decision', 'unknown')}",
                name="human_reviewer"
            )
            # Note: This will be automatically added to state via add_messages reducer
            state.messages.append(decision_message)
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
    Create full workflow with all agents + human interrupt + router + persistence
    Flow: email_processor -> supervisor -> (calendar/rag/crm) -> adaptive_writer -> human_review -> router
    
    Features:
    - LangGraph 0.6+ modern patterns
    - MemorySaver for conversation persistence 
    - Thread-based memory for multi-session conversations
    - Time travel debugging capabilities
    - Fault tolerance with checkpoint recovery
    """
    global email_processor_agent, supervisor_agent, adaptive_writer_agent
    global rag_agent, crm_agent, email_sender_agent  # Removed calendar_agent from globals

    logger.info("🚀 Creating Agent Inbox Modern Workflow with LangGraph 0.6+ Features...")

    # Initialize all agents
    email_processor_agent = EmailProcessorAgent()
    supervisor_agent = SupervisorAgent()
    adaptive_writer_agent = AdaptiveWriterAgent()
    email_sender_agent = EmailSenderAgent()
    # REMOVED: calendar_agent = CalendarAgent()
    rag_agent = RAGAgent()
    crm_agent = CRMAgent()

    # Create the calendar subgraph
    calendar_subgraph = create_calendar_subgraph()  # <-- NEW LINE

    # Create workflow with modern LangGraph pattern - using AgentState as Pydantic schema
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("email_processor", email_processor_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("adaptive_writer", adaptive_writer_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("router", router_node)

    # Add specialized agent nodes with traceable wrappers
    # REMOVED the calendar_node function - no longer needed

    @traceable
    async def rag_node(state: AgentState) -> AgentState:
        logger.info("📄 RAG Agent Node")
        return await rag_agent.ainvoke(state)

    @traceable
    async def crm_node(state: AgentState) -> AgentState:
        logger.info("👥 CRM Agent Node")
        return await crm_agent.ainvoke(state)

    # Add nodes - calendar_subgraph is added directly
    workflow.add_node("calendar_agent", calendar_subgraph)  # This line stays as is!
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
        """Determine next node based on supervisor's routing decision with completion checks"""
        routing = state.response_metadata.get("routing", {})
        next_agent = routing.get("next", "adaptive_writer")
        completed_agents = routing.get("completed_agents", [])

        # CRITICAL: Override supervisor decision ONLY if no human feedback AND work is complete
        # Check if calendar agent already completed successfully BUT respect human feedback
        has_human_feedback = (
            state.human_feedback or 
            state.response_metadata.get("human_feedback") or
            "HUMAN FEEDBACK FROM AGENT INBOX" in str(state.messages[-3:]) if state.messages else False
        )
        
        if (next_agent == "calendar_agent" and 
            "calendar_agent" in completed_agents and 
            state.calendar_data and 
            state.calendar_data.action_taken and 
            ("successfully" in state.calendar_data.action_taken.lower() or 
             "meeting_booked" in state.calendar_data.action_taken.lower() or
             "event has been created" in state.calendar_data.action_taken.lower()) and
            not has_human_feedback):  # ← ONLY override if NO human feedback
            
            logger.info("🛑 OVERRIDE: Calendar work already complete, forcing route to adaptive_writer")
            next_agent = "adaptive_writer"
            # Update routing to reflect override
            routing["next"] = "adaptive_writer"
            routing["override_reason"] = "Calendar agent already completed successfully"
            state.response_metadata["routing"] = routing
        elif has_human_feedback and next_agent == "calendar_agent":
            logger.info("👤 HUMAN FEEDBACK DETECTED: Respecting supervisor's calendar_agent routing decision")
            routing["override_reason"] = "Human feedback requires calendar modification"
            state.response_metadata["routing"] = routing

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
    # MODERN: LangGraph API handles persistence automatically
    logger.info("🔧 Compiling workflow for LangGraph API compatibility")
    
    # Compile without custom checkpointer (API provides built-in persistence)
    app = workflow.compile()

    logger.info("✅ Modern Workflow compiled successfully with:")
    logger.info("    - Multi-agent support")
    logger.info("    - API-managed persistence")
    logger.info("    - Built-in time travel debugging")
    logger.info("    - Fault tolerance with checkpoint recovery")
    logger.info("    - Thread-based memory management")

    return app
