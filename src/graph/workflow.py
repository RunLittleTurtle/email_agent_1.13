"""
Agent Inbox MVP Workflow
Simplified workflow using only implemented agents: email_processor -> supervisor -> adaptive_writer
With interrupt_before for human-in-the-loop via Agent Inbox
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime
from langsmith import traceable

from src.models.state import AgentState
from src.models.context import RuntimeContext
from src.agents.email_processor import EmailProcessorAgent
from src.agents.supervisor import SupervisorAgent
from src.agents.adaptive_writer import AdaptiveWriterAgent
from src.agents.calendar_subgraph import create_calendar_subgraph
from src.agents.rag_agent import RAGAgent
from src.agents.crm_agent import CRMAgent
from src.agents.email_sender import EmailSenderAgent
from src.agents.router import router_node
from src.memory.store_manager import StoreManager
from src.memory.memory_utils import MemoryUtils

import structlog

logger = structlog.get_logger()


# Global variables (will be initialized in create_workflow)
email_processor_agent = None
supervisor_agent = None
adaptive_writer_agent = None
calendar_agent = None
rag_agent = None
crm_agent = None
email_sender_agent = None

# Memory management
store_manager = None
memory_utils = None


@traceable
async def email_processor_node(state: AgentState, runtime: Optional[Runtime[RuntimeContext]] = None) -> AgentState:
    """Process incoming email and extract context with memory enrichment"""
    logger.info("📧 Email Processor Node")
    
    # Enrich state with user memory if runtime context available
    if runtime and store_manager:
        try:
            # Runtime contains the context data directly
            user_id = getattr(runtime, 'user_id', 'default_user')
            state = await memory_utils.enrich_state_with_memory(state, user_id)
        except Exception as e:
            logger.warning(f"Could not enrich with memory: {e}")
    
    return await email_processor_agent.ainvoke(state, runtime)


@traceable
async def supervisor_node(state: AgentState, runtime: Optional[Runtime[RuntimeContext]] = None) -> AgentState:
    """Analyze email and determine next action with contextual recommendations"""
    logger.info("🔍 Supervisor Node")
    
    # Get contextual recommendations from memory
    if runtime and memory_utils:
        try:
            user_id = getattr(runtime, 'user_id', 'default_user')
            recommendations = await memory_utils.get_contextual_recommendations(state, user_id)
            if recommendations:
                state.add_insight(f"Memory recommendations: {recommendations}")
        except Exception as e:
            logger.warning(f"Could not get memory recommendations: {e}")
    
    return await supervisor_agent.ainvoke(state, runtime)


@traceable
async def adaptive_writer_node(state: AgentState, runtime: Optional[Runtime[RuntimeContext]] = None) -> AgentState:
    """Generate draft response (this will be interrupted for human review)"""
    logger.info("✍️ Adaptive Writer Node")
    
    result = await adaptive_writer_agent.ainvoke(state, runtime)
    
    # Extract insights from interaction for learning
    if runtime and memory_utils:
        try:
            user_id = getattr(runtime, 'user_id', 'default_user')
            await memory_utils.extract_insights_from_email(result, user_id)
        except Exception as e:
            logger.warning(f"Could not extract insights: {e}")
    
    return result


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
        action_name = f"📧 Review: {subject_short}"
        description_text = f"Review draft response for email from {sender_name}"
    else:
        action_name = "📧 Review Email Draft"
        description_text = "Review draft response for email"

    # Build detailed review information (matching working calendar booking pattern)
    review_details = f"""📧 Email Response Review Required

Original Email:
• From: {state.email.sender if state.email else 'Unknown'}
• Subject: {state.email.subject if state.email else 'Unknown'}
• Received: {date_received}

Draft Response:
{state.draft_response or 'No draft response available'}

Status: ✅ Draft response is ready to send

Would you like to proceed with sending this email response?"""

    # Create the interrupt for human review (EXACT MATCH of working calendar booking pattern)
    human_response = interrupt({
        "action_request": {
            "action": action_name,
            "args": {
                "review_details": review_details,
                "message": "Please approve or reject this email response",
                "draft_response": state.draft_response,
                "email_context": email_context_str
            }
        },
        "config": {
            "allow_accept": True,   # Sends the email
            "allow_ignore": True,   # Cancels sending
            "allow_respond": True,  # Future: Allow modifications
            "timeout": 300          # 5 minute timeout
        },
        "description": description_text
    })

    # Process the human response (using EXACT working pattern from calendar booking)
    email_approved = _process_human_email_response(human_response, state)
    state.response_metadata["email_approved"] = email_approved

    if email_approved:
        logger.info("✅ Human APPROVED sending the email")
        state.response_metadata["decision"] = "accept"
    else:
        logger.info("❌ Human REJECTED sending the email")
        state.response_metadata["decision"] = "ignore"

    return state


def _process_human_email_response(human_response: Any, state: AgentState) -> bool:
    """
    Process the human response from the interrupt.
    Returns True if email sending is approved, False otherwise.
    (EXACT COPY of working _process_human_booking_response pattern)
    """
    # Handle list responses (Agent Inbox can return updates as a list)
    if isinstance(human_response, list):
        human_response = human_response[-1] if human_response else None

    if not human_response:
        logger.warning("No human response received - defaulting to reject")
        return False

    if isinstance(human_response, dict):
        response_type = human_response.get("type", "ignore")

        if response_type == "accept":
            return True
        elif response_type == "ignore":
            return False
        elif response_type in ["response", "edit"]:
            # Future enhancement: Handle modifications
            response_args = human_response.get("args", {})
            if response_args:
                state.human_feedback = (
                    response_args if isinstance(response_args, str)
                    else response_args.get("feedback", "")
                )
                logger.info(f"Human provided modifications: {state.human_feedback}")
            return False  # For now, treat modifications as rejection
        else:
            logger.warning(f"Unknown response type: {response_type}")
            return False
    else:
        logger.warning(f"Unexpected response format: {type(human_response)}")
        return False


def create_workflow(store: Optional[InMemoryStore] = None) -> StateGraph:
    """
    Create full workflow with all agents + human interrupt + router + persistence
    Flow: email_processor -> supervisor -> (calendar/rag/crm) -> adaptive_writer -> human_review -> router
    """
    global email_processor_agent, supervisor_agent, adaptive_writer_agent
    global calendar_agent, rag_agent, crm_agent, email_sender_agent
    global store_manager, memory_utils

    logger.info("🚀 Creating Enhanced Agent Inbox Workflow with Memory")

    # Initialize memory system
    if store is None:
        store = InMemoryStore()
    store_manager = StoreManager(store)
    memory_utils = MemoryUtils(store_manager)

    # Initialize agents
    email_processor_agent = EmailProcessorAgent()
    supervisor_agent = SupervisorAgent()
    adaptive_writer_agent = AdaptiveWriterAgent()
    
    # Initialize other agents (not used in MVP but needed for full workflow)
    calendar_agent = None  # create_calendar_subgraph()  # TODO: Implement
    rag_agent = RAGAgent()
    crm_agent = CRMAgent()
    email_sender_agent = EmailSenderAgent()

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

    # Conditional store usage: detect if running in LangGraph API context
    import os
    import sys
    
    # Check if we're running via LangGraph API by looking for specific indicators
    is_langgraph_api = (
        "langgraph" in " ".join(sys.argv) or 
        os.getenv("LANGGRAPH_API_MODE") == "true" or
        "langgraph_api" in str(sys.modules.keys())
    )
    
    if is_langgraph_api:
        logger.info("☁️ Detected LangGraph API context - using built-in persistence")
        app = workflow.compile()
    else:
        logger.info("📝 Local development context - using custom InMemoryStore")
        app = workflow.compile(store=store)

    logger.info("✅ Enhanced Workflow compiled successfully with:")
    logger.info("    - Multi-agent support with memory integration")
    logger.info("    - Long-term memory stores")
    logger.info("    - API-managed persistence")
    logger.info("    - Built-in time travel debugging")
    logger.info("    - Fault tolerance with checkpoint recovery")
    logger.info("    - Thread-based memory management")

    return app


def create_runtime_context(
    user_id: str,
    user_email: str,
    preferences: Optional[Dict[str, Any]] = None
) -> RuntimeContext:
    """
    Create runtime context for workflow invocation with user context
    
    Args:
        user_id: User identifier
        user_email: User email address
        preferences: User preferences
        
    Returns:
        Runtime context for agent execution
    """
    if memory_utils:
        return memory_utils.create_runtime_context(user_id, user_email, preferences)
    
    # Fallback if memory_utils not initialized
    return RuntimeContext(
        user_id=user_id,
        user_email=user_email,
        user_preferences=preferences or {},
        available_tools=["gmail", "calendar", "documents", "contacts"],
        timezone="UTC",
        language="en"
    )
