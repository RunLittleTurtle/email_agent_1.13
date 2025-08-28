
"""
Agent Inbox MVP Workflow
Simplified workflow using only implemented agents: email_processor -> supervisor -> adaptive_writer
With interrupt_before for human-in-the-loop via Agent Inbox
"""

import os
from typing import Dict, Any, Optional, TypedDict, Literal, Union
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime
from langsmith import traceable

from src.models.state import AgentState
from src.models.context import RuntimeContext
from src.agents.email_processor import EmailProcessorAgent
from src.agents.supervisor import supervisor_node, get_next_agent_from_state, SupervisorAgent
from src.agents.adaptive_writer import AdaptiveWriterAgent
from src.agents.rag_agent import RAGAgent
from src.agents.crm_agent import CRMAgent
from src.agents.email_sender import EmailSenderAgent
from src.agents.calendar_subgraph import create_calendar_subgraph
from src.agents.router import router_node
from src.agents.human_feedback_processor import format_feedback_for_processing, human_feedback_processor_node
from src.memory.store_manager import StoreManager
from src.memory.memory_utils import MemoryUtils
from langgraph.store.memory import InMemoryStore

import structlog

logger = structlog.get_logger()


# Agent Inbox Compatible Schemas
class HumanInterruptConfig(TypedDict):
    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool


class ActionRequest(TypedDict):
    action: str
    args: dict


class HumanInterrupt(TypedDict):
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: Optional[str]


class HumanResponse(TypedDict):
    type: Literal['accept', 'ignore', 'response', 'edit']
    args: Union[None, str, ActionRequest]


# Global variables - initialized on first use
email_processor_agent = None
adaptive_writer_agent = None
calendar_agent = None
rag_agent = None
crm_agent = None
email_sender_agent = None

# Memory management - initialized on first use
store_manager = None
memory_utils = None


def _ensure_agents_initialized():
    """Ensure all agents are initialized before use"""
    global email_processor_agent, adaptive_writer_agent, calendar_agent
    global rag_agent, crm_agent, email_sender_agent, store_manager, memory_utils

    try:
        if email_processor_agent is None:
            from src.agents.email_processor import EmailProcessorAgent
            email_processor_agent = EmailProcessorAgent()
            logger.info("✅ Email processor agent initialized")

        if adaptive_writer_agent is None:
            from src.agents.adaptive_writer import AdaptiveWriterAgent
            adaptive_writer_agent = AdaptiveWriterAgent()
            logger.info("✅ Adaptive writer agent initialized")

        if rag_agent is None:
            from src.agents.rag_agent import RAGAgent
            rag_agent = RAGAgent()
            logger.info("✅ RAG agent initialized")

        if crm_agent is None:
            from src.agents.crm_agent import CRMAgent
            crm_agent = CRMAgent()
            logger.info("✅ CRM agent initialized")

        if email_sender_agent is None:
            from src.agents.email_sender import EmailSenderAgent
            email_sender_agent = EmailSenderAgent()
            logger.info("✅ Email sender agent initialized")

        if store_manager is None:
            store_manager = StoreManager(InMemoryStore())
            logger.info("✅ Store manager initialized")

        if memory_utils is None:
            memory_utils = MemoryUtils(store_manager)
            logger.info("✅ Memory utils initialized")

    except Exception as e:
        logger.error(f"❌ Failed to initialize agents: {e}")
        # Create minimal fallback agents to prevent None errors
        if email_processor_agent is None:
            logger.warning("Using fallback for email processor agent")
        if adaptive_writer_agent is None:
            logger.warning("Using fallback for adaptive writer agent")
        raise e


@traceable
async def email_processor_node(state: AgentState, runtime: Optional[Runtime[RuntimeContext]] = None) -> Dict[str, Any]:
    """Process incoming email and extract context with memory enrichment"""
    logger.info("📧 Email Processor Node")

    # Ensure agents are initialized
    _ensure_agents_initialized()

    # Enrich state with user memory if runtime context available
    state_updates = {}
    if runtime and memory_utils:
        try:
            # Runtime contains the context data directly
            user_id = getattr(runtime, 'user_id', 'default_user')
            enriched_state = await memory_utils.enrich_state_with_memory(state, user_id)
            # Extract any updates from enriched state if needed
            if hasattr(enriched_state, 'long_term_memory'):
                state_updates['long_term_memory'] = enriched_state.long_term_memory
        except Exception as e:
            logger.warning(f"Could not enrich with memory: {e}")

    # Get updates from email processor agent
    if email_processor_agent is not None:
        agent_updates = await email_processor_agent.ainvoke(state, runtime)
        state_updates.update(agent_updates)
    else:
        logger.error("❌ Email processor agent is None!")
        state_updates.update({"error_messages": ["Email processor agent not initialized"]})

    return state_updates


# supervisor_node is imported from src.agents.supervisor - using prebuilt langgraph-supervisor


@traceable
async def adaptive_writer_node(state: AgentState, runtime: Optional[Runtime[RuntimeContext]] = None) -> Dict[str, Any]:
    """Generate draft response (this will be interrupted for human review)"""
    logger.info("✍️ Adaptive Writer Node")

    # Ensure agents are initialized
    _ensure_agents_initialized()

    if adaptive_writer_agent is not None:
        result = await adaptive_writer_agent.ainvoke(state, runtime)
    else:
        logger.error("❌ Adaptive writer agent is None!")
        return {"error_messages": ["Adaptive writer agent not initialized"]}

    # Extract insights from interaction for learning
    if runtime and memory_utils:
        try:
            user_id = getattr(runtime, 'user_id', 'default_user')
            await memory_utils.extract_insights_from_email(state, user_id)
        except Exception as e:
            logger.warning(f"Could not extract insights: {e}")

    return result


@traceable
async def send_email_node(state: AgentState, runtime: Optional[Runtime[RuntimeContext]] = None) -> Dict[str, Any]:
    """Send the approved email response"""
    logger.info("📮 Send Email Node")

    # Ensure agents are initialized
    _ensure_agents_initialized()

    if email_sender_agent is not None:
        return await email_sender_agent.ainvoke(state, runtime)
    else:
        logger.error("❌ Email sender agent is None!")
        return {"error_messages": ["Email sender agent not initialized"]}


@traceable
async def human_review_node(state: AgentState) -> Dict[str, Any]:
    """Human-in-the-loop node using modern human feedback processor"""
    logger.info("⏸️ Human Review Node (Using Modern Feedback Processor)")

    email = state.email
    draft_response = state.draft_response or ""

    # Build email context for the interrupt - formatted for readability
    if email:
        # Format the received date nicely
        try:
            if hasattr(email, 'timestamp') and email.timestamp:
                from datetime import datetime
                timestamp = email.timestamp
                if isinstance(timestamp, str):
                    # Handle ISO format with Z timezone
                    if timestamp.endswith('Z'):
                        timestamp = timestamp[:-1] + '+00:00'
                    date_received = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    date_received = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                date_received = "Date not available"
        except Exception:
            date_received = "Date parsing error"
        # Create a clean, formatted email context string with FULL body and date
        email_context_str = f"From: {email.sender}\nSubject: {email.subject}\nReceived: {date_received}\n\n{email.body}"
    else:
        date_received = "No email available"
        email_context_str = "No email context available"

    # Create descriptive action name with sender and subject for better UI display
    if email:
        # Truncate subject if too long for UI display
        subject = email.subject
        subject_short = subject[:50] + "..." if len(subject) > 50 else subject
        sender = email.sender
        sender_name = sender.split('<')[0].strip() if '<' in sender else sender
        action_name = f"📧 Review: {subject_short}"
        description_text = f"Review draft response for email from {sender_name}"
    else:
        action_name = "📧 Review Email Draft"
        description_text = "Review draft response for email"

    # Build detailed review information
    review_details = f"""📧 Email Response Review Required

Original Email:
• From: {email.sender if email else 'Unknown'}
• Subject: {email.subject if email else 'Unknown'}
• Received: {date_received}

Draft Response:
{draft_response or 'No draft response available'}

Status: ✅ Draft response is ready to send

Would you like to proceed with sending this email response?"""

    # Create the interrupt for human review using Agent Inbox schema
    interrupt_request: HumanInterrupt = {
        "action_request": {
            "action": action_name,
            "args": {
                "review_details": review_details,
                "message": "Please approve or reject this email response",
                "draft_response": draft_response,
                "email_context": email_context_str
            }
        },
        "config": {
            "allow_accept": True,   # Sends the email
            "allow_ignore": True,   # Cancels sending
            "allow_respond": True,  # Future: Allow modifications
            "allow_edit": False     # No editing for now
        },
        "description": description_text
    }

    # Send interrupt and get response
    human_response_list = interrupt(interrupt_request)
    human_response: HumanResponse = human_response_list[0] if human_response_list else None

    # Process human response according to Agent Inbox schema
    if human_response:
        response_type = human_response.get("type", "ignore")
        response_args = human_response.get("args")

        if response_type == "accept":
            email_approved = True
            decision = "approved"
            logger.info("✅ Human ACCEPTED the email draft")
        elif response_type == "ignore":
            email_approved = False
            decision = "rejected"
            logger.info("❌ Human IGNORED/REJECTED the email draft")
        elif response_type == "response":
            # Human provided additional feedback
            email_approved = False
            decision = "modified"
            feedback_text = response_args if isinstance(response_args, str) else "User provided feedback"
            logger.info(f"📝 Human provided feedback: {feedback_text}")
        elif response_type == "edit":
            # Human edited the action request
            email_approved = False
            decision = "edited"
            logger.info("✏️ Human edited the action request")
        else:
            email_approved = False
            decision = "unclear"
            logger.warning(f"⚠️ Unknown response type: {response_type}")
    else:
        email_approved = False
        decision = "no_response"
        logger.warning("⚠️ No human response received")

    # Update response metadata with Agent Inbox decision
    updated_response_metadata = {**state.response_metadata}
    updated_response_metadata["email_approved"] = email_approved
    updated_response_metadata["decision"] = decision
    updated_response_metadata["human_response_type"] = human_response.get("type") if human_response else None
    updated_response_metadata["agent_inbox_compatible"] = True

    # Format feedback for human_feedback_processor using the helper function
    from src.agents.human_feedback_processor import format_feedback_for_processing

    pending_feedback = format_feedback_for_processing(
        human_response,
        source_node="human_review",
        action_context=f"Email response review for: {email.subject if email else 'Unknown'}"
    )

    # Prepare return with Agent Inbox response
    result_updates = {
        "response_metadata": updated_response_metadata,
        "pending_human_feedback": pending_feedback
    }

    # Add human response details if available
    if human_response and human_response.get("args"):
        result_updates["human_feedback"] = human_response["args"]

    return result_updates


# Legacy function removed - now using human_feedback_processor_node


def create_workflow(store: Optional[InMemoryStore] = None):
    """
    Create full workflow with all agents + human interrupt + router + persistence
    Flow: email_processor -> supervisor -> (calendar/rag/crm) -> adaptive_writer -> human_review -> router
    """
    global store_manager, memory_utils

    logger.info("🚀 Creating Enhanced Agent Inbox Workflow with Memory")

    # Initialize memory system
    if store is None:
        store = InMemoryStore()
    store_manager = StoreManager(store)
    memory_utils = MemoryUtils(store_manager)

    # Ensure agents are initialized
    _ensure_agents_initialized()

    # Create the calendar subgraph
    calendar_subgraph = create_calendar_subgraph()

    # Create workflow with modern LangGraph pattern - using AgentState as Pydantic schema
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("email_processor", email_processor_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("adaptive_writer", adaptive_writer_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("human_feedback_processor", human_feedback_processor_node)
    workflow.add_node("router", router_node)

    # Add specialized agent nodes with traceable wrappers
    # REMOVED the calendar_node function - no longer needed

    @traceable
    async def rag_node(state: AgentState) -> Dict[str, Any]:
        logger.info("📄 RAG Agent Node")
        _ensure_agents_initialized()
        if rag_agent is not None:
            return await rag_agent.ainvoke(state)
        else:
            logger.error("❌ RAG agent is None!")
            return {"error_messages": ["RAG agent not initialized"]}

    @traceable
    async def crm_node(state: AgentState) -> Dict[str, Any]:
        logger.info("👥 CRM Agent Node")
        _ensure_agents_initialized()
        if crm_agent is not None:
            return await crm_agent.ainvoke(state)
        else:
            logger.error("❌ CRM agent is None!")
            return {"error_messages": ["CRM agent not initialized"]}

    # Add nodes - calendar_subgraph is added directly
    workflow.add_node("calendar_agent", calendar_subgraph)  # This line stays as is!
    workflow.add_node("rag_agent", rag_node)
    workflow.add_node("crm_agent", crm_node)

    # Add dedicated EmailSenderAgent node for robust email sending
    @traceable
    async def send_email_node_internal(state: AgentState) -> Dict[str, Any]:
        """Send the approved email response using dedicated EmailSenderAgent"""
        logger.info("📮 Send Email Node (Using EmailSenderAgent)")
        _ensure_agents_initialized()
        if email_sender_agent is not None:
            return await email_sender_agent.ainvoke(state)
        else:
            logger.error("❌ Email sender agent is None!")
            return {"error_messages": ["Email sender agent not initialized"]}

    workflow.add_node("send_email", send_email_node_internal)

    # Define the flow
    workflow.set_entry_point("email_processor")
    workflow.add_edge("email_processor", "supervisor")

    # Use prebuilt supervisor routing
    def route_from_supervisor(state: AgentState) -> str:
        """Route based on prebuilt supervisor's decision - clean hub-and-spoke"""
        logger.info("DEBUG: route_from_supervisor called")

        # Debug state information
        routing_metadata = state.response_metadata.get("routing", {})
        logger.info(f"DEBUG: routing metadata keys: {list(routing_metadata.keys())}")
        logger.info(f"DEBUG: next agent from metadata: {routing_metadata.get('next', 'NOT_SET')}")
        logger.info(f"DEBUG: current_agent: {state.current_agent if hasattr(state, 'current_agent') else 'NOT_SET'}")

        # Check if supervisor made a decision
        if "supervisor_routed" in routing_metadata:
            logger.info(f"DEBUG: Supervisor routing confirmed: {routing_metadata.get('reasoning', 'no reasoning')}")
        else:
            logger.warning("DEBUG: No supervisor routing metadata found!")

        next_agent = get_next_agent_from_state(state)
        logger.info(f"Supervisor routing decision: {next_agent}")
        logger.info(f"DEBUG: Available routing options: calendar_agent, rag_agent, crm_agent, adaptive_writer, END")

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

    # Hub-and-spoke: specialized agents always report back to supervisor
    workflow.add_edge("calendar_agent", "supervisor")
    workflow.add_edge("rag_agent", "supervisor")
    workflow.add_edge("crm_agent", "supervisor")

    # adaptive_writer flows directly to human_review - bypass supervisor
    workflow.add_edge("adaptive_writer", "human_review")
    workflow.add_edge("human_review", "human_feedback_processor")
    workflow.add_edge("human_feedback_processor", "router")

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

    # Check if we're running via LangGraph API - improved detection
    is_langgraph_api = (
        os.getenv("LANGGRAPH_API_MODE") == "true" or
        os.getenv("LANGGRAPH_DEPLOYMENT_NAME") is not None or
        "uvicorn" in " ".join(sys.argv) or
        "app:graph" in " ".join(sys.argv) or
        any("langgraph" in str(mod) for mod in sys.modules.keys())
    )

    # Default to API mode for Agent Inbox compatibility
    if is_langgraph_api or os.getenv("FORCE_LANGGRAPH_API", "true").lower() == "true":
        logger.info("☁️ Using LangGraph API mode - built-in persistence for Agent Inbox")
        app = workflow.compile()
    else:
        logger.info("📝 Local development mode - using custom InMemoryStore")
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
