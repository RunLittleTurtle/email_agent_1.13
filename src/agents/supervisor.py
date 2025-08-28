"""
Supervisor Agent - Hybrid LangGraph Supervisor
Makes intelligent routing decisions using LLM, works with existing workflow nodes
"""

from typing import Dict, Any, List
from langsmith import traceable
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
import json

from src.models.state import AgentState
from src.agents.base_agent import BaseAgent
import structlog

logger = structlog.get_logger(__name__)


class SupervisorAgent(BaseAgent):
    """
    Hybrid supervisor that makes intelligent routing decisions.
    Uses LLM to analyze context and decide which agent should work next.
    Works with existing workflow nodes instead of recreating agents.
    """

    def __init__(self):
        super().__init__(
            name="supervisor",
            model="gpt-4o",
            temperature=0.1
        )

    async def process(self, state: AgentState, runtime=None) -> Dict[str, Any]:
        """
        Analyze current state and decide which agent should work next.
        Hub-and-spoke supervisor that coordinates all agent work.
        """
        logger.info("🧭 Supervisor analyzing state for routing decision")

        try:
            # Check if this is initial routing or returning from an agent
            is_returning_from_agent = self._is_returning_from_agent(state)

            if is_returning_from_agent:
                logger.info("📥 Agent completed work, supervisor analyzing results")
            else:
                logger.info("🎯 Initial routing decision needed")

            # Build context for routing decision
            context = self._build_context_summary(state)

            # Make routing decision using LLM
            routing_decision = await self._make_routing_decision(context, state, is_returning_from_agent)

            # Create routing metadata
            routing_metadata = {
                "next": routing_decision["next_agent"],
                "reasoning": routing_decision["reasoning"],
                "supervisor_routed": True,
                "is_returning_from_agent": is_returning_from_agent,
                "routed_at": "now"
            }

            # Create supervisor message
            supervisor_msg = self.create_ai_message(
                f"Routing to {routing_decision['next_agent']}: {routing_decision['reasoning']}",
                metadata={"routing_decision": routing_decision}
            )

            logger.info(f"📍 Supervisor routing to: {routing_decision['next_agent']}")

            return {
                "messages": [supervisor_msg],
                "response_metadata": {"routing": routing_metadata},
                "current_agent": routing_decision['next_agent']
            }

        except Exception as e:
            logger.error(f"❌ Supervisor routing failed: {e}", exc_info=True)

            # Fallback to adaptive_writer
            fallback_msg = self.create_ai_message(
                f"Supervisor error, defaulting to adaptive_writer: {str(e)}"
            )

            return {
                "messages": [fallback_msg],
                "response_metadata": {"routing": {"next": "adaptive_writer", "error": str(e)}},
                "current_agent": "adaptive_writer"
            }

    def _is_returning_from_agent(self, state: AgentState) -> bool:
        """Check if supervisor is receiving results from a completed agent"""
        if not state.messages:
            return False

        # Look for recent agent messages indicating completed work
        for msg in state.messages[-3:]:
            if hasattr(msg, 'name'):
                agent_name = getattr(msg, 'name', '')
                if agent_name in ['calendar_agent', 'rag_agent', 'crm_agent', 'adaptive_writer']:
                    logger.info(f"🔍 Detected recent work from: {agent_name}")
                    return True

        # Check if adaptive_writer completed (has draft_response)
        if state.draft_response and state.current_agent == 'adaptive_writer':
            logger.info("🔍 Detected adaptive_writer completion - has draft_response")
            return True

        # Also check if we have new data that suggests agent work was completed
        has_new_data = bool(state.calendar_data or state.document_data or state.contact_data)
        return has_new_data

    def _build_context_summary(self, state: AgentState) -> str:
        """Build context summary for routing decision"""
        context_parts = []

        # Email context
        if state.email:
            context_parts.append(f"EMAIL:")
            context_parts.append(f"- From: {state.email.sender}")
            context_parts.append(f"- Subject: {state.email.subject}")
            context_parts.append(f"- Body preview: {state.email.body[:200]}...")

        # Extracted context from email processor
        if state.extracted_context:
            context_parts.append(f"\nEMAIL REQUIREMENTS:")
            if hasattr(state.extracted_context, 'requested_actions') and state.extracted_context.requested_actions:
                context_parts.append(f"- Actions requested: {state.extracted_context.requested_actions}")
            if hasattr(state.extracted_context, 'dates_mentioned') and state.extracted_context.dates_mentioned:
                context_parts.append(f"- Dates mentioned: {state.extracted_context.dates_mentioned}")

        # Current data status - what agents have provided
        completed_work = []
        calendar_work_complete = False
        if state.calendar_data:
            calendar_info = str(state.calendar_data.action_taken if hasattr(state.calendar_data, 'action_taken') else 'data available')
            completed_work.append(f"✅ Calendar: {calendar_info[:100]}")
            # Check if calendar work is actually complete (found conflicts or available slots)
            if any(keyword in calendar_info.lower() for keyword in ["conflict", "alternative", "available", "suggested", "slots"]):
                calendar_work_complete = True

        if state.document_data:
            completed_work.append(f"✅ Documents: {str(state.document_data)[:100]}...")
        if state.contact_data:
            completed_work.append(f"✅ Contacts: {str(state.contact_data)[:100]}...")

        if completed_work:
            context_parts.append(f"\nCOMPLETED AGENT WORK:")
            for work in completed_work:
                context_parts.append(f"- {work}")

        # Add calendar completion status
        if calendar_work_complete:
            context_parts.append(f"📅 CALENDAR STATUS: Analysis complete - conflicts/alternatives identified")

        # Recent work done - focus on agent outputs
        if state.messages:
            context_parts.append(f"\nRECENT AGENT MESSAGES:")
            agent_messages = [msg for msg in state.messages[-5:]
                            if hasattr(msg, 'name') and getattr(msg, 'name') in ['calendar_agent', 'rag_agent', 'crm_agent', 'adaptive_writer']]
            for msg in agent_messages:
                name = getattr(msg, 'name', 'unknown')
                content = str(getattr(msg, 'content', ''))[:150]
                context_parts.append(f"- {name}: {content}...")

                # Special check for calendar completion
                if name == 'calendar_agent' and any(keyword in content.lower() for keyword in ["conflict", "alternative slots", "available", "suggested", "feel free to choose"]):
                    context_parts.append(f"  ⚠️ CALENDAR ANALYSIS COMPLETE - Ready for response writing")

        # Human feedback - be specific about what type of changes are requested
        if state.human_feedback or state.response_metadata.get("human_feedback_processed"):
            context_parts.append(f"\nHUMAN FEEDBACK: Present - requires agent re-work")

            # Check if feedback is about scheduling/calendar changes
            feedback_text = ""
            if state.human_feedback:
                feedback_text += str(state.human_feedback).lower()
            if state.response_metadata.get("human_feedback_processed"):
                hf_data = state.response_metadata["human_feedback_processed"]
                if isinstance(hf_data, dict):
                    if "decision" in hf_data:
                        context_parts.append(f"- Decision: {hf_data.get('decision', 'unknown')}")
                    # Add any feedback content for analysis
                    for key in ["modifications_requested", "content", "human_readable"]:
                        if key in hf_data and hf_data[key]:
                            feedback_text += str(hf_data[key]).lower()

            # Identify type of feedback for better routing
            if any(word in feedback_text for word in ["time", "schedule", "meeting", "appointment", "calendar", "date", "pm", "am", "hour", "reschedule", "change time"]):
                context_parts.append(f"- TYPE: CALENDAR/SCHEDULING feedback - needs calendar_agent")
            elif any(word in feedback_text for word in ["contact", "person", "people", "invite", "attendee"]):
                context_parts.append(f"- TYPE: CONTACT feedback - needs crm_agent")
            elif any(word in feedback_text for word in ["document", "information", "search", "find", "lookup"]):
                context_parts.append(f"- TYPE: INFORMATION feedback - needs rag_agent")
            else:
                context_parts.append(f"- TYPE: RESPONSE feedback - may need adaptive_writer")

        # Current draft status
        if state.draft_response:
            context_parts.append(f"\nCURRENT DRAFT: {state.draft_response[:150]}...")

        return "\n".join(context_parts)

    async def _make_routing_decision(self, context: str, state: AgentState, is_returning: bool = False) -> Dict[str, Any]:
        """Use LLM to make intelligent routing decision"""

        # Check which agents have already completed their work to prevent loops
        completed_agents = self._get_completed_agents(state)
        logger.info(f"🔍 Already completed agents: {completed_agents}")

        if is_returning:
            system_prompt = """You are a supervisor analyzing results from completed agents:

- calendar_agent: Handles scheduling, meetings, appointments, availability checks, time coordination, BOOKING CHANGES
- rag_agent: Retrieves documents, searches knowledge base, finds information
- crm_agent: Manages contacts, customer data, relationship information
- adaptive_writer: Composes final email responses (only use when all needed data is gathered)

CRITICAL RULES:
1. If calendar_agent has already provided conflict analysis and alternative times → DO NOT route back to calendar_agent
2. If calendar_agent shows "alternatives suggested" or "conflicts identified" → route to adaptive_writer
3. Only route to calendar_agent if NO calendar analysis exists yet
4. If human feedback requests time changes AND calendar_agent hasn't analyzed yet → route to calendar_agent
5. If calendar work is complete → route to adaptive_writer to compose response

An agent just completed work and reported back. Analyze what was accomplished and decide the next step.

Respond in JSON format with your routing decision."""

            user_prompt = f"""An agent just completed work. Analyze the results and decide next steps:

{context}

AGENTS THAT HAVE ALREADY COMPLETED THEIR WORK: {completed_agents}
⚠️ DO NOT route to agents that have already completed their core work!

CRITICAL ANALYSIS:
1. What specific work was just completed by the agent?
2. What NEW information is now available?
3. Has calendar_agent already provided conflict analysis and alternatives?
4. Does HUMAN FEEDBACK request changes that require NEW agent work?
5. Are there any gaps still remaining for the email request?
6. Is all necessary data now gathered to write a complete response?

ROUTING PRIORITY (AVOID INFINITE LOOPS):
- If calendar_agent already provided alternatives/conflicts → route to adaptive_writer (NOT calendar_agent again)
- If calendar_agent hasn't analyzed yet AND scheduling needed → route to calendar_agent
- If human feedback requests contact changes → route to crm_agent
- If human feedback requests information lookup → route to rag_agent
- If all required information is available → route to adaptive_writer
- If task is fully complete → use FINISH

Return JSON:
{{
    "next_agent": "agent_name_or_FINISH",
    "reasoning": "detailed analysis of completed work and why this next step is needed",
    "confidence": 0.0-1.0
}}"""
        else:
            system_prompt = """You are a supervisor routing emails to specialized agents:

- calendar_agent: Handles scheduling, meetings, appointments, availability checks, time coordination
- rag_agent: Retrieves documents, searches knowledge base, finds information
- crm_agent: Manages contacts, customer data, relationship information
- adaptive_writer: Composes final email responses (only use when all needed data is gathered)

Analyze the email context and decide which ONE agent should work first.

Respond in JSON format with your routing decision."""

            user_prompt = f"""Analyze this email and decide initial routing:

{context}

Consider:
1. What does the email request?
2. What agents need to gather information before a response can be written?
3. Start with data gathering agents, save adaptive_writer for last

Return JSON:
{{
    "next_agent": "agent_name",
    "reasoning": "why you chose this agent to work first",
    "confidence": 0.0-1.0
}}"""

        try:
            response = await self._call_llm(user_prompt, system_prompt)
            decision = json.loads(response)

            # Validate next_agent is valid
            valid_agents = ["calendar_agent", "rag_agent", "crm_agent", "adaptive_writer", "FINISH"]
            if decision["next_agent"] not in valid_agents:
                logger.warning(f"Invalid agent {decision['next_agent']}, defaulting to adaptive_writer")
                decision["next_agent"] = "adaptive_writer"
                decision["reasoning"] = "Invalid routing corrected to adaptive_writer"

            # Prevent infinite loops - don't route to already completed agents
            if decision["next_agent"] in completed_agents and decision["next_agent"] != "adaptive_writer":
                logger.warning(f"🛑 LOOP PREVENTION: {decision['next_agent']} already completed, routing to adaptive_writer")
                decision["next_agent"] = "adaptive_writer"
                decision["reasoning"] = f"Loop prevention: {decision['next_agent']} already completed their work, routing to response writer"

            # Special case: if adaptive_writer is complete (has draft_response), route to FINISH
            if state.draft_response and state.current_agent == "adaptive_writer":
                logger.info("✅ Adaptive writer completed - draft response ready, routing to FINISH")
                decision["next_agent"] = "FINISH"
                decision["reasoning"] = "Adaptive writer completed with draft response - ready for human review"

            return decision

        except Exception as e:
            logger.error(f"Failed to parse routing decision: {e}")
            return {
                "next_agent": "adaptive_writer",
                "reasoning": f"Routing decision failed: {str(e)}",
                "confidence": 0.5
            }

    def _get_completed_agents(self, state: AgentState) -> List[str]:
        """
        Determine which agents have already completed their core work
        to prevent infinite routing loops.
        """
        completed = []

        # Check calendar agent completion
        if state.calendar_data and hasattr(state.calendar_data, 'action_taken'):
            action = str(state.calendar_data.action_taken).lower()
            if any(keyword in action for keyword in ["conflict", "alternative", "available", "suggested", "slots", "feel free to choose"]):
                completed.append("calendar_agent")
                logger.info("✅ Calendar agent marked as completed - provided conflict analysis/alternatives")

        # Check if calendar agent completed via messages
        if state.messages:
            for msg in state.messages[-5:]:  # Check recent messages
                if hasattr(msg, 'name') and getattr(msg, 'name') == 'calendar_agent':
                    content = str(getattr(msg, 'content', '')).lower()
                    if any(keyword in content for keyword in ["conflict", "alternative slots", "available", "suggested", "feel free to choose"]):
                        if "calendar_agent" not in completed:
                            completed.append("calendar_agent")
                            logger.info("✅ Calendar agent marked as completed via message analysis")

        # Check RAG agent completion
        if state.document_data:
            completed.append("rag_agent")
            logger.info("✅ RAG agent marked as completed - document data available")

        # Check CRM agent completion
        if state.contact_data:
            completed.append("crm_agent")
            logger.info("✅ CRM agent marked as completed - contact data available")

        return completed


# Lazy supervisor instance (created when needed to avoid initialization issues)
supervisor_agent = None


@traceable(name="supervisor_node", tags=["supervisor", "routing", "hub"])
async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Main supervisor node for workflow integration.
    Uses hybrid supervisor to make intelligent routing decisions.
    Hub-and-spoke coordinator for all agent work.
    """
    global supervisor_agent
    if supervisor_agent is None:
        supervisor_agent = SupervisorAgent()

    logger.info("🧭 Supervisor node processing - hub-and-spoke coordination")
    result = await supervisor_agent.process(state)
    logger.info(f"📍 Supervisor routing decision: {result.get('current_agent', 'unknown')}")
    return result


def get_next_agent_from_state(state: AgentState) -> str:
    """
    Helper for workflow routing.
    Gets next agent from supervisor's routing decision.
    """
    logger.info("🔍 DEBUG: get_next_agent_from_state called")

    # Debug response_metadata structure
    response_metadata = state.response_metadata or {}
    logger.info(f"🔍 DEBUG: response_metadata keys: {list(response_metadata.keys())}")

    routing = response_metadata.get("routing", {})
    logger.info(f"🔍 DEBUG: routing keys: {list(routing.keys())}")
    logger.info(f"🔍 DEBUG: full routing object: {routing}")

    next_agent = routing.get("next")
    logger.info(f"🔍 DEBUG: raw next_agent value: {next_agent}")

    # Handle None case explicitly - always return a valid route
    if next_agent is None:
        logger.warning("❌ DEBUG: next_agent is None - defaulting to adaptive_writer")
        next_agent = "adaptive_writer"

    # Check if supervisor actually made a routing decision
    if routing.get("supervisor_routed"):
        logger.info(f"✅ DEBUG: Supervisor routing confirmed with reasoning: {routing.get('reasoning', 'no reasoning')}")
    else:
        logger.warning("❌ DEBUG: No supervisor routing found - defaulting to adaptive_writer")
        logger.warning(f"❌ DEBUG: Available metadata: {list(response_metadata.keys())}")
        # If no supervisor routing, default to adaptive_writer to continue workflow
        next_agent = "adaptive_writer"

    # Convert FINISH to END for workflow compatibility
    final_result = next_agent if next_agent != "FINISH" else "END"
    
    # Ensure we never return None - final safety check
    if final_result is None or final_result == "":
        logger.error("❌ CRITICAL: final_result is None/empty - forcing adaptive_writer")
        final_result = "adaptive_writer"
    
    logger.info(f"📍 Final routing result: {final_result}")

    return final_result
