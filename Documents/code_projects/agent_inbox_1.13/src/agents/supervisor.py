"""
Supervisor Agent
Routes emails to appropriate specialized agents based on deep contextual understanding
from email_processor's structured analysis.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from langsmith import traceable
from src.agents.base_agent import BaseAgent
from src.models.state import AgentState, EmailIntent


class SupervisorAgent(BaseAgent):
    """
    Central routing agent that leverages email_processor's rich context
    for intelligent agent selection using LLM heuristics.
    """

    def __init__(self):
        super().__init__(
            name="supervisor",
            model="gpt-4o",
            temperature=0.1
        )

    @traceable(name="supervisor_process", tags=["agent", "supervisor"])
    async def process(self, state: AgentState) -> AgentState:
        """
        Analyze email_processor output for intelligent routing decisions.
        Track execution state and ensure proper agent sequencing.
        """

        # Handle feedback refinement
        if self._has_feedback(state):
            return self._handle_feedback_refinement(state)

        # Check if returning from agent execution
        if self._is_returning_from_agent(state):
            return self._update_agent_progress(state)

        # Initial routing based on email_processor analysis
        if not state.response_metadata.get("routing"):
            return await self._analyze_and_route(state)

        # Check progress and route to next agent
        return self._check_and_route_next(state)

    async def _analyze_and_route(self, state: AgentState) -> AgentState:
        """Analyze email_processor output for comprehensive routing decision."""
        try:
            if not state.email or not state.extracted_context:
                state.add_error("Missing email or extracted context")
                return self._route_to_adaptive_writer(state, "Missing prerequisites")

            # Get rich context from email_processor
            parsing = state.response_metadata.get("email_parsing", {})
            context = state.response_metadata.get("context_extraction", {})

            self.logger.info("Analyzing email_processor output for routing")

            system_prompt = """You are an intelligent routing system that makes decisions based on comprehensive email analysis.

            Available specialized agents:
            - calendar_agent: Handles ALL scheduling, meetings, appointments, availability checks, time coordination
            - rag_agent: Retrieves documents, searches knowledge base, finds specific information, answers factual questions
            - crm_agent: Manages contacts, customer data, relationship information, interaction history
            - adaptive_writer: ONLY runs AFTER other agents complete OR for simple direct responses with no special agents needs

            Routing principles:
            1. If dates/times are mentioned for scheduling → calendar_agent
            2. If specific information/documents are requested → rag_agent
            3. If contact/customer info is needed → crm_agent
            4. Multiple needs = multiple agents in sequence
            5. adaptive_writer ALWAYS runs last to compose final response

            Be comprehensive - identify ALL agents needed based on the extracted context."""

            prompt = f"""Based on the email_processor's analysis, determine routing:

EMAIL PARSING SUMMARY:
- Summary: {parsing.get('summary', 'N/A')}
- Main Request: {parsing.get('main_request', 'N/A')}
- Questions Asked: {json.dumps(parsing.get('questions_asked', []))}
- Key Points: {json.dumps(parsing.get('key_points', []))}
- Requires Response: {parsing.get('requires_response', True)}

EXTRACTED CONTEXT:
- Key Entities: {json.dumps(context.get('key_entities', [])[:10])}
- Dates Mentioned: {json.dumps(context.get('dates_mentioned', []))}
- Requested Actions: {json.dumps(context.get('requested_actions', []))}
- Requested Information: {json.dumps(context.get('requested_information', []))}
- Requested Data: {json.dumps(context.get('requested_data', []))}
- Requested Dates: {json.dumps(context.get('requested_dates', []))}
- References: {json.dumps(context.get('references', []))}
- Deadlines: {json.dumps(context.get('deadlines', []))}
- Urgency: {context.get('urgency_level', 'medium')}

ORIGINAL EMAIL:
Subject: {state.email.subject}
From: {state.email.sender}
Body preview: {state.email.body[:500]}...

Analyze ALL aspects and determine:
1. What specialized agents are needed (before adaptive_writer)?
2. What is the optimal execution order?
3. What specific task should each agent perform?

Return JSON:
{{
    "detailed_analysis": {{
        "scheduling_needs": ["any calendar/time related needs"],
        "information_needs": ["any document/data retrieval needs"],
        "contact_needs": ["any CRM/contact related needs"],
        "detected_patterns": ["patterns suggesting specific agents"]
    }},
    "agent_assignments": {{
        "calendar_agent": {{"needed": true/false, "task": "specific task if needed"}},
        "rag_agent": {{"needed": true/false, "task": "specific task if needed"}},
        "crm_agent": {{"needed": true/false, "task": "specific task if needed"}}
    }},
    "execution_plan": ["ordered list of agents to execute"],
    "routing_rationale": "detailed explanation of routing logic",
    "confidence": 0.0-1.0
}}"""

            response = await self._call_llm(prompt, system_prompt)

            try:
                decision = json.loads(response)
                self.logger.info("Routing decision made", decision=decision)

                # Build execution order from agent assignments
                execution_order = []
                assignments = decision.get("agent_assignments", {})

                # Add agents in logical order
                for agent in ["calendar_agent", "rag_agent", "crm_agent"]:
                    if assignments.get(agent, {}).get("needed", False):
                        execution_order.append(agent)

                # Use provided plan if more comprehensive
                if decision.get("execution_plan"):
                    execution_order = [a for a in decision["execution_plan"]
                                     if a in ["calendar_agent", "rag_agent", "crm_agent"]]

                # Always add adaptive_writer last if other agents are needed
                if execution_order:
                    execution_order.append("adaptive_writer")
                else:
                    # No special agents needed, go straight to adaptive_writer
                    execution_order = ["adaptive_writer"]

                # Store comprehensive routing state
                state.response_metadata["routing"] = {
                    "analysis": decision.get("detailed_analysis", {}),
                    "assignments": assignments,
                    "execution_order": execution_order,
                    "completed_agents": [],
                    "failed_agents": [],
                    "agent_results": {},
                    "current_index": 0,
                    "started_at": datetime.now().isoformat(),
                    "rationale": decision.get("routing_rationale", ""),
                    "confidence": decision.get("confidence", 0.8),
                    "email_context": {
                        "summary": parsing.get("summary"),
                        "main_request": parsing.get("main_request"),
                        "questions": parsing.get("questions_asked", []),
                        "actions": context.get("requested_actions", [])
                    }
                }

                # Set first agent to execute
                if execution_order:
                    state.response_metadata["routing"]["next"] = execution_order[0]
                    state.response_metadata["routing"]["last_routed_to"] = execution_order[0]

                # Log routing decision
                agent_tasks = [
                    f"{agent}: {assignments.get(agent, {}).get('task', 'N/A')}"
                    for agent in execution_order[:-1]  # Exclude adaptive_writer from task list
                ]

                self._add_message(
                    state,
                    f"Routing plan: {' → '.join(execution_order)}. "
                    f"Tasks: {'; '.join(agent_tasks) if agent_tasks else 'Direct response'}",
                    metadata=decision
                )

                return state

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse routing decision: {e}")
                return self._route_to_adaptive_writer(state, "Routing parse error")

        except Exception as e:
            self.logger.error(f"Routing analysis failed: {e}", exc_info=True)
            return self._route_to_adaptive_writer(state, f"Analysis error: {e}")

    def _is_returning_from_agent(self, state: AgentState) -> bool:
        """Check if returning from agent execution."""
        routing = state.response_metadata.get("routing", {})
        last_routed = routing.get("last_routed_to")
        completed = routing.get("completed_agents", [])

        # Check if we have evidence of agent execution
        has_agent_output = (
            (last_routed == "calendar_agent" and
             (state.calendar_data or "calendar" in str(state.messages[-1:]))),
            (last_routed == "rag_agent" and
             (state.document_data or "document" in str(state.messages[-1:]))),
            (last_routed == "crm_agent" and
             (state.contact_data or "contact" in str(state.messages[-1:])))
        )

        return (
            last_routed and
            last_routed not in completed and
            last_routed != "supervisor" and
            (any(has_agent_output) or state.status == "processing")
        )

    def _update_agent_progress(self, state: AgentState) -> AgentState:
        """Update progress after agent completes."""
        routing = state.response_metadata["routing"]
        last_agent = routing.get("last_routed_to")

        if not last_agent:
            return state

        self.logger.info(f"Agent {last_agent} completed execution")

        # Mark as completed
        if last_agent not in routing["completed_agents"]:
            routing["completed_agents"].append(last_agent)

            # Store agent-specific results
            assignments = routing.get("assignments", {})
            agent_task = assignments.get(last_agent, {}).get("task", "N/A")

            routing["agent_results"][last_agent] = {
                "completed": True,
                "task": agent_task,
                "timestamp": datetime.now().isoformat()
            }

            # Capture agent output data
            if last_agent == "calendar_agent" and state.calendar_data:
                routing["agent_results"][last_agent]["data"] = state.calendar_data.dict()
            elif last_agent == "rag_agent" and state.document_data:
                routing["agent_results"][last_agent]["data"] = state.document_data.dict()
            elif last_agent == "crm_agent" and state.contact_data:
                routing["agent_results"][last_agent]["data"] = state.contact_data.dict()

        # Advance to next agent
        execution_order = routing.get("execution_order", [])
        current_idx = routing.get("current_index", 0)

        if current_idx + 1 < len(execution_order):
            routing["current_index"] = current_idx + 1
            next_agent = execution_order[current_idx + 1]
            routing["next"] = next_agent
            routing["last_routed_to"] = next_agent

            self.logger.info(
                f"Progress: {len(routing['completed_agents'])}/{len(execution_order)-1} "
                f"specialized agents complete. Next: {next_agent}"
            )
        else:
            routing["next"] = "END"
            state.status = "ready_for_response"
            self.logger.info("All agents completed, ready for final response")

        state.response_metadata["routing"] = routing
        return state

    def _check_and_route_next(self, state: AgentState) -> AgentState:
        """Determine next step in execution flow."""
        routing = state.response_metadata["routing"]
        next_agent = routing.get("next")

        if next_agent and next_agent != "END":
            self.logger.info(f"Continuing to: {next_agent}")
        else:
            state.status = "completed"

        return state

    def _has_feedback(self, state: AgentState) -> bool:
        """Check for human feedback."""
        return bool(
            state.human_feedback or
            state.response_metadata.get("human_feedback") or
            state.response_metadata.get("decision") == "instruction"
        )

    def _handle_feedback_refinement(self, state: AgentState) -> AgentState:
        """Route feedback to adaptive_writer."""
        self.logger.info("Processing human feedback")

        feedback_list = []
        if state.human_feedback:
            feedback_list.append(state.human_feedback)
        if "human_feedback" in state.response_metadata:
            historical = state.response_metadata["human_feedback"]
            feedback_list.extend(historical if isinstance(historical, list) else [historical])

        state.response_metadata["feedback_context"] = {
            "feedback_count": len(feedback_list),
            "all_feedback": feedback_list,
            "refinement_iteration": state.response_metadata.get("refinement_iteration", 0) + 1,
            "previous_draft": state.draft_response
        }

        state.response_metadata["routing"] = {
            "execution_order": ["adaptive_writer"],
            "next": "adaptive_writer",
            "last_routed_to": "adaptive_writer",
            "is_refinement": True,
            "completed_agents": []
        }

        return state

    def _route_to_adaptive_writer(self, state: AgentState, reason: str) -> AgentState:
        """Direct route to adaptive_writer."""
        state.response_metadata["routing"] = {
            "execution_order": ["adaptive_writer"],
            "next": "adaptive_writer",
            "last_routed_to": "adaptive_writer",
            "completed_agents": [],
            "rationale": reason
        }
        return state

    def get_next_agents(self, state: AgentState) -> List[str]:
        """Return next agent(s) to execute."""
        routing = state.response_metadata.get("routing", {})
        next_agent = routing.get("next")

        if next_agent and next_agent != "END":
            return [next_agent]
        return []
