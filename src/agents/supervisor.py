"""
Supervisor Agent
Routes emails to appropriate specialized agents based on intent classification.
Now directly handles multiple intents (no task_decomposer).
"""

import json
from typing import List, Dict, Any
from datetime import datetime

from langsmith import traceable
from src.agents.base_agent import BaseAgent
from src.models.state import AgentState, EmailIntent


class SupervisorAgent(BaseAgent):
    """
    Central routing agent that classifies email intent and decides
    which specialized agent(s) to invoke.
    Tracks progress and only forwards to adaptive_writer once all
    required agents have completed.
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
        Classify email intent and determine routing.
        If multiple intents, decompose directly and queue multiple agents.
        Handles human feedback for refinement loops.
        """

        # Check if this is a feedback refinement loop
        has_feedback = (
            state.human_feedback or 
            state.response_metadata.get("human_feedback") or
            state.response_metadata.get("decision") == "instruction"
        )
        
        if has_feedback:
            return self._handle_feedback_refinement(state)
        
        # If already has a routing plan, just check completion
        if "routing" in state.response_metadata:
            return self._check_progress(state)

        try:
            if not state.email or not state.extracted_context:
                state.add_error("Missing email or context for supervision")
                return state

            self.logger.info("Classifying email intent for routing")

            # System prompt for classification
            system_prompt = """You are an email intent classifier and router.
            Analyze emails and determine the appropriate handling path.
            Always respond in valid JSON format.

            Intent types:
            - meeting_request: Email requests scheduling a meeting
            - document_request: Email requests documents or information retrieval
            - task_delegation: Email delegates tasks or requests action from others
            - simple_direct: Simple response needed, no special handling
            - multiple_intents: Email contains multiple distinct requests"""

            # Build classification prompt
            prompt = f"""Classify this email's intent and determine routing:

Subject: {state.email.subject}
From: {state.email.sender}
Body: {state.email.body}

Extracted Context:
- Key Entities: {', '.join(state.extracted_context.key_entities)}
- Requested Actions: {', '.join(state.extracted_context.requested_actions)}
- Urgency: {state.extracted_context.urgency_level}

Return JSON:
{{
    "primary_intent": "meeting_request|document_request|task_delegation|simple_direct|multiple_intents",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of classification",
    "secondary_intents": ["list of other intents if multiple_intents"],
    "recommended_agents": ["list of agent names to invoke"],
    "special_instructions": "Any special handling instructions"
}}"""

            # Call LLM
            response = await self._call_llm(prompt, system_prompt)

            try:
                classification = json.loads(response)
                self.logger.info("Intent classified", classification=classification)

                primary_intent = classification.get("primary_intent", "simple_direct")
                state.intent = EmailIntent(primary_intent)

                # Determine which agents are required
                required_agents = self._determine_agents(classification, primary_intent)

                # Store routing plan in state with timeout configuration
                state.response_metadata["routing"] = {
                    "classification": classification,
                    "required_agents": required_agents,
                    "completed_agents": [],
                    "failed_agents": [],
                    "agent_timeouts": {
                        "calendar_agent": 30,  # 30 seconds
                        "rag_agent": 45,      # 45 seconds for document search
                        "crm_agent": 30       # 30 seconds
                    },
                    "started_at": datetime.now().isoformat(),
                    "next": required_agents[0] if required_agents else "adaptive_writer"
                }

                self._add_message(
                    state,
                    f"Email classified as '{primary_intent}' "
                    f"with {classification.get('confidence', 0):.0%} confidence. "
                    f"Routing to: {', '.join(required_agents)}",
                    metadata=classification,
                )

                return state

            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse classification: {e}")
                # Default fallback
                state.intent = EmailIntent.SIMPLE_DIRECT
                state.response_metadata["routing"] = {
                    "required_agents": ["adaptive_writer"],
                    "completed_agents": []
                }
                state.add_error(f"Intent classification parsing failed, fallback to adaptive_writer: {str(e)}")
                return state

        except Exception as e:
            self.logger.error(f"Supervision failed: {str(e)}", exc_info=True)
            state.add_error(f"Supervision failed: {str(e)}")
            # Fallback
            state.intent = EmailIntent.SIMPLE_DIRECT
            state.response_metadata["routing"] = {
                "required_agents": ["adaptive_writer"],
                "completed_agents": []
            }
            return state

    def _determine_agents(self, classification: Dict[str, Any], primary_intent: str) -> List[str]:
        """Map intents to required agents (handles multiple intents inline)."""

        routing_map = {
            EmailIntent.MEETING_REQUEST: ["calendar_agent"],
            EmailIntent.DOCUMENT_REQUEST: ["rag_agent"],
            EmailIntent.TASK_DELEGATION: ["crm_agent"],
            EmailIntent.SIMPLE_DIRECT: ["adaptive_writer"],
        }

        # If multiple intents, expand
        if primary_intent == "multiple_intents":
            secondary = classification.get("secondary_intents", [])
            all_intents = secondary or []
            agents = []
            for intent in all_intents:
                mapped = routing_map.get(EmailIntent(intent), [])
                agents.extend(mapped)
            return agents or ["adaptive_writer"]

        return routing_map.get(EmailIntent(primary_intent), ["adaptive_writer"])

    def _handle_feedback_refinement(self, state: AgentState) -> AgentState:
        """
        Handle human feedback refinement - route back to adaptive_writer with feedback context.
        """
        self.logger.info("🔄 Processing human feedback for refinement")
        
        # Collect all feedback sources
        feedback_list = []
        
        # Current feedback
        if state.human_feedback:
            feedback_list.append(state.human_feedback)
            
        # Historical feedback from response_metadata
        if "human_feedback" in state.response_metadata:
            historical = state.response_metadata["human_feedback"]
            if isinstance(historical, list):
                feedback_list.extend(historical)
            else:
                feedback_list.append(historical)
        
        # Store comprehensive feedback context
        state.response_metadata["feedback_context"] = {
            "feedback_count": len(feedback_list),
            "all_feedback": feedback_list,
            "refinement_iteration": state.response_metadata.get("refinement_iteration", 0) + 1,
            "previous_draft": state.draft_response
        }
        
        # Set routing to go directly to adaptive_writer for refinement
        state.response_metadata["routing"] = {
            "required_agents": ["adaptive_writer"],
            "completed_agents": [],
            "next": "adaptive_writer",
            "is_refinement": True
        }
        
        self._add_message(
            state,
            f"Processing feedback refinement (iteration {state.response_metadata['feedback_context']['refinement_iteration']}). "
            f"Routing to adaptive_writer with {len(feedback_list)} feedback items.",
            metadata={
                "feedback_items": len(feedback_list),
                "is_refinement": True
            }
        )
        
        return state

    def _check_progress(self, state: AgentState) -> AgentState:
        """
        Check which required agents are done.
        Handle timeouts and errors.
        If all completed or critical failure, forward to adaptive_writer.
        """

        routing = state.response_metadata["routing"]
        required = routing.get("required_agents", [])
        completed = routing.get("completed_agents", [])
        failed = routing.get("failed_agents", [])
        timeouts = routing.get("agent_timeouts", {})
        started_at = datetime.fromisoformat(routing.get("started_at", datetime.now().isoformat()))

        # Check which agent just finished execution
        # When we return to supervisor, we need to check which agent data was updated
        last_executed_agent = None
        
        # Determine which agent just executed based on new data or routing context
        if routing.get("last_routed_to"):
            last_executed_agent = routing["last_routed_to"]
        
        # Debug logging
        self.logger.info(f"🔍 DEBUG - current_agent: {state.current_agent}")
        self.logger.info(f"🔍 DEBUG - last_executed_agent: {last_executed_agent}")
        self.logger.info(f"🔍 DEBUG - required: {required}")
        self.logger.info(f"🔍 DEBUG - completed: {completed}")
        self.logger.info(f"🔍 DEBUG - state.status: {state.status}")
        self.logger.info(f"🔍 DEBUG - has calendar_data: {bool(state.calendar_data)}")
        self.logger.info(f"🔍 DEBUG - errors: {state.error_messages}")
        
        # Check if the last executed agent completed successfully
        if last_executed_agent and last_executed_agent in required and last_executed_agent not in completed:
            # Check if agent has returned from execution with data/results
            agent_has_data = (
                (last_executed_agent == "calendar_agent" and state.calendar_data) or
                (last_executed_agent == "rag_agent" and state.document_data) or
                (last_executed_agent == "crm_agent" and state.contact_data) or
                last_executed_agent in ["calendar_agent", "rag_agent", "crm_agent"]  # Allow completion even without data
            )
            
            self.logger.info(f"🔍 DEBUG - agent_has_data: {agent_has_data}")
            
            # If agent executed without errors, mark as completed
            if state.status != "error" and not any(last_executed_agent in str(err) for err in state.error_messages) and agent_has_data:
                completed.append(last_executed_agent)
                routing["completed_agents"] = completed
                self.logger.info(f"✅ Marking {last_executed_agent} as completed")
                
                # Add agent results to state metadata
                if last_executed_agent == "calendar_agent" and state.calendar_data:
                    routing["calendar_results"] = state.calendar_data.dict()
                elif last_executed_agent == "rag_agent" and state.document_data:
                    routing["document_results"] = state.document_data.dict()
                elif last_executed_agent == "crm_agent" and state.contact_data:
                    routing["contact_results"] = state.contact_data.dict()
            else:
                self.logger.info(f"🔍 DEBUG - Agent {last_executed_agent} not marked complete. Reasons:")
                self.logger.info(f"  - status != error: {state.status != 'error'}")
                self.logger.info(f"  - no errors: {not any(last_executed_agent in str(err) for err in state.error_messages)}")
                self.logger.info(f"  - has_data: {agent_has_data}")
                
                # Agent failed
                if last_executed_agent not in failed:
                    failed.append(last_executed_agent)
                    routing["failed_agents"] = failed
                    self.logger.error(f"❌ Agent {last_executed_agent} failed")

        # Check for timeouts
        elapsed_seconds = (datetime.now() - started_at).total_seconds()
        for agent in required:
            if agent not in completed and agent not in failed:
                agent_timeout = timeouts.get(agent, 60)  # Default 60 seconds
                if elapsed_seconds > agent_timeout:
                    failed.append(agent)
                    routing["failed_agents"] = failed
                    self.logger.warning(f"⏱️ Agent {agent} timed out after {agent_timeout}s")
                    state.add_error(f"Agent {agent} timed out")

        # Determine next step
        all_done = set(completed + failed) >= set(required)
        if all_done:
            # All agents have either completed or failed
            success_rate = len(completed) / len(required) if required else 0
            
            if success_rate >= 0.5:  # At least 50% success
                self.logger.info(
                    f"✅ Workflow proceeding to adaptive_writer "
                    f"({len(completed)}/{len(required)} agents succeeded)"
                )
                routing["next"] = "adaptive_writer"
                
                # Add summary of what succeeded/failed
                routing["completion_summary"] = {
                    "total_required": len(required),
                    "completed": completed,
                    "failed": failed,
                    "success_rate": success_rate
                }
            else:
                # Too many failures
                self.logger.error("❌ Too many agent failures, workflow cannot continue")
                state.status = "error"
                state.add_error(f"Workflow failed: only {len(completed)}/{len(required)} agents succeeded")
                routing["next"] = "END"
        else:
            # Still have agents to run
            remaining = set(required) - set(completed) - set(failed)
            if remaining:
                next_agent = list(remaining)[0]
                self.logger.info(f"⏳ Next agent: {next_agent} (remaining: {remaining})")
                routing["next"] = next_agent
            else:
                # No agents left to run but not all done? Shouldn't happen
                routing["next"] = "adaptive_writer"

        state.response_metadata["routing"] = routing
        return state

    def get_next_agents(self, state: AgentState) -> List[str]:
        """
        Decide next agent(s) to run based on progress.
        """
        routing = state.response_metadata.get("routing", {})
        if not routing:
            return ["adaptive_writer"]

        next_agent = routing.get("next")
        if next_agent == "adaptive_writer":
            return ["adaptive_writer"]

        return [next_agent]
