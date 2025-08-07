"""
Supervisor Agent
Routes emails to appropriate specialized agents based on intent classification.
Now directly handles multiple intents (no task_decomposer).
"""

import json
from typing import List, Dict, Any

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

                # Store routing plan in state
                state.response_metadata["routing"] = {
                    "classification": classification,
                    "required_agents": required_agents,
                    "completed_agents": [],
                    "confidence": classification.get("confidence", 0.0),
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
        If all completed, forward to adaptive_writer.
        """

        routing = state.response_metadata["routing"]
        required = routing.get("required_agents", [])
        completed = routing.get("completed_agents", [])

        # Add any agent results that just finished
        last_agent = state.last_agent if hasattr(state, "last_agent") else None
        if last_agent and last_agent not in completed:
            completed.append(last_agent)
            routing["completed_agents"] = completed
            self.logger.info(f"Marking {last_agent} as completed")

        # Check if all done
        if set(completed) >= set(required):
            self.logger.info("✅ All required agents completed, forwarding to adaptive_writer")
            routing["next"] = "adaptive_writer"
        else:
            remaining = set(required) - set(completed)
            self.logger.info(f"⏳ Still waiting on agents: {remaining}")
            routing["next"] = list(remaining)[0]

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
