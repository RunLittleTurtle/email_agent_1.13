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
        Route email based on extracted context to appropriate agents.
        Handles human feedback from agent_inbox for re-routing decisions.
        """
        self.logger.info("🎯 Supervisor Agent analyzing routing options")

        # Priority 1: Check for human feedback from agent_inbox
        human_feedback = state.human_feedback 
        human_feedback_list = state.response_metadata.get("human_feedback", [])
        
        # Log both for debugging
        self.logger.info("Checking human feedback", 
                        direct=human_feedback, 
                        list_feedback=human_feedback_list)
        
        if human_feedback or human_feedback_list:
            # CRITICAL: Check if all work is already complete before processing feedback
            routing = state.response_metadata.get("routing", {})
            completed_agents = routing.get("completed_agents", [])
            
            # If calendar agent completed and we have successful booking data, skip to adaptive_writer
            if ("calendar_agent" in completed_agents and 
                state.calendar_data and 
                state.calendar_data.action_taken and 
                "successfully" in state.calendar_data.action_taken.lower()):
                
                self.logger.info("🎯 Calendar work already complete, routing directly to adaptive_writer")
                state.response_metadata["routing"] = {
                    "source": "completion_detected",
                    "next": "adaptive_writer",
                    "completed_agents": completed_agents,
                    "execution_order": ["adaptive_writer"],
                    "rationale": "All specialized agents complete, proceeding to response generation"
                }
                return state
            
            self.logger.info("📥 Processing human feedback from agent_inbox", feedback=human_feedback)
            return await self._process_human_feedback(state)

        # Priority 2: Check if returning from agent execution
        if self._is_returning_from_agent(state):
            return self._update_agent_progress(state)

        # Priority 3: Initial routing based on email_processor analysis
        if not state.response_metadata.get("routing"):
            return await self._analyze_and_route(state)

        # Priority 4: Check progress and route to next agent
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
- Conversation Type: {parsing.get('conversation_type', 'unknown')}

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
Body: {state.email.body}

CONVERSATION HISTORY:
{self._get_conversation_summary(state)}

CALENDAR MODIFICATION DETECTION:
Look for these key phrases that indicate calendar modifications:
- "change the event" / "change the meeting"
- "modify the booking" / "update the booking"
- "reschedule" / "move the meeting"
- "new date" / "different time"
- "cancel and reschedule"

CRITICAL DATE PARSING RULES:
- Current year is {datetime.now().year}
- Current date context: {datetime.now().strftime('%B %d, %Y')}
- When parsing dates like "the 27th" or "september 3", use current year {datetime.now().year}
- Do NOT use past years like 2023

Analyze ALL aspects and determine:
1. What specialized agents are needed (before adaptive_writer)?
2. What is the optimal execution order?
3. What specific task should each agent perform?
4. IMPORTANT:Be very specific about the task each agent should perform.
5. If it's a booking and a date, be very SPECIFIC and CLEARLY state the booking details.
6. CRITICAL: If conversation history shows existing bookings/meetings, specify whether this is UPDATE/MODIFY existing booking vs CREATE new booking.

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

                # Create routing message using new LangGraph patterns
                routing_msg = f"Routing plan: {' → '.join(execution_order)}. " \
                             f"Tasks: {'; '.join(agent_tasks) if agent_tasks else 'Direct response'}"
                message_update = self._add_message_to_state(routing_msg, metadata=decision)

                # Apply message update to state and return
                if "messages" in message_update:
                    state.messages.extend(message_update["messages"])
                return state

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse routing decision: {e}")
                return self._route_to_adaptive_writer(state, "Routing parse error")

        except Exception as e:
            self.logger.error(f"Routing failed: {str(e)}", exc_info=True)
            state.add_error(f"Routing failed: {str(e)}")
            return state

    def _get_conversation_summary(self, state: AgentState) -> str:
        """Extract relevant context from conversation history for routing decisions"""
        if not state.messages:
            return "No previous conversation context."

        summary_parts = []

        # Look for calendar/booking related messages
        calendar_context = []
        booking_mentions = []

        for msg in state.messages[-10:]:  # Last 10 messages
            if hasattr(msg, 'content') and msg.content:
                content = msg.content.lower()

                # Check for calendar/booking keywords
                if any(keyword in content for keyword in ['meeting', 'calendar', 'scheduled', 'booking', 'event', 'appointment']):
                    calendar_context.append(f"- {msg.content[:150]}...")

                # Check for date mentions
                import re
                date_patterns = [r'\b\d{1,2}(st|nd|rd|th)\b', r'\d{4}-\d{2}-\d{2}', r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b']
                if any(re.search(pattern, content, re.IGNORECASE) for pattern in date_patterns):
                    booking_mentions.append(f"- Date mentioned: {msg.content[:100]}...")

        if calendar_context:
            summary_parts.append("PREVIOUS CALENDAR/BOOKING CONTEXT:")
            summary_parts.extend(calendar_context[:3])  # Top 3 most recent

        if booking_mentions:
            summary_parts.append("PREVIOUS DATE/TIME MENTIONS:")
            summary_parts.extend(booking_mentions[:3])  # Top 3 most recent

        # Look for current calendar data from state
        if state.calendar_data and state.calendar_data.meeting_request:
            meeting = state.calendar_data.meeting_request
            summary_parts.append("CURRENT CALENDAR STATE:")
            summary_parts.append(f"- Existing meeting: {meeting.get('title', 'Untitled')}")
            summary_parts.append(f"- Current date: {meeting.get('requested_datetime', 'Not set')}")

        return "\n".join(summary_parts) if summary_parts else "No relevant conversation context found."

    async def _process_human_feedback(self, state: AgentState) -> AgentState:
        """Process human feedback from agent_inbox and route accordingly."""
        # Check both locations for human feedback
        human_feedback_list = state.response_metadata.get("human_feedback", [])
        direct_feedback = state.human_feedback

        # Use direct feedback first, then list
        latest_feedback = direct_feedback or (human_feedback_list[-1] if human_feedback_list else "")

        self.logger.info("Analyzing human feedback for routing", feedback=latest_feedback)

        # Add human feedback to output for transparency
        if "human_feedback_processed" not in state.response_metadata:
            state.response_metadata["human_feedback_processed"] = []
        state.response_metadata["human_feedback_processed"].append({
            "feedback": latest_feedback,
            "timestamp": "current",
            "status": "processing"
        })

        system_prompt = """You are analyzing human feedback from an agent inbox system.
        The human has reviewed a draft response and provided feedback/instructions.

        Your job is to determine what agent should handle this feedback and provide specific task instructions.

        CRITICAL: Human feedback always requires re-routing to appropriate agents with clear instructions."""

        # Get conversation context
        conversation_summary = self._get_conversation_summary(state)

        # Get current completion status
        routing = state.response_metadata.get("routing", {})
        completed_agents = routing.get("completed_agents", [])
        agent_results = routing.get("agent_results", {})
        
        # Check if calendar work is already complete
        calendar_complete = (
            "calendar_agent" in completed_agents and 
            state.calendar_data and 
            state.calendar_data.action_taken and 
            ("successfully" in state.calendar_data.action_taken.lower() or 
             "meeting_booked" in state.calendar_data.action_taken.lower() or
             "event has been created" in state.calendar_data.action_taken.lower())
        )

        prompt = f"""HUMAN FEEDBACK FROM AGENT INBOX:
"{latest_feedback}"

PREVIOUS CONVERSATION CONTEXT:
{conversation_summary}

CURRENT WORK STATUS:
- Completed Agents: {completed_agents}
- Calendar Agent Status: {"✅ COMPLETED SUCCESSFULLY" if calendar_complete else "❌ Not completed"}
{f"- Calendar Action: {state.calendar_data.action_taken}" if state.calendar_data and state.calendar_data.action_taken else "- No calendar data"}

ORIGINAL EMAIL:
Subject: {state.email.subject}
Body: {state.email.body}

CRITICAL ROUTING RULES:
1. If calendar_agent is COMPLETED SUCCESSFULLY, DO NOT route to calendar_agent again
2. If all requested work is complete, route to "adaptive_writer" for final response
3. Only route to specialized agents if work is genuinely incomplete

CALENDAR MODIFICATION DETECTION:
Look for these key phrases that indicate calendar modifications:
- "change the event/meeting", "reschedule", "move the meeting"
- "update/modify the booking", "new date/time", "different day"

CRITICAL DATE PARSING RULES:
- Current year is {datetime.now().year}
- Current date context: {datetime.now().strftime('%B %d, %Y')}
- When user mentions dates like "the 27th" or "2 of september", assume current year {datetime.now().year}
- Parse all dates relative to current date context: {datetime.now().strftime('%B %Y')}

Analyze the human feedback and determine routing:

Return JSON:
{{
    "feedback_analysis": {{
        "intent": "calendar_modification|document_request|contact_update|simple_response",
        "specific_request": "clear description of what human wants",
        "urgency": "low|medium|high"
    }},
    "agent_assignments": {{
        "calendar_agent": {{"needed": true/false, "task": "SPECIFIC task with clear instructions INCLUDING EXACT DATE from human feedback"}},
        "rag_agent": {{"needed": true/false, "task": "SPECIFIC task with clear instructions"}},
        "crm_agent": {{"needed": true/false, "task": "SPECIFIC task with clear instructions"}}
    }},
    "execution_plan": ["ordered list of agents needed"],
    "routing_rationale": "detailed explanation why this routing was chosen"
}}"""

        response = await self._call_llm(prompt, system_prompt)

        try:
            decision = json.loads(response)
            self.logger.info("Human feedback routing decision", decision=decision)

            # PRESERVE completed agents when processing human feedback
            existing_routing = state.response_metadata.get("routing", {})
            existing_completed = existing_routing.get("completed_agents", [])
            
            # Only re-route if calendar agent hasn't completed successfully
            calendar_completed = "calendar_agent" in existing_completed
            calendar_has_data = bool(state.calendar_data and 
                                   state.calendar_data.action_taken and 
                                   "successfully" in state.calendar_data.action_taken.lower())
            
            if calendar_completed or calendar_has_data:
                # Calendar work is done, route directly to adaptive_writer
                state.response_metadata["routing"] = {
                    "source": "human_feedback_completion",
                    "feedback_analyzed": latest_feedback,
                    "execution_plan": ["adaptive_writer"],
                    "next": "adaptive_writer",
                    "completed_agents": existing_completed,
                    "routing_rationale": "Calendar agent completed, proceeding to response generation"
                }
                self.logger.info("Calendar work complete, routing to adaptive_writer")
            else:
                # Calendar work incomplete, allow re-routing
                state.response_metadata["routing"] = {
                    "source": "human_feedback",
                    "feedback_analyzed": latest_feedback,
                    "agent_assignments": decision.get("agent_assignments", {}),
                    "execution_plan": decision.get("execution_plan", []),
                    "routing_rationale": decision.get("routing_rationale", ""),
                    "completed_agents": existing_completed,  # Preserve completed agents
                    "last_routed_to": None
                }

            # Store human feedback in output for Agent Inbox visibility
            state.response_metadata["supervisor_output"] = f"Processing human feedback: '{latest_feedback}'"

            # Override any existing system messages to ensure human feedback is visible
            state.response_metadata["system_output"] = f"HUMAN FEEDBACK: {latest_feedback}"

            # Determine next agent and UPDATE TASK with human feedback
            execution_plan = decision.get("execution_plan", [])
            if execution_plan:
                next_agent = execution_plan[0]
                state.response_metadata["routing"]["next"] = next_agent
                
                # CRITICAL: Update agent task with human feedback details
                if next_agent == "calendar_agent":
                    calendar_task = decision.get("agent_assignments", {}).get("calendar_agent", {}).get("task", "")
                    state.response_metadata["routing"]["current_task"] = f"HUMAN FEEDBACK MODIFICATION: {latest_feedback}. {calendar_task}"
                    self.logger.info(f"Updated calendar agent task with feedback: {latest_feedback}")
                
                self.logger.info(f"Routing human feedback to: {next_agent}")
            else:
                state.response_metadata["routing"]["next"] = "adaptive_writer"
                self.logger.info("No specific agents needed, routing to adaptive_writer")

            # Mark human feedback as processed
            state.response_metadata["human_feedback_processed"][-1]["status"] = "routed"

            return state

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse human feedback routing: {e}")
            # Fallback to adaptive writer
            state.response_metadata["routing"] = {"next": "adaptive_writer"}
            return state

    def _is_returning_from_agent(self, state: AgentState) -> bool:
        """Check if returning from agent execution."""
        routing = state.response_metadata.get("routing", {})
        last_routed = routing.get("last_routed_to")
        completed = routing.get("completed_agents", [])

        # Enhanced evidence of agent execution with success detection
        calendar_success = (
            last_routed == "calendar_agent" and
            state.calendar_data and 
            state.calendar_data.action_taken and
            ("successfully" in state.calendar_data.action_taken.lower() or 
             "meeting_booked" in state.calendar_data.action_taken.lower() or
             "event has been created" in state.calendar_data.action_taken.lower())
        )
        
        has_agent_output = (
            calendar_success or
            (last_routed == "rag_agent" and
             (state.document_data or "document" in str(state.messages[-1:]))) or
            (last_routed == "crm_agent" and
             (state.contact_data or "contact" in str(state.messages[-1:])))
        )

        return (
            last_routed and
            last_routed not in completed and
            last_routed != "supervisor" and
            (has_agent_output or state.status == "processing")
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
