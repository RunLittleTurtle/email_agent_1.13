"""
Email Processor Agent
First agent in the workflow - parses and structures incoming emails,
and extracts contextual information (entities, dates, actions, urgency).
"""

import json
from datetime import datetime
from typing import Dict, Any

from langsmith import traceable
from src.agents.base_agent import BaseAgent
from src.models.state import AgentState, ExtractedContext


class EmailProcessorAgent(BaseAgent):
    """
    Agent responsible for parsing and structuring email data,
    including context extraction (entities, actions, dates, urgency).
    This is the entry point for all email processing.
    """

    def __init__(self):
        super().__init__(
            name="email_processor",
            model="gpt-4o",
            temperature=0.0
        )

    @traceable(name="email_processor_process", tags=["agent", "processor"])
    async def process(self, state: AgentState) -> AgentState:
        """
        Generate parsed + structured state for incoming email.
        LangSmith traceable.
        """
        try:
            if not state.email:
                self.logger.error("No email data provided to process")
                state.add_error("No email data provided")
                return state

            self.logger.info(
                "Processing email",
                subject=state.email.subject,
                sender=state.email.sender
            )

            # System prompt for parsing + context extraction
            system_prompt = """You are an email parsing and context extraction assistant.
            Extract and structure key information from emails, including entities, actions, and urgency.
            IMPORTANT: Be very clear when you formulate your response, since the supervisor and the rest of the graph will use your output.
            
            CRITICAL: Pay special attention to calendar modification language:
            - "change the event/meeting" = modification
            - "reschedule" = modification  
            - "move the meeting" = modification
            - "update the booking" = modification
            - "new date/time" = modification
            - "different day" = modification
            
            Always respond in valid JSON format.
            For dates, use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."""

            # Check for conversation history in thread memory
            conversation_context = ""
            if state.messages:
                recent_messages = state.messages[-5:]  # Last 5 messages for context
                conversation_context = "\n\nPREVIOUS CONVERSATION CONTEXT:\n"
                for msg in recent_messages:
                    if hasattr(msg, 'content') and msg.content:
                        msg_type = msg.__class__.__name__.replace('Message', '')
                        conversation_context += f"[{msg_type}]: {msg.content[:200]}...\n"

            # Combined parsing + extraction prompt
            prompt = f"""Parse and extract context from this email:

Subject: {state.email.subject}
From: {state.email.sender}
To: {', '.join(state.email.recipients)}
Body: {state.email.body}{conversation_context}

Return JSON with two sections:
{{
    "parsing": {{
        "summary": "Brief summary of the email",
        "main_request": "What is being asked or communicated",
        "additional_info": "Any additional information or context",
        "has_attachments": true/false,
        "requires_response": true/false,
        "urgency_indicators": ["list of urgency indicators if any"],
        "key_points": ["list of main points"],
        "questions_asked": ["list of specific questions if any"],
        "past_emails_summary": ["list of summaries of past emails"],
        "conversation_type": "new_request|follow_up|modification|confirmation",
        "previous_meeting_context": "any context about existing meetings or bookings from conversation history"

    }},
    "context": {{
        "key_entities": ["list of people, companies, projects mentioned"],
        "dates_mentioned": ["list of dates in ISO format"],
        "requested_actions": ["list of specific actions requested"],
        "urgency_level": "low|medium|high",
        "sentiment": "positive|neutral|negative",
        "urgency_indicators": ["words/phrases indicating urgency"],
        "deadlines": ["any mentioned deadlines"],
        "references": ["any referenced documents, meetings, or previous communications"]
    }}
}}"""

            # Call LLM
            response = await self._call_llm(prompt, system_prompt)

            try:
                parsed = json.loads(response)
                self.logger.info("Successfully processed email", parsed=parsed)

                # --- Parsing results ---
                parsing = parsed.get("parsing", {})
                context = parsed.get("context", {})

                # Store results in state with consistent field names
                state.response_metadata["email_parsing"] = parsing
                state.response_metadata["context_extraction"] = context

                extracted_context = ExtractedContext(
                    key_entities=context.get("key_entities", []),
                    dates_mentioned=[
                        datetime.fromisoformat(date) if date else None
                        for date in context.get("dates_mentioned", [])
                        if date
                    ],
                    requested_actions=context.get("requested_actions", []),
                    urgency_level=context.get("urgency_level", "medium"),
                    sentiment=context.get("sentiment", "neutral")
                )

                state.extracted_context = extracted_context

                # Add message using new LangGraph patterns
                process_msg = f"Email processed. Summary: {parsing.get('summary', 'N/A')}, " \
                             f"Context: {len(extracted_context.key_entities)} entities, " \
                             f"{len(extracted_context.requested_actions)} actions, urgency={extracted_context.urgency_level}"
                
                message_update = self._add_message_to_state(process_msg, metadata=parsed)
                
                # Apply message update to state
                if "messages" in message_update:
                    state.messages.extend(message_update["messages"])

                state.status = "processing"

            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse LLM response: {e}")
                state.add_error(f"Failed to parse email/context: {str(e)}")

            return state

        except Exception as e:
            self.logger.error(f"Email processing failed: {str(e)}", exc_info=True)
            state.add_error(f"Email processing failed: {str(e)}")
            return state
