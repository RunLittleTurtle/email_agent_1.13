"""
Adaptive Writer Agent
Generates email responses based on context and intent
"""

import json
from typing import Dict, Any, Optional

from langsmith import traceable
from src.agents.base_agent import BaseAgent
from src.models.state import AgentState, EmailIntent


class AdaptiveWriterAgent(BaseAgent):
    """
    Agent responsible for generating email responses.
    Adapts writing style based on context and intent.
    """

    def __init__(self):
        super().__init__(
            name="adaptive_writer",
            model="gpt-4o",
            temperature=0.7  # Higher temperature for more creative writing
        )

    @traceable(name="adaptive_writer_process", tags=["agent", "writer"])
    async def process(self, state: AgentState) -> AgentState:
        """
        Generate email response based on all gathered context

        Args:
            state: Current workflow state

        Returns:
            Updated state with draft response
        """
        try:
            if not state.email:
                state.add_error("No email to respond to")
                return state

            self.logger.info("Generating email response")

            # Check if this is a feedback refinement iteration
            is_refinement = state.response_metadata.get("routing", {}).get("is_refinement", False)
            feedback_context = state.response_metadata.get("feedback_context", {})

            if is_refinement:
                self.logger.info(f"🔄 Processing feedback refinement iteration {feedback_context.get('refinement_iteration', 1)}")

            # Gather all context for response generation
            context_parts = []

            # Add email parsing results
            if "email_parsing" in state.response_metadata:
                parsing = state.response_metadata["email_parsing"]
                context_parts.append(f"Email Summary: {parsing.get('summary', 'N/A')}")
                context_parts.append(f"Main Request: {parsing.get('main_request', 'N/A')}")

            # Add extracted context
            if state.extracted_context:
                context_parts.append(f"Key Entities: {', '.join(state.extracted_context.key_entities)}")
                context_parts.append(f"Urgency: {state.extracted_context.urgency_level}")
                context_parts.append(f"Actions Requested: {', '.join(state.extracted_context.requested_actions)}")

            # Add detailed specialized agent results
            routing_info = state.response_metadata.get("routing", {})
            completion_summary = routing_info.get("completion_summary", {})

            # Add calendar agent results
            if state.calendar_data:
                context_parts.append("\n📅 Calendar Information:")

                if state.calendar_data.meeting_request:
                    meeting = state.calendar_data.meeting_request
                    context_parts.append(f"  - Meeting: {meeting.get('title', 'Untitled')}")
                    context_parts.append(f"  - Requested time: {meeting.get('requested_datetime', 'Not specified')}")

                # Check for conflicts and alternative times
                if state.calendar_data.action_taken == "conflict_detected":
                    context_parts.append(f"  - STATUS: CONFLICT DETECTED at requested time")
                    if state.calendar_data.suggested_times:
                        context_parts.append(f"  - Alternative times available:")
                        for i, time in enumerate(state.calendar_data.suggested_times[:3], 1):
                            context_parts.append(f"    {i}. {time}")
                elif state.calendar_data.action_taken == "meeting_booked":
                    context_parts.append(f"  - STATUS: Meeting successfully scheduled")
                    if state.calendar_data.booked_event:
                        event = state.calendar_data.booked_event
                        context_parts.append(f"  - Confirmed: {event.get('summary', 'Meeting')}")
                        context_parts.append(f"  - Date/Time: {event.get('datetime', 'Not specified')}")
                        if event.get('meeting_link'):
                            context_parts.append(f"  - Meeting Link: {event.get('meeting_link')}")
                        if event.get('attendees'):
                            context_parts.append(f"  - Attendees: {', '.join(event.get('attendees', []))}")
                else:
                    context_parts.append(f"  - STATUS: {state.calendar_data.action_taken}")

                if state.calendar_data.availability:
                    avail = state.calendar_data.availability
                    context_parts.append(f"  - Available slots: {len(avail.get('available', []))}")
                    context_parts.append(f"  - Conflicts: {len(avail.get('conflicts', []))}")

            # Add document search results
            if state.document_data and state.document_data.found_documents:
                context_parts.append("\n📄 Document Search Results:")
                for doc in state.document_data.found_documents[:3]:  # Top 3
                    context_parts.append(f"  - {doc['name']} ({doc['type']})")
                    if doc.get('content_summary'):
                        context_parts.append(f"    Summary: {doc['content_summary'][:100]}...")
                if state.document_data.missing_documents:
                    context_parts.append(f"  - Not found: {', '.join(state.document_data.missing_documents)}")

            # Add contact/CRM results
            if state.contact_data and state.contact_data.contacts:
                context_parts.append("\n👥 Contact Information:")
                for contact in state.contact_data.contacts[:3]:  # Top 3
                    context_parts.append(f"  - {contact['name']}: {contact.get('summary', 'No details')}")
                if state.contact_data.relationship_context:
                    ctx = state.contact_data.relationship_context
                    if ctx.get('delegation_ready'):
                        context_parts.append(f"  - Task delegation ready for {len(ctx.get('assignees', []))} people")

            # Add completion summary if some agents failed
            if completion_summary:
                failed = completion_summary.get("failed", [])
                if failed:
                    context_parts.append(f"\n⚠️ Note: Some information may be incomplete ({', '.join(failed)} failed)")

            # Add full agent outputs for detailed context
            if hasattr(state, 'output') and state.output:
                context_parts.append("\n🤖 Detailed Agent Outputs:")
                for output in state.output:
                    agent_name = output.get("agent", "Unknown Agent")
                    message = output.get("message", "")
                    if message:
                        context_parts.append(f"  • {agent_name}: {message}")
                        # Extract meeting links from agent output messages if calendar data doesn't have it
                        if agent_name == "calendar_agent" and "https://" in message:
                            import re
                            link_patterns = [
                                r'https://meet\.google\.com/[a-z0-9-]+',
                                r'https://zoom\.us/j/\d+',
                                r'https://teams\.microsoft\.com/[^\s]+',
                                r'https://[^\s]+meet[^\s]*'
                            ]
                            for pattern in link_patterns:
                                match = re.search(pattern, message, re.IGNORECASE)
                                if match:
                                    meeting_link = match.group(0)
                                    context_parts.append(f"  📎 Meeting Link Found: {meeting_link}")
                                    break

            # System prompt for response generation
            system_prompt = """You are a professional email response writer.
            Generate appropriate, contextual email responses.
            Match the tone and formality of the original email.
            Be concise but complete.
            Always maintain a professional and helpful tone.
            Important: Do not include any subjectinformation of already scheduled events in the response. Only tell the time unavailable, but DO NOT say why."""

            # Build feedback context for refinement iterations
            feedback_prompt_section = ""
            if is_refinement and feedback_context:
                all_feedback = feedback_context.get("all_feedback", [])
                previous_draft = feedback_context.get("previous_draft", "")
                iteration = feedback_context.get("refinement_iteration", 1)

                feedback_prompt_section = f"""

FEEDBACK REFINEMENT - Iteration {iteration}:
Previous Draft:
{previous_draft}

Human Feedback to Address:
{chr(10).join(f"- {fb}" for fb in all_feedback)}

IMPORTANT: This is a refinement iteration. Please improve the previous draft based on the human feedback above.
Address all feedback points while maintaining the email's core purpose and professionalism."""

            # Create response generation prompt
            prompt = f"""Generate a response to this email:

Original Email:
From: {state.email.sender}
Subject: {state.email.subject}
Body: {state.email.body}

Context Information:
{chr(10).join(context_parts)}

Intent Classification: {state.intent.value if state.intent else 'unknown'}{feedback_prompt_section}

Generate a professional email response that:
1. Acknowledges the sender's message
2. Addresses all requested actions or questions using the context information above
3. Maintains appropriate tone and formality
4. Is clear and concise
5. Includes a proper greeting and closing{' and addresses all human feedback' if is_refinement else ''}
6. Incorporates relevant information from calendar, documents, and contacts as needed
7. Use the complete OUTPUT of the specialized agents like the calendar agent, document agent, and contact agent
8. Always format the draft email so it is easy to read and nice to the eye
9. If some information is missing due to agent failures, acknowledge this appropriately
10. IMPORTANT: If a meeting link is found in the context (like Google Meet, Zoom, Teams links), ALWAYS include it in the response
11. Include ALL relevant details from successful calendar bookings, including meeting links, date/time confirmations, and attendee information

Return the response in JSON format:
{{
    "subject": "Response subject line",
    "body": "Full email response body",
    "tone": "formal|casual|friendly|professional",
    "confidence": 0.0-1.0
}}"""

            # Call LLM
            response = await self._call_llm(prompt, system_prompt)

            try:
                response_data = json.loads(response)
                self.logger.info(
                    "Response generated",
                    tone=response_data.get("tone"),
                    confidence=response_data.get("confidence")
                )

                # Format the complete response
                formatted_response = response_data.get("body", "")

                # Store draft response
                state.draft_response = formatted_response
                state.response_metadata["generated_response"] = response_data

                # Add message using new LangGraph patterns
                confidence_msg = f"Draft response generated with {response_data.get('confidence', 0):.0%} confidence"
                message_update = self._add_message_to_state(confidence_msg, metadata=response_data)
                
                # Apply message update to state
                if "messages" in message_update:
                    state.messages.extend(message_update["messages"])

                return state

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse response generation: {e}")
                # Try to extract response from raw text
                state.draft_response = response
                state.add_error(f"Response parsing failed, using raw response: {str(e)}")
                return state

        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}", exc_info=True)
            state.add_error(f"Response generation failed: {str(e)}")
            return state
