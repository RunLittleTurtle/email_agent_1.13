"""
Adaptive Writer Agent
Generates email responses based on context and intent
"""

import json
from typing import Dict, Any, Optional

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
            
            # Add specialized agent results
            if state.calendar_data:
                context_parts.append("Calendar data available for meeting scheduling")
            if state.document_data:
                context_parts.append("Document search results available")
            if state.contact_data:
                context_parts.append("Contact information available")
            
            # System prompt for response generation
            system_prompt = """You are a professional email response writer.
            Generate appropriate, contextual email responses.
            Match the tone and formality of the original email.
            Be concise but complete.
            Always maintain a professional and helpful tone."""
            
            # Create response generation prompt
            prompt = f"""Generate a response to this email:

Original Email:
From: {state.email.sender}
Subject: {state.email.subject}
Body: {state.email.body}

Context Information:
{chr(10).join(context_parts)}

Intent Classification: {state.intent.value if state.intent else 'unknown'}

Generate a professional email response that:
1. Acknowledges the sender's message
2. Addresses all requested actions or questions
3. Maintains appropriate tone and formality
4. Is clear and concise
5. Includes a proper greeting and closing

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
                
                # Add message
                self._add_message(
                    state,
                    f"Draft response generated with {response_data.get('confidence', 0):.0%} confidence",
                    metadata=response_data
                )
                
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
