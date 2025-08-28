"""
Human Feedback Processor
Dedicated node for processing and analyzing human feedback from any workflow node
Uses LLM to intelligently format feedback for downstream agents
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable
import structlog

from .base_agent import BaseAgent

# Set up module-level logger
logger = structlog.get_logger(__name__)


class HumanFeedbackProcessor(BaseAgent):
    """
    Dedicated processor for human feedback from any workflow node.
    Uses LLM to analyze and format feedback for downstream consumption.
    """

    def __init__(self):
        super().__init__(
            name="human_feedback_processor",
            model="gpt-4o",
            temperature=0.1  # Low temperature for consistent formatting
        )

    async def process(self, state, runtime=None) -> Dict[str, Any]:
        """Required abstract method implementation - not used directly"""
        return {}

    @traceable(name="process_human_feedback", tags=["human_feedback", "processing"])
    async def process_feedback(
        self,
        feedback_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process human feedback and return formatted output for state.messages

        Args:
            feedback_data: Raw human feedback data
            context: Context about what the feedback is for (node, action, etc.)

        Returns:
            State updates with formatted messages
        """
        try:
            self.logger.info("Starting feedback processing", feedback_data=feedback_data, context_keys=list(context.keys()))

            # Extract feedback details
            feedback_type = feedback_data.get("type", "unknown")
            feedback_content = feedback_data.get("content", "")
            source_node = context.get("source_node", "unknown")
            action_context = context.get("action_context", "")

            self.logger.info(f"Processing human feedback from {source_node}: {feedback_type}", content=feedback_content)

            # Use LLM to analyze and format feedback
            formatted_feedback = await self._analyze_feedback_with_llm(
                feedback_data, context
            )

            # Create human message for workflow visibility
            human_message = HumanMessage(
                content=formatted_feedback["human_readable"],
                name="human_feedback"
            )

            # Create AI analysis message
            ai_analysis = AIMessage(
                content=formatted_feedback["ai_analysis"],
                name="feedback_processor"
            )

            return {
                "messages": [human_message, ai_analysis],
                "response_metadata": {
                    "human_feedback_processed": {
                        "timestamp": datetime.now().isoformat(),
                        "source_node": source_node,
                        "decision": formatted_feedback["decision"],
                        "confidence": formatted_feedback["confidence"],
                        "next_actions": formatted_feedback["next_actions"]
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to process human feedback: {e}",
                            feedback_data=feedback_data,
                            context=context,
                            exc_info=True)
            return {
                "error_messages": [f"Feedback processing failed: {str(e)}"],
                "messages": [HumanMessage(
                    content=f"Human feedback received but processing failed: {feedback_content}",
                    name="human_feedback"
                )]
            }

    async def _analyze_feedback_with_llm(
        self,
        feedback_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to intelligently analyze human feedback"""

        self.logger.info("Starting LLM analysis of feedback")

        system_prompt = """You are a human feedback analyzer. Your job is to:
1. Interpret human feedback in context of the ENTIRE conversation
2. Determine the clear decision/intent
3. Format it for AI agents to understand
4. Suggest next actions based on all prior context

Always respond in JSON format."""

        # Format message history for context - safe access
        message_summary = ""
        try:
            messages = context.get('message_history', [])
            if messages and isinstance(messages, list):
                for i, msg in enumerate(messages[-5:]):  # Last 5 messages for context
                    try:
                        if isinstance(msg, dict):
                            content = str(msg.get('content', ''))[:200]
                            name = str(msg.get('name', 'unknown'))
                            message_summary += f"\n  - {name}: {content}..."
                        else:
                            message_summary += f"\n  - {str(msg)[:200]}..."
                    except Exception:
                        message_summary += f"\n  - [message processing error]"
        except Exception:
            message_summary = "\n  - [error accessing message history]"

        # Safe context extraction
        try:
            extracted_context_str = json.dumps(context.get('extracted_context', {}), indent=2)[:500]
        except Exception:
            extracted_context_str = str(context.get('extracted_context', {}))[:500]

        try:
            args_str = json.dumps(feedback_data.get('args', {}), indent=2)
        except Exception:
            args_str = str(feedback_data.get('args', {}))

        user_prompt = f"""Analyze this human feedback WITH FULL CONVERSATION CONTEXT:

CONVERSATION CONTEXT:
- Source Node: {context.get('source_node', 'unknown')}
- Action Context: {context.get('action_context', '')}
- Original Email Request: {context.get('original_request', '')}
- Draft Response Being Reviewed: {str(context.get('draft_response', ''))[:500]}...
- Extracted Context from Email: {extracted_context_str}...
- Recent Messages:{message_summary}

HUMAN FEEDBACK:
- Type: {feedback_data.get('type', 'unknown')}
- Content: {feedback_data.get('content', '')}
- Args: {args_str}

Provide analysis in this JSON format:
{{
    "decision": "approved|rejected|modified|unclear",
    "confidence": 0.0-1.0,
    "human_readable": "Clear human-readable summary of the feedback",
    "ai_analysis": "Detailed analysis for AI agents including implications and context from ENTIRE conversation",
    "key_points": ["list", "of", "key", "points"],
    "next_actions": ["recommended", "next", "steps"],
    "modifications_requested": "any specific changes requested (be specific about what was requested like time changes, people to exclude, etc)"
}}"""

        try:
            self.logger.info("Calling LLM for feedback analysis", prompt_length=len(user_prompt))
            response = await self._call_llm(user_prompt, system_prompt)
            self.logger.info("LLM response received", response_length=len(response))
            result = json.loads(response)
            self.logger.info("LLM analysis completed successfully", decision=result.get("decision"))
            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in LLM response: {e}", llm_response=response[:500] if 'response' in locals() else 'No response')
            # Fallback if LLM doesn't return valid JSON
            return {
                "decision": "unclear",
                "confidence": 0.5,
                "human_readable": f"Human feedback received: {feedback_data.get('content', 'No content')}",
                "ai_analysis": f"Raw feedback from {context.get('source_node', 'unknown')}: {feedback_data}",
                "key_points": ["Human provided feedback"],
                "next_actions": ["Review feedback and proceed accordingly"],
                "modifications_requested": feedback_data.get('content') if feedback_data.get('type') == 'modify' else None
            }
        except Exception as e:
            self.logger.error(f"Unexpected error in LLM analysis: {e}",
                            feedback_data=feedback_data,
                            context=context,
                            exc_info=True)
            # Handle any other errors
            return {
                "decision": "unclear",
                "confidence": 0.3,
                "human_readable": f"Human feedback received: {feedback_data.get('content', 'No content')}",
                "ai_analysis": f"Error processing feedback: {str(e)}",
                "key_points": ["Error in processing"],
                "next_actions": ["Manual review needed"],
                "modifications_requested": feedback_data.get('content') if feedback_data.get('type') == 'modify' else None
            }


@traceable(name="human_feedback_processor_node", tags=["human_feedback", "node"])
async def human_feedback_processor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone node function for processing human feedback
    Can be added to any workflow where human feedback processing is needed
    """
    # Check if there's pending human feedback to process
    pending_feedback = state.get("pending_human_feedback")
    if not pending_feedback:
        logger.info("No pending human feedback to process")
        return {}  # No feedback to process

    logger.info("Creating HumanFeedbackProcessor instance")
    processor = HumanFeedbackProcessor()

    # Extract context from state - now including full message history
    context = {
        "source_node": pending_feedback.get("source_node", "unknown"),
        "action_context": pending_feedback.get("action_context", ""),
        "original_request": state.get("email", {}).get("subject", "") if state.get("email") else "",
        "message_history": state.get("messages", []),
        "draft_response": state.get("draft_response", ""),
        "extracted_context": state.get("extracted_context", {})
    }

    # Process the feedback
    result = await processor.process_feedback(
        feedback_data=pending_feedback.get("feedback_data", {}),
        context=context
    )

    # Clear the pending feedback
    result["pending_human_feedback"] = None

    return result


def format_feedback_for_processing(
    feedback_response: Any,
    source_node: str,
    action_context: str = ""
) -> Dict[str, Any]:
    """
    Helper function to format raw human feedback for processing
    Use this in any node that receives human feedback
    """
    if isinstance(feedback_response, list):
        feedback_response = feedback_response[-1] if feedback_response else {}

    if not isinstance(feedback_response, dict):
        return {
            "source_node": source_node,
            "action_context": action_context,
            "feedback_data": {
                "type": "unknown",
                "content": str(feedback_response),
                "args": {}
            }
        }

    # Map response types to standard format
    response_type = feedback_response.get("type", "unknown")
    content = ""

    if response_type == "accept":
        content = "Approved"
    elif response_type == "ignore":
        content = "Rejected/Ignored"
    elif response_type in ["response", "edit"]:
        args = feedback_response.get("args", {})
        content = args.get("feedback", "") if isinstance(args, dict) else str(args)
        response_type = "modify"

    return {
        "source_node": source_node,
        "action_context": action_context,
        "feedback_data": {
            "type": response_type,
            "content": content,
            "args": feedback_response.get("args", {}),
            "raw_response": feedback_response
        }
    }


# Modern usage example:
"""
# In any node that gets human feedback:

human_response = interrupt({...})

# Format feedback for processing
pending_feedback = format_feedback_for_processing(
    human_response,
    source_node="calendar_booking_review",
    action_context="Calendar booking approval for pickleball game"
)

return {
    "pending_human_feedback": pending_feedback,
    # ... other state updates
}

# Then add human_feedback_processor_node to your workflow
"""
