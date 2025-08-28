"""
Base Agent class for Ambient Email Agent
All agents inherit from this abstract base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import time
import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langsmith import traceable
from langgraph.runtime import Runtime
import structlog

from src.models.state import AgentState, AgentOutput
from src.models.context import RuntimeContext


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the workflow.
    Provides common functionality and enforces interface.
    """

    def __init__(
        self,
        name: str,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        provider: str = "openai"
    ):
        """
        Initialize base agent

        Args:
            name: Agent name for identification
            model: Model to use (gpt-4o, claude-3-sonnet, etc.)
            temperature: Temperature for model
            provider: LLM provider (openai or anthropic)
        """
        self.name = name
        self.model_name = model
        self.temperature = temperature
        self.provider = provider

        # Initialize LLM based on provider
        if provider == "openai":
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
        elif provider == "anthropic":
            self.llm = ChatAnthropic(
                model=model,
                temperature=temperature
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Setup structured logging
        self.logger = structlog.get_logger().bind(agent=name)

    @abstractmethod
    async def process(self, state: AgentState, runtime: Optional[Runtime[RuntimeContext]] = None) -> Dict[str, Any]:
        """
        Process the current state and return state updates (LangGraph 0.6+ pattern).
        This must be implemented by each specific agent.

        Args:
            state: Current workflow state
            runtime: Runtime context with user preferences, tools, etc.

        Returns:
            Dictionary with state updates (messages, output, etc.)
        """
        pass

    @traceable(name="agent_invoke", tags=["agent"])
    async def ainvoke(self, state: AgentState, runtime: Optional[Runtime[RuntimeContext]] = None) -> Dict[str, Any]:
        """
        Modern LangGraph 0.6+ invoke pattern with comprehensive state tracking.
        This wraps the process method with full LangSmith visibility.

        Args:
            state: Current workflow state
            runtime: Runtime context with user preferences, tools, etc.

        Returns:
            State updates dictionary for reducers
        """
        start_time = time.time()

        # === COMPREHENSIVE INPUT STATE TRACKING ===
        input_summary = self._serialize_state_for_tracking(state)
        runtime_summary = self._serialize_runtime_for_tracking(runtime)

        self.logger.info(
            f"🎯 Starting {self.name} processing",
            agent=self.name,
            input_state_keys=list(input_summary.keys()),
            has_email=bool(state.email),
            has_context=bool(state.extracted_context),
            message_count=len(state.messages),
            current_phase=state.dynamic_context.current_phase if state.dynamic_context else "none",
            execution_step=state.dynamic_context.execution_step if state.dynamic_context else 0,
            runtime_context=bool(runtime)
        )

        try:
            # === PRE-PROCESSING STATE CAPTURE ===
            pre_processing_state = {
                "agent": self.name,
                "timestamp": datetime.now().isoformat(),
                "input_state": input_summary,
                "runtime_context": runtime_summary,
                "execution_metadata": {
                    "step": state.dynamic_context.execution_step if state.dynamic_context else 0,
                    "phase": state.dynamic_context.current_phase if state.dynamic_context else "unknown"
                }
            }

            # Process state and get updates with full tracking
            self.logger.info(f"🔄 {self.name} calling process method")
            updates = await self.process(state, runtime) or {}

            # Calculate execution time
            execution_time = time.time() - start_time

            # === POST-PROCESSING STATE TRACKING ===
            self.logger.info(
                f"✅ {self.name} process completed",
                duration=execution_time,
                updates_applied=list(updates.keys()),
                update_count=len(updates)
            )

            # Prepare all state updates for reducers with enhanced tracking
            state_updates = {}

            # Add current agent
            state_updates["current_agent"] = self.name

            # Add enhanced context updates with full tracking
            context_updates = state.update_dynamic_context(
                execution_step=state.dynamic_context.execution_step + 1 if state.dynamic_context else 1,
                current_phase=f"{self.name}_completed",
                accumulated_insights=[f"{self.name} processed at {datetime.now().isoformat()}"]
            )
            state_updates.update(context_updates)

            # Add agent processing updates
            state_updates.update(updates)

            # Add comprehensive agent output with full state tracking
            output_updates = state.add_agent_output(
                agent=self.name,
                message=f"{self.name} completed successfully with {len(updates)} state updates",
                confidence=0.9,
                execution_time=execution_time,
                data={
                    "updates_applied": list(updates.keys()),
                    "pre_processing_state": pre_processing_state,
                    "processing_results": self._serialize_updates_for_tracking(updates),
                    "state_changes": {
                        "before": input_summary,
                        "changes": updates,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                input_context=pre_processing_state,
                state_changes=updates,
                next_recommendations=[f"State ready for next agent or {self.name} processing complete"]
            )
            state_updates.update(output_updates)

            # === FINAL STATE LOGGING FOR LANGSMITH ===
            self.logger.info(
                f"🎉 {self.name} completed successfully - FULL STATE TRACKED",
                agent=self.name,
                duration=execution_time,
                updates=list(updates.keys()),
                state_changes_count=len(updates),
                execution_phase="completed",
                langsmith_metadata={
                    "agent_name": self.name,
                    "execution_time": execution_time,
                    "updates_count": len(updates),
                    "success": True,
                    "input_state_keys": list(input_summary.keys()),
                    "output_state_keys": list(updates.keys())
                }
            )

            return state_updates

        except Exception as e:
            execution_time = time.time() - start_time

            # === COMPREHENSIVE ERROR TRACKING ===
            self.logger.error(
                f"❌ {self.name} failed - FULL ERROR TRACKED",
                agent=self.name,
                error=str(e),
                error_type=type(e).__name__,
                duration=execution_time,
                input_state=input_summary,
                runtime_context=runtime_summary,
                langsmith_metadata={
                    "agent_name": self.name,
                    "execution_time": execution_time,
                    "success": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )

            # Prepare comprehensive error state updates for reducers
            state_updates = {}

            # Add current agent
            state_updates["current_agent"] = self.name

            # Add error updates with full context
            error_updates = state.add_error(f"[{self.name}] {type(e).__name__}: {str(e)}")
            state_updates.update(error_updates)

            # Add comprehensive failed agent output
            output_updates = state.add_agent_output(
                agent=self.name,
                message=f"{self.name} failed: {type(e).__name__}: {str(e)}",
                confidence=0.0,
                execution_time=execution_time,
                data={
                    "error_details": {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "input_state": input_summary,
                        "runtime_context": runtime_summary,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                errors=[f"{type(e).__name__}: {str(e)}"],
                input_context={"error_occurred_during": "agent_processing"},
                state_changes={"status": "error", "error_agent": self.name}
            )
            state_updates.update(output_updates)

            return state_updates

    def _serialize_state_for_tracking(self, state: AgentState) -> Dict[str, Any]:
        """
        Serialize agent state for comprehensive tracking in LangSmith

        Args:
            state: Current agent state

        Returns:
            Serialized state summary for logging
        """
        try:
            return {
                "has_email": bool(state.email),
                "email_subject": state.email.subject if state.email else None,
                "email_sender": state.email.sender if state.email else None,
                "intent": state.intent.value if state.intent else None,
                "extracted_context_available": bool(state.extracted_context),
                "urgency_level": state.extracted_context.urgency_level if state.extracted_context else None,
                "calendar_data_available": bool(state.calendar_data),
                "document_data_available": bool(state.document_data),
                "contact_data_available": bool(state.contact_data),
                "draft_response_available": bool(state.draft_response),
                "draft_length": len(state.draft_response) if state.draft_response else 0,
                "message_count": len(state.messages),
                "output_count": len(state.output),
                "error_count": len(state.error_messages),
                "current_agent": state.current_agent,
                "status": state.status,
                "workflow_id": state.workflow_id,
                "dynamic_context": {
                    "execution_step": state.dynamic_context.execution_step if state.dynamic_context else 0,
                    "current_phase": state.dynamic_context.current_phase if state.dynamic_context else "unknown",
                    "insights_count": len(state.dynamic_context.accumulated_insights) if state.dynamic_context else 0
                },
                "response_metadata_keys": list(state.response_metadata.keys()) if state.response_metadata else []
            }
        except Exception as e:
            return {"serialization_error": str(e)}

    def _serialize_runtime_for_tracking(self, runtime: Optional[Runtime[RuntimeContext]]) -> Dict[str, Any]:
        """
        Serialize runtime context for tracking

        Args:
            runtime: Runtime context

        Returns:
            Serialized runtime summary
        """
        if not runtime:
            return {"runtime_available": False}

        try:
            return {
                "runtime_available": True,
                "has_user_id": hasattr(runtime, 'user_id'),
                "user_id": getattr(runtime, 'user_id', None),
                "context_type": str(type(runtime))
            }
        except Exception as e:
            return {"runtime_available": True, "serialization_error": str(e)}

    def _serialize_updates_for_tracking(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize state updates for tracking

        Args:
            updates: State updates dictionary

        Returns:
            Serialized updates summary
        """
        try:
            serialized = {}
            for key, value in updates.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    serialized[key] = value
                elif isinstance(value, list):
                    serialized[key] = f"list_length_{len(value)}"
                elif isinstance(value, dict):
                    serialized[key] = f"dict_keys_{list(value.keys())}"
                else:
                    serialized[key] = f"object_{type(value).__name__}"
            return serialized
        except Exception as e:
            return {"serialization_error": str(e)}

    def create_ai_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> AIMessage:
        """
        Modern helper to create AI messages for LangGraph 0.6+ pattern
        Returns message for inclusion in state updates

        Args:
            content: Message content
            metadata: Optional metadata

        Returns:
            AIMessage for state updates
        """
        message = AIMessage(content=content, name=self.name)
        if metadata:
            message.additional_kwargs.update(metadata)
        return message

    async def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Helper method to call LLM with consistent error handling

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            LLM response as string
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise

    def format_prompt(self, template: str, **kwargs) -> str:
        """
        Helper method to format prompts with variables

        Args:
            template: Prompt template with {variable} placeholders
            **kwargs: Variables to insert

        Returns:
            Formatted prompt
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            self.logger.error(f"Missing prompt variable: {e}")
            raise
