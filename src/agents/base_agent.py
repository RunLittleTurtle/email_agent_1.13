"""
Base Agent class for Ambient Email Agent
All agents inherit from this abstract base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import time

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
    async def ainvoke(self, state: AgentState, runtime: Optional[Runtime[RuntimeContext]] = None) -> AgentState:
        """
        Modern LangGraph 0.6+ invoke pattern with context support.
        This wraps the process method with common functionality.

        Args:
            state: Current workflow state
            runtime: Runtime context with user preferences, tools, etc.

        Returns:
            Updated workflow state
        """
        start_time = time.time()
        self.logger.info(f"Starting {self.name} processing")

        try:
            # Update current agent and execution context
            state.current_agent = self.name
            state.update_dynamic_context(
                execution_step=state.dynamic_context.execution_step + 1,
                current_phase=f"{self.name}_processing"
            )

            # Process state and get updates
            updates = await self.process(state, runtime)
            
            # Apply updates to state
            for key, value in updates.items():
                if key == "messages" and isinstance(value, list):
                    # LangGraph will handle message merging via add_messages reducer
                    state.messages.extend(value)
                elif hasattr(state, key):
                    setattr(state, key, value)

            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create rich agent output
            state.add_agent_output(
                agent=self.name,
                message=f"{self.name} completed successfully",
                confidence=0.9,
                execution_time=execution_time,
                data={"updates_applied": list(updates.keys())}
            )

            # Log success
            self.logger.info(
                f"{self.name} completed successfully",
                duration=execution_time,
                updates=list(updates.keys())
            )

            return state

        except Exception as e:
            # Log error
            self.logger.error(
                f"{self.name} failed",
                error=str(e),
                exc_info=True
            )

            # Add error to state with rich context
            state.add_error(f"{self.name} failed: {str(e)}")
            state.add_agent_output(
                agent=self.name,
                message=f"{self.name} failed: {str(e)}",
                confidence=0.0,
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
            return state

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
