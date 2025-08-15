"""
Base Agent class for Ambient Email Agent
All agents inherit from this abstract base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langsmith import traceable
import structlog

from src.models.state import AgentState


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
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the current state and return updated state.
        This must be implemented by each specific agent.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        pass

    @traceable(name="agent_invoke", tags=["agent"])
    async def ainvoke(self, state: AgentState) -> AgentState:
        """
        Invoke the agent with error handling and logging.
        This wraps the process method with common functionality.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        start_time = datetime.now()
        self.logger.info(f"Starting {self.name} processing")

        try:
            # Update current agent
            state.current_agent = self.name

            # Process state
            updated_state = await self.process(state)

            # Log success
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"{self.name} completed successfully",
                duration=duration
            )

            return updated_state

        except Exception as e:
            # Log error
            self.logger.error(
                f"{self.name} failed",
                error=str(e),
                exc_info=True
            )

            # Update state with error
            state.add_error(f"{self.name} failed: {str(e)}")
            return state

    def _add_message(
        self,
        state: AgentState,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Helper method to add a message to state

        Args:
            state: Current state
            content: Message content
            metadata: Optional metadata
        """
        state.add_message(
            role=self.name,
            content=content,
            metadata=metadata
        )

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
