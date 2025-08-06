"""
Models package for Ambient Email Agent
Contains Pydantic models for state management
"""

from .state import (
    EmailIntent,
    EmailMessage,
    ExtractedContext,
    TaskDecomposition,
    AgentState,
)

__all__ = [
    "EmailIntent",
    "EmailMessage",
    "ExtractedContext",
    "TaskDecomposition",
    "AgentState",
]
