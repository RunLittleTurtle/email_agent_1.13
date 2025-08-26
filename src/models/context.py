"""
Context schemas for LangGraph 0.6+ modern architecture
Defines runtime context and dynamic context management
https://langchain-ai.github.io/langgraph/agents/context/
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass


@dataclass
class RuntimeContext:
    """
    Static runtime context - immutable data passed at invocation
    Contains user metadata, tools, DB connections, preferences
    """
    user_id: str
    user_email: str
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    available_tools: List[str] = Field(default_factory=list)
    timezone: str = "UTC"
    language: str = "en"
    company_context: Optional[Dict[str, Any]] = None


class DynamicContext(BaseModel):
    """
    Dynamic runtime context - mutable data that evolves during execution
    Represents the evolving execution state and accumulated insights
    """
    execution_step: int = 0
    current_phase: str = "initialization"
    accumulated_insights: List[str] = Field(default_factory=list)
    execution_metadata: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class LongTermMemory(BaseModel):
    """
    Cross-conversation memory structure
    Persistent data that spans multiple sessions and threads
    """
    user_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Persistent user preferences and profile data"
    )
    interaction_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical interaction summaries"
    )
    learned_patterns: Dict[str, Any] = Field(
        default_factory=dict,
        description="AI-discovered patterns about user behavior"
    )
    contact_relationships: Dict[str, Any] = Field(
        default_factory=dict,
        description="Known relationships and contact preferences"
    )
    scheduling_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="Learned scheduling patterns and preferences"
    )
    last_updated: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
