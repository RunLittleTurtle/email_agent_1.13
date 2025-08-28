"""
State models for Ambient Email Agent
Defines the shared state structure for all agents in the LangGraph workflow
"""

import operator
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Literal, Sequence, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .context import DynamicContext, LongTermMemory


def merge_dynamic_context(left, right) -> DynamicContext:
    """
    Custom reducer for DynamicContext that safely merges updates
    from multiple agents updating concurrently
    """
    if left is None:
        left = DynamicContext()
    if right is None:
        return left

    # Handle dict updates from helper methods
    if isinstance(right, dict):
        # Create a new context with merged data
        merged_insights = list(left.accumulated_insights)
        if "accumulated_insights" in right:
            new_insights = right["accumulated_insights"]
            if isinstance(new_insights, list):
                merged_insights.extend(new_insights)
            else:
                merged_insights.append(new_insights)

        # Merge metadata
        merged_metadata = {**left.execution_metadata}
        if "execution_metadata" in right:
            merged_metadata.update(right["execution_metadata"])

        # Create updated context
        return DynamicContext(
            execution_step=right.get("execution_step", left.execution_step),
            current_phase=right.get("current_phase", left.current_phase),
            accumulated_insights=merged_insights,
            execution_metadata=merged_metadata,
            performance_metrics={**left.performance_metrics, **right.get("performance_metrics", {})}
        )

    # Handle DynamicContext object merging
    if isinstance(right, DynamicContext):
        merged_insights = list(left.accumulated_insights)
        for insight in right.accumulated_insights:
            if insight not in merged_insights:
                merged_insights.append(insight)

        return DynamicContext(
            execution_step=max(left.execution_step, right.execution_step),
            current_phase=right.current_phase,
            accumulated_insights=merged_insights,
            execution_metadata={**left.execution_metadata, **right.execution_metadata},
            performance_metrics={**left.performance_metrics, **right.performance_metrics}
        )

    return left




class EmailIntent(str, Enum):
    """Email intent classification"""
    MEETING_REQUEST = "meeting_request"
    DOCUMENT_REQUEST = "document_request"
    TASK_DELEGATION = "task_delegation"
    SIMPLE_DIRECT = "simple_direct"
    MULTIPLE_INTENTS = "multiple_intents"


class EmailMessage(BaseModel):
    """Represents an email message"""
    id: str
    subject: str
    body: str
    sender: str
    recipients: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)
    attachments: Optional[List[str]] = Field(default_factory=list)
    thread_id: Optional[str] = None
    message_id: Optional[str] = Field(None, description="Gmail Message-ID for reply threading (e.g., <CAG41pbv...@mail.gmail.com>)")


class ExtractedContext(BaseModel):
    """Context extracted from email"""
    key_entities: List[str] = Field(
        default_factory=list,
        description="People, companies, projects mentioned"
    )
    dates_mentioned: Optional[List[datetime]] = Field(default_factory=list)
    requested_actions: List[str] = Field(default_factory=list)
    urgency_level: Literal["low", "medium", "high"] = "medium"
    sentiment: Optional[Literal["positive", "neutral", "negative"]] = "neutral"


class TaskDecomposition(BaseModel):
    """Decomposed tasks for multi-intent emails"""
    tasks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of decomposed tasks"
    )
    parallel_executable: bool = True


class CalendarData(BaseModel):
    """Calendar-specific data from CalendarAgent"""
    meeting_request: Optional[Dict[str, Any]] = None
    availability: Optional[Dict[str, Any]] = None
    suggested_times: List[Dict[str, Any]] = Field(default_factory=list)
    action_taken: Optional[str] = None
    availability_status: Optional[str] = None
    message: Optional[str] = None
    booked_event: Optional[Dict[str, Any]] = None
    attendees_notified: List[str] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)


class DocumentData(BaseModel):
    """Document search results from RAGAgent"""
    found_documents: List[Dict[str, Any]] = Field(default_factory=list)
    missing_documents: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)


class ContactData(BaseModel):
    """Contact information from CRMAgent"""
    contacts: List[Dict[str, Any]] = Field(default_factory=list)
    unknown_contacts: List[str] = Field(default_factory=list)
    relationship_context: Optional[Dict[str, Any]] = None


class AgentOutput(BaseModel):
    """
    Rich output structure for agent results
    Provides detailed context about agent execution and results
    """
    agent: str = Field(description="Name of the agent that produced this output")
    timestamp: datetime = Field(default_factory=datetime.now)
    message: str = Field(description="Human-readable summary of agent's work")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score for the results")
    execution_time_seconds: Optional[float] = Field(None, description="Time taken for agent execution")

    # Detailed results and metadata
    data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Structured data produced by agent")
    tools_used: List[str] = Field(default_factory=list, description="Tools/APIs called during execution")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Any warnings or issues")

    # Context and state awareness
    input_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Input context when agent started")
    state_changes: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Changes made to state")
    next_recommendations: List[str] = Field(default_factory=list, description="Recommended next steps")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentState(BaseModel):
    """Shared state for all agents in the LangGraph workflow"""
    # Core state - LangGraph native message handling
    messages: Annotated[Sequence[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="Message history using LangGraph add_messages reducer"
    )
    email: Optional[EmailMessage] = None
    extracted_context: Optional[ExtractedContext] = None
    intent: Optional[EmailIntent] = None

    # Agent-specific data
    calendar_data: Optional[CalendarData] = None
    document_data: Optional[DocumentData] = None
    contact_data: Optional[ContactData] = None
    decomposed_tasks: Optional[TaskDecomposition] = None

    # Response generation
    draft_response: Optional[str] = None
    response_metadata: Annotated[Dict[str, Any], operator.or_] = Field(
        default_factory=dict,
        description="Response metadata that can be updated by multiple agents"
    )
    output: Annotated[List[AgentOutput], operator.add] = Field(
        default_factory=list,
        description="Rich agent output with execution details and context"
    )

    # Modern LangGraph 0.6+ Context Management
    dynamic_context: Annotated[DynamicContext, merge_dynamic_context] = Field(
        default_factory=DynamicContext,
        description="Evolving execution context with concurrent merge support"
    )
    long_term_memory: Optional[LongTermMemory] = Field(
        None,
        description="Cross-conversation memory and user profile data"
    )

    # Workflow control
    current_agent: Optional[str] = None
    status: Literal["processing", "approved", "rejected", "error"] = "processing"
    human_feedback: Optional[str] = None
    pending_human_feedback: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pending human feedback data to be processed by human_feedback_processor"
    )
    error_messages: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="Error messages that can be added by multiple agents concurrently"
    )

    # Tracking
    workflow_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)



    def add_error(self, error: str) -> Dict[str, Any]:
        """Helper method to create error state updates for reducers"""
        return {
            "error_messages": [f"[{self.current_agent or 'unknown'}] {error}"],
            "status": "error",
            "updated_at": datetime.now()
        }

    def update_dynamic_context(self, **updates) -> Dict[str, Any]:
        """Create dynamic context updates for reducers"""
        return {
            "dynamic_context": updates,
            "updated_at": datetime.now()
        }

    def add_insight(self, insight: str) -> Dict[str, Any]:
        """Create insight updates for reducers"""
        return {
            "dynamic_context": {
                "accumulated_insights": [insight]
            },
            "updated_at": datetime.now()
        }

    def add_agent_output(
        self,
        agent: str,
        message: str,
        confidence: float = 0.8,
        execution_time: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        tools_used: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Helper to create agent output updates for reducers"""
        output = AgentOutput(
            agent=agent,
            message=message,
            confidence=confidence,
            execution_time_seconds=execution_time,
            data=data or {},
            tools_used=tools_used or [],
            **kwargs
        )
        return {
            "output": [output],
            "updated_at": datetime.now()
        }

    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
