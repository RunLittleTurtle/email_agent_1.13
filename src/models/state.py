"""
State models for Ambient Email Agent
Defines the shared state structure for all agents in the LangGraph workflow
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Literal, Sequence, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


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


class AgentState(BaseModel):
    """
    Shared state for all agents in the workflow.
    This is the central state object that gets passed between agents.
    """
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
    response_metadata: Dict[str, Any] = Field(default_factory=dict)
    output: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Agent output messages for downstream consumption"
    )
    
    # Workflow control
    current_agent: Optional[str] = None
    status: Literal["processing", "approved", "rejected", "error"] = "processing"
    human_feedback: Optional[str] = None
    error_messages: List[str] = Field(default_factory=list)
    
    # Tracking
    workflow_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_ai_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Helper method to create an AI message (use sparingly - prefer LangGraph's add_messages)"""
        from langchain_core.messages import AIMessage
        message = AIMessage(content=content)
        if metadata:
            message.additional_kwargs.update(metadata)
        # Note: State updates should use return {"messages": [message]} in nodes
        self.updated_at = datetime.now()
        return message
    
    def add_human_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Helper method to create a human message (use sparingly - prefer LangGraph's add_messages)"""
        from langchain_core.messages import HumanMessage
        message = HumanMessage(content=content)
        if metadata:
            message.additional_kwargs.update(metadata)
        # Note: State updates should use return {"messages": [message]} in nodes
        self.updated_at = datetime.now()
        return message
    
    def add_error(self, error: str):
        """Helper method to add an error message"""
        self.error_messages.append(f"[{self.current_agent or 'unknown'}] {error}")
        self.status = "error"
        self.updated_at = datetime.now()
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
