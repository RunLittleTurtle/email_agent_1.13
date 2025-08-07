"""
Calendar Agent
Handles meeting requests and calendar operations using Google Calendar API
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os

from googleapiclient.errors import HttpError
from src.utils.google_auth import GoogleAuthHelper

from langsmith import traceable
from src.agents.base_agent import BaseAgent
from src.models.state import AgentState, CalendarData


class CalendarAgent(BaseAgent):
    """
    Agent responsible for calendar operations:
    - Check availability
    - Schedule meetings
    - Suggest meeting times
    - Handle rescheduling requests
    """
    
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    
    def __init__(self):
        super().__init__(
            name="calendar_agent",
            model="gpt-4o",
            temperature=0.1
        )
        self.service = None
        self._initialize_calendar_service()
    
    def _initialize_calendar_service(self):
        """Initialize Google Calendar service with OAuth2"""
        try:
            creds = GoogleAuthHelper.get_credentials(self.SCOPES, 'token_calendar.pickle')
            if creds:
                from googleapiclient.discovery import build
                self.service = build('calendar', 'v3', credentials=creds)
                self.logger.info("Google Calendar service initialized")
            else:
                self.logger.warning("Failed to get Google Calendar credentials, using mock service")
                self.service = GoogleAuthHelper.create_mock_service('calendar', 'v3')
        except Exception as e:
            self.logger.error(f"Error initializing Calendar service: {e}")
            self.service = GoogleAuthHelper.create_mock_service('calendar', 'v3')
    
    @traceable(name="calendar_process", tags=["agent", "calendar"])
    async def process(self, state: AgentState) -> AgentState:
        """
        Process calendar-related requests from emails
        """
        try:
            self.logger.info("Processing calendar request")
            
            # Extract meeting details from email context
            meeting_details = await self._extract_meeting_details(state)
            
            # Initialize calendar data if not exists
            if not state.calendar_data:
                state.calendar_data = CalendarData()
            
            # Store the meeting request
            state.calendar_data.meeting_request = meeting_details
            
            # Check availability if dates are mentioned
            if meeting_details.get("proposed_times"):
                availability = await self._check_availability(
                    meeting_details["proposed_times"]
                )
                state.calendar_data.availability = availability
                
                # Suggest alternative times if conflicts exist
                if availability.get("conflicts"):
                    suggestions = await self._suggest_alternative_times(
                        meeting_details,
                        availability["conflicts"]
                    )
                    state.calendar_data.suggested_times = suggestions
            
            # Add summary message
            self._add_message(
                state,
                self._generate_calendar_summary(state.calendar_data),
                metadata={"calendar_data": state.calendar_data.dict()}
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Calendar agent failed: {str(e)}")
            state.add_error(f"Calendar processing failed: {str(e)}")
            return state
    
    async def _extract_meeting_details(self, state: AgentState) -> Dict[str, Any]:
        """Extract meeting details from email using LLM"""
        prompt = f"""Extract meeting details from this email:

Subject: {state.email.subject}
From: {state.email.sender}
Body: {state.email.body}

Extracted Context:
- Dates mentioned: {state.extracted_context.dates_mentioned if state.extracted_context else 'None'}
- Key entities: {state.extracted_context.key_entities if state.extracted_context else 'None'}

Return JSON with:
{{
    "title": "Meeting title",
    "description": "Meeting description",
    "attendees": ["email1@example.com"],
    "proposed_times": [
        {{"start": "2024-01-15T10:00:00", "end": "2024-01-15T11:00:00"}}
    ],
    "duration_minutes": 60,
    "location": "Location or 'Virtual'",
    "is_recurring": false,
    "urgency": "high/medium/low"
}}"""

        response = await self._call_llm(prompt)
        return json.loads(response)
    
    async def _check_availability(self, proposed_times: List[Dict[str, str]]) -> Dict[str, Any]:
        """Check calendar availability for proposed times"""
        availability = {"available": [], "conflicts": []}
        
        try:
            for time_slot in proposed_times:
                start_time = datetime.fromisoformat(time_slot["start"])
                end_time = datetime.fromisoformat(time_slot["end"])
                
                # Query calendar for conflicts
                events_result = self.service.events().list(
                    calendarId='primary',
                    timeMin=start_time.isoformat() + 'Z',
                    timeMax=end_time.isoformat() + 'Z',
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()
                
                events = events_result.get('items', [])
                
                if events:
                    availability["conflicts"].append({
                        "time_slot": time_slot,
                        "conflicting_events": [
                            {"summary": e.get("summary", "Busy"), 
                             "start": e["start"].get("dateTime", e["start"].get("date"))}
                            for e in events
                        ]
                    })
                else:
                    availability["available"].append(time_slot)
                    
        except HttpError as error:
            self.logger.error(f"Calendar API error: {error}")
            
        return availability
    
    async def _suggest_alternative_times(
        self, 
        meeting_details: Dict[str, Any],
        conflicts: List[Dict[str, Any]]
    ) -> List[datetime]:
        """Suggest alternative meeting times based on calendar availability"""
        suggestions = []
        duration_minutes = meeting_details.get("duration_minutes", 60)
        
        # Get the next 5 business days
        base_date = datetime.now()
        for days_ahead in range(1, 6):
            check_date = base_date + timedelta(days=days_ahead)
            
            # Skip weekends
            if check_date.weekday() >= 5:
                continue
                
            # Check standard meeting slots
            for hour in [9, 10, 11, 14, 15, 16]:
                slot_start = check_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                slot_end = slot_start + timedelta(minutes=duration_minutes)
                
                # Check if slot is available
                is_available = await self._is_slot_available(slot_start, slot_end)
                if is_available:
                    suggestions.append(slot_start)
                    
                if len(suggestions) >= 3:  # Limit suggestions
                    return suggestions
                    
        return suggestions
    
    async def _is_slot_available(self, start: datetime, end: datetime) -> bool:
        """Check if a specific time slot is available"""
        try:
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=start.isoformat() + 'Z',
                timeMax=end.isoformat() + 'Z',
                singleEvents=True
            ).execute()
            
            return len(events_result.get('items', [])) == 0
            
        except HttpError:
            return False
    
    def _generate_calendar_summary(self, calendar_data: CalendarData) -> str:
        """Generate a summary of calendar processing results"""
        summary_parts = []
        
        if calendar_data.meeting_request:
            summary_parts.append(
                f"📅 Meeting request identified: {calendar_data.meeting_request.get('title', 'Untitled')}"
            )
        
        if calendar_data.availability:
            available = len(calendar_data.availability.get("available", []))
            conflicts = len(calendar_data.availability.get("conflicts", []))
            summary_parts.append(f"Availability check: {available} slots available, {conflicts} conflicts")
        
        if calendar_data.suggested_times:
            summary_parts.append(
                f"Suggested {len(calendar_data.suggested_times)} alternative time slots"
            )
        
        return " | ".join(summary_parts) if summary_parts else "Calendar processing completed"
