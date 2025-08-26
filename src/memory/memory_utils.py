"""
Memory utilities for LangGraph 0.6+ workflow integration
Helper functions for memory management and context enrichment
"""

from typing import Dict, Any, Optional
from datetime import datetime
import structlog

from src.models.state import AgentState
from src.models.context import LongTermMemory, RuntimeContext
from .store_manager import StoreManager

logger = structlog.get_logger()


class MemoryUtils:
    """
    Utility functions for integrating memory with LangGraph workflow
    Provides helpers for context enrichment and memory updates
    """

    def __init__(self, store_manager: StoreManager):
        """
        Initialize with store manager

        Args:
            store_manager: Store manager instance
        """
        self.store_manager = store_manager
        self.logger = logger.bind(component="memory_utils")

    async def enrich_state_with_memory(
        self,
        state: AgentState,
        user_id: str
    ) -> AgentState:
        """
        Enrich agent state with user's long-term memory

        Args:
            state: Current agent state
            user_id: User identifier

        Returns:
            State enriched with memory context
        """
        try:
            # Load user's long-term memory
            memory = await self.store_manager.get_user_memory(user_id)
            if memory:
                state.long_term_memory = memory
                
                # Add memory insights to dynamic context
                state.add_insight(f"Loaded memory for user {user_id}")
                if memory.user_profile:
                    state.add_insight(f"User profile: {len(memory.user_profile)} preferences")
                if memory.interaction_history:
                    state.add_insight(f"History: {len(memory.interaction_history)} interactions")

            return state

        except Exception as e:
            self.logger.error(f"Failed to enrich state with memory: {e}")
            return state

    async def extract_insights_from_email(
        self,
        state: AgentState,
        user_id: str
    ):
        """
        Extract insights from current email interaction for learning

        Args:
            state: Current agent state
            user_id: User identifier
        """
        if not state.email:
            return

        try:
            # Extract interaction summary
            interaction = {
                "type": "email_interaction",
                "subject": state.email.subject,
                "sender": state.email.sender,
                "intent": state.intent.value if state.intent else "unknown",
                "urgency": state.extracted_context.urgency_level if state.extracted_context else "medium",
                "agent_outputs": [output.dict() for output in state.output],
                "response_generated": bool(state.draft_response),
                "execution_time": (datetime.now() - state.created_at).total_seconds()
            }

            # Store interaction history
            await self.store_manager.add_interaction_history(user_id, interaction)

            # Learn patterns if applicable
            await self._learn_patterns_from_interaction(state, user_id)

        except Exception as e:
            self.logger.error(f"Failed to extract insights: {e}")

    async def _learn_patterns_from_interaction(
        self,
        state: AgentState,
        user_id: str
    ):
        """
        Learn patterns from the current interaction

        Args:
            state: Current agent state
            user_id: User identifier
        """
        # Learn scheduling patterns
        if state.calendar_data and state.calendar_data.meeting_request:
            await self._learn_scheduling_pattern(state, user_id)

        # Learn communication patterns
        if state.email:
            await self._learn_communication_pattern(state, user_id)

    async def _learn_scheduling_pattern(
        self,
        state: AgentState,
        user_id: str
    ):
        """Learn scheduling preferences from calendar interactions"""
        try:
            meeting_data = state.calendar_data.meeting_request
            pattern_data = {
                "preferred_times": [],
                "meeting_duration_preference": meeting_data.get("duration", 30),
                "meeting_types": [meeting_data.get("type", "general")],
                "conflict_resolution": state.calendar_data.action_taken
            }

            if state.calendar_data.suggested_times:
                pattern_data["alternative_time_preferences"] = state.calendar_data.suggested_times

            await self.store_manager.learn_user_pattern(
                user_id, "scheduling", pattern_data
            )

        except Exception as e:
            self.logger.error(f"Failed to learn scheduling pattern: {e}")

    async def _learn_communication_pattern(
        self,
        state: AgentState,
        user_id: str
    ):
        """Learn communication preferences from email interactions"""
        try:
            pattern_data = {
                "typical_response_time": "immediate",  # Could be calculated
                "communication_style": state.response_metadata.get("generated_response", {}).get("tone", "professional"),
                "frequent_contacts": [state.email.sender],
                "common_requests": [state.intent.value if state.intent else "unknown"]
            }

            await self.store_manager.learn_user_pattern(
                user_id, "communication", pattern_data
            )

        except Exception as e:
            self.logger.error(f"Failed to learn communication pattern: {e}")

    async def get_contextual_recommendations(
        self,
        state: AgentState,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get contextual recommendations based on memory

        Args:
            state: Current agent state
            user_id: User identifier

        Returns:
            Contextual recommendations
        """
        try:
            memory = await self.store_manager.get_user_memory(user_id)
            if not memory:
                return {}

            recommendations = {}

            # Scheduling recommendations
            if state.intent and "meeting" in state.intent.value.lower():
                scheduling_prefs = await self.store_manager.get_scheduling_preferences(user_id)
                if scheduling_prefs:
                    recommendations["scheduling"] = {
                        "preferred_duration": scheduling_prefs.get("meeting_duration_preference", 30),
                        "typical_times": scheduling_prefs.get("preferred_times", []),
                        "conflict_strategy": scheduling_prefs.get("conflict_resolution", "suggest_alternatives")
                    }

            # Communication recommendations
            comm_patterns = memory.learned_patterns.get("communication", {})
            if comm_patterns:
                recommendations["communication"] = {
                    "preferred_tone": comm_patterns.get("communication_style", "professional"),
                    "response_urgency": comm_patterns.get("typical_response_time", "normal")
                }

            return recommendations

        except Exception as e:
            self.logger.error(f"Failed to get recommendations: {e}")
            return {}

    def create_runtime_context(
        self,
        user_id: str,
        user_email: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> RuntimeContext:
        """
        Create runtime context for LangGraph 0.6+ pattern

        Args:
            user_id: User identifier
            user_email: User email
            preferences: User preferences

        Returns:
            Runtime context for agent invocation
        """
        return RuntimeContext(
            user_id=user_id,
            user_email=user_email,
            user_preferences=preferences or {},
            available_tools=["gmail", "calendar", "documents", "contacts"],
            timezone="UTC",  # Should be determined from user profile
            language="en"
        )
