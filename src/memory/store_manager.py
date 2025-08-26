"""
Store Manager for LangGraph 0.6+ long-term memory
Manages cross-conversation memory using LangGraph stores
https://langchain-ai.github.io/langgraph/concepts/memory/
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import get_runtime

from src.models.context import LongTermMemory, RuntimeContext

logger = structlog.get_logger()


class StoreManager:
    """
    Manages long-term memory across conversations using LangGraph stores
    Handles user profiles, preferences, and interaction history
    """

    def __init__(self, store: Optional[InMemoryStore] = None):
        """
        Initialize store manager with LangGraph store

        Args:
            store: LangGraph store instance (will create if None)
        """
        self.store = store or InMemoryStore()
        self.logger = logger.bind(component="store_manager")

    async def get_user_memory(self, user_id: str) -> Optional[LongTermMemory]:
        """
        Retrieve user's long-term memory from store

        Args:
            user_id: User identifier

        Returns:
            User's long-term memory or None if not found
        """
        try:
            # Use LangGraph store to get user memory
            memory_data = await self.store.aget(
                namespace="user_memory",
                key=user_id
            )
            
            if memory_data:
                return LongTermMemory(**memory_data.value)
            return None

        except Exception as e:
            self.logger.error(f"Failed to retrieve user memory: {e}")
            return None

    async def save_user_memory(self, user_id: str, memory: LongTermMemory):
        """
        Save user's long-term memory to store

        Args:
            user_id: User identifier
            memory: Long-term memory to save
        """
        try:
            memory.last_updated = datetime.now()
            
            await self.store.aput(
                namespace="user_memory",
                key=user_id,
                value=memory.dict()
            )
            
            self.logger.info(f"Saved user memory for {user_id}")

        except Exception as e:
            self.logger.error(f"Failed to save user memory: {e}")

    async def update_user_profile(
        self, 
        user_id: str, 
        profile_updates: Dict[str, Any]
    ):
        """
        Update specific fields in user profile

        Args:
            user_id: User identifier
            profile_updates: Fields to update
        """
        memory = await self.get_user_memory(user_id)
        if not memory:
            memory = LongTermMemory()

        # Update profile fields
        memory.user_profile.update(profile_updates)
        await self.save_user_memory(user_id, memory)

    async def add_interaction_history(
        self,
        user_id: str,
        interaction: Dict[str, Any]
    ):
        """
        Add new interaction to user's history

        Args:
            user_id: User identifier
            interaction: Interaction details to store
        """
        memory = await self.get_user_memory(user_id)
        if not memory:
            memory = LongTermMemory()

        # Add timestamp to interaction
        interaction["timestamp"] = datetime.now().isoformat()
        
        # Add to history (keep last 100 interactions)
        memory.interaction_history.append(interaction)
        if len(memory.interaction_history) > 100:
            memory.interaction_history = memory.interaction_history[-100:]

        await self.save_user_memory(user_id, memory)

    async def learn_user_pattern(
        self,
        user_id: str,
        pattern_type: str,
        pattern_data: Dict[str, Any]
    ):
        """
        Store learned patterns about user behavior

        Args:
            user_id: User identifier
            pattern_type: Type of pattern (scheduling, communication, etc.)
            pattern_data: Pattern details
        """
        memory = await self.get_user_memory(user_id)
        if not memory:
            memory = LongTermMemory()

        # Store pattern with timestamp
        pattern_data["learned_at"] = datetime.now().isoformat()
        memory.learned_patterns[pattern_type] = pattern_data

        await self.save_user_memory(user_id, memory)
        self.logger.info(f"Learned {pattern_type} pattern for user {user_id}")

    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get user preferences for context-aware processing

        Args:
            user_id: User identifier

        Returns:
            User preferences dictionary
        """
        memory = await self.get_user_memory(user_id)
        if memory:
            return memory.user_profile.get("preferences", {})
        return {}

    async def get_scheduling_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's scheduling patterns and preferences

        Args:
            user_id: User identifier

        Returns:
            Scheduling preferences
        """
        memory = await self.get_user_memory(user_id)
        if memory:
            return memory.scheduling_preferences
        return {}

    async def search_interaction_history(
        self,
        user_id: str,
        query_type: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search user's interaction history for relevant context

        Args:
            user_id: User identifier
            query_type: Type of interactions to find
            limit: Maximum results to return

        Returns:
            Relevant interaction history
        """
        memory = await self.get_user_memory(user_id)
        if not memory:
            return []

        # Simple filtering - could be enhanced with semantic search
        relevant = [
            interaction for interaction in memory.interaction_history
            if query_type.lower() in str(interaction).lower()
        ]

        return relevant[-limit:]  # Return most recent matches
