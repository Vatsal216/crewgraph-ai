"""
Base Memory Interface for CrewGraph AI
Defines the contract for all memory backends
"""

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..config import get_current_user, get_formatted_timestamp
from ..utils.exceptions import MemoryError
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class MemoryOperation(Enum):
    """Memory operation types for tracking"""

    SAVE = "save"
    LOAD = "load"
    DELETE = "delete"
    EXISTS = "exists"
    CLEAR = "clear"
    LIST_KEYS = "list_keys"
    GET_SIZE = "get_size"
    SEARCH = "search"


@dataclass
class MemoryStats:
    """Memory backend statistics"""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_keys: int = 0
    total_size_bytes: int = 0
    average_operation_time: float = 0.0
    last_operation_time: float = 0.0
    backend_type: str = ""
    created_at: str = ""
    created_by: str = ""
    
    def __post_init__(self):
        """Initialize dynamic fields after creation"""
        if not self.created_at:
            self.created_at = get_formatted_timestamp()
        if not self.created_by:
            self.created_by = get_current_user()


class BaseMemory(ABC):
    """
    Abstract base class for all memory backends in CrewGraph AI.

    This class defines the standard interface that all memory backends
    must implement, ensuring consistency across different storage systems.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base memory backend.

        Args:
            config: Backend-specific configuration
        """
        self.config = config or {}
        self.stats = MemoryStats(backend_type=self.__class__.__name__)
        self._lock = threading.RLock()
        self._connected = False

        logger.info(f"Initializing {self.__class__.__name__} memory backend")
        logger.info(f"User: {get_current_user()}, Time: {get_formatted_timestamp()}")

    def _record_operation(self, operation: MemoryOperation, success: bool, duration: float):
        """Record operation metrics and statistics"""
        with self._lock:
            self.stats.total_operations += 1
            self.stats.last_operation_time = time.time()

            if success:
                self.stats.successful_operations += 1
            else:
                self.stats.failed_operations += 1

            # Update average operation time
            if self.stats.total_operations > 0:
                total_time = self.stats.average_operation_time * (self.stats.total_operations - 1)
                self.stats.average_operation_time = (
                    total_time + duration
                ) / self.stats.total_operations

        # Record global metrics
        metrics.record_duration(
            f"memory_operation_{operation.value}_duration_seconds",
            duration,
            labels={
                "backend": self.__class__.__name__,
                "success": str(success),
                "user": get_current_user(),
            },
        )

        metrics.increment_counter(
            f"memory_operations_total",
            labels={
                "backend": self.__class__.__name__,
                "operation": operation.value,
                "success": str(success),
                "user": get_current_user(),
            },
        )

    def _execute_with_metrics(self, operation: MemoryOperation, func, *args, **kwargs):
        """Execute operation with automatic metrics recording"""
        start_time = time.time()
        success = False

        try:
            result = func(*args, **kwargs)
            success = True
            return result
        except Exception as e:
            logger.error(f"Memory operation {operation.value} failed: {e}")
            raise MemoryError(
                f"Memory operation failed: {operation.value}",
                operation=operation.value,
                backend=self.__class__.__name__,
                original_error=e,
            )
        finally:
            duration = time.time() - start_time
            self._record_operation(operation, success, duration)

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the memory backend"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the memory backend"""
        pass

    @abstractmethod
    def save(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Save value to memory with optional TTL.

        Args:
            key: Storage key
            value: Value to store
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def load(self, key: str) -> Any:
        """
        Load value from memory.

        Args:
            key: Storage key

        Returns:
            Stored value or None if not found
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete value from memory.

        Args:
            key: Storage key

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in memory.

        Args:
            key: Storage key

        Returns:
            True if key exists, False otherwise
        """
        pass

    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all data from memory.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        List all keys in memory, optionally filtered by pattern.

        Args:
            pattern: Optional pattern to filter keys

        Returns:
            List of keys
        """
        pass

    @abstractmethod
    def get_size(self) -> int:
        """
        Get total size of stored data in bytes.

        Returns:
            Size in bytes
        """
        pass

    def batch_save(self, data: Dict[str, Any], ttl: Optional[int] = None) -> Dict[str, bool]:
        """
        Save multiple key-value pairs in batch.

        Args:
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds

        Returns:
            Dictionary of key -> success status
        """

        def _batch_save():
            results = {}
            for key, value in data.items():
                try:
                    results[key] = self.save(key, value, ttl)
                except Exception as e:
                    logger.error(f"Batch save failed for key '{key}': {e}")
                    results[key] = False
            return results

        return self._execute_with_metrics(MemoryOperation.SAVE, _batch_save)

    def batch_load(self, keys: List[str]) -> Dict[str, Any]:
        """
        Load multiple values in batch.

        Args:
            keys: List of keys to load

        Returns:
            Dictionary of key -> value (None for missing keys)
        """

        def _batch_load():
            results = {}
            for key in keys:
                try:
                    results[key] = self.load(key)
                except Exception as e:
                    logger.error(f"Batch load failed for key '{key}': {e}")
                    results[key] = None
            return results

        return self._execute_with_metrics(MemoryOperation.LOAD, _batch_load)

    def batch_delete(self, keys: List[str]) -> Dict[str, bool]:
        """
        Delete multiple keys in batch.

        Args:
            keys: List of keys to delete

        Returns:
            Dictionary of key -> success status
        """

        def _batch_delete():
            results = {}
            for key in keys:
                try:
                    results[key] = self.delete(key)
                except Exception as e:
                    logger.error(f"Batch delete failed for key '{key}': {e}")
                    results[key] = False
            return results

        return self._execute_with_metrics(MemoryOperation.DELETE, _batch_delete)

    def get_stats(self) -> MemoryStats:
        """Get memory backend statistics"""
        with self._lock:
            # Update current stats
            self.stats.total_keys = len(self.list_keys())
            self.stats.total_size_bytes = self.get_size()
            return self.stats

    def get_health(self) -> Dict[str, Any]:
        """Get memory backend health status"""
        try:
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            test_value = "health_check_value"

            # Test save/load/delete
            save_success = self.save(test_key, test_value, ttl=60)
            load_success = self.load(test_key) == test_value
            delete_success = self.delete(test_key)

            healthy = save_success and load_success and delete_success

            return {
                "status": "healthy" if healthy else "unhealthy",
                "backend_type": self.__class__.__name__,
                "connected": self._connected,
                "operations_test": {
                    "save": save_success,
                    "load": load_success,
                    "delete": delete_success,
                },
                "stats": self.get_stats().__dict__,
                "timestamp": get_formatted_timestamp(),
                "checked_by": get_current_user(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "backend_type": self.__class__.__name__,
                "connected": self._connected,
                "error": str(e),
                "timestamp": get_formatted_timestamp(),
                "checked_by": get_current_user(),
            }

    # ============= MESSAGE HANDLING METHODS =============

    def save_conversation(self, conversation_id: str, messages: List[BaseMessage]) -> bool:
        """Save conversation with proper message types"""
        try:
            # Serialize messages to a format suitable for storage
            serialized_messages = []
            for msg in messages:
                msg_data = {
                    "type": msg.__class__.__name__,
                    "content": msg.content,
                    "additional_kwargs": getattr(msg, "additional_kwargs", {}),
                    "timestamp": time.time(),
                }
                serialized_messages.append(msg_data)

            conversation_data = {
                "conversation_id": conversation_id,
                "messages": serialized_messages,
                "created_at": time.time(),
                "message_count": len(messages),
            }

            key = f"conversation:{conversation_id}"
            return self.save(key, conversation_data)

        except Exception as e:
            logger.error(f"Failed to save conversation {conversation_id}: {e}")
            return False

    def load_conversation(self, conversation_id: str) -> List[BaseMessage]:
        """Load conversation messages"""
        try:
            key = f"conversation:{conversation_id}"
            conversation_data = self.load(key)

            if not conversation_data:
                return []

            # Deserialize messages back to proper types
            messages = []
            for msg_data in conversation_data.get("messages", []):
                msg_type = msg_data.get("type", "AIMessage")
                content = msg_data.get("content", "")
                additional_kwargs = msg_data.get("additional_kwargs", {})

                if msg_type == "HumanMessage":
                    message = HumanMessage(content=content, additional_kwargs=additional_kwargs)
                elif msg_type == "AIMessage":
                    message = AIMessage(content=content, additional_kwargs=additional_kwargs)
                else:
                    # Default to AIMessage for unknown types
                    message = AIMessage(content=content, additional_kwargs=additional_kwargs)

                messages.append(message)

            return messages

        except Exception as e:
            logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return []

    def search_messages(
        self, query: str, message_type: Optional[type] = None, limit: int = 10
    ) -> List[BaseMessage]:
        """Search messages by content and type"""
        try:
            all_messages = []

            # Get all conversation keys
            conversation_keys = [k for k in self.list_keys() if k.startswith("conversation:")]

            for key in conversation_keys:
                messages = self.load_conversation(key.split(":", 1)[1])

                for msg in messages:
                    # Filter by message type if specified
                    if message_type and not isinstance(msg, message_type):
                        continue

                    # Simple text search in content
                    if query.lower() in msg.content.lower():
                        all_messages.append(msg)

                        if len(all_messages) >= limit:
                            return all_messages

            return all_messages

        except Exception as e:
            logger.error(f"Failed to search messages: {e}")
            return []

    def append_message_to_conversation(self, conversation_id: str, message: BaseMessage) -> bool:
        """Append a new message to an existing conversation"""
        try:
            # Load existing conversation
            existing_messages = self.load_conversation(conversation_id)

            # Add new message
            existing_messages.append(message)

            # Save updated conversation
            return self.save_conversation(conversation_id, existing_messages)

        except Exception as e:
            logger.error(f"Failed to append message to conversation {conversation_id}: {e}")
            return False

    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get comprehensive summary information about a conversation"""
        try:
            messages = self.load_conversation(conversation_id)

            if not messages:
                return {
                    "conversation_id": conversation_id,
                    "message_count": 0,
                    "human_messages": 0,
                    "ai_messages": 0,
                    "total_content_length": 0,
                    "agents_involved": [],
                    "conversation_duration": 0,
                    "topics_discussed": [],
                    "error_count": 0,
                    "success_rate": 0.0,
                }

            # Basic message analysis
            human_count = len([m for m in messages if isinstance(m, HumanMessage)])
            ai_count = len([m for m in messages if isinstance(m, AIMessage)])
            total_content = sum(len(m.content) for m in messages)

            # Extract agents involved and analyze success/errors
            agents_involved = set()
            error_count = 0
            successful_responses = 0
            timestamps = []

            for msg in messages:
                if isinstance(msg, AIMessage) and hasattr(msg, "additional_kwargs"):
                    kwargs = msg.additional_kwargs

                    # Extract agent information
                    agent = kwargs.get("agent", kwargs.get("task_name", "unknown"))
                    if agent != "unknown":
                        agents_involved.add(agent)

                    # Count errors and successes
                    if kwargs.get("error", False):
                        error_count += 1
                    elif kwargs.get("success", True):
                        successful_responses += 1

                    # Collect timestamps
                    if "timestamp" in kwargs:
                        timestamps.append(kwargs["timestamp"])

            # Calculate metrics
            success_rate = (successful_responses / ai_count) if ai_count > 0 else 0.0
            conversation_duration = (
                (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0
            )

            # Extract topics (simple keyword-based)
            topics_discussed = self._extract_conversation_topics(messages)

            summary = {
                "conversation_id": conversation_id,
                "message_count": len(messages),
                "human_messages": human_count,
                "ai_messages": ai_count,
                "total_content_length": total_content,
                "agents_involved": list(agents_involved),
                "conversation_duration": conversation_duration,
                "topics_discussed": topics_discussed,
                "error_count": error_count,
                "success_rate": success_rate,
                "average_message_length": total_content / len(messages) if messages else 0,
                "interaction_ratio": human_count / ai_count if ai_count > 0 else 0,
                "first_message_time": min(timestamps) if timestamps else None,
                "last_message_time": max(timestamps) if timestamps else None,
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to get conversation summary for {conversation_id}: {e}")
            return {"conversation_id": conversation_id, "error": str(e), "message_count": 0}

    def _extract_conversation_topics(self, messages: List[BaseMessage]) -> List[str]:
        """Extract main topics from conversation (simple keyword-based)"""
        try:
            common_topics = [
                "strategy",
                "analysis",
                "research",
                "technology",
                "business",
                "ai",
                "machine learning",
                "data",
                "innovation",
                "planning",
                "market",
                "product",
                "development",
                "implementation",
                "solution",
            ]

            topics_found = set()
            all_content = " ".join(msg.content.lower() for msg in messages)

            for topic in common_topics:
                if topic in all_content:
                    topics_found.add(topic)

            return list(topics_found)[:10]  # Return top 10 topics

        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        try:
            key = f"conversation:{conversation_id}"
            return self.delete(key)

        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            return False

    def list_conversations(self) -> List[str]:
        """List all conversation IDs"""
        try:
            conversation_keys = [k for k in self.list_keys() if k.startswith("conversation:")]
            return [k.split(":", 1)[1] for k in conversation_keys]

        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return []

    # ============= END MESSAGE HANDLING METHODS =============

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(connected={self._connected}, keys={self.stats.total_keys})"
        )
