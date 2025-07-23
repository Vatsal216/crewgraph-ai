"""
CrewGraph AI State Management System
Advanced state management for workflows with persistence and synchronization

Author: Vatsal216
Created: 2025-07-22 12:23:59 UTC
"""

import json
import pickle
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ..utils.exceptions import StateError
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class StateChangeType(Enum):
    """Types of state changes"""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESET = "reset"
    TRANSITION = "transition"


class StatePersistenceMode(Enum):
    """State persistence modes"""

    NONE = "none"  # No persistence
    MEMORY = "memory"  # In-memory persistence
    FILE = "file"  # File-based persistence
    DATABASE = "database"  # Database persistence
    REDIS = "redis"  # Redis persistence


@dataclass
class StateChange:
    """Represents a state change event"""

    change_id: str
    change_type: StateChangeType
    key: str
    old_value: Any
    new_value: Any
    timestamp: float
    user: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "change_id": self.change_id,
            "change_type": self.change_type.value,
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp,
            "user": self.user,
            "metadata": self.metadata,
        }


@dataclass
class StateSnapshot:
    """Represents a complete state snapshot"""

    snapshot_id: str
    state_data: Dict[str, Any]
    timestamp: float
    user: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "snapshot_id": self.snapshot_id,
            "state_data": self.state_data,
            "timestamp": self.timestamp,
            "user": self.user,
            "description": self.description,
            "metadata": self.metadata,
        }


class StateValidator:
    """Validates state changes and transitions"""

    def __init__(self):
        """Initialize state validator"""
        self.validation_rules: Dict[str, Callable[[Any], bool]] = {}
        self.transition_rules: Dict[str, Dict[str, List[str]]] = {}

        logger.info("StateValidator initialized by Vatsal216 at 2025-07-22 12:23:59")

    def add_validation_rule(self, key: str, validator: Callable[[Any], bool]) -> None:
        """Add validation rule for state key"""
        self.validation_rules[key] = validator
        logger.debug(f"Added validation rule for key: {key}")

    def add_transition_rule(self, entity: str, from_state: str, allowed_states: List[str]) -> None:
        """Add state transition rule"""
        if entity not in self.transition_rules:
            self.transition_rules[entity] = {}
        self.transition_rules[entity][from_state] = allowed_states
        logger.debug(f"Added transition rule for {entity}: {from_state} -> {allowed_states}")

    def validate_value(self, key: str, value: Any) -> bool:
        """Validate a value against its validation rule"""
        if key not in self.validation_rules:
            return True  # No rule = valid

        try:
            return self.validation_rules[key](value)
        except Exception as e:
            logger.error(f"Validation error for key '{key}': {e}")
            return False

    def validate_transition(self, entity: str, from_state: str, to_state: str) -> bool:
        """Validate state transition"""
        if entity not in self.transition_rules:
            return True  # No rules = allowed

        if from_state not in self.transition_rules[entity]:
            return True  # No rules for this from_state = allowed

        allowed_states = self.transition_rules[entity][from_state]
        return to_state in allowed_states


class StateManager:
    """
    Advanced state management system for CrewGraph AI workflows.

    Provides comprehensive state management with features like:
    - Thread-safe state operations
    - State change tracking and history
    - State persistence and recovery
    - State validation and transition rules
    - Event-driven state notifications
    - State snapshots and rollback
    - Performance monitoring and metrics

    Created by: Vatsal216
    Date: 2025-07-22 12:23:59 UTC
    """

    def __init__(
        self,
        workflow_id: str = None,
        persistence_mode: StatePersistenceMode = StatePersistenceMode.MEMORY,
        enable_history: bool = True,
        max_history_size: int = 1000,
        enable_validation: bool = True,
        enable_snapshots: bool = True,
        auto_snapshot_interval: Optional[int] = None,
        memory_backend=None,  # For backward compatibility
    ):
        """
        Initialize state manager.

        Args:
            workflow_id: Unique identifier for the workflow
            persistence_mode: How to persist state
            enable_history: Enable state change history
            max_history_size: Maximum history entries
            enable_validation: Enable state validation
            enable_snapshots: Enable state snapshots
            auto_snapshot_interval: Automatic snapshot interval in seconds
            memory_backend: Memory backend (for backward compatibility)
        """
        self.workflow_id = workflow_id or "default_workflow"
        self.memory_backend = memory_backend  # For backward compatibility
        self.persistence_mode = persistence_mode
        self.enable_history = enable_history
        self.max_history_size = max_history_size
        self.enable_validation = enable_validation
        self.enable_snapshots = enable_snapshots
        self.auto_snapshot_interval = auto_snapshot_interval

        # Core state storage
        self._state: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # State change tracking
        self.change_history: List[StateChange] = []
        self.change_listeners: List[Callable[[StateChange], None]] = []

        # State snapshots
        self.snapshots: Dict[str, StateSnapshot] = {}
        self.current_snapshot_id: Optional[str] = None

        # Validation
        self.validator = StateValidator() if enable_validation else None

        # Auto-snapshot timer
        self._snapshot_timer: Optional[threading.Timer] = None

        # Performance tracking
        self._operation_count = 0
        self._last_operation_time = time.time()

        # Initialize persistence
        self._initialize_persistence()

        # Start auto-snapshot if enabled
        if self.auto_snapshot_interval:
            self._start_auto_snapshot()

        # Record initialization metrics
        metrics.increment_counter(
            "crewgraph_state_managers_created_total",
            labels={
                "workflow_id": workflow_id,
                "persistence_mode": persistence_mode.value,
                "user": "Vatsal216",
            },
        )

        logger.info(f"StateManager initialized for workflow: {workflow_id}")
        logger.info(f"Persistence mode: {persistence_mode.value}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:23:59")

    def _initialize_persistence(self):
        """Initialize persistence backend"""
        if self.persistence_mode == StatePersistenceMode.FILE:
            self._persistence_file = f"state_{self.workflow_id}.json"
            self._load_from_file()
        elif self.persistence_mode == StatePersistenceMode.DATABASE:
            self._initialize_database_persistence()
        elif self.persistence_mode == StatePersistenceMode.REDIS:
            self._initialize_redis_persistence()

    def _load_from_file(self):
        """Load state from file"""
        try:
            import os

            if os.path.exists(self._persistence_file):
                with open(self._persistence_file, "r") as f:
                    data = json.load(f)
                    self._state = data.get("state", {})
                    self._metadata = data.get("metadata", {})
                    logger.info(f"State loaded from file: {self._persistence_file}")
        except Exception as e:
            logger.error(f"Failed to load state from file: {e}")

    def _save_to_file(self):
        """Save state to file"""
        if self.persistence_mode != StatePersistenceMode.FILE:
            return

        try:
            data = {
                "state": self._state,
                "metadata": self._metadata,
                "timestamp": time.time(),
                "user": "Vatsal216",
            }
            with open(self._persistence_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"State saved to file: {self._persistence_file}")
        except Exception as e:
            logger.error(f"Failed to save state to file: {e}")

    def _initialize_database_persistence(self):
        """Initialize database persistence (placeholder)"""
        logger.info("Database persistence initialized (implementation pending)")

    def _initialize_redis_persistence(self):
        """Initialize Redis persistence (placeholder)"""
        logger.info("Redis persistence initialized (implementation pending)")

    def _record_change(
        self, change_type: StateChangeType, key: str, old_value: Any, new_value: Any
    ):
        """Record state change"""
        if not self.enable_history:
            return

        change = StateChange(
            change_id=str(uuid.uuid4()),
            change_type=change_type,
            key=key,
            old_value=old_value,
            new_value=new_value,
            timestamp=time.time(),
            user="Vatsal216",
            metadata={"workflow_id": self.workflow_id},
        )

        with self._lock:
            self.change_history.append(change)

            # Maintain history size limit
            if len(self.change_history) > self.max_history_size:
                self.change_history = self.change_history[-self.max_history_size :]

        # Notify listeners
        for listener in self.change_listeners:
            try:
                listener(change)
            except Exception as e:
                logger.error(f"Error in change listener: {e}")

        # Record metrics
        metrics.increment_counter(
            "crewgraph_state_changes_total",
            labels={
                "workflow_id": self.workflow_id,
                "change_type": change_type.value,
                "user": "Vatsal216",
            },
        )

    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Set state value with validation and change tracking.

        Args:
            key: State key
            value: State value
            metadata: Optional metadata for the state entry

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()

        try:
            # Validate value if validation is enabled
            if self.validator and not self.validator.validate_value(key, value):
                logger.error(f"Validation failed for key '{key}'")
                return False

            with self._lock:
                old_value = self._state.get(key)

                # Set the value
                self._state[key] = value

                # Set metadata
                if metadata:
                    self._metadata[key] = metadata

                # Record change
                change_type = (
                    StateChangeType.UPDATE if key in self._state else StateChangeType.CREATE
                )
                self._record_change(change_type, key, old_value, value)

                # Update operation tracking
                self._operation_count += 1
                self._last_operation_time = time.time()

            # Persist if needed
            self._save_to_file()

            # Record performance metrics
            operation_time = time.time() - start_time
            metrics.record_duration(
                "crewgraph_state_operation_duration_seconds",
                operation_time,
                labels={"workflow_id": self.workflow_id, "operation": "set", "user": "Vatsal216"},
            )

            logger.debug(f"State set: {key} = {type(value).__name__}")
            return True

        except Exception as e:
            logger.error(f"Failed to set state '{key}': {e}")
            metrics.increment_counter(
                "crewgraph_state_operation_errors_total",
                labels={
                    "workflow_id": self.workflow_id,
                    "operation": "set",
                    "error_type": type(e).__name__,
                },
            )
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get state value.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            State value or default
        """
        start_time = time.time()

        try:
            with self._lock:
                value = self._state.get(key, default)
                self._operation_count += 1
                self._last_operation_time = time.time()

            # Record performance metrics
            operation_time = time.time() - start_time
            metrics.record_duration(
                "crewgraph_state_operation_duration_seconds",
                operation_time,
                labels={"workflow_id": self.workflow_id, "operation": "get", "user": "Vatsal216"},
            )

            logger.debug(f"State get: {key} = {type(value).__name__}")
            return value

        except Exception as e:
            logger.error(f"Failed to get state '{key}': {e}")
            return default

    def delete(self, key: str) -> bool:
        """
        Delete state value.

        Args:
            key: State key to delete

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()

        try:
            with self._lock:
                if key not in self._state:
                    return False

                old_value = self._state[key]
                del self._state[key]

                # Remove metadata if exists
                if key in self._metadata:
                    del self._metadata[key]

                # Record change
                self._record_change(StateChangeType.DELETE, key, old_value, None)

                # Update operation tracking
                self._operation_count += 1
                self._last_operation_time = time.time()

            # Persist if needed
            self._save_to_file()

            # Record performance metrics
            operation_time = time.time() - start_time
            metrics.record_duration(
                "crewgraph_state_operation_duration_seconds",
                operation_time,
                labels={
                    "workflow_id": self.workflow_id,
                    "operation": "delete",
                    "user": "Vatsal216",
                },
            )

            logger.debug(f"State deleted: {key}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete state '{key}': {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if state key exists"""
        with self._lock:
            return key in self._state

    def keys(self) -> List[str]:
        """Get all state keys"""
        with self._lock:
            return list(self._state.keys())

    def items(self) -> Dict[str, Any]:
        """Get all state items"""
        with self._lock:
            return self._state.copy()

    def clear(self) -> bool:
        """Clear all state data"""
        try:
            with self._lock:
                old_state = self._state.copy()
                self._state.clear()
                self._metadata.clear()

                # Record change
                self._record_change(StateChangeType.RESET, "*", old_state, {})

                # Update operation tracking
                self._operation_count += 1
                self._last_operation_time = time.time()

            # Persist if needed
            self._save_to_file()

            logger.info("State cleared")
            return True

        except Exception as e:
            logger.error(f"Failed to clear state: {e}")
            return False

    def bulk_set(self, data: Dict[str, Any]) -> bool:
        """
        Set multiple state values at once.
        
        Args:
            data: Dictionary of key-value pairs to set
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._lock:
                for key, value in data.items():
                    self.set(key, value)
            logger.info(f"Bulk set {len(data)} state values")
            return True
        except Exception as e:
            logger.error(f"Failed to bulk set state: {e}")
            return False

    def export_state(self) -> Dict[str, Any]:
        """
        Export all state data as a dictionary.
        
        Returns:
            Dict[str, Any]: Complete state dictionary
        """
        with self._lock:
            return self._state.copy()

    def list_keys(self) -> List[str]:
        """
        List all state keys (alias for keys method).
        
        Returns:
            List[str]: List of all state keys
        """
        return self.keys()

    def import_state(self, data: Dict[str, Any]) -> bool:
        """
        Import state data from dictionary.
        
        Args:
            data: Dictionary of state data to import
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._lock:
                for key, value in data.items():
                    self.set(key, value)
            logger.info(f"Imported {len(data)} state values")
            return True
        except Exception as e:
            logger.error(f"Failed to import state: {e}")
            return False

    def create_snapshot(self, description: str = "") -> str:
        """
        Create state snapshot.

        Args:
            description: Optional description for the snapshot

        Returns:
            Snapshot ID
        """
        if not self.enable_snapshots:
            raise StateError("Snapshots are disabled")

        snapshot_id = f"snapshot_{int(time.time())}_{str(uuid.uuid4())[:8]}"

        with self._lock:
            snapshot = StateSnapshot(
                snapshot_id=snapshot_id,
                state_data=self._state.copy(),
                timestamp=time.time(),
                user="Vatsal216",
                description=description,
                metadata={
                    "workflow_id": self.workflow_id,
                    "operation_count": self._operation_count,
                },
            )

            self.snapshots[snapshot_id] = snapshot
            self.current_snapshot_id = snapshot_id

        metrics.increment_counter(
            "crewgraph_state_snapshots_created_total",
            labels={"workflow_id": self.workflow_id, "user": "Vatsal216"},
        )

        logger.info(f"State snapshot created: {snapshot_id}")
        return snapshot_id

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore state from snapshot.

        Args:
            snapshot_id: Snapshot ID to restore from

        Returns:
            True if successful, False otherwise
        """
        if not self.enable_snapshots:
            raise StateError("Snapshots are disabled")

        if snapshot_id not in self.snapshots:
            logger.error(f"Snapshot not found: {snapshot_id}")
            return False

        try:
            snapshot = self.snapshots[snapshot_id]

            with self._lock:
                old_state = self._state.copy()
                self._state = snapshot.state_data.copy()

                # Record change
                self._record_change(StateChangeType.RESET, "*", old_state, self._state)

                # Update operation tracking
                self._operation_count += 1
                self._last_operation_time = time.time()
                self.current_snapshot_id = snapshot_id

            # Persist if needed
            self._save_to_file()

            logger.info(f"State restored from snapshot: {snapshot_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore snapshot '{snapshot_id}': {e}")
            return False

    def get_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Get snapshot by ID"""
        return self.snapshots.get(snapshot_id)

    def list_snapshots(self) -> List[StateSnapshot]:
        """List all snapshots"""
        return list(self.snapshots.values())

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete snapshot by ID"""
        if snapshot_id in self.snapshots:
            del self.snapshots[snapshot_id]
            logger.info(f"Snapshot deleted: {snapshot_id}")
            return True
        return False

    def add_change_listener(self, listener: Callable[[StateChange], None]) -> None:
        """Add state change listener"""
        self.change_listeners.append(listener)
        logger.debug("State change listener added")

    def remove_change_listener(self, listener: Callable[[StateChange], None]) -> None:
        """Remove state change listener"""
        if listener in self.change_listeners:
            self.change_listeners.remove(listener)
            logger.debug("State change listener removed")

    def get_change_history(self, limit: Optional[int] = None) -> List[StateChange]:
        """Get state change history"""
        with self._lock:
            if limit:
                return self.change_history[-limit:]
            return self.change_history.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get state manager statistics"""
        with self._lock:
            return {
                "workflow_id": self.workflow_id,
                "state_keys_count": len(self._state),
                "operation_count": self._operation_count,
                "change_history_size": len(self.change_history),
                "snapshots_count": len(self.snapshots),
                "current_snapshot_id": self.current_snapshot_id,
                "last_operation_time": self._last_operation_time,
                "persistence_mode": self.persistence_mode.value,
                "features": {
                    "history_enabled": self.enable_history,
                    "validation_enabled": self.enable_validation,
                    "snapshots_enabled": self.enable_snapshots,
                    "auto_snapshot_enabled": self.auto_snapshot_interval is not None,
                },
                "created_by": "Vatsal216",
                "created_at": "2025-07-22 12:23:59",
            }

    def _start_auto_snapshot(self):
        """Start automatic snapshot timer"""

        def auto_snapshot():
            try:
                self.create_snapshot("Automatic snapshot")
                logger.debug("Automatic snapshot created")
            except Exception as e:
                logger.error(f"Auto-snapshot failed: {e}")
            finally:
                # Schedule next snapshot
                if self.auto_snapshot_interval:
                    self._snapshot_timer = threading.Timer(
                        self.auto_snapshot_interval, auto_snapshot
                    )
                    self._snapshot_timer.daemon = True
                    self._snapshot_timer.start()

        self._snapshot_timer = threading.Timer(self.auto_snapshot_interval, auto_snapshot)
        self._snapshot_timer.daemon = True
        self._snapshot_timer.start()

        logger.info(f"Auto-snapshot started (interval: {self.auto_snapshot_interval}s)")

    def stop_auto_snapshot(self):
        """Stop automatic snapshot timer"""
        if self._snapshot_timer:
            self._snapshot_timer.cancel()
            self._snapshot_timer = None
            logger.info("Auto-snapshot stopped")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_auto_snapshot()
        self._save_to_file()

    def clear_all(self) -> bool:
        """Clear all state data (alias for clear)"""
        return self.clear()

    # Additional methods expected by tests
    def get_state(self, context: str = "default") -> Dict[str, Any]:
        """Get state for a specific context"""
        if context == "default":
            return self._state.copy()
        else:
            return self._state.get(f"context:{context}", {})

    def switch_context(self, context: str) -> bool:
        """Switch to a different state context"""
        # Store current context if needed
        if not hasattr(self, '_current_context'):
            self._current_context = "default"
        
        # Save current context state
        current_state = self._state.copy()
        self.set(f"context:{self._current_context}", current_state)
        
        # Load new context state
        new_state = self.get(f"context:{context}", {})
        self._state.clear()
        self._state.update(new_state)
        
        self._current_context = context
        return True

    def set_shared(self, key: str, value: Any) -> bool:
        """Set a value that is shared across all contexts"""
        return self.set(f"shared:{key}", value)

    def __repr__(self) -> str:
        return f"StateManager(workflow_id='{self.workflow_id}', keys={len(self._state)}, operations={self._operation_count})"


def create_state_manager(workflow_id: str, **kwargs) -> StateManager:
    """
    Factory function to create state manager.

    Args:
        workflow_id: Unique workflow identifier
        **kwargs: Additional configuration options

    Returns:
        Configured StateManager instance
    """
    logger.info(f"Creating state manager for workflow: {workflow_id}")
    logger.info(f"User: Vatsal216, Time: 2025-07-22 12:23:59")

    return StateManager(workflow_id, **kwargs)


class SharedState:
    """
    Simplified shared state interface for CrewGraph AI.

    This is a convenience wrapper around StateManager that provides
    a simpler API for common state operations in workflows.

    Created by: Vatsal216
    Date: 2025-07-22 12:23:59 UTC
    """

    def __init__(self, memory=None, workflow_id: str = "default_workflow", memory_backend=None, **kwargs):
        """
        Initialize shared state.

        Args:
            memory: Memory backend (optional)
            workflow_id: Workflow identifier
            memory_backend: Alternative name for memory (for backward compatibility)
            **kwargs: Additional StateManager options
        """
        # Support both 'memory' and 'memory_backend' parameters for backward compatibility
        self.memory = memory or memory_backend
        self.memory_backend = self.memory  # For backward compatibility
        self.workflow_id = workflow_id

        # Create underlying state manager
        self._state_manager = StateManager(workflow_id=workflow_id, memory_backend=self.memory, **kwargs)

        # Add metrics for compatibility
        from ..utils.metrics import get_metrics_collector
        self.metrics = get_metrics_collector()

        logger.info(f"SharedState initialized for workflow: {workflow_id}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from shared state"""
        return self._state_manager.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        """Set value in shared state"""
        return self._state_manager.set(key, value)

    def update(self, key_or_data, value=None) -> bool:
        """Update single value or multiple values in shared state"""
        if isinstance(key_or_data, dict) and value is None:
            # Multiple values update: update({"key1": "value1", "key2": "value2"})
            return self._state_manager.bulk_set(key_or_data)
        elif isinstance(key_or_data, str) and value is not None:
            # Single value update: update("key", "value")
            return self.set(key_or_data, value)
        else:
            raise ValueError("Invalid arguments: use update(key, value) or update({key: value, ...})")

    def delete(self, key: str) -> bool:
        """Delete key from shared state"""
        return self._state_manager.delete(key)

    def exists(self, key: str) -> bool:
        """Check if key exists in shared state"""
        return self._state_manager.exists(key)

    def keys(self) -> List[str]:
        """Get all keys in shared state"""
        return self._state_manager.list_keys()

    def clear(self) -> bool:
        """Clear all state"""
        return self._state_manager.clear()

    def reset(self) -> bool:
        """Reset state (alias for clear)"""
        return self.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export state as dictionary"""
        return self._state_manager.export_state()

    def from_dict(self, data: Dict[str, Any]) -> bool:
        """Import state from dictionary"""
        return self._state_manager.import_state(data)

    def get_context(self, key: str = None) -> Dict[str, Any]:
        """
        Get context information for state access.
        
        Args:
            key: Optional key to get context for specific item
            
        Returns:
            Dict[str, Any]: Context information
        """
        if key:
            return {
                "key": key,
                "value": self.get(key),
                "exists": self.exists(key),
                "workflow": self.workflow_id
            }
        else:
            return {
                "workflow": self.workflow_id,
                "keys": self.keys(),
                "size": len(self.keys()),
                "memory_backend": str(type(self.memory).__name__) if self.memory else "None"
            }

    def save(self, filename: str) -> bool:
        """Save state to file"""
        try:
            data = self.to_dict()
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def load(self, filename: str) -> bool:
        """Load state from file"""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    def create_snapshot(self, description: str = "") -> str:
        """Create state snapshot"""
        return self._state_manager.create_snapshot(description)

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore from snapshot"""
        return self._state_manager.restore_snapshot(snapshot_id)

    def get_state_manager(self) -> StateManager:
        """Get underlying state manager for advanced operations"""
        return self._state_manager

    def set_agent_state(self, agent_id: str, state_data: Dict[str, Any]) -> bool:
        """Set agent-specific state data"""
        return self.set(f"agent:{agent_id}", state_data)

    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get agent-specific state data"""
        return self.get(f"agent:{agent_id}", {})

    def set_task_state(self, task_id: str, state_data: Dict[str, Any]) -> bool:
        """Set task-specific state data"""
        return self.set(f"task:{task_id}", state_data)

    def get_task_state(self, task_id: str) -> Dict[str, Any]:
        """Get task-specific state data"""
        return self.get(f"task:{task_id}", {})

    def set_workflow_state(self, workflow_id: str, state_data: Dict[str, Any]) -> bool:
        """Set workflow-specific state data"""
        return self.set(f"workflow:{workflow_id}", state_data)

    def get_workflow_state(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow-specific state data"""
        return self.get(f"workflow:{workflow_id}", {})

    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access for getting values"""
        value = self.get(key)
        if value is None and not self.exists(key):
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-like access for setting values"""
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        """Dictionary-like access for deleting values"""
        if not self.delete(key):
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Dictionary-like access for checking existence"""
        return self.exists(key)

    def __len__(self) -> int:
        """Get number of keys"""
        return len(self.keys())

    def __repr__(self) -> str:
        return f"SharedState(workflow_id='{self.workflow_id}', keys={len(self)})"

    # Additional methods expected by tests
    def update_agent_status(self, agent_id: str, status: str) -> bool:
        """Update agent status"""
        agent_state = self.get_agent_state(agent_id)
        agent_state['status'] = status
        agent_state['last_updated'] = time.time()
        return self.set_agent_state(agent_id, agent_state)

    def update_task_progress(self, task_id: str, progress: float, status: str = None) -> bool:
        """Update task progress"""
        task_state = self.get_task_state(task_id)
        task_state['progress'] = progress
        task_state['last_updated'] = time.time()
        if status:
            task_state['status'] = status
        return self.set_task_state(task_id, task_state)

    def update_workflow_step(self, workflow_id: str, step: int) -> bool:
        """Update workflow step"""
        workflow_state = self.get_workflow_state(workflow_id)
        workflow_state['current_step'] = step
        workflow_state['last_updated'] = time.time()
        return self.set_workflow_state(workflow_id, workflow_state)

    def get_nested(self, path: str, default: Any = None) -> Any:
        """Get nested value using dot notation (e.g., 'user.profile.name')"""
        keys = path.split('.')
        value = self._state_manager._state
        
        try:
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default

    def set_nested(self, path: str, value: Any) -> bool:
        """Set nested value using dot notation (e.g., 'user.profile.name')"""
        keys = path.split('.')
        current = self._state_manager._state
        
        # Navigate to the parent of the final key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
        return True

    def enable_history(self, enabled: bool = True) -> bool:
        """Enable state change history tracking"""
        self._state_manager.enable_history = enabled
        return True

    def serialize(self) -> str:
        """Serialize state to JSON string"""
        try:
            return json.dumps(self.to_dict(), default=str)
        except Exception as e:
            logger.error(f"Failed to serialize state: {e}")
            return "{}"

    def deserialize(self, data: str) -> bool:
        """Deserialize state from JSON string"""
        try:
            state_dict = json.loads(data)
            return self.from_dict(state_dict)
        except Exception as e:
            logger.error(f"Failed to deserialize state: {e}")
            return False

    def subscribe(self, event_type: str, callback: Callable[[str, Any, Any], None]) -> str:
        """Subscribe to state changes"""
        # Generate a subscription ID
        subscription_id = str(uuid.uuid4())
        
        # Add to state manager's subscribers if it has them
        if not hasattr(self._state_manager, '_subscribers'):
            self._state_manager._subscribers = {}
        
        self._state_manager._subscribers[subscription_id] = {
            'event_type': event_type,
            'callback': callback
        }
        return subscription_id

    def get_history(self, key: str = None) -> List[Dict[str, Any]]:
        """Get state change history for a specific key or all changes"""
        if not hasattr(self._state_manager, 'change_history'):
            return []
        
        history = self._state_manager.change_history
        if key is None:
            return [change.to_dict() for change in history]
        else:
            return [change.to_dict() for change in history if change.key == key]
