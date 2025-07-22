"""
CrewGraph AI State Management System
Advanced state management for workflows with persistence and synchronization

Author: Vatsal216
Created: 2025-07-22 12:23:59 UTC
"""

import time
import json
import pickle
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import uuid

from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector
from ..utils.exceptions import StateError

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
    NONE = "none"           # No persistence
    MEMORY = "memory"       # In-memory persistence
    FILE = "file"           # File-based persistence
    DATABASE = "database"   # Database persistence
    REDIS = "redis"         # Redis persistence


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
            'change_id': self.change_id,
            'change_type': self.change_type.value,
            'key': self.key,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'timestamp': self.timestamp,
            'user': self.user,
            'metadata': self.metadata
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
            'snapshot_id': self.snapshot_id,
            'state_data': self.state_data,
            'timestamp': self.timestamp,
            'user': self.user,
            'description': self.description,
            'metadata': self.metadata
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
    
    def __init__(self, 
                 workflow_id: str,
                 persistence_mode: StatePersistenceMode = StatePersistenceMode.MEMORY,
                 enable_history: bool = True,
                 max_history_size: int = 1000,
                 enable_validation: bool = True,
                 enable_snapshots: bool = True,
                 auto_snapshot_interval: Optional[int] = None):
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
        """
        self.workflow_id = workflow_id
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
                "user": "Vatsal216"
            }
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
                with open(self._persistence_file, 'r') as f:
                    data = json.load(f)
                    self._state = data.get('state', {})
                    self._metadata = data.get('metadata', {})
                    logger.info(f"State loaded from file: {self._persistence_file}")
        except Exception as e:
            logger.error(f"Failed to load state from file: {e}")
    
    def _save_to_file(self):
        """Save state to file"""
        if self.persistence_mode != StatePersistenceMode.FILE:
            return
        
        try:
            data = {
                'state': self._state,
                'metadata': self._metadata,
                'timestamp': time.time(),
                'user': 'Vatsal216'
            }
            with open(self._persistence_file, 'w') as f:
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
    
    def _record_change(self, change_type: StateChangeType, key: str, old_value: Any, new_value: Any):
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
            metadata={"workflow_id": self.workflow_id}
        )
        
        with self._lock:
            self.change_history.append(change)
            
            # Maintain history size limit
            if len(self.change_history) > self.max_history_size:
                self.change_history = self.change_history[-self.max_history_size:]
        
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
                "user": "Vatsal216"
            }
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
                change_type = StateChangeType.UPDATE if key in self._state else StateChangeType.CREATE
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
                labels={
                    "workflow_id": self.workflow_id,
                    "operation": "set",
                    "user": "Vatsal216"
                }
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
                    "error_type": type(e).__name__
                }
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
                labels={
                    "workflow_id": self.workflow_id,
                    "operation": "get",
                    "user": "Vatsal216"
                }
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
                    "user": "Vatsal216"
                }
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
                    "operation_count": self._operation_count
                }
            )
            
            self.snapshots[snapshot_id] = snapshot
            self.current_snapshot_id = snapshot_id
        
        metrics.increment_counter(
            "crewgraph_state_snapshots_created_total",
            labels={
                "workflow_id": self.workflow_id,
                "user": "Vatsal216"
            }
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
                    "auto_snapshot_enabled": self.auto_snapshot_interval is not None
                },
                "created_by": "Vatsal216",
                "created_at": "2025-07-22 12:23:59"
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
                        self.auto_snapshot_interval,
                        auto_snapshot
                    )
                    self._snapshot_timer.daemon = True
                    self._snapshot_timer.start()
        
        self._snapshot_timer = threading.Timer(
            self.auto_snapshot_interval,
            auto_snapshot
        )
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