"""
Audit Logger - Comprehensive audit logging system for CrewGraph AI

This module provides audit logging functionality including:
- Security event logging
- User action tracking
- System event monitoring
- Compliance reporting
- Log retention and archival

Features:
- Structured audit events
- Configurable log levels
- Multiple output formats
- Log rotation and retention
- Performance optimized
- Compliance ready

Created by: Vatsal216
Date: 2025-07-23
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Any, TextIO
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
from pathlib import Path
import threading
from collections import deque
import gzip

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError

logger = get_logger(__name__)


class AuditLevel(Enum):
    """Audit log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventCategory(Enum):
    """Audit event categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    WORKFLOW = "workflow"
    COMMUNICATION = "communication"
    SECURITY = "security"
    SYSTEM = "system"
    USER_ACTION = "user_action"
    ADMIN = "admin"


@dataclass
class AuditEvent:
    """Audit event structure"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str = ""
    category: EventCategory = EventCategory.SYSTEM
    level: AuditLevel = AuditLevel.INFO
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: str = ""
    action: str = ""
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['category'] = self.category.value
        result['level'] = self.level.value
        return result
    
    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Create event from dictionary"""
        event = cls()
        event.event_id = data.get("event_id", str(uuid.uuid4()))
        event.timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now(timezone.utc).isoformat()))
        event.event_type = data.get("event_type", "")
        event.category = EventCategory(data.get("category", "system"))
        event.level = AuditLevel(data.get("level", "info"))
        event.user_id = data.get("user_id")
        event.session_id = data.get("session_id")
        event.ip_address = data.get("ip_address")
        event.user_agent = data.get("user_agent")
        event.resource = data.get("resource", "")
        event.action = data.get("action", "")
        event.success = data.get("success", True)
        event.details = data.get("details", {})
        event.duration_ms = data.get("duration_ms")
        event.error_message = data.get("error_message")
        event.correlation_id = data.get("correlation_id")
        return event


@dataclass
class AuditConfig:
    """Audit logger configuration"""
    enabled: bool = True
    log_file: Optional[str] = None
    log_level: AuditLevel = AuditLevel.INFO
    max_memory_events: int = 10000
    buffer_size: int = 100
    flush_interval: int = 5  # seconds
    enable_file_rotation: bool = True
    max_file_size_mb: int = 100
    max_files: int = 10
    compress_old_files: bool = True
    retention_days: int = 365
    include_sensitive_data: bool = False
    custom_fields: List[str] = field(default_factory=list)


class AuditLogger:
    """
    Comprehensive audit logging system.
    
    Provides structured audit logging with configurable outputs,
    retention policies, and compliance features.
    """
    
    def __init__(self, config: Optional[AuditConfig] = None):
        """
        Initialize audit logger.
        
        Args:
            config: Audit configuration
        """
        self.config = config or AuditConfig()
        
        # Event storage
        self._memory_events: deque = deque(maxlen=self.config.max_memory_events)
        self._event_buffer: List[AuditEvent] = []
        
        # File handling
        self._log_file: Optional[TextIO] = None
        self._current_file_size = 0
        
        # Statistics
        self._event_count = 0
        self._events_by_category: Dict[EventCategory, int] = {}
        self._events_by_level: Dict[AuditLevel, int] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._flush_timer: Optional[threading.Timer] = None
        
        # Initialize
        self._initialize_logging()
        self._start_flush_timer()
        
        logger.info("AuditLogger initialized")
        logger.info(f"Config: file={self.config.log_file}, level={self.config.log_level.value}")
    
    def _initialize_logging(self):
        """Initialize file logging if configured"""
        if not self.config.log_file:
            return
        
        try:
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            self._log_file = open(log_path, 'a', encoding='utf-8')
            self._current_file_size = log_path.stat().st_size if log_path.exists() else 0
            
            logger.info(f"Audit log file initialized: {log_path}")
        except Exception as e:
            logger.error(f"Failed to initialize audit log file: {e}")
            self._log_file = None
    
    def _start_flush_timer(self):
        """Start periodic flush timer"""
        if self.config.flush_interval > 0:
            self._flush_timer = threading.Timer(self.config.flush_interval, self._periodic_flush)
            self._flush_timer.daemon = True
            self._flush_timer.start()
    
    def _periodic_flush(self):
        """Periodic flush of buffered events"""
        self.flush()
        self._start_flush_timer()  # Restart timer
    
    def log_event(self, event: AuditEvent):
        """
        Log an audit event.
        
        Args:
            event: Audit event to log
        """
        if not self.config.enabled:
            return
        
        # Filter by log level
        if event.level.value < self.config.log_level.value:
            return
        
        with self._lock:
            # Add to memory storage
            self._memory_events.append(event)
            
            # Update statistics
            self._event_count += 1
            self._events_by_category[event.category] = self._events_by_category.get(event.category, 0) + 1
            self._events_by_level[event.level] = self._events_by_level.get(event.level, 0) + 1
            
            # Add to buffer for file writing
            if self._log_file:
                self._event_buffer.append(event)
                
                # Flush if buffer is full
                if len(self._event_buffer) >= self.config.buffer_size:
                    self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush event buffer to file"""
        if not self._log_file or not self._event_buffer:
            return
        
        try:
            for event in self._event_buffer:
                line = event.to_json() + '\n'
                self._log_file.write(line)
                self._current_file_size += len(line.encode('utf-8'))
            
            self._log_file.flush()
            self._event_buffer.clear()
            
            # Check for file rotation
            if (self.config.enable_file_rotation and 
                self._current_file_size > self.config.max_file_size_mb * 1024 * 1024):
                self._rotate_log_file()
                
        except Exception as e:
            logger.error(f"Failed to flush audit events to file: {e}")
    
    def _rotate_log_file(self):
        """Rotate log file when it gets too large"""
        if not self._log_file or not self.config.log_file:
            return
        
        try:
            self._log_file.close()
            
            log_path = Path(self.config.log_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Rotate existing files
            for i in range(self.config.max_files - 1, 0, -1):
                old_file = log_path.with_suffix(f'.{i}{log_path.suffix}')
                new_file = log_path.with_suffix(f'.{i+1}{log_path.suffix}')
                if old_file.exists():
                    if i == self.config.max_files - 1:
                        # Delete oldest file
                        old_file.unlink()
                    else:
                        old_file.rename(new_file)
            
            # Move current file
            rotated_file = log_path.with_suffix(f'.1{log_path.suffix}')
            if log_path.exists():
                log_path.rename(rotated_file)
                
                # Compress if configured
                if self.config.compress_old_files:
                    self._compress_file(rotated_file)
            
            # Create new file
            self._log_file = open(log_path, 'w', encoding='utf-8')
            self._current_file_size = 0
            
            logger.info(f"Audit log file rotated: {log_path}")
            
        except Exception as e:
            logger.error(f"Failed to rotate audit log file: {e}")
            # Try to reopen original file
            try:
                self._log_file = open(self.config.log_file, 'a', encoding='utf-8')
            except:
                self._log_file = None
    
    def _compress_file(self, file_path: Path):
        """Compress rotated log file"""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            file_path.unlink()  # Remove original
            logger.info(f"Compressed audit log: {compressed_path}")
            
        except Exception as e:
            logger.error(f"Failed to compress audit log file: {e}")
    
    def flush(self):
        """Manually flush buffered events"""
        with self._lock:
            self._flush_buffer()
    
    def query_events(self,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    user_id: Optional[str] = None,
                    event_type: Optional[str] = None,
                    category: Optional[EventCategory] = None,
                    level: Optional[AuditLevel] = None,
                    success: Optional[bool] = None,
                    limit: int = 1000) -> List[AuditEvent]:
        """
        Query audit events with filters.
        
        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            user_id: Filter by user ID
            event_type: Filter by event type
            category: Filter by category
            level: Filter by level
            success: Filter by success status
            limit: Maximum number of events to return
            
        Returns:
            List of matching audit events
        """
        with self._lock:
            events = list(self._memory_events)
        
        # Apply filters
        filtered_events = []
        for event in events:
            # Time filters
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            
            # Attribute filters
            if user_id and event.user_id != user_id:
                continue
            if event_type and event.event_type != event_type:
                continue
            if category and event.category != category:
                continue
            if level and event.level != level:
                continue
            if success is not None and event.success != success:
                continue
            
            filtered_events.append(event)
            
            if len(filtered_events) >= limit:
                break
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return filtered_events
    
    def export_events(self,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     user_id: Optional[str] = None,
                     format: str = "json") -> List[Dict[str, Any]]:
        """
        Export audit events for compliance reporting.
        
        Args:
            start_time: Export events after this time
            end_time: Export events before this time
            user_id: Filter by user ID
            format: Export format (json, csv)
            
        Returns:
            List of event dictionaries
        """
        events = self.query_events(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            limit=100000  # Large limit for export
        )
        
        if format.lower() == "json":
            return [event.to_dict() for event in events]
        else:
            # For now, return dict format (CSV conversion can be added later)
            return [event.to_dict() for event in events]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics"""
        with self._lock:
            stats = {
                "total_events": self._event_count,
                "memory_events": len(self._memory_events),
                "buffered_events": len(self._event_buffer),
                "events_by_category": {cat.value: count for cat, count in self._events_by_category.items()},
                "events_by_level": {level.value: count for level, count in self._events_by_level.items()},
                "current_file_size_mb": round(self._current_file_size / (1024 * 1024), 2) if self._log_file else 0,
                "logging_enabled": self.config.enabled,
                "file_logging": self._log_file is not None
            }
        
        return stats
    
    def cleanup_old_events(self, retention_days: Optional[int] = None):
        """
        Clean up old events beyond retention period.
        
        Args:
            retention_days: Days to retain events (uses config if not specified)
        """
        if retention_days is None:
            retention_days = self.config.retention_days
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=retention_days)
        
        with self._lock:
            # Clean memory events
            original_count = len(self._memory_events)
            while self._memory_events and self._memory_events[0].timestamp < cutoff_time:
                self._memory_events.popleft()
            
            cleaned_count = original_count - len(self._memory_events)
            
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old audit events")
    
    def create_security_event(self,
                            event_type: str,
                            user_id: Optional[str] = None,
                            session_id: Optional[str] = None,
                            resource: str = "",
                            action: str = "",
                            success: bool = True,
                            details: Optional[Dict[str, Any]] = None,
                            level: AuditLevel = AuditLevel.INFO) -> AuditEvent:
        """
        Create and log a security event.
        
        Args:
            event_type: Type of security event
            user_id: User involved in event
            session_id: Session ID
            resource: Resource accessed
            action: Action performed
            success: Whether action succeeded
            details: Additional event details
            level: Event level
            
        Returns:
            Created audit event
        """
        event = AuditEvent(
            event_type=event_type,
            category=EventCategory.SECURITY,
            level=level,
            user_id=user_id,
            session_id=session_id,
            resource=resource,
            action=action,
            success=success,
            details=details or {}
        )
        
        self.log_event(event)
        return event
    
    def create_workflow_event(self,
                            event_type: str,
                            workflow_id: str,
                            user_id: Optional[str] = None,
                            action: str = "",
                            success: bool = True,
                            details: Optional[Dict[str, Any]] = None,
                            duration_ms: Optional[float] = None) -> AuditEvent:
        """
        Create and log a workflow event.
        
        Args:
            event_type: Type of workflow event
            workflow_id: Workflow identifier
            user_id: User who initiated the workflow
            action: Action performed
            success: Whether action succeeded
            details: Additional event details
            duration_ms: Execution duration in milliseconds
            
        Returns:
            Created audit event
        """
        event = AuditEvent(
            event_type=event_type,
            category=EventCategory.WORKFLOW,
            level=AuditLevel.INFO,
            user_id=user_id,
            resource=f"workflow:{workflow_id}",
            action=action,
            success=success,
            details=details or {},
            duration_ms=duration_ms
        )
        
        self.log_event(event)
        return event
    
    def close(self):
        """Close audit logger and flush remaining events"""
        if self._flush_timer:
            self._flush_timer.cancel()
        
        self.flush()
        
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        
        logger.info("AuditLogger closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()