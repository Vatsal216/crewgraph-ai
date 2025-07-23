"""
Enhanced Error Recovery System for CrewGraph AI
Provides intelligent error handling with context awareness, automatic retry logic, and error pattern recognition.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict
import json

from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ESCALATE = "escalate"
    RESTART = "restart"


@dataclass
class ErrorContext:
    """Error context information"""
    error_id: str
    timestamp: float
    node_id: str
    workflow_id: str
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    retry_count: int = 0
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Recovery action definition"""
    strategy: RecoveryStrategy
    max_retries: int = 3
    backoff_factor: float = 2.0
    initial_delay: float = 1.0
    timeout: Optional[float] = None
    fallback_function: Optional[Callable] = None
    conditions: List[Callable[[ErrorContext], bool]] = field(default_factory=list)


class ErrorPattern:
    """Error pattern for recognition and handling"""
    
    def __init__(
        self,
        name: str,
        error_types: List[str],
        message_patterns: List[str] = None,
        recovery_action: RecoveryAction = None,
        learning_enabled: bool = True
    ):
        self.name = name
        self.error_types = error_types
        self.message_patterns = message_patterns or []
        self.recovery_action = recovery_action or RecoveryAction(RecoveryStrategy.RETRY)
        self.learning_enabled = learning_enabled
        self.occurrence_count = 0
        self.success_rate = 0.0
        self.last_seen = None
    
    def matches(self, error_context: ErrorContext) -> bool:
        """Check if error matches this pattern"""
        # Check error type
        if error_context.error_type not in self.error_types:
            return False
        
        # Check message patterns if defined
        if self.message_patterns:
            message = error_context.error_message.lower()
            return any(pattern.lower() in message for pattern in self.message_patterns)
        
        return True
    
    def update_stats(self, success: bool):
        """Update pattern statistics"""
        if self.learning_enabled:
            self.occurrence_count += 1
            self.last_seen = time.time()
            
            # Update success rate with exponential moving average
            alpha = 0.1  # Learning rate
            if self.occurrence_count == 1:
                self.success_rate = 1.0 if success else 0.0
            else:
                new_rate = 1.0 if success else 0.0
                self.success_rate = alpha * new_rate + (1 - alpha) * self.success_rate


class AdvancedErrorRecovery:
    """
    Advanced error recovery system with intelligent handling, pattern recognition, and learning.
    
    Features:
    - Context-aware error handling
    - Automatic retry with exponential backoff
    - Error pattern recognition and learning
    - Custom recovery strategies
    - Performance metrics and analytics
    """
    
    def __init__(
        self,
        max_global_retries: int = 10,
        default_backoff_factor: float = 2.0,
        max_backoff_delay: float = 300.0,
        enable_learning: bool = True,
        error_history_size: int = 1000
    ):
        self.max_global_retries = max_global_retries
        self.default_backoff_factor = default_backoff_factor
        self.max_backoff_delay = max_backoff_delay
        self.enable_learning = enable_learning
        self.error_history_size = error_history_size
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.error_patterns: List[ErrorPattern] = []
        self.global_retry_count = 0
        self.recovery_stats = defaultdict(int)
        
        # Context tracking
        self.active_contexts: Dict[str, ErrorContext] = {}
        self.node_error_counts: Dict[str, int] = defaultdict(int)
        self.workflow_error_counts: Dict[str, int] = defaultdict(int)
        
        # Initialize default error patterns
        self._initialize_default_patterns()
        
        logger.info("AdvancedErrorRecovery system initialized")
    
    def _initialize_default_patterns(self):
        """Initialize default error patterns"""
        # Network/Connection errors
        self.add_pattern(ErrorPattern(
            name="network_errors",
            error_types=["ConnectionError", "TimeoutError", "NetworkError"],
            message_patterns=["connection", "timeout", "network", "unreachable"],
            recovery_action=RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_retries=5,
                backoff_factor=2.0,
                initial_delay=2.0
            )
        ))
        
        # Resource errors
        self.add_pattern(ErrorPattern(
            name="resource_errors",
            error_types=["MemoryError", "ResourceExhausted", "DiskError"],
            message_patterns=["memory", "resource", "disk", "space"],
            recovery_action=RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_retries=3,
                backoff_factor=3.0,
                initial_delay=5.0
            )
        ))
        
        # Rate limiting errors
        self.add_pattern(ErrorPattern(
            name="rate_limit_errors",
            error_types=["RateLimitError", "TooManyRequests"],
            message_patterns=["rate limit", "too many requests", "quota"],
            recovery_action=RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_retries=10,
                backoff_factor=2.5,
                initial_delay=10.0
            )
        ))
        
        # Critical system errors
        self.add_pattern(ErrorPattern(
            name="critical_errors",
            error_types=["SystemError", "RuntimeError", "InternalError"],
            message_patterns=["critical", "fatal", "system"],
            recovery_action=RecoveryAction(
                strategy=RecoveryStrategy.ESCALATE,
                max_retries=1
            )
        ))
    
    def add_pattern(self, pattern: ErrorPattern):
        """Add custom error pattern"""
        self.error_patterns.append(pattern)
        logger.info(f"Added error pattern: {pattern.name}")
    
    def handle_error(
        self,
        error: Exception,
        node_id: str,
        workflow_id: str,
        context: Dict[str, Any] = None,
        custom_recovery: Optional[RecoveryAction] = None
    ) -> Tuple[bool, Any]:
        """
        Handle error with intelligent recovery.
        
        Args:
            error: The exception that occurred
            node_id: ID of the node where error occurred
            workflow_id: ID of the workflow
            context: Additional context information
            custom_recovery: Custom recovery action
        
        Returns:
            Tuple of (should_retry, result_or_none)
        """
        # Create error context
        error_context = ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=time.time(),
            node_id=node_id,
            workflow_id=workflow_id,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=self._get_stack_trace(error),
            metadata=context or {}
        )
        
        # Add to history
        self._add_to_history(error_context)
        
        # Update tracking
        self.node_error_counts[node_id] += 1
        self.workflow_error_counts[workflow_id] += 1
        
        # Find matching pattern or use custom recovery
        recovery_action = custom_recovery or self._find_recovery_action(error_context)
        
        # Determine severity
        error_context.severity = self._determine_severity(error_context)
        
        # Record metrics
        self._record_error_metrics(error_context)
        
        # Execute recovery strategy
        success, result = self._execute_recovery(error_context, recovery_action)
        
        # Update pattern statistics if applicable
        pattern = self._find_matching_pattern(error_context)
        if pattern:
            pattern.update_stats(success)
        
        logger.info(
            f"Error recovery {'succeeded' if success else 'failed'} for {error_context.error_type} "
            f"in node {node_id} (strategy: {recovery_action.strategy.value})"
        )
        
        return success, result
    
    def _find_recovery_action(self, error_context: ErrorContext) -> RecoveryAction:
        """Find appropriate recovery action for error"""
        # Find matching pattern
        pattern = self._find_matching_pattern(error_context)
        if pattern:
            return pattern.recovery_action
        
        # Default recovery based on error type
        return self._get_default_recovery(error_context)
    
    def _find_matching_pattern(self, error_context: ErrorContext) -> Optional[ErrorPattern]:
        """Find pattern matching the error"""
        for pattern in self.error_patterns:
            if pattern.matches(error_context):
                return pattern
        return None
    
    def _get_default_recovery(self, error_context: ErrorContext) -> RecoveryAction:
        """Get default recovery action for error type"""
        error_type = error_context.error_type
        
        # Critical errors should escalate
        if any(critical in error_type.lower() for critical in ['critical', 'fatal', 'system']):
            return RecoveryAction(RecoveryStrategy.ESCALATE, max_retries=1)
        
        # Network/connection errors should retry with backoff
        if any(net in error_type.lower() for net in ['connection', 'network', 'timeout']):
            return RecoveryAction(RecoveryStrategy.RETRY, max_retries=5, backoff_factor=2.0)
        
        # Default: retry with standard backoff
        return RecoveryAction(RecoveryStrategy.RETRY, max_retries=3, backoff_factor=1.5)
    
    def _determine_severity(self, error_context: ErrorContext) -> ErrorSeverity:
        """Determine error severity based on context"""
        error_type = error_context.error_type.lower()
        error_message = error_context.error_message.lower()
        
        # Critical indicators
        if any(word in error_type + error_message for word in ['critical', 'fatal', 'system', 'memory']):
            return ErrorSeverity.CRITICAL
        
        # High severity indicators
        if any(word in error_type + error_message for word in ['security', 'auth', 'permission']):
            return ErrorSeverity.HIGH
        
        # Check frequency - frequent errors are more severe
        node_errors = self.node_error_counts[error_context.node_id]
        if node_errors > 10:
            return ErrorSeverity.HIGH
        elif node_errors > 5:
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def _execute_recovery(self, error_context: ErrorContext, recovery_action: RecoveryAction) -> Tuple[bool, Any]:
        """Execute recovery strategy"""
        strategy = recovery_action.strategy
        
        if strategy == RecoveryStrategy.RETRY:
            return self._execute_retry(error_context, recovery_action)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._execute_fallback(error_context, recovery_action)
        elif strategy == RecoveryStrategy.SKIP:
            return self._execute_skip(error_context)
        elif strategy == RecoveryStrategy.ESCALATE:
            return self._execute_escalate(error_context)
        elif strategy == RecoveryStrategy.RESTART:
            return self._execute_restart(error_context)
        
        return False, None
    
    def _execute_retry(self, error_context: ErrorContext, recovery_action: RecoveryAction) -> Tuple[bool, Any]:
        """Execute retry strategy with exponential backoff"""
        if error_context.retry_count >= recovery_action.max_retries:
            logger.warning(f"Max retries ({recovery_action.max_retries}) exceeded for {error_context.error_id}")
            return False, None
        
        # Calculate delay with exponential backoff
        delay = min(
            recovery_action.initial_delay * (recovery_action.backoff_factor ** error_context.retry_count),
            self.max_backoff_delay
        )
        
        logger.info(f"Retrying in {delay:.2f}s (attempt {error_context.retry_count + 1}/{recovery_action.max_retries})")
        
        # Update retry count
        error_context.retry_count += 1
        self.recovery_stats['retries'] += 1
        
        # For now, return success to indicate retry should happen
        # In a real implementation, this would coordinate with the execution engine
        return True, {"action": "retry", "delay": delay}
    
    def _execute_fallback(self, error_context: ErrorContext, recovery_action: RecoveryAction) -> Tuple[bool, Any]:
        """Execute fallback strategy"""
        if recovery_action.fallback_function:
            try:
                result = recovery_action.fallback_function(error_context)
                self.recovery_stats['fallbacks'] += 1
                return True, result
            except Exception as e:
                logger.error(f"Fallback function failed: {e}")
        
        return False, None
    
    def _execute_skip(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Execute skip strategy"""
        logger.info(f"Skipping failed node {error_context.node_id}")
        self.recovery_stats['skips'] += 1
        return True, {"action": "skip"}
    
    def _execute_escalate(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Execute escalate strategy"""
        logger.error(f"Escalating critical error {error_context.error_id}")
        self.recovery_stats['escalations'] += 1
        # In a real implementation, this would notify administrators
        return False, {"action": "escalate", "error_context": error_context}
    
    def _execute_restart(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Execute restart strategy"""
        logger.warning(f"Requesting workflow restart for {error_context.workflow_id}")
        self.recovery_stats['restarts'] += 1
        return True, {"action": "restart"}
    
    def _add_to_history(self, error_context: ErrorContext):
        """Add error to history with size limit"""
        self.error_history.append(error_context)
        
        # Maintain size limit
        if len(self.error_history) > self.error_history_size:
            self.error_history = self.error_history[-self.error_history_size:]
    
    def _get_stack_trace(self, error: Exception) -> Optional[str]:
        """Get stack trace from exception"""
        import traceback
        try:
            return traceback.format_exc()
        except:
            return None
    
    def _record_error_metrics(self, error_context: ErrorContext):
        """Record error metrics"""
        metrics.increment_counter(
            "crewgraph_errors_total",
            labels={
                "error_type": error_context.error_type,
                "severity": error_context.severity.value,
                "node_id": error_context.node_id,
                "workflow_id": error_context.workflow_id
            }
        )
    
    def get_error_analytics(self) -> Dict[str, Any]:
        """Get comprehensive error analytics"""
        if not self.error_history:
            return {"message": "No errors recorded"}
        
        total_errors = len(self.error_history)
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
        
        # Error type distribution
        error_types = defaultdict(int)
        severity_distribution = defaultdict(int)
        
        for error in self.error_history:
            error_types[error.error_type] += 1
            severity_distribution[error.severity.value] += 1
        
        # Pattern performance
        pattern_stats = []
        for pattern in self.error_patterns:
            if pattern.occurrence_count > 0:
                pattern_stats.append({
                    "name": pattern.name,
                    "occurrences": pattern.occurrence_count,
                    "success_rate": pattern.success_rate,
                    "last_seen": pattern.last_seen
                })
        
        return {
            "total_errors": total_errors,
            "recent_errors": len(recent_errors),
            "error_types": dict(error_types),
            "severity_distribution": dict(severity_distribution),
            "recovery_stats": dict(self.recovery_stats),
            "pattern_performance": pattern_stats,
            "top_error_nodes": dict(sorted(self.node_error_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "workflow_error_counts": dict(self.workflow_error_counts)
        }
    
    def export_error_report(self, format: str = "json") -> str:
        """Export comprehensive error report"""
        analytics = self.get_error_analytics()
        
        if format.lower() == "json":
            return json.dumps(analytics, indent=2, default=str)
        
        # Text format
        lines = []
        lines.append("=== CrewGraph AI Error Recovery Report ===")
        lines.append(f"Total Errors: {analytics['total_errors']}")
        lines.append(f"Recent Errors (1h): {analytics['recent_errors']}")
        lines.append("")
        
        lines.append("Error Types:")
        for error_type, count in analytics['error_types'].items():
            lines.append(f"  {error_type}: {count}")
        lines.append("")
        
        lines.append("Recovery Statistics:")
        for action, count in analytics['recovery_stats'].items():
            lines.append(f"  {action}: {count}")
        
        return "\n".join(lines)


# Global error recovery instance
_global_error_recovery: Optional[AdvancedErrorRecovery] = None


def get_error_recovery() -> AdvancedErrorRecovery:
    """Get global error recovery instance"""
    global _global_error_recovery
    if _global_error_recovery is None:
        _global_error_recovery = AdvancedErrorRecovery()
    return _global_error_recovery


def handle_workflow_error(
    error: Exception,
    node_id: str,
    workflow_id: str,
    context: Dict[str, Any] = None
) -> Tuple[bool, Any]:
    """Convenience function for handling workflow errors"""
    recovery_system = get_error_recovery()
    return recovery_system.handle_error(error, node_id, workflow_id, context)