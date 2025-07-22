"""
Custom exception classes for CrewGraph AI
"""

from typing import Optional, Dict, Any, List


class CrewGraphError(Exception):
    """Base exception for all CrewGraph AI errors"""
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 suggestions: Optional[List[str]] = None):
        """
        Initialize CrewGraph error.
        
        Args:
            message: Error message
            error_code: Unique error code
            details: Additional error details
            suggestions: Suggestions for resolution
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "CREWGRAPH_ERROR"
        self.details = details or {}
        self.suggestions = suggestions or []
        
        # Add timestamp and user context
        import time
        import os
        self.timestamp = time.time()
        self.user = os.getenv('USER', 'unknown')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details,
            'suggestions': self.suggestions,
            'timestamp': self.timestamp,
            'user': self.user
        }
    
    def __str__(self) -> str:
        base_msg = f"[{self.error_code}] {self.message}"
        
        if self.details:
            base_msg += f" | Details: {self.details}"
        
        if self.suggestions:
            base_msg += f" | Suggestions: {', '.join(self.suggestions)}"
        
        return base_msg


class ValidationError(CrewGraphError):
    """Raised when validation fails"""
    
    def __init__(self, 
                 message: str,
                 field: Optional[str] = None,
                 value: Optional[Any] = None,
                 expected_type: Optional[str] = None,
                 **kwargs):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            expected_type: Expected type/format
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        if expected_type:
            details['expected_type'] = expected_type
        
        suggestions = kwargs.get('suggestions', [])
        if expected_type and not suggestions:
            suggestions.append(f"Ensure {field or 'value'} is of type {expected_type}")
        
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            details=details,
            suggestions=suggestions
        )


class ExecutionError(CrewGraphError):
    """Raised when task or workflow execution fails"""
    
    def __init__(self, 
                 message: str,
                 task_name: Optional[str] = None,
                 agent_name: Optional[str] = None,
                 execution_stage: Optional[str] = None,
                 original_error: Optional[Exception] = None,
                 **kwargs):
        """
        Initialize execution error.
        
        Args:
            message: Error message
            task_name: Name of failed task
            agent_name: Name of agent that failed
            execution_stage: Stage where failure occurred
            original_error: Original exception that caused the failure
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        
        if task_name:
            details['task_name'] = task_name
        if agent_name:
            details['agent_name'] = agent_name
        if execution_stage:
            details['execution_stage'] = execution_stage
        if original_error:
            details['original_error'] = str(original_error)
            details['original_error_type'] = type(original_error).__name__
        
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions.extend([
                "Check task configuration and dependencies",
                "Verify agent has required tools and permissions",
                "Review execution logs for more details"
            ])
        
        super().__init__(
            message,
            error_code="EXECUTION_ERROR", 
            details=details,
            suggestions=suggestions
        )


class ConfigurationError(CrewGraphError):
    """Raised when configuration is invalid"""
    
    def __init__(self, 
                 message: str,
                 config_section: Optional[str] = None,
                 config_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_section: Configuration section with error
            config_key: Configuration key with error
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        
        if config_section:
            details['config_section'] = config_section
        if config_key:
            details['config_key'] = config_key
        
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions.extend([
                "Check configuration file syntax",
                "Verify all required configuration keys are present",
                "Ensure configuration values are of correct type"
            ])
        
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            details=details,
            suggestions=suggestions
        )


class MemoryError(CrewGraphError):
    """Raised when memory operations fail"""
    
    def __init__(self, 
                 message: str,
                 memory_backend: Optional[str] = None,
                 operation: Optional[str] = None,
                 key: Optional[str] = None,
                 **kwargs):
        """
        Initialize memory error.
        
        Args:
            message: Error message
            memory_backend: Memory backend type
            operation: Memory operation that failed
            key: Memory key involved
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        
        if memory_backend:
            details['memory_backend'] = memory_backend
        if operation:
            details['operation'] = operation
        if key:
            details['key'] = key
        
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions.extend([
                "Check memory backend connection",
                "Verify memory backend configuration",
                "Ensure sufficient memory/storage space"
            ])
        
        super().__init__(
            message,
            error_code="MEMORY_ERROR",
            details=details,
            suggestions=suggestions
        )


class PlanningError(CrewGraphError):
    """Raised when workflow planning fails"""
    
    def __init__(self, 
                 message: str,
                 planning_stage: Optional[str] = None,
                 strategy: Optional[str] = None,
                 task_count: Optional[int] = None,
                 **kwargs):
        """
        Initialize planning error.
        
        Args:
            message: Error message
            planning_stage: Planning stage where error occurred
            strategy: Planning strategy being used
            task_count: Number of tasks being planned
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        
        if planning_stage:
            details['planning_stage'] = planning_stage
        if strategy:
            details['strategy'] = strategy
        if task_count:
            details['task_count'] = task_count
        
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions.extend([
                "Check task dependencies for circular references",
                "Verify resource constraints are reasonable",
                "Try a different planning strategy"
            ])
        
        super().__init__(
            message,
            error_code="PLANNING_ERROR",
            details=details,
            suggestions=suggestions
        )


class ToolError(CrewGraphError):
    """Raised when tool operations fail"""
    
    def __init__(self, 
                 message: str,
                 tool_name: Optional[str] = None,
                 tool_operation: Optional[str] = None,
                 **kwargs):
        """
        Initialize tool error.
        
        Args:
            message: Error message
            tool_name: Name of tool that failed
            tool_operation: Tool operation that failed
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        
        if tool_name:
            details['tool_name'] = tool_name
        if tool_operation:
            details['tool_operation'] = tool_operation
        
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions.extend([
                "Verify tool is properly registered",
                "Check tool parameters and permissions",
                "Ensure tool dependencies are available"
            ])
        
        super().__init__(
            message,
            error_code="TOOL_ERROR",
            details=details,
            suggestions=suggestions
        )


class AgentError(CrewGraphError):
    """Raised when agent operations fail"""
    
    def __init__(self, 
                 message: str,
                 agent_name: Optional[str] = None,
                 agent_operation: Optional[str] = None,
                 **kwargs):
        """
        Initialize agent error.
        
        Args:
            message: Error message
            agent_name: Name of agent that failed
            agent_operation: Agent operation that failed
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        
        if agent_name:
            details['agent_name'] = agent_name
        if agent_operation:
            details['agent_operation'] = agent_operation
        
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions.extend([
                "Verify agent configuration and permissions",
                "Check agent memory and state",
                "Ensure required tools are available to agent"
            ])
        
        super().__init__(
            message,
            error_code="AGENT_ERROR",
            details=details,
            suggestions=suggestions
        )


# Exception hierarchy for easy catching
class CrewGraphSystemError(CrewGraphError):
    """Base class for system-level errors"""
    pass


class CrewGraphUserError(CrewGraphError):
    """Base class for user-caused errors"""
    pass


# Categorize existing exceptions
ValidationError.__bases__ = (CrewGraphUserError,)
ConfigurationError.__bases__ = (CrewGraphUserError,)
ExecutionError.__bases__ = (CrewGraphSystemError,)
MemoryError.__bases__ = (CrewGraphSystemError,)
PlanningError.__bases__ = (CrewGraphSystemError,)
ToolError.__bases__ = (CrewGraphSystemError,)
AgentError.__bases__ = (CrewGraphSystemError,)


def handle_exception(func):
    """Decorator to handle and log exceptions consistently"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CrewGraphError:
            # Re-raise CrewGraph errors as-is
            raise
        except Exception as e:
            # Wrap other exceptions in CrewGraphError
            raise ExecutionError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                original_error=e,
                details={'function': func.__name__, 'module': func.__module__}
            )
    
    return wrapper