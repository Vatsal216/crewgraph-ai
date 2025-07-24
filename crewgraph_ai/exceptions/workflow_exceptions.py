"""Comprehensive workflow exception handling."""

from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)

class CrewGraphError(Exception):
    """Base exception for all CrewGraph AI errors."""
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        logger.error(f"CrewGraphError [{error_code}]: {message}", extra=self.context)

class WorkflowExecutionError(CrewGraphError):
    """Errors during workflow execution."""
    def __init__(self, message: str, workflow_id: Optional[str] = None, 
                 task_id: Optional[str] = None, **kwargs):
        context = {'workflow_id': workflow_id, 'task_id': task_id}
        super().__init__(message, error_code='WORKFLOW_EXECUTION', context=context)

class AgentError(CrewGraphError):
    """Errors related to agent operations."""
    def __init__(self, message: str, agent_id: Optional[str] = None, 
                 agent_type: Optional[str] = None, **kwargs):
        context = {'agent_id': agent_id, 'agent_type': agent_type}
        super().__init__(message, error_code='AGENT_ERROR', context=context)

class MemoryBackendError(CrewGraphError):
    """Errors related to memory backend operations."""
    def __init__(self, message: str, backend_type: Optional[str] = None, 
                 operation: Optional[str] = None, **kwargs):
        context = {'backend_type': backend_type, 'operation': operation}
        super().__init__(message, error_code='MEMORY_BACKEND', context=context)

class ConfigurationError(CrewGraphError):
    """Configuration validation and loading errors."""
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 config_value: Optional[Any] = None, **kwargs):
        context = {'config_key': config_key, 'config_value': str(config_value) if config_value else None}
        super().__init__(message, error_code='CONFIGURATION', context=context)

class ValidationError(CrewGraphError):
    """Input validation errors."""
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 field_value: Optional[Any] = None, **kwargs):
        context = {'field_name': field_name, 'field_value': str(field_value) if field_value else None}
        super().__init__(message, error_code='VALIDATION', context=context)

class ResourceError(CrewGraphError):
    """Resource allocation and management errors."""
    def __init__(self, message: str, resource_type: Optional[str] = None, 
                 resource_id: Optional[str] = None, **kwargs):
        context = {'resource_type': resource_type, 'resource_id': resource_id}
        super().__init__(message, error_code='RESOURCE', context=context)

def handle_error_with_fallback(func, fallback_value=None, exception_types=(Exception,)):
    """Decorator for graceful error handling with fallback."""
    def decorator(original_func):
        def wrapper(*args, **kwargs):
            try:
                return original_func(*args, **kwargs)
            except exception_types as e:
                logger.warning(f"Function {original_func.__name__} failed: {e}. Using fallback.")
                if callable(fallback_value):
                    return fallback_value(*args, **kwargs)
                return fallback_value
        return wrapper
    
    if callable(func):
        return decorator(func)
    else:
        fallback_value = func
        return decorator