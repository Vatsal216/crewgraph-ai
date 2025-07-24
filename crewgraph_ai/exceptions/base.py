"""Base exception classes for CrewGraph AI."""

class CrewGraphError(Exception):
    """Base exception for all CrewGraph AI errors."""
    pass

class WorkflowError(CrewGraphError):
    """Errors related to workflow execution."""
    pass

class AgentError(CrewGraphError):
    """Errors related to agent operations."""
    pass

class MemoryError(CrewGraphError):
    """Errors related to memory operations."""
    pass

class ConfigurationError(CrewGraphError):
    """Errors related to configuration."""
    pass

class ValidationError(CrewGraphError):
    """Errors related to input validation."""
    pass