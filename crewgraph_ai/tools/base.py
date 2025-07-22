"""
Base tool interface and metadata structures
"""

import inspect
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError, ValidationError

logger = get_logger(__name__)


class ToolType(Enum):
    """Tool type classifications"""
    FUNCTION = "function"
    API = "api"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    WEB_SCRAPER = "web_scraper"
    AI_MODEL = "ai_model"
    UTILITY = "utility"
    CUSTOM = "custom"


class ParameterType(Enum):
    """Parameter type definitions"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    FILE = "file"
    URL = "url"
    JSON = "json"


@dataclass
class ParameterSpec:
    """Tool parameter specification"""
    name: str
    type: ParameterType
    description: str = ""
    required: bool = True
    default: Optional[Any] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex pattern for validation


@dataclass
class ToolMetadata:
    """Comprehensive tool metadata"""
    name: str
    description: str
    tool_type: ToolType
    version: str = "1.0.0"
    author: str = "Unknown"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Parameters and usage
    parameters: List[ParameterSpec] = field(default_factory=list)
    return_type: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Operational metadata
    timeout: float = 30.0
    rate_limit: Optional[int] = None  # Calls per minute
    retry_count: int = 3
    
    # Dependencies and requirements
    dependencies: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    
    # Documentation and tags
    documentation_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    
    # Usage statistics
    usage_count: int = 0
    success_rate: float = 1.0
    average_execution_time: float = 0.0


class BaseTool(ABC):
    """
    Abstract base class for all tools with standardized interface.
    
    Provides a unified interface for tool execution, validation,
    and metadata management across different tool types.
    """
    
    def __init__(self, metadata: ToolMetadata):
        """
        Initialize base tool.
        
        Args:
            metadata: Tool metadata
        """
        self.metadata = metadata
        self._execution_history: List[Dict[str, Any]] = []
        self._is_initialized = False
        
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate tool parameters.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def initialize(self) -> bool:
        """
        Initialize tool resources.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._initialize_impl()
            self._is_initialized = True
            logger.info(f"Tool '{self.metadata.name}' initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize tool '{self.metadata.name}': {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up tool resources."""
        try:
            self._cleanup_impl()
            self._is_initialized = False
            logger.info(f"Tool '{self.metadata.name}' cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup tool '{self.metadata.name}': {e}")
    
    def _initialize_impl(self) -> None:
        """Implementation-specific initialization."""
        pass
    
    def _cleanup_impl(self) -> None:
        """Implementation-specific cleanup."""
        pass
    
    def get_parameter_spec(self, param_name: str) -> Optional[ParameterSpec]:
        """Get parameter specification by name."""
        for param in self.metadata.parameters:
            if param.name == param_name:
                return param
        return None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            'name': self.metadata.name,
            'usage_count': self.metadata.usage_count,
            'success_rate': self.metadata.success_rate,
            'average_execution_time': self.metadata.average_execution_time,
            'total_executions': len(self._execution_history),
            'last_used': max([h['timestamp'] for h in self._execution_history], default=None)
        }
    
    def update_stats(self, execution_time: float, success: bool) -> None:
        """Update tool usage statistics."""
        self.metadata.usage_count += 1
        
        # Update success rate
        total_executions = len(self._execution_history) + 1
        current_successes = sum(1 for h in self._execution_history if h['success']) + (1 if success else 0)
        self.metadata.success_rate = current_successes / total_executions
        
        # Update average execution time
        total_time = (self.metadata.average_execution_time * (total_executions - 1)) + execution_time
        self.metadata.average_execution_time = total_time / total_executions
        
        # Add to history
        self._execution_history.append({
            'timestamp': time.time(),
            'execution_time': execution_time,
            'success': success
        })
        
        # Keep only recent history (last 1000 executions)
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-1000:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            'metadata': {
                'name': self.metadata.name,
                'description': self.metadata.description,
                'tool_type': self.metadata.tool_type.value,
                'version': self.metadata.version,
                'author': self.metadata.author,
                'parameters': [
                    {
                        'name': p.name,
                        'type': p.type.value,
                        'description': p.description,
                        'required': p.required,
                        'default': p.default
                    }
                    for p in self.metadata.parameters
                ],
                'tags': self.metadata.tags,
                'category': self.metadata.category
            },
            'stats': self.get_usage_stats(),
            'initialized': self._is_initialized
        }
    
    def __repr__(self) -> str:
        return f"BaseTool(name='{self.metadata.name}', type={self.metadata.tool_type.value})"