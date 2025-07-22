"""
Tool wrapper for integrating functions and external tools
"""

import asyncio
import inspect
import time
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps

from crewai.tools import BaseTool as CrewBaseTool

from .base import BaseTool, ToolMetadata, ToolType, ParameterSpec, ParameterType
from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError, ValidationError, ExecutionError

logger = get_logger(__name__)


class ToolWrapper(BaseTool):
    """
    Universal tool wrapper that can wrap:
    - Python functions
    - CrewAI tools
    - External APIs
    - Custom implementations
    
    Provides enhanced functionality on top of original tools while
    maintaining full compatibility.
    """
    
    def __init__(self,
                 name: str,
                 func: Optional[Callable] = None,
                 crew_tool: Optional[CrewBaseTool] = None,
                 description: str = "",
                 tool_type: ToolType = ToolType.FUNCTION,
                 **metadata_kwargs):
        """
        Initialize tool wrapper.
        
        Args:
            name: Tool name
            func: Python function to wrap
            crew_tool: CrewAI tool to wrap
            description: Tool description
            tool_type: Tool type classification
            **metadata_kwargs: Additional metadata
        """
        # Auto-detect parameters from function if available
        parameters = []
        if func:
            parameters = self._extract_parameters_from_function(func)
            if not description and func.__doc__:
                description = func.__doc__.strip()
        elif crew_tool:
            parameters = self._extract_parameters_from_crew_tool(crew_tool)
            if not description and hasattr(crew_tool, 'description'):
                description = crew_tool.description
        
        # Create metadata
        metadata = ToolMetadata(
            name=name,
            description=description,
            tool_type=tool_type,
            parameters=parameters,
            **metadata_kwargs
        )
        
        super().__init__(metadata)
        
        self.func = func
        self.crew_tool = crew_tool
        self._is_async = func and asyncio.iscoroutinefunction(func)
        
        # Enhanced features
        self._middleware: List[Callable] = []
        self._pre_hooks: List[Callable] = []
        self._post_hooks: List[Callable] = []
        self._error_handlers: List[Callable] = []
        
        logger.info(f"ToolWrapper '{name}' created for {tool_type.value}")
    
    @classmethod
    def from_function(cls, 
                      func: Callable,
                      name: Optional[str] = None,
                      description: Optional[str] = None,
                      **kwargs) -> 'ToolWrapper':
        """
        Create ToolWrapper from Python function.
        
        Args:
            func: Function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            **kwargs: Additional metadata
            
        Returns:
            ToolWrapper instance
        """
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__.strip() if func.__doc__ else "")
        
        return cls(
            name=tool_name,
            func=func,
            description=tool_description,
            tool_type=ToolType.FUNCTION,
            **kwargs
        )
    
    @classmethod
    def from_crew_tool(cls,
                       crew_tool: CrewBaseTool,
                       name: Optional[str] = None,
                       **kwargs) -> 'ToolWrapper':
        """
        Create ToolWrapper from CrewAI tool.
        
        Args:
            crew_tool: CrewAI tool instance
            name: Tool name (defaults to tool name)
            **kwargs: Additional metadata
            
        Returns:
            ToolWrapper instance with full CrewAI compatibility
        """
        tool_name = name or getattr(crew_tool, 'name', crew_tool.__class__.__name__)
        
        return cls(
            name=tool_name,
            crew_tool=crew_tool,
            description=getattr(crew_tool, 'description', ''),
            tool_type=ToolType.CUSTOM,
            **kwargs
        )
    
    def execute(self, **kwargs) -> Any:
        """
        Execute tool with comprehensive error handling and monitoring.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        start_time = time.time()
        success = False
        result = None
        
        try:
            # Validate parameters
            if not self.validate_parameters(**kwargs):
                raise ValidationError(f"Invalid parameters for tool '{self.metadata.name}'")
            
            # Run pre-hooks
            for hook in self._pre_hooks:
                try:
                    hook(self, kwargs)
                except Exception as e:
                    logger.warning(f"Pre-hook failed for tool '{self.metadata.name}': {e}")
            
            # Apply middleware
            execution_chain = self._build_execution_chain()
            result = execution_chain(kwargs)
            
            success = True
            
            # Run post-hooks
            for hook in self._post_hooks:
                try:
                    hook(self, kwargs, result)
                except Exception as e:
                    logger.warning(f"Post-hook failed for tool '{self.metadata.name}': {e}")
            
            return result
            
        except Exception as e:
            # Handle errors with registered handlers
            for handler in self._error_handlers:
                try:
                    handled_result = handler(self, e, kwargs)
                    if handled_result is not None:
                        result = handled_result
                        success = True
                        break
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {handler_error}")
            
            if not success:
                logger.error(f"Tool '{self.metadata.name}' execution failed: {e}")
                raise ExecutionError(f"Tool execution failed: {e}")
        
        finally:
            # Update statistics
            execution_time = time.time() - start_time
            self.update_stats(execution_time, success)
    
    async def execute_async(self, **kwargs) -> Any:
        """Execute tool asynchronously."""
        if self._is_async:
            # Direct async execution
            return await self._execute_async_impl(**kwargs)
        else:
            # Run sync tool in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.execute, **kwargs)
    
    async def _execute_async_impl(self, **kwargs) -> Any:
        """Implementation for async tool execution."""
        start_time = time.time()
        success = False
        result = None
        
        try:
            if not self.validate_parameters(**kwargs):
                raise ValidationError(f"Invalid parameters for tool '{self.metadata.name}'")
            
            if self.func:
                result = await self.func(**kwargs)
            elif self.crew_tool:
                # Run CrewAI tool in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._execute_crew_tool, kwargs)
            else:
                raise ExecutionError("No execution method available")
            
            success = True
            return result
            
        except Exception as e:
            logger.error(f"Async tool '{self.metadata.name}' execution failed: {e}")
            raise ExecutionError(f"Async tool execution failed: {e}")
        
        finally:
            execution_time = time.time() - start_time
            self.update_stats(execution_time, success)
    
    def _build_execution_chain(self) -> Callable:
        """Build middleware execution chain."""
        def execute_core(kwargs: Dict[str, Any]) -> Any:
            if self.func:
                return self.func(**kwargs)
            elif self.crew_tool:
                return self._execute_crew_tool(kwargs)
            else:
                raise ExecutionError("No execution method available")
        
        # Apply middleware in reverse order
        chain = execute_core
        for middleware in reversed(self._middleware):
            chain = self._wrap_with_middleware(chain, middleware)
        
        return chain
    
    def _wrap_with_middleware(self, next_func: Callable, middleware: Callable) -> Callable:
        """Wrap function with middleware."""
        def wrapped(kwargs: Dict[str, Any]) -> Any:
            return middleware(kwargs, next_func)
        return wrapped
    
    def _execute_crew_tool(self, kwargs: Dict[str, Any]) -> Any:
        """Execute CrewAI tool with full compatibility."""
        if not self.crew_tool:
            raise ExecutionError("No CrewAI tool available")
        
        try:
            # CrewAI tools expect different parameter formats
            if hasattr(self.crew_tool, '_run'):
                # Use internal _run method
                return self.crew_tool._run(**kwargs)
            elif hasattr(self.crew_tool, 'run'):
                # Use public run method
                return self.crew_tool.run(**kwargs)
            elif callable(self.crew_tool):
                # Direct call
                return self.crew_tool(**kwargs)
            else:
                raise ExecutionError("CrewAI tool is not callable")
                
        except Exception as e:
            logger.error(f"CrewAI tool execution failed: {e}")
            raise
    
    def validate_parameters(self, **kwargs) -> bool:
        """
        Comprehensive parameter validation.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            for param_spec in self.metadata.parameters:
                param_name = param_spec.name
                param_value = kwargs.get(param_name)
                
                # Check required parameters
                if param_spec.required and param_value is None:
                    logger.error(f"Required parameter '{param_name}' missing")
                    return False
                
                # Skip validation if parameter not provided and not required
                if param_value is None:
                    continue
                
                # Type validation
                if not self._validate_parameter_type(param_value, param_spec):
                    logger.error(f"Parameter '{param_name}' type validation failed")
                    return False
                
                # Value validation
                if not self._validate_parameter_value(param_value, param_spec):
                    logger.error(f"Parameter '{param_name}' value validation failed")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return False
    
    def _validate_parameter_type(self, value: Any, spec: ParameterSpec) -> bool:
        """Validate parameter type."""
        type_validators = {
            ParameterType.STRING: lambda v: isinstance(v, str),
            ParameterType.INTEGER: lambda v: isinstance(v, int),
            ParameterType.FLOAT: lambda v: isinstance(v, (int, float)),
            ParameterType.BOOLEAN: lambda v: isinstance(v, bool),
            ParameterType.LIST: lambda v: isinstance(v, list),
            ParameterType.DICT: lambda v: isinstance(v, dict),
            ParameterType.FILE: lambda v: isinstance(v, str),  # File path
            ParameterType.URL: lambda v: isinstance(v, str) and (v.startswith('http') or v.startswith('https')),
            ParameterType.JSON: lambda v: isinstance(v, (dict, list, str))
        }
        
        validator = type_validators.get(spec.type)
        return validator(value) if validator else True
    
    def _validate_parameter_value(self, value: Any, spec: ParameterSpec) -> bool:
        """Validate parameter value constraints."""
        # Min/max value validation
        if spec.min_value is not None and hasattr(value, '__lt__') and value < spec.min_value:
            return False
        
        if spec.max_value is not None and hasattr(value, '__gt__') and value > spec.max_value:
            return False
        
        # Allowed values validation
        if spec.allowed_values and value not in spec.allowed_values:
            return False
        
        # Pattern validation for strings
        if spec.pattern and isinstance(value, str):
            import re
            if not re.match(spec.pattern, value):
                return False
        
        return True
    
    def _extract_parameters_from_function(self, func: Callable) -> List[ParameterSpec]:
        """Extract parameter specifications from function signature."""
        parameters = []
        
        try:
            sig = inspect.signature(func)
            
            for param_name, param in sig.parameters.items():
                # Skip self and cls parameters
                if param_name in ('self', 'cls'):
                    continue
                
                # Determine parameter type from annotation
                param_type = ParameterType.STRING  # Default
                if param.annotation != inspect.Parameter.empty:
                    param_type = self._annotation_to_parameter_type(param.annotation)
                
                # Determine if required (no default value)
                required = param.default == inspect.Parameter.empty
                default = None if required else param.default
                
                param_spec = ParameterSpec(
                    name=param_name,
                    type=param_type,
                    description=f"Parameter {param_name}",
                    required=required,
                    default=default
                )
                
                parameters.append(param_spec)
                
        except Exception as e:
            logger.warning(f"Failed to extract parameters from function: {e}")
        
        return parameters
    
    def _extract_parameters_from_crew_tool(self, crew_tool: CrewBaseTool) -> List[ParameterSpec]:
        """Extract parameter specifications from CrewAI tool."""
        parameters = []
        
        try:
            # Try to get parameters from tool schema
            if hasattr(crew_tool, 'args_schema') and crew_tool.args_schema:
                schema = crew_tool.args_schema
                
                if hasattr(schema, '__fields__'):
                    # Pydantic model
                    for field_name, field_info in schema.__fields__.items():
                        param_type = self._python_type_to_parameter_type(field_info.type_)
                        required = field_info.is_required() if hasattr(field_info, 'is_required') else True
                        default = field_info.default if hasattr(field_info, 'default') else None
                        
                        param_spec = ParameterSpec(
                            name=field_name,
                            type=param_type,
                            description=field_info.field_info.description if hasattr(field_info, 'field_info') else "",
                            required=required,
                            default=default
                        )
                        
                        parameters.append(param_spec)
            
            # Fallback: try to extract from tool's _run method
            elif hasattr(crew_tool, '_run'):
                parameters = self._extract_parameters_from_function(crew_tool._run)
                
        except Exception as e:
            logger.warning(f"Failed to extract parameters from CrewAI tool: {e}")
        
        return parameters
    
    def _annotation_to_parameter_type(self, annotation: Any) -> ParameterType:
        """Convert Python type annotation to ParameterType."""
        type_mapping = {
            str: ParameterType.STRING,
            int: ParameterType.INTEGER,
            float: ParameterType.FLOAT,
            bool: ParameterType.BOOLEAN,
            list: ParameterType.LIST,
            dict: ParameterType.DICT,
        }
        
        # Handle typing module types
        origin = getattr(annotation, '__origin__', None)
        if origin:
            return type_mapping.get(origin, ParameterType.STRING)
        
        return type_mapping.get(annotation, ParameterType.STRING)
    
    def _python_type_to_parameter_type(self, python_type: Any) -> ParameterType:
        """Convert Python type to ParameterType."""
        return self._annotation_to_parameter_type(python_type)
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware function."""
        self._middleware.append(middleware)
        logger.debug(f"Middleware added to tool '{self.metadata.name}'")
    
    def add_pre_hook(self, hook: Callable) -> None:
        """Add pre-execution hook."""
        self._pre_hooks.append(hook)
        logger.debug(f"Pre-hook added to tool '{self.metadata.name}'")
    
    def add_post_hook(self, hook: Callable) -> None:
        """Add post-execution hook."""
        self._post_hooks.append(hook)
        logger.debug(f"Post-hook added to tool '{self.metadata.name}'")
    
    def add_error_handler(self, handler: Callable) -> None:
        """Add error handler."""
        self._error_handlers.append(handler)
        logger.debug(f"Error handler added to tool '{self.metadata.name}'")
    
    def get_crew_tool(self) -> Optional[CrewBaseTool]:
        """Get the original CrewAI tool."""
        return self.crew_tool
    
    def get_function(self) -> Optional[Callable]:
        """Get the original function."""
        return self.func
    
    def __call__(self, **kwargs) -> Any:
        """Make tool callable directly."""
        return self.execute(**kwargs)
    
    def __repr__(self) -> str:
        return f"ToolWrapper(name='{self.metadata.name}', type={self.metadata.tool_type.value})"