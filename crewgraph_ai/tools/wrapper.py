"""
Tool Wrapper for CrewGraph AI
Wraps existing functions and CrewAI tools with enhanced functionality

Author: Vatsal216
Created: 2025-07-22 12:40:39 UTC
"""

import inspect
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

from crewai.tools import BaseTool as CrewAIBaseTool

from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector
from .base import BaseTool, ToolCategory, ToolMetadata, ToolStatus

logger = get_logger(__name__)
metrics = get_metrics_collector()


class ToolWrapper(BaseTool):
    """
    Wrapper class that converts functions and CrewAI Tools into BaseTool instances.

    Provides enhanced functionality including:
    - Automatic metadata extraction
    - Performance monitoring
    - Error handling and retry logic
    - Input/output validation
    - Caching capabilities
    - Usage analytics

    Created by: Vatsal216
    Date: 2025-07-22 12:40:39 UTC
    """

    def __init__(
        self,
        tool_or_function: Union[CrewAIBaseTool, Callable],
        metadata: Optional[ToolMetadata] = None,
        enable_caching: bool = False,
        cache_ttl: int = 3600,
        enable_retry: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize tool wrapper.

        Args:
            tool_or_function: CrewAI BaseTool or callable function to wrap
            metadata: Optional metadata (will be auto-generated if not provided)
            enable_caching: Enable result caching
            cache_ttl: Cache time-to-live in seconds
            enable_retry: Enable automatic retry on failure
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        # Extract function and metadata
        if isinstance(tool_or_function, CrewAIBaseTool):
            self.wrapped_tool = tool_or_function
            self.wrapped_function = tool_or_function.func
            auto_metadata = self._extract_metadata_from_tool(tool_or_function)
        else:
            self.wrapped_tool = None
            self.wrapped_function = tool_or_function
            auto_metadata = self._extract_metadata_from_function(tool_or_function)

        # Use provided metadata or auto-generated
        final_metadata = metadata or auto_metadata

        # Initialize base tool
        super().__init__(final_metadata)

        # Wrapper configuration
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Cache storage
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Signature analysis
        self.signature = inspect.signature(self.wrapped_function)
        self.parameters = dict(self.signature.parameters)

        logger.info(f"ToolWrapper initialized for: {self.metadata.name}")
        logger.info(f"Caching: {enable_caching}, Retry: {enable_retry}")
        logger.info(f"Created by: Vatsal216 at 2025-07-22 12:40:39")

    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the wrapped function with enhanced functionality.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function execution result
        """
        # Validate inputs
        if not self.validate_inputs(*args, **kwargs):
            raise ValueError("Input validation failed")

        # Generate cache key if caching enabled
        cache_key = None
        if self.enable_caching:
            cache_key = self._generate_cache_key(*args, **kwargs)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {self.metadata.name}")
                metrics.increment_counter(
                    "crewgraph_tool_cache_hits_total",
                    labels={"tool_name": self.metadata.name, "user": "Vatsal216"},
                )
                return cached_result

        # Execute with retry logic
        if self.enable_retry:
            result = self._execute_with_retry(*args, **kwargs)
        else:
            result = self.wrapped_function(*args, **kwargs)

        # Cache result if caching enabled
        if self.enable_caching and cache_key:
            self._cache_result(cache_key, result)
            metrics.increment_counter(
                "crewgraph_tool_cache_stores_total",
                labels={"tool_name": self.metadata.name, "user": "Vatsal216"},
            )

        return result

    def _execute_with_retry(self, *args, **kwargs) -> Any:
        """Execute with retry logic"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = self.wrapped_function(*args, **kwargs)

                if attempt > 0:
                    logger.info(f"Tool {self.metadata.name} succeeded on attempt {attempt + 1}")
                    metrics.increment_counter(
                        "crewgraph_tool_retry_successes_total",
                        labels={
                            "tool_name": self.metadata.name,
                            "attempt": str(attempt + 1),
                            "user": "Vatsal216",
                        },
                    )

                return result

            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    logger.warning(
                        f"Tool {self.metadata.name} failed on attempt {attempt + 1}, retrying..."
                    )
                    metrics.increment_counter(
                        "crewgraph_tool_retry_attempts_total",
                        labels={
                            "tool_name": self.metadata.name,
                            "attempt": str(attempt + 1),
                            "error_type": type(e).__name__,
                            "user": "Vatsal216",
                        },
                    )

                    if self.retry_delay > 0:
                        time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"Tool {self.metadata.name} failed after {self.max_retries + 1} attempts"
                    )
                    metrics.increment_counter(
                        "crewgraph_tool_retry_failures_total",
                        labels={
                            "tool_name": self.metadata.name,
                            "error_type": type(e).__name__,
                            "user": "Vatsal216",
                        },
                    )

        # Re-raise the last exception
        raise last_exception

    def _extract_metadata_from_tool(self, tool: CrewAIBaseTool) -> ToolMetadata:
        """Extract metadata from CrewAI Tool"""
        return ToolMetadata(
            name=tool.name,
            description=tool.description or "CrewAI Tool",
            author="Vatsal216",
            created_at="2025-07-22 12:40:39",
            category=ToolCategory.GENERAL,
            tags=["crewai", "wrapped"],
            input_types=self._analyze_input_types(tool.func),
            output_types=self._analyze_output_types(tool.func),
        )

    def _extract_metadata_from_function(self, func: Callable) -> ToolMetadata:
        """Extract metadata from function"""
        # Get function name and docstring
        name = getattr(func, "__name__", "unknown_function")
        description = inspect.getdoc(func) or f"Wrapped function: {name}"

        # Analyze function signature
        sig = inspect.signature(func)

        # Extract parameter information
        input_types = []
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                input_types.append(str(param.annotation))

        # Extract return type
        output_types = []
        if sig.return_annotation != inspect.Signature.empty:
            output_types.append(str(sig.return_annotation))

        # Determine category based on function name and docstring
        category = self._determine_category(name, description)

        return ToolMetadata(
            name=name,
            description=description,
            author="Vatsal216",
            created_at="2025-07-22 12:40:39",
            category=category,
            tags=["function", "wrapped"],
            input_types=input_types,
            output_types=output_types,
            python_requirements=self._extract_requirements(func),
        )

    def _analyze_input_types(self, func: Callable) -> List[str]:
        """Analyze function input types"""
        try:
            sig = inspect.signature(func)
            types = []

            for param in sig.parameters.values():
                if param.annotation != inspect.Parameter.empty:
                    types.append(str(param.annotation))
                else:
                    types.append("Any")

            return types
        except Exception:
            return ["Any"]

    def _analyze_output_types(self, func: Callable) -> List[str]:
        """Analyze function output types"""
        try:
            sig = inspect.signature(func)
            if sig.return_annotation != inspect.Signature.empty:
                return [str(sig.return_annotation)]
            else:
                return ["Any"]
        except Exception:
            return ["Any"]

    def _determine_category(self, name: str, description: str) -> ToolCategory:
        """Determine tool category based on name and description"""
        text = (name + " " + description).lower()

        if any(word in text for word in ["file", "read", "write", "save", "load"]):
            return ToolCategory.FILE_OPERATIONS
        elif any(word in text for word in ["web", "http", "api", "request", "url"]):
            return ToolCategory.API_INTEGRATION
        elif any(word in text for word in ["analyze", "analysis", "process", "calculate"]):
            return ToolCategory.ANALYSIS
        elif any(word in text for word in ["data", "database", "sql", "query"]):
            return ToolCategory.DATA_PROCESSING
        elif any(word in text for word in ["scrape", "crawl", "extract", "parse"]):
            return ToolCategory.WEB_SCRAPING
        elif any(word in text for word in ["email", "send", "notify", "message"]):
            return ToolCategory.COMMUNICATION
        elif any(word in text for word in ["monitor", "check", "watch", "alert"]):
            return ToolCategory.MONITORING
        elif any(word in text for word in ["security", "encrypt", "decrypt", "secure"]):
            return ToolCategory.SECURITY
        else:
            return ToolCategory.GENERAL

    def _extract_requirements(self, func: Callable) -> List[str]:
        """Extract Python requirements from function"""
        requirements = []

        try:
            # Get function source and look for import statements
            source = inspect.getsource(func)
            lines = source.split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    # Extract module name (simplified)
                    if line.startswith("import "):
                        module = line.replace("import ", "").split()[0].split(".")[0]
                    else:
                        module = line.split(" from ")[1].split()[0].split(".")[0]

                    if module not in ["os", "sys", "time", "json", "typing"]:
                        requirements.append(module)

        except Exception:
            pass

        return list(set(requirements))

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        try:
            import hashlib

            # Create a string representation of arguments
            arg_str = str(args) + str(sorted(kwargs.items()))

            # Generate hash
            return hashlib.md5(arg_str.encode()).hexdigest()

        except Exception:
            # Fallback to simple string
            return f"{len(args)}_{len(kwargs)}_{hash(str(args)[:100])}"

    def _get_cached_result(self, cache_key: str) -> Any:
        """Get cached result if available and not expired"""
        if cache_key not in self._cache:
            return None

        cache_entry = self._cache[cache_key]

        # Check if expired
        if time.time() - cache_entry["timestamp"] > self.cache_ttl:
            del self._cache[cache_key]
            return None

        return cache_entry["result"]

    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache execution result"""
        self._cache[cache_key] = {"result": result, "timestamp": time.time()}

        # Limit cache size (simple LRU)
        if len(self._cache) > 1000:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate input arguments against function signature"""
        try:
            # Bind arguments to signature
            bound_args = self.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            return True

        except TypeError as e:
            logger.warning(f"Input validation failed for {self.metadata.name}: {e}")
            return False

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema with parameter information"""
        schema = super().get_schema()

        # Add parameter details from signature
        parameters = {}
        required = []
        optional = []

        for param_name, param in self.parameters.items():
            param_info = {
                "type": (
                    str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
                ),
                "default": param.default if param.default != inspect.Parameter.empty else None,
            }
            parameters[param_name] = param_info

            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            else:
                optional.append(param_name)

        schema.update(
            {
                "parameters": parameters,
                "required_parameters": required,
                "optional_parameters": optional,
                "signature": str(self.signature),
            }
        )

        return schema

    def clear_cache(self) -> None:
        """Clear cached results"""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {cache_size} cached results for {self.metadata.name}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._cache),
            "cache_enabled": self.enable_caching,
            "cache_ttl": self.cache_ttl,
            "oldest_entry": min(
                (entry["timestamp"] for entry in self._cache.values()), default=None
            ),
            "newest_entry": max(
                (entry["timestamp"] for entry in self._cache.values()), default=None
            ),
        }

    def to_crewai_tool(self) -> CrewAIBaseTool:
        """Convert back to CrewAI Tool"""
        if self.wrapped_tool:
            return self.wrapped_tool
        else:
            return CrewAIBaseTool(
                name=self.metadata.name,
                func=self.wrapped_function,
                description=self.metadata.description,
            )

    def __repr__(self) -> str:
        return (
            f"ToolWrapper(name='{self.metadata.name}', "
            f"function='{self.wrapped_function.__name__}', "
            f"caching={self.enable_caching}, "
            f"retry={self.enable_retry})"
        )


def wrap_function(func: Callable, **kwargs) -> ToolWrapper:
    """
    Convenience function to wrap a function.

    Args:
        func: Function to wrap
        **kwargs: ToolWrapper configuration options

    Returns:
        ToolWrapper instance
    """
    return ToolWrapper(func, **kwargs)


def wrap_tool(tool: CrewAIBaseTool, **kwargs) -> ToolWrapper:
    """
    Convenience function to wrap a CrewAI Tool.

    Args:
        tool: CrewAI Tool to wrap
        **kwargs: ToolWrapper configuration options

    Returns:
        ToolWrapper instance
    """
    return ToolWrapper(tool, **kwargs)


def tool_decorator(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[ToolCategory] = None,
    **wrapper_kwargs,
):
    """
    Decorator to automatically wrap functions as tools.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        category: Tool category
        **wrapper_kwargs: ToolWrapper configuration options

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> ToolWrapper:
        # Create metadata
        metadata = ToolMetadata(
            name=name or func.__name__,
            description=description or inspect.getdoc(func) or f"Tool: {func.__name__}",
            category=category or ToolCategory.GENERAL,
            author="Vatsal216",
            created_at="2025-07-22 12:40:39",
        )

        # Create wrapper
        wrapper = ToolWrapper(func, metadata=metadata, **wrapper_kwargs)

        # Preserve function attributes
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__

        return wrapper

    return decorator
