"""
Base Tool Classes and Metadata for CrewGraph AI
Foundation classes for all tool implementations

Author: Vatsal216
Created: 2025-07-22 12:40:39 UTC
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class ToolStatus(Enum):
    """Tool status enumeration"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    MAINTENANCE = "maintenance"


class ToolCategory(Enum):
    """Tool category enumeration"""

    GENERAL = "general"
    DATA_PROCESSING = "data_processing"
    WEB_SCRAPING = "web_scraping"
    FILE_OPERATIONS = "file_operations"
    API_INTEGRATION = "api_integration"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    AUTOMATION = "automation"
    SECURITY = "security"
    MONITORING = "monitoring"


class ToolType(Enum):
    """Tool type enumeration"""

    FUNCTION = "function"
    CLASS = "class"
    EXTERNAL = "external"
    BUILTIN = "builtin"
    CUSTOM = "custom"
    WEBHOOK = "webhook"
    API = "api"


@dataclass
class ToolMetadata:
    """
    Comprehensive metadata for CrewGraph AI tools.

    Stores detailed information about tool capabilities,
    requirements, performance characteristics, and usage patterns.

    Created by: Vatsal216
    Date: 2025-07-22 12:40:39 UTC
    """

    # Basic information
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "Vatsal216"
    created_at: str = "2025-07-22 12:40:39"

    # Categorization
    category: ToolCategory = ToolCategory.GENERAL
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Status and lifecycle
    status: ToolStatus = ToolStatus.ACTIVE
    deprecated_since: Optional[str] = None
    replacement_tool: Optional[str] = None

    # Requirements and dependencies
    python_requirements: List[str] = field(default_factory=list)
    system_requirements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Performance characteristics
    estimated_execution_time: Optional[float] = None
    memory_usage_mb: Optional[int] = None
    cpu_intensive: bool = False
    io_intensive: bool = False
    network_required: bool = False

    # Usage information
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)

    # Documentation
    documentation_url: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)

    # Security and permissions
    requires_permissions: List[str] = field(default_factory=list)
    security_level: str = "low"  # low, medium, high
    data_privacy_impact: str = "none"  # none, low, medium, high

    # Analytics and monitoring
    usage_count: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    last_used: Optional[str] = None
    error_count: int = 0

    # Custom metadata
    custom_properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing"""
        if not self.created_at:
            self.created_at = "2025-07-22 12:40:39"

        # Auto-generate tags from category if none provided
        if not self.tags and self.category:
            self.tags = [self.category.value.replace("_", " ")]

    def add_tag(self, tag: str) -> None:
        """Add a tag to the tool metadata"""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the tool metadata"""
        if tag in self.tags:
            self.tags.remove(tag)

    def add_keyword(self, keyword: str) -> None:
        """Add a keyword to the tool metadata"""
        if keyword not in self.keywords:
            self.keywords.append(keyword)

    def update_usage_stats(self, execution_time: float, success: bool) -> None:
        """Update usage statistics"""
        self.usage_count += 1
        self.last_used = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        if success:
            # Update average execution time
            if self.average_execution_time == 0:
                self.average_execution_time = execution_time
            else:
                self.average_execution_time = (
                    self.average_execution_time * (self.usage_count - 1) + execution_time
                ) / self.usage_count

            # Update success rate
            successful_uses = int(self.success_rate * (self.usage_count - 1) / 100) + 1
            self.success_rate = (successful_uses / self.usage_count) * 100
        else:
            self.error_count += 1
            # Update success rate
            successful_uses = int(self.success_rate * (self.usage_count - 1) / 100)
            self.success_rate = (successful_uses / self.usage_count) * 100

    def is_deprecated(self) -> bool:
        """Check if tool is deprecated"""
        return self.status == ToolStatus.DEPRECATED

    def is_experimental(self) -> bool:
        """Check if tool is experimental"""
        return self.status == ToolStatus.EXPERIMENTAL

    def requires_network(self) -> bool:
        """Check if tool requires network access"""
        return self.network_required

    def get_risk_level(self) -> str:
        """Get overall risk level based on security and privacy settings"""
        security_levels = {"low": 1, "medium": 2, "high": 3}
        privacy_levels = {"none": 0, "low": 1, "medium": 2, "high": 3}

        security_score = security_levels.get(self.security_level, 1)
        privacy_score = privacy_levels.get(self.data_privacy_impact, 0)

        total_score = security_score + privacy_score

        if total_score <= 2:
            return "low"
        elif total_score <= 4:
            return "medium"
        else:
            return "high"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at,
            "category": self.category.value,
            "tags": self.tags,
            "keywords": self.keywords,
            "status": self.status.value,
            "deprecated_since": self.deprecated_since,
            "replacement_tool": self.replacement_tool,
            "python_requirements": self.python_requirements,
            "system_requirements": self.system_requirements,
            "dependencies": self.dependencies,
            "estimated_execution_time": self.estimated_execution_time,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_intensive": self.cpu_intensive,
            "io_intensive": self.io_intensive,
            "network_required": self.network_required,
            "input_types": self.input_types,
            "output_types": self.output_types,
            "supported_formats": self.supported_formats,
            "documentation_url": self.documentation_url,
            "examples": self.examples,
            "use_cases": self.use_cases,
            "requires_permissions": self.requires_permissions,
            "security_level": self.security_level,
            "data_privacy_impact": self.data_privacy_impact,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "average_execution_time": self.average_execution_time,
            "last_used": self.last_used,
            "error_count": self.error_count,
            "custom_properties": self.custom_properties,
            "risk_level": self.get_risk_level(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolMetadata":
        """Create metadata from dictionary"""
        # Convert enum fields
        if "category" in data and isinstance(data["category"], str):
            data["category"] = ToolCategory(data["category"])

        if "status" in data and isinstance(data["status"], str):
            data["status"] = ToolStatus(data["status"])

        # Remove computed fields
        computed_fields = ["risk_level"]
        for field in computed_fields:
            data.pop(field, None)

        return cls(**data)


class BaseTool(ABC):
    """
    Abstract base class for all CrewGraph AI tools.

    Provides a standardized interface and common functionality
    for tool implementations, including metadata management,
    execution tracking, error handling, and performance monitoring.

    Created by: Vatsal216
    Date: 2025-07-22 12:40:39 UTC
    """

    def __init__(self, metadata: Optional[ToolMetadata] = None):
        """
        Initialize base tool.

        Args:
            metadata: Tool metadata
        """
        self.tool_id = str(uuid.uuid4())
        self.metadata = metadata or ToolMetadata(
            name=self.__class__.__name__,
            description="Base tool implementation",
            author="Vatsal216",
            created_at="2025-07-22 12:40:39",
        )

        # Execution tracking
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_execution_time = None
        self._error_count = 0

        logger.info(f"BaseTool initialized: {self.metadata.name}")
        logger.info(f"Tool ID: {self.tool_id}")
        logger.info(f"Created by: Vatsal216 at 2025-07-22 12:40:39")

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the tool with given arguments.

        This method must be implemented by all tool subclasses.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool execution result
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """
        Make tool callable and add execution tracking.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool execution result
        """
        start_time = time.time()
        success = False
        result = None

        try:
            # Record execution start
            self._execution_count += 1

            # Execute the tool
            result = self.execute(*args, **kwargs)
            success = True

            # Record success metrics
            metrics.increment_counter(
                "crewgraph_tool_executions_total",
                labels={
                    "tool_name": self.metadata.name,
                    "tool_id": self.tool_id,
                    "success": "true",
                    "user": "Vatsal216",
                },
            )

            return result

        except Exception as e:
            # Record error
            self._error_count += 1
            self.metadata.error_count += 1

            # Record error metrics
            metrics.increment_counter(
                "crewgraph_tool_executions_total",
                labels={
                    "tool_name": self.metadata.name,
                    "tool_id": self.tool_id,
                    "success": "false",
                    "error_type": type(e).__name__,
                    "user": "Vatsal216",
                },
            )

            logger.error(f"Tool execution failed: {self.metadata.name} - {e}")
            raise

        finally:
            # Update execution statistics
            execution_time = time.time() - start_time
            self._total_execution_time += execution_time
            self._last_execution_time = time.time()

            # Update metadata statistics
            self.metadata.update_usage_stats(execution_time, success)

            # Record performance metrics
            metrics.record_duration(
                "crewgraph_tool_execution_duration_seconds",
                execution_time,
                labels={
                    "tool_name": self.metadata.name,
                    "tool_id": self.tool_id,
                    "user": "Vatsal216",
                },
            )

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive tool information"""
        return {
            "tool_id": self.tool_id,
            "metadata": self.metadata.to_dict(),
            "runtime_stats": {
                "execution_count": self._execution_count,
                "total_execution_time": self._total_execution_time,
                "average_execution_time": (
                    self._total_execution_time / self._execution_count
                    if self._execution_count > 0
                    else 0
                ),
                "last_execution_time": self._last_execution_time,
                "error_count": self._error_count,
                "success_rate": (
                    ((self._execution_count - self._error_count) / self._execution_count * 100)
                    if self._execution_count > 0
                    else 0
                ),
            },
            "status": {
                "is_active": self.metadata.status == ToolStatus.ACTIVE,
                "is_deprecated": self.metadata.is_deprecated(),
                "is_experimental": self.metadata.is_experimental(),
                "risk_level": self.metadata.get_risk_level(),
            },
            "created_by": "Vatsal216",
            "created_at": "2025-07-22 12:40:39",
        }

    def validate_inputs(self, *args, **kwargs) -> bool:
        """
        Validate input parameters.

        Override this method to implement custom input validation.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            True if inputs are valid, False otherwise
        """
        return True

    def get_schema(self) -> Dict[str, Any]:
        """
        Get tool input/output schema.

        Override this method to provide tool schema information.

        Returns:
            Schema dictionary
        """
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "input_types": self.metadata.input_types,
            "output_types": self.metadata.output_types,
            "parameters": {},
            "required_parameters": [],
            "optional_parameters": [],
        }

    def cleanup(self) -> None:
        """
        Cleanup resources used by the tool.

        Override this method to implement custom cleanup logic.
        """
        pass

    def reset_stats(self) -> None:
        """Reset execution statistics"""
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_execution_time = None
        self._error_count = 0

        # Reset metadata stats
        self.metadata.usage_count = 0
        self.metadata.success_rate = 0.0
        self.metadata.average_execution_time = 0.0
        self.metadata.error_count = 0
        self.metadata.last_used = None

        logger.info(f"Statistics reset for tool: {self.metadata.name}")

    def __str__(self) -> str:
        return f"{self.metadata.name} (v{self.metadata.version})"

    def __repr__(self) -> str:
        return (
            f"BaseTool(name='{self.metadata.name}', "
            f"id='{self.tool_id[:8]}...', "
            f"executions={self._execution_count})"
        )


def create_tool_metadata(name: str, description: str, **kwargs) -> ToolMetadata:
    """
    Factory function to create tool metadata.

    Args:
        name: Tool name
        description: Tool description
        **kwargs: Additional metadata properties

    Returns:
        ToolMetadata instance
    """
    return ToolMetadata(
        name=name,
        description=description,
        author="Vatsal216",
        created_at="2025-07-22 12:40:39",
        **kwargs,
    )
