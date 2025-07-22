"""
Common type definitions for CrewGraph AI

Provides type aliases, protocols, and type definitions used throughout
the CrewGraph AI library for better type safety and IDE support.
"""

from typing import (
    TypeVar, Generic, Protocol, Union, Dict, List, Any, Optional, Callable,
    Awaitable, Iterator, AsyncIterator, TYPE_CHECKING
)
from typing_extensions import TypedDict, Literal, ParamSpec
from abc import abstractmethod
import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

# Type variables
T = TypeVar('T')
P = ParamSpec('P')
StateType = TypeVar('StateType', bound=Dict[str, Any])
ResultType = TypeVar('ResultType')

# Common type aliases
StateDict: TypeAlias = Dict[str, Any]
TaskResult: TypeAlias = Dict[str, Any]
AgentResponse: TypeAlias = Union[str, Dict[str, Any]]
ToolFunction: TypeAlias = Callable[..., Any]
AsyncToolFunction: TypeAlias = Callable[..., Awaitable[Any]]
ExecutionCallback: TypeAlias = Callable[[str, Dict[str, Any]], None]
AsyncExecutionCallback: TypeAlias = Callable[[str, Dict[str, Any]], Awaitable[None]]

# Configuration types
MemoryConfig: TypeAlias = Dict[str, Any]
AgentConfig: TypeAlias = Dict[str, Any]
TaskConfig: TypeAlias = Dict[str, Any]
WorkflowConfig: TypeAlias = Dict[str, Any]

# Execution types
NodeId: TypeAlias = str
WorkflowId: TypeAlias = str
ExecutionId: TypeAlias = str
Timestamp: TypeAlias = float

# Status literals
TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
WorkflowStatus = Literal["pending", "running", "completed", "failed", "paused", "cancelled"]
NodeStatus = Literal["pending", "running", "completed", "failed", "skipped"]

# Visualization types
VisualizationFormat = Literal["html", "png", "svg", "pdf", "json"]
ChartType = Literal["timeline", "graph", "heatmap", "metrics", "memory"]

# Protocol definitions for better typing
class AgentProtocol(Protocol):
    """Protocol for agent implementations."""
    
    @abstractmethod
    def execute(self, prompt: str, context: Dict[str, Any]) -> Any:
        """Execute agent with given prompt and context."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name."""
        ...


class AsyncAgentProtocol(Protocol):
    """Protocol for async agent implementations."""
    
    @abstractmethod
    async def execute(self, prompt: str, context: Dict[str, Any]) -> Any:
        """Execute agent asynchronously with given prompt and context."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name."""
        ...


class MemoryProtocol(Protocol):
    """Protocol for memory backend implementations."""
    
    @abstractmethod
    def store(self, key: str, value: Any) -> None:
        """Store value with key."""
        ...
    
    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """Load value by key."""
        ...
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value by key."""
        ...
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored data."""
        ...


class AsyncMemoryProtocol(Protocol):
    """Protocol for async memory backend implementations."""
    
    @abstractmethod
    async def store(self, key: str, value: Any) -> None:
        """Store value with key asynchronously."""
        ...
    
    @abstractmethod
    async def load(self, key: str) -> Optional[Any]:
        """Load value by key asynchronously."""
        ...
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value by key asynchronously."""
        ...
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists asynchronously."""
        ...
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored data asynchronously."""
        ...


class ToolProtocol(Protocol):
    """Protocol for tool implementations."""
    
    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute tool with given arguments."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        ...
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        ...


class StateProtocol(Protocol):
    """Protocol for state management implementations."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        ...
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        ...
    
    @abstractmethod
    def update(self, data: Dict[str, Any]) -> None:
        """Update state with dictionary."""
        ...
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        ...


# TypedDict definitions for structured data
class ExecutionEvent(TypedDict, total=False):
    """Execution event structure."""
    event_id: str
    timestamp: Timestamp
    node_id: NodeId
    event_type: str
    message: str
    duration: Optional[float]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class NodeInfo(TypedDict, total=False):
    """Node information structure."""
    id: NodeId
    name: str
    status: NodeStatus
    type: str
    description: Optional[str]
    metadata: Dict[str, Any]


class EdgeInfo(TypedDict, total=False):
    """Edge information structure."""
    source: NodeId
    target: NodeId
    type: str
    metadata: Dict[str, Any]


class WorkflowData(TypedDict, total=False):
    """Workflow data structure for visualization."""
    nodes: List[NodeInfo]
    edges: List[EdgeInfo]
    metadata: Dict[str, Any]


class MemoryStats(TypedDict, total=False):
    """Memory statistics structure."""
    total_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    python_memory_mb: float
    object_count: int
    gc_stats: Dict[str, int]
    backend_stats: Dict[str, Any]


class PerformanceMetrics(TypedDict, total=False):
    """Performance metrics structure."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    error_rate: float
    custom_metrics: Dict[str, float]


class ValidationIssue(TypedDict, total=False):
    """Validation issue structure."""
    severity: Literal["error", "warning", "info"]
    category: str
    component: str
    message: str
    details: Dict[str, Any]
    suggestions: List[str]


class DebugReport(TypedDict, total=False):
    """Debug report structure."""
    workflow_id: WorkflowId
    timestamp: Timestamp
    validation_issues: List[ValidationIssue]
    performance_metrics: PerformanceMetrics
    memory_analysis: MemoryStats
    recommendations: List[str]


# Generic types for better type safety
class ExecutionResult(Generic[T]):
    """Generic execution result wrapper."""
    
    def __init__(self, 
                 success: bool, 
                 data: Optional[T] = None, 
                 error: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
    
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.success
    
    def get_data(self) -> Optional[T]:
        """Get execution data."""
        return self.data
    
    def get_error(self) -> Optional[str]:
        """Get error message if any."""
        return self.error


# Callback type definitions
BeforeExecutionCallback: TypeAlias = Callable[[NodeId, StateDict], None]
AfterExecutionCallback: TypeAlias = Callable[[NodeId, StateDict, Any], None]
ErrorCallback: TypeAlias = Callable[[NodeId, StateDict, Exception], None]

AsyncBeforeExecutionCallback: TypeAlias = Callable[[NodeId, StateDict], Awaitable[None]]
AsyncAfterExecutionCallback: TypeAlias = Callable[[NodeId, StateDict, Any], Awaitable[None]]
AsyncErrorCallback: TypeAlias = Callable[[NodeId, StateDict, Exception], Awaitable[None]]

# Configuration schemas
class CrewGraphConfig(TypedDict, total=False):
    """CrewGraph configuration schema."""
    memory_backend: Optional[MemoryProtocol]
    enable_planning: bool
    max_concurrent_tasks: int
    task_timeout: float
    enable_logging: bool
    log_level: str
    enable_visualization: bool
    visualization_output_dir: str
    enable_metrics: bool
    metrics_collection_interval: float


class VisualizationConfig(TypedDict, total=False):
    """Visualization configuration schema."""
    output_dir: str
    default_format: VisualizationFormat
    enable_interactive: bool
    enable_real_time: bool
    chart_theme: str
    color_scheme: Dict[str, str]


# Error types
class CrewGraphException(Exception):
    """Base exception for CrewGraph errors."""
    pass


class ValidationException(CrewGraphException):
    """Exception for validation errors."""
    pass


class ExecutionException(CrewGraphException):
    """Exception for execution errors."""
    pass


class VisualizationException(CrewGraphException):
    """Exception for visualization errors."""
    pass


# Export commonly used types
__all__ = [
    # Type variables
    "T", "P", "StateType", "ResultType",
    
    # Type aliases
    "StateDict", "TaskResult", "AgentResponse", "ToolFunction", "AsyncToolFunction",
    "ExecutionCallback", "AsyncExecutionCallback", "MemoryConfig", "AgentConfig",
    "TaskConfig", "WorkflowConfig", "NodeId", "WorkflowId", "ExecutionId", "Timestamp",
    
    # Status literals
    "TaskStatus", "WorkflowStatus", "NodeStatus",
    
    # Visualization types
    "VisualizationFormat", "ChartType",
    
    # Protocols
    "AgentProtocol", "AsyncAgentProtocol", "MemoryProtocol", "AsyncMemoryProtocol",
    "ToolProtocol", "StateProtocol",
    
    # TypedDict structures
    "ExecutionEvent", "NodeInfo", "EdgeInfo", "WorkflowData", "MemoryStats",
    "PerformanceMetrics", "ValidationIssue", "DebugReport",
    
    # Configuration schemas
    "CrewGraphConfig", "VisualizationConfig",
    
    # Generic types
    "ExecutionResult",
    
    # Callbacks
    "BeforeExecutionCallback", "AfterExecutionCallback", "ErrorCallback",
    "AsyncBeforeExecutionCallback", "AsyncAfterExecutionCallback", "AsyncErrorCallback",
    
    # Exceptions
    "CrewGraphException", "ValidationException", "ExecutionException", "VisualizationException"
]