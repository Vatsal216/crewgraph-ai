"""
CrewGraph AI - Production-ready library combining CrewAI and LangGraph

This library provides advanced agent orchestration capabilities by combining
the power of CrewAI for agent definition and LangGraph for workflow orchestration.

Key Features:
    - Advanced agent orchestration with CrewAI integration
    - Powerful workflow management using LangGraph
    - Production-ready memory backends (Dict, Redis, FAISS, SQL)
    - Comprehensive visualization and debugging tools
    - Real-time execution monitoring and performance analytics
    - Type-safe APIs with complete type annotations
    - Scalable and fault-tolerant execution

Basic Usage:
    ```python
    from crewgraph_ai import CrewGraph
    
    # Create a workflow
    workflow = CrewGraph("my_workflow")
    
    # Add agents and tasks
    workflow.add_agent(my_agent, "data_analyst")
    workflow.add_task("analyze_data", "Analyze the provided dataset")
    
    # Execute workflow
    result = workflow.execute({"data": my_data})
    ```

Advanced Usage:
    ```python
    from crewgraph_ai import GraphOrchestrator, WorkflowVisualizer
    
    # Create orchestrator with visualization
    orchestrator = GraphOrchestrator("advanced_workflow")
    
    # Build workflow with LangGraph
    orchestrator.create_state_graph()
    orchestrator.add_node("task1", my_function)
    orchestrator.add_edge("task1", "task2")
    
    # Visualize workflow
    viz_path = orchestrator.visualize_workflow(format="html")
    print(f"Workflow visualization: {viz_path}")
    ```

For more information, visit: https://github.com/Vatsal216/crewgraph-ai
"""

__version__ = "1.0.0"
__author__ = "Vatsal216"
__created__ = "2025-07-22 11:25:03"
__license__ = "MIT"

# Core imports
from .core.agents import AgentWrapper, AgentPool
from .core.tasks import TaskWrapper, TaskChain
from .core.orchestrator import GraphOrchestrator, WorkflowBuilder
from .core.state import SharedState, StateManager

# Memory imports
from .memory.base import BaseMemory
from .memory.dict_memory import DictMemory

# Optional memory imports
try:
    from .memory.redis_memory import RedisMemory
except ImportError:
    RedisMemory = None

try:
    from .memory.faiss_memory import FAISSMemory
except ImportError:
    FAISSMemory = None

# Tools imports
from .tools.registry import ToolRegistry
from .tools.wrapper import ToolWrapper
from .tools.builtin import BuiltinTools

# Planning imports
from .planning.planner import DynamicPlanner
from .planning.strategies import PlanningStrategy

# Utility imports
from .utils.exceptions import CrewGraphError, ValidationError, ExecutionError
from .utils.logging import setup_logging, get_logger

# Type definitions
from .types import (
    StateDict, TaskResult, AgentResponse, ToolFunction, 
    NodeId, WorkflowId, TaskStatus, WorkflowStatus, NodeStatus,
    AgentProtocol, MemoryProtocol, ToolProtocol, StateProtocol,
    ExecutionResult, VisualizationConfig
)

# Main orchestration class for easy usage
from .crewgraph import CrewGraph, CrewGraphConfig
from .utils.metrics import get_metrics_collector, MetricsCollector, PerformanceMonitor

# Communication imports
from .communication import AgentCommunicationHub, Message, MessageType, MessagePriority, Channel, CommunicationProtocol

# Templates imports
from .templates import (
    WorkflowTemplate, TemplateRegistry, TemplateCategory, TemplateBuilder,
    DataPipelineTemplate, ResearchWorkflowTemplate, ContentGenerationTemplate
)

# Security imports
from .security import (
    SecurityManager, RoleManager, Role, Permission, User,
    AuditLogger, AuditEvent, EncryptionManager, CryptoConfig
)

# Intelligence imports
from .intelligence import (
    WorkflowOptimizer, OptimizationStrategy, 
    PerformancePredictor, ResourcePredictor,
    BottleneckAnalyzer, ResourceAnalyzer,
    MLModelManager, ModelType
)

# NLP imports
from .nlp import (
    RequirementsParser, WorkflowParser,
    NLToWorkflowConverter, WorkflowToNLConverter,
    ConversationalWorkflowBuilder,
    DocumentationGenerator, CodeGenerator
)
try:
    from .visualization import WorkflowVisualizer, ExecutionTracer, MemoryInspector, DebugTools
    VISUALIZATION_AVAILABLE = True
except ImportError:
    WorkflowVisualizer = None
    ExecutionTracer = None
    MemoryInspector = None
    DebugTools = None
    VISUALIZATION_AVAILABLE = False

__all__ = [
    # Core classes
    "CrewGraph",
    "AgentWrapper",
    "AgentPool", 
    "TaskWrapper",
    "TaskChain",
    "GraphOrchestrator",
    "WorkflowBuilder",
    "SharedState",
    "StateManager",
    
    # Memory classes (always available)
    "BaseMemory",
    "DictMemory",
    
    # Tool classes
    "ToolRegistry",
    "ToolWrapper",
    "BuiltinTools",
    
    # Planning classes
    "DynamicPlanner",
    "PlanningStrategy",
    
    # Communication classes
    "AgentCommunicationHub",
    "Message",
    "MessageType",
    "MessagePriority", 
    "Channel",
    "CommunicationProtocol",
    
    # Template classes
    "WorkflowTemplate",
    "TemplateRegistry",
    "TemplateCategory", 
    "TemplateBuilder",
    "DataPipelineTemplate",
    "ResearchWorkflowTemplate",
    "ContentGenerationTemplate",
    
    # Security classes
    "SecurityManager",
    "RoleManager",
    "Role",
    "Permission",
    "User",
    "AuditLogger", 
    "AuditEvent",
    "EncryptionManager",
    "CryptoConfig",
    
    # Intelligence classes
    "WorkflowOptimizer",
    "OptimizationStrategy",
    "PerformancePredictor", 
    "ResourcePredictor",
    "BottleneckAnalyzer",
    "ResourceAnalyzer",
    "MLModelManager",
    "ModelType",
    
    # NLP classes
    "RequirementsParser",
    "WorkflowParser",
    "NLToWorkflowConverter",
    "WorkflowToNLConverter", 
    "ConversationalWorkflowBuilder",
    "DocumentationGenerator",
    "CodeGenerator",
    
    # Utilities
    "CrewGraphError",
    "ValidationError", 
    "ExecutionError",
    "setup_logging",
    "get_logger",
    "get_metrics_collector",
    "MetricsCollector", 
    "PerformanceMonitor",
    
    # Type definitions
    "StateDict", "TaskResult", "AgentResponse", "ToolFunction",
    "NodeId", "WorkflowId", "TaskStatus", "WorkflowStatus", "NodeStatus",
    "AgentProtocol", "MemoryProtocol", "ToolProtocol", "StateProtocol",
    "ExecutionResult", "VisualizationConfig", "CrewGraphConfig",
]

# Add optional memory backends to __all__ if available
if RedisMemory:
    __all__.append("RedisMemory")
if FAISSMemory:
    __all__.append("FAISSMemory")

# Add visualization classes to __all__ if available
if VISUALIZATION_AVAILABLE:
    __all__.extend(["WorkflowVisualizer", "ExecutionTracer", "MemoryInspector", "DebugTools"])

# Setup default logging
setup_logging()

# Initialize global metrics on import
_metrics = get_metrics_collector()
_metrics.record_metric("crewgraph_library_imports_total", 1.0, {"version": __version__, "user": "Vatsal216"})

print(f"🎉 CrewGraph AI v{__version__} loaded with built-in metrics!")
print(f"📊 Metrics tracking enabled for user: Vatsal216")
print(f"📅 Created by {__author__} on {__created__}")