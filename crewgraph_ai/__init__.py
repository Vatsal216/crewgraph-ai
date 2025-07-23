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

For more information, visit: https://github.com/crewgraph/crewgraph-ai
"""

__version__ = "1.0.0"
__author__ = "CrewGraph AI Team"
__created__ = "Production Release"
__license__ = "MIT"

# Core imports
from .core.agents import AgentPool, AgentWrapper
from .core.orchestrator import GraphOrchestrator, WorkflowBuilder
from .core.state import SharedState, StateManager
from .core.tasks import TaskChain, TaskWrapper

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

# Communication imports
from .communication import (
    AgentCommunicationHub,
    Channel,
    CommunicationProtocol,
    Message,
    MessagePriority,
    MessageType,
)

# Main orchestration class for easy usage (temporarily commented out due to dependency issues)
# from .crewgraph import CrewGraph, CrewGraphConfig
from .config import CrewGraphSettings, get_settings, configure, quick_setup

# Intelligence imports (commented out due to missing numpy dependency)
# from .intelligence import (
#     BottleneckAnalyzer,
#     MLModelManager,
#     ModelType,
#     OptimizationStrategy,
#     PerformancePredictor,
#     ResourceAnalyzer,
#     ResourcePredictor,
#     WorkflowOptimizer,
# )

# NLP imports
from .nlp import (
    CodeGenerator,
    ConversationalWorkflowBuilder,
    DocumentationGenerator,
    NLToWorkflowConverter,
    RequirementsParser,
    WorkflowParser,
    WorkflowToNLConverter,
)

# Planning imports
from .planning.planner import DynamicPlanner
from .planning.strategies import PlanningStrategy

# Security imports
from .security import (
    AuditEvent,
    AuditLogger,
    CryptoConfig,
    EncryptionManager,
    Permission,
    Role,
    RoleManager,
    SecurityManager,
    User,
)

# Templates imports (commented out due to missing crewai dependency)
# from .templates import (
#     ContentGenerationTemplate,
#     DataPipelineTemplate,
#     ResearchWorkflowTemplate,
#     TemplateBuilder,
#     TemplateCategory,
#     TemplateRegistry,
#     WorkflowTemplate,
# )
# Tools imports (commented out due to missing crewai dependency)
# from .tools.builtin import BuiltinTools
# from .tools.registry import ToolRegistry
# from .tools.wrapper import ToolWrapper

# Type definitions
from .types import (
    AgentProtocol,
    AgentResponse,
    ExecutionResult,
    MemoryProtocol,
    NodeId,
    NodeStatus,
    StateDict,
    StateProtocol,
    TaskResult,
    TaskStatus,
    ToolFunction,
    ToolProtocol,
    VisualizationConfig,
    WorkflowId,
    WorkflowStatus,
)

# Utility imports
from .utils.exceptions import CrewGraphError, ExecutionError, ValidationError
from .utils.logging import get_logger, setup_logging
from .utils.metrics import MetricsCollector, PerformanceMonitor, get_metrics_collector

# Intelligence imports (AI-driven optimization)
try:
    from .intelligence import (
        AdaptivePlanner,
        OptimizationResult,
        PatternAnalyzer,
        PlanningRecommendation,
        ResourceOptimizer,
        WorkflowMetrics,
        WorkflowPattern,
    )

    INTELLIGENCE_AVAILABLE = True
except ImportError:
    WorkflowMetrics = ResourceOptimizer = OptimizationResult = None
    AdaptivePlanner = PlanningRecommendation = PatternAnalyzer = WorkflowPattern = None
    INTELLIGENCE_AVAILABLE = False

# NLP imports (Natural Language Processing)
try:
    from .nlp import (
        ExtractedTask,
        IntentClassifier,
        ParsedWorkflow,
        TaskExtractor,
        WorkflowDocumentation,
        WorkflowIntent,
    )

    NLP_AVAILABLE = True
except ImportError:
    ParsedWorkflow = IntentClassifier = WorkflowIntent = None
    TaskExtractor = ExtractedTask = WorkflowDocumentation = None
    NLP_AVAILABLE = False

# Analytics imports (Advanced Analytics & Visualization)
try:
    from .analytics import AnalysisReport, DashboardConfig, PerformanceDashboard, WorkflowAnalyzer

    ANALYTICS_AVAILABLE = True
except ImportError:
    PerformanceDashboard = DashboardConfig = WorkflowAnalyzer = AnalysisReport = None
    ANALYTICS_AVAILABLE = False

# Optimization imports (Performance Optimization)
try:
    from .optimization import PerformanceTuner, ResourceScheduler

    OPTIMIZATION_AVAILABLE = True
except ImportError:
    ResourceScheduler = PerformanceTuner = None
    OPTIMIZATION_AVAILABLE = False

__all__ = [
    # Configuration
    "CrewGraphSettings",
    "get_settings", 
    "configure",
    "quick_setup",
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
    "StateDict",
    "TaskResult",
    "AgentResponse",
    "ToolFunction",
    "NodeId",
    "WorkflowId",
    "TaskStatus",
    "WorkflowStatus",
    "NodeStatus",
    "AgentProtocol",
    "MemoryProtocol",
    "ToolProtocol",
    "StateProtocol",
    "ExecutionResult",
    "VisualizationConfig",
    "CrewGraphConfig",
]

# Add optional memory backends to __all__ if available
if RedisMemory:
    __all__.append("RedisMemory")
if FAISSMemory:
    __all__.append("FAISSMemory")

# Add intelligence classes to __all__ if available
if INTELLIGENCE_AVAILABLE:
    __all__.extend(
        [
            "WorkflowMetrics",
            "ResourceOptimizer",
            "OptimizationResult",
            "AdaptivePlanner",
            "PlanningRecommendation",
            "PatternAnalyzer",
            "WorkflowPattern",
        ]
    )

# Add NLP classes to __all__ if available
if NLP_AVAILABLE:
    __all__.extend(
        [
            "ParsedWorkflow",
            "IntentClassifier",
            "WorkflowIntent",
            "TaskExtractor",
            "ExtractedTask",
            "WorkflowDocumentation",
        ]
    )

# Add analytics classes to __all__ if available
if ANALYTICS_AVAILABLE:
    __all__.extend(
        ["PerformanceDashboard", "DashboardConfig", "WorkflowAnalyzer", "AnalysisReport"]
    )

# Add optimization classes to __all__ if available
if OPTIMIZATION_AVAILABLE:
    __all__.extend(["ResourceScheduler", "PerformanceTuner"])

# Setup default logging
setup_logging()

# Import configuration for dynamic user
from .config import get_current_user

# Initialize global metrics on import
_metrics = get_metrics_collector()
current_user = get_current_user()
_metrics.record_metric(
    "crewgraph_library_imports_total", 1.0, {"version": __version__, "user": current_user}
)

print(f"🎉 CrewGraph AI v{__version__} loaded with built-in metrics!")
print(f"📊 Metrics tracking enabled for user: {current_user}")
print(f"📅 Created by {__author__} - {__created__}")
