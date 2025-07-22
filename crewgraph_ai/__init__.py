"""
CrewGraph AI - Production-ready library combining CrewAI and LangGraph

This library provides advanced agent orchestration capabilities by combining
the power of CrewAI for agent definition and LangGraph for workflow orchestration.
"""

__version__ = "1.0.0"
__author__ = "Vatsal216"
__license__ = "MIT"

# Core imports
from .core.agents import AgentWrapper, AgentPool
from .core.tasks import TaskWrapper, TaskChain
from .core.orchestrator import GraphOrchestrator, WorkflowBuilder
from .core.state import SharedState, StateManager

# Memory imports
from .memory.base import BaseMemory
from .memory.dict_memory import DictMemory
from .memory.redis_memory import RedisMemory
from .memory.faiss_memory import FAISSMemory

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

# Main orchestration class for easy usage
from .crewgraph import CrewGraph

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
    
    # Memory classes
    "BaseMemory",
    "DictMemory", 
    "RedisMemory",
    "FAISSMemory",
    
    # Tool classes
    "ToolRegistry",
    "ToolWrapper",
    "BuiltinTools",
    
    # Planning classes
    "DynamicPlanner",
    "PlanningStrategy",
    
    # Utilities
    "CrewGraphError",
    "ValidationError", 
    "ExecutionError",
    "setup_logging",
    "get_logger",
]

# Setup default logging
setup_logging()