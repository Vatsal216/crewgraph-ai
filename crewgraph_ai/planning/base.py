"""
Base planning interfaces and data structures
"""

import uuid
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logging import get_logger

logger = get_logger(__name__)


class NodeType(Enum):
    """Plan node types"""
    TASK = "task"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    MERGE = "merge"


class EdgeType(Enum):
    """Plan edge types"""
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional" 
    PARALLEL = "parallel"
    FALLBACK = "fallback"


@dataclass
class PlanNode:
    """Execution plan node"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_name: str = ""
    node_type: NodeType = NodeType.TASK
    priority: int = 1
    estimated_duration: float = 0.0
    estimated_resources: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_wrapper: Optional[Any] = None


@dataclass
class PlanEdge:
    """Execution plan edge"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_node: str = ""
    to_node: str = ""
    edge_type: EdgeType = EdgeType.SEQUENTIAL
    condition: Optional[str] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Complete execution plan"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    nodes: List[PlanNode] = field(default_factory=list)
    edges: List[PlanEdge] = field(default_factory=list)
    constraints: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    estimated_total_duration: float = 0.0
    parent_plan_id: Optional[str] = None
    replan_reason: Optional[str] = None


class PlanningStrategy(ABC):
    """Abstract base class for planning strategies"""
    
    @abstractmethod
    def create_plan(self,
                   tasks: List[Any],
                   state: Any,
                   task_analysis: Dict[str, Any],
                   resource_analysis: Dict[str, Any],
                   constraints: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """Create execution plan using this strategy"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        pass


class BasePlanner(ABC):
    """Abstract base class for planners"""
    
    @abstractmethod
    def create_plan(self, 
                   tasks: List[Any],
                   state: Any,
                   constraints: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """Create execution plan"""
        pass
    
    @abstractmethod
    def replan(self,
              plan_id: str,
              current_state: Any,
              execution_feedback: Dict[str, Any]) -> Optional[ExecutionPlan]:
        """Replan based on execution feedback"""
        pass