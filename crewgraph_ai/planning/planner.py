"""
Advanced dynamic planner for intelligent workflow orchestration
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from .base import BasePlanner, PlanningStrategy, ExecutionPlan, PlanNode, PlanEdge
from .strategies import SequentialStrategy, ParallelStrategy, ConditionalStrategy, OptimalStrategy
from ..core.tasks import TaskWrapper
from ..core.state import SharedState
from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError, ValidationError, ExecutionError

logger = get_logger(__name__)


class PlanningMode(Enum):
    """Planning mode options"""
    STATIC = "static"           # Plan once, execute
    DYNAMIC = "dynamic"         # Replan during execution
    ADAPTIVE = "adaptive"       # Learn and adapt from execution
    REACTIVE = "reactive"       # React to changes in real-time


class ResourceType(Enum):
    """Resource type classifications"""
    COMPUTE = "compute"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    API_CALLS = "api_calls"
    TIME = "time"


@dataclass
class ResourceConstraint:
    """Resource constraint definition"""
    resource_type: ResourceType
    max_value: float
    current_usage: float = 0.0
    priority: int = 1  # 1-10, higher is more important


@dataclass
class PlannerConfig:
    """Dynamic planner configuration"""
    mode: PlanningMode = PlanningMode.DYNAMIC
    strategy: str = "optimal"
    max_planning_time: float = 30.0
    replan_threshold: float = 0.3  # Replan if success rate drops below this
    look_ahead_steps: int = 3
    enable_parallel_execution: bool = True
    max_parallel_tasks: int = 5
    resource_constraints: List[ResourceConstraint] = field(default_factory=list)
    learning_enabled: bool = True
    optimization_enabled: bool = True


class DynamicPlanner(BasePlanner):
    """
    Advanced dynamic planner with intelligent workflow orchestration.
    
    Features:
    - Multiple planning strategies (sequential, parallel, conditional, optimal)
    - Resource-aware planning and optimization
    - Real-time replanning based on execution feedback
    - Machine learning-based task scheduling
    - Performance prediction and optimization
    - Fault tolerance and recovery planning
    """
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialize dynamic planner.
        
        Args:
            config: Planner configuration
        """
        self.config = config or PlannerConfig()
        
        # Planning strategies
        self._strategies: Dict[str, PlanningStrategy] = {
            'sequential': SequentialStrategy(),
            'parallel': ParallelStrategy(),
            'conditional': ConditionalStrategy(),
            'optimal': OptimalStrategy()
        }
        
        # Current strategy
        self._current_strategy = self._strategies.get(
            self.config.strategy, 
            self._strategies['optimal']
        )
        
        # Execution tracking
        self._execution_history: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, Dict[str, float]] = {}
        self._resource_usage: Dict[ResourceType, float] = {}
        
        # Learning and optimization
        self._task_performance_model: Dict[str, Dict[str, float]] = {}
        self._dependency_model: Dict[str, List[str]] = {}
        self._success_predictions: Dict[str, float] = {}
        
        # Thread safety and execution
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_parallel_tasks)
        self._active_plans: Dict[str, ExecutionPlan] = {}
        
        # Callbacks
        self._on_plan_created: Optional[Callable] = None
        self._on_replan_triggered: Optional[Callable] = None
        self._on_execution_complete: Optional[Callable] = None
        
        logger.info(f"DynamicPlanner initialized with strategy: {self.config.strategy}")
    
    def create_plan(self, 
                   tasks: List[TaskWrapper],
                   state: SharedState,
                   constraints: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """
        Create optimized execution plan for tasks.
        
        Args:
            tasks: List of tasks to plan
            state: Current workflow state
            constraints: Additional constraints
            
        Returns:
            Optimized execution plan
        """
        with self._lock:
            start_time = time.time()
            
            try:
                logger.info(f"Creating execution plan for {len(tasks)} tasks")
                
                # Analyze tasks and dependencies
                task_analysis = self._analyze_tasks(tasks, state)
                
                # Apply resource constraints
                resource_analysis = self._analyze_resources(tasks, constraints)
                
                # Generate plan using current strategy
                plan = self._current_strategy.create_plan(
                    tasks=tasks,
                    state=state,
                    task_analysis=task_analysis,
                    resource_analysis=resource_analysis,
                    constraints=constraints
                )
                
                # Optimize the plan
                if self.config.optimization_enabled:
                    plan = self._optimize_plan(plan, task_analysis)
                
                # Add predictions
                self._add_performance_predictions(plan, tasks)
                
                # Store plan
                self._active_plans[plan.id] = plan
                
                planning_time = time.time() - start_time
                logger.info(f"Execution plan created in {planning_time:.2f}s")
                
                if self._on_plan_created:
                    self._on_plan_created(plan, planning_time)
                
                return plan
                
            except Exception as e:
                logger.error(f"Failed to create execution plan: {e}")
                raise ExecutionError(f"Planning failed: {e}")
    
    async def create_plan_async(self,
                               tasks: List[TaskWrapper],
                               state: SharedState,
                               constraints: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """Create execution plan asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.create_plan,
            tasks,
            state,
            constraints
        )
    
    def replan(self, 
               plan_id: str,
               current_state: SharedState,
               execution_feedback: Dict[str, Any]) -> Optional[ExecutionPlan]:
        """
        Replan workflow based on execution feedback.
        
        Args:
            plan_id: Current plan ID
            current_state: Current execution state
            execution_feedback: Feedback from execution
            
        Returns:
            New execution plan or None if no replanning needed
        """
        with self._lock:
            current_plan = self._active_plans.get(plan_id)
            if not current_plan:
                logger.warning(f"Plan {plan_id} not found for replanning")
                return None
            
            try:
                # Analyze execution feedback
                should_replan = self._should_replan(current_plan, execution_feedback)
                
                if not should_replan:
                    logger.debug(f"No replanning needed for plan {plan_id}")
                    return None
                
                logger.info(f"Replanning triggered for plan {plan_id}")
                
                # Get remaining tasks
                remaining_tasks = self._get_remaining_tasks(current_plan, execution_feedback)
                
                # Create new plan for remaining tasks
                new_plan = self.create_plan(
                    tasks=remaining_tasks,
                    state=current_state,
                    constraints=current_plan.constraints
                )
                
                # Update plan relationships
                new_plan.parent_plan_id = plan_id
                new_plan.replan_reason = execution_feedback.get('reason', 'Performance optimization')
                
                if self._on_replan_triggered:
                    self._on_replan_triggered(current_plan, new_plan, execution_feedback)
                
                return new_plan
                
            except Exception as e:
                logger.error(f"Failed to replan workflow: {e}")
                return None
    
    def update_performance_model(self, 
                                task_name: str,
                                execution_result: Dict[str, Any]) -> None:
        """
        Update performance model with execution results.
        
        Args:
            task_name: Task that was executed
            execution_result: Execution result data
        """
        with self._lock:
            if task_name not in self._task_performance_model:
                self._task_performance_model[task_name] = {}
            
            model = self._task_performance_model[task_name]
            
            # Update execution time model
            execution_time = execution_result.get('execution_time', 0.0)
            if 'avg_execution_time' in model:
                count = model.get('execution_count', 1)
                model['avg_execution_time'] = (
                    (model['avg_execution_time'] * count + execution_time) / (count + 1)
                )
                model['execution_count'] = count + 1
            else:
                model['avg_execution_time'] = execution_time
                model['execution_count'] = 1
            
            # Update success rate model
            success = execution_result.get('success', False)
            if 'success_rate' in model:
                success_count = model.get('success_count', 0)
                total_count = model.get('execution_count', 1)
                if success:
                    success_count += 1
                model['success_rate'] = success_count / total_count
                model['success_count'] = success_count
            else:
                model['success_rate'] = 1.0 if success else 0.0
                model['success_count'] = 1 if success else 0
            
            # Update resource usage model
            resource_usage = execution_result.get('resource_usage', {})
            for resource_type, usage in resource_usage.items():
                key = f'{resource_type}_usage'
                if key in model:
                    count = model.get('execution_count', 1)
                    model[key] = (model[key] * (count - 1) + usage) / count
                else:
                    model[key] = usage
            
            logger.debug(f"Performance model updated for task '{task_name}'")
    
    def predict_execution_time(self, task_name: str, context: Dict[str, Any]) -> float:
        """
        Predict execution time for a task.
        
        Args:
            task_name: Task name
            context: Execution context
            
        Returns:
            Predicted execution time in seconds
        """
        model = self._task_performance_model.get(task_name, {})
        base_time = model.get('avg_execution_time', 30.0)  # Default 30 seconds
        
        # Apply context-based adjustments
        complexity_factor = context.get('complexity_factor', 1.0)
        data_size_factor = context.get('data_size_factor', 1.0)
        
        predicted_time = base_time * complexity_factor * data_size_factor
        
        logger.debug(f"Predicted execution time for '{task_name}': {predicted_time:.2f}s")
        return predicted_time
    
    def predict_success_probability(self, task_name: str, context: Dict[str, Any]) -> float:
        """
        Predict success probability for a task.
        
        Args:
            task_name: Task name
            context: Execution context
            
        Returns:
            Success probability (0.0 to 1.0)
        """
        model = self._task_performance_model.get(task_name, {})
        base_success_rate = model.get('success_rate', 0.9)  # Default 90%
        
        # Apply context-based adjustments
        error_conditions = context.get('error_conditions', [])
        error_penalty = len(error_conditions) * 0.1
        
        predicted_success = max(0.0, base_success_rate - error_penalty)
        
        logger.debug(f"Predicted success probability for '{task_name}': {predicted_success:.2f}")
        return predicted_success
    
    def get_optimal_task_order(self, 
                              tasks: List[TaskWrapper],
                              constraints: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Get optimal task execution order.
        
        Args:
            tasks: List of tasks to order
            constraints: Execution constraints
            
        Returns:
            Optimal task order (list of task names)
        """
        with self._lock:
            try:
                # Build dependency graph
                dependency_graph = self._build_dependency_graph(tasks)
                
                # Calculate task priorities
                task_priorities = self._calculate_task_priorities(tasks, constraints)
                
                # Apply topological sorting with priorities
                ordered_tasks = self._topological_sort_with_priorities(
                    dependency_graph, 
                    task_priorities
                )
                
                logger.info(f"Optimal task order determined: {ordered_tasks}")
                return ordered_tasks
                
            except Exception as e:
                logger.error(f"Failed to determine optimal task order: {e}")
                # Fallback to simple dependency order
                return [task.name for task in tasks]
    
    def analyze_workflow_bottlenecks(self, 
                                    plan: ExecutionPlan) -> List[Dict[str, Any]]:
        """
        Analyze workflow for potential bottlenecks.
        
        Args:
            plan: Execution plan to analyze
            
        Returns:
            List of potential bottlenecks
        """
        bottlenecks = []
        
        try:
            # Analyze task execution times
            for node in plan.nodes:
                task_name = node.task_name
                model = self._task_performance_model.get(task_name, {})
                
                avg_time = model.get('avg_execution_time', 0.0)
                success_rate = model.get('success_rate', 1.0)
                
                # Check for slow tasks
                if avg_time > 60.0:  # Tasks taking more than 1 minute
                    bottlenecks.append({
                        'type': 'slow_task',
                        'task_name': task_name,
                        'avg_execution_time': avg_time,
                        'severity': 'high' if avg_time > 300.0 else 'medium',
                        'recommendation': 'Consider optimizing or parallelizing this task'
                    })
                
                # Check for unreliable tasks
                if success_rate < 0.8:  # Tasks with less than 80% success rate
                    bottlenecks.append({
                        'type': 'unreliable_task',
                        'task_name': task_name,
                        'success_rate': success_rate,
                        'severity': 'high' if success_rate < 0.5 else 'medium',
                        'recommendation': 'Add error handling and retry logic'
                    })
            
            # Analyze dependencies
            dependency_chains = self._find_long_dependency_chains(plan)
            for chain in dependency_chains:
                if len(chain) > 5:  # Long dependency chains
                    bottlenecks.append({
                        'type': 'long_dependency_chain',
                        'chain': chain,
                        'length': len(chain),
                        'severity': 'medium',
                        'recommendation': 'Consider breaking down dependencies or adding parallelization'
                    })
            
            logger.info(f"Found {len(bottlenecks)} potential bottlenecks")
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Failed to analyze workflow bottlenecks: {e}")
            return []
    
    def set_strategy(self, strategy_name: str) -> bool:
        """
        Set planning strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            True if strategy set successfully
        """
        if strategy_name not in self._strategies:
            logger.error(f"Unknown strategy: {strategy_name}")
            return False
        
        self._current_strategy = self._strategies[strategy_name]
        self.config.strategy = strategy_name
        
        logger.info(f"Planning strategy set to: {strategy_name}")
        return True
    
    def add_custom_strategy(self, name: str, strategy: PlanningStrategy) -> None: