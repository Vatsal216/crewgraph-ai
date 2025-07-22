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
        """
        Add custom planning strategy.
        
        Args:
            name: Strategy name
            strategy: Strategy implementation
        """
        self._strategies[name] = strategy
        logger.info(f"Custom strategy '{name}' added")
    
    def set_callbacks(self,
                      on_plan_created: Optional[Callable] = None,
                      on_replan_triggered: Optional[Callable] = None,
                      on_execution_complete: Optional[Callable] = None):
        """Set planner callbacks."""
        self._on_plan_created = on_plan_created
        self._on_replan_triggered = on_replan_triggered
        self._on_execution_complete = on_execution_complete
    
    def get_planner_stats(self) -> Dict[str, Any]:
        """Get comprehensive planner statistics."""
        with self._lock:
            total_plans = len(self._execution_history)
            
            # Calculate success metrics
            successful_plans = sum(
                1 for h in self._execution_history 
                if h.get('success', False)
            )
            
            success_rate = successful_plans / max(total_plans, 1)
            
            # Calculate average planning time
            avg_planning_time = sum(
                h.get('planning_time', 0.0) 
                for h in self._execution_history
            ) / max(total_plans, 1)
            
            # Get strategy usage
            strategy_usage = {}
            for history in self._execution_history:
                strategy = history.get('strategy', 'unknown')
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            return {
                'total_plans_created': total_plans,
                'success_rate': success_rate,
                'avg_planning_time': avg_planning_time,
                'active_plans': len(self._active_plans),
                'strategy_usage': strategy_usage,
                'performance_models': len(self._task_performance_model),
                'current_strategy': self.config.strategy,
                'config': {
                    'mode': self.config.mode.value,
                    'max_parallel_tasks': self.config.max_parallel_tasks,
                    'learning_enabled': self.config.learning_enabled,
                    'optimization_enabled': self.config.optimization_enabled
                }
            }
    
    def _analyze_tasks(self, 
                      tasks: List[TaskWrapper],
                      state: SharedState) -> Dict[str, Any]:
        """Analyze tasks for planning optimization."""
        analysis = {
            'task_count': len(tasks),
            'has_dependencies': False,
            'estimated_total_time': 0.0,
            'complexity_score': 0.0,
            'parallelizable_tasks': [],
            'sequential_tasks': [],
            'critical_path': []
        }
        
        # Analyze each task
        for task in tasks:
            # Estimate execution time
            estimated_time = self.predict_execution_time(
                task.name, 
                {'state': state.to_dict()}
            )
            analysis['estimated_total_time'] += estimated_time
            
            # Check dependencies
            if task.dependencies:
                analysis['has_dependencies'] = True
                analysis['sequential_tasks'].append(task.name)
            else:
                analysis['parallelizable_tasks'].append(task.name)
            
            # Calculate complexity score
            complexity = len(task.dependencies) + len(task.tools)
            analysis['complexity_score'] += complexity
        
        # Find critical path
        if analysis['has_dependencies']:
            analysis['critical_path'] = self._find_critical_path(tasks)
        
        return analysis
    
    def _analyze_resources(self, 
                          tasks: List[TaskWrapper],
                          constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resource requirements and constraints."""
        resource_analysis = {
            'total_estimated_compute': 0.0,
            'total_estimated_memory': 0.0,
            'peak_resource_usage': {},
            'resource_conflicts': [],
            'constraint_violations': []
        }
        
        # Analyze resource usage for each task
        for task in tasks:
            model = self._task_performance_model.get(task.name, {})
            
            # Estimate resource usage
            compute_usage = model.get('compute_usage', 1.0)
            memory_usage = model.get('memory_usage', 100.0)  # MB
            
            resource_analysis['total_estimated_compute'] += compute_usage
            resource_analysis['total_estimated_memory'] += memory_usage
        
        # Check against constraints
        for constraint in self.config.resource_constraints:
            resource_type = constraint.resource_type
            max_value = constraint.max_value
            
            if resource_type == ResourceType.COMPUTE:
                if resource_analysis['total_estimated_compute'] > max_value:
                    resource_analysis['constraint_violations'].append({
                        'type': resource_type.value,
                        'required': resource_analysis['total_estimated_compute'],
                        'available': max_value
                    })
            
            elif resource_type == ResourceType.MEMORY:
                if resource_analysis['total_estimated_memory'] > max_value:
                    resource_analysis['constraint_violations'].append({
                        'type': resource_type.value,
                        'required': resource_analysis['total_estimated_memory'],
                        'available': max_value
                    })
        
        return resource_analysis
    
    def _optimize_plan(self, 
                      plan: ExecutionPlan,
                      task_analysis: Dict[str, Any]) -> ExecutionPlan:
        """Optimize execution plan for better performance."""
        try:
            # Optimize task ordering
            if self.config.enable_parallel_execution:
                plan = self._optimize_parallelization(plan, task_analysis)
            
            # Optimize resource allocation
            plan = self._optimize_resource_allocation(plan)
            
            # Add error recovery paths
            plan = self._add_error_recovery(plan)
            
            logger.debug("Execution plan optimized")
            return plan
            
        except Exception as e:
            logger.warning(f"Plan optimization failed: {e}")
            return plan
    
    def _optimize_parallelization(self, 
                                 plan: ExecutionPlan,
                                 task_analysis: Dict[str, Any]) -> ExecutionPlan:
        """Optimize plan for parallel execution."""
        parallelizable_tasks = task_analysis.get('parallelizable_tasks', [])
        
        if len(parallelizable_tasks) > 1:
            # Group parallelizable tasks
            parallel_groups = []
            current_group = []
            
            for task_name in parallelizable_tasks:
                current_group.append(task_name)
                
                # Create groups of max_parallel_tasks size
                if len(current_group) >= self.config.max_parallel_tasks:
                    parallel_groups.append(current_group)
                    current_group = []
            
            if current_group:
                parallel_groups.append(current_group)
            
            # Update plan metadata
            plan.metadata['parallel_groups'] = parallel_groups
            plan.metadata['parallelization_optimized'] = True
        
        return plan
    
    def _optimize_resource_allocation(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize resource allocation for plan."""
        # Calculate resource requirements per task
        resource_requirements = {}
        
        for node in plan.nodes:
            task_name = node.task_name
            model = self._task_performance_model.get(task_name, {})
            
            resource_requirements[task_name] = {
                'compute': model.get('compute_usage', 1.0),
                'memory': model.get('memory_usage', 100.0),
                'priority': node.priority
            }
        
        # Store in plan metadata
        plan.metadata['resource_requirements'] = resource_requirements
        plan.metadata['resource_optimized'] = True
        
        return plan
    
    def _add_error_recovery(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Add error recovery mechanisms to plan."""
        recovery_strategies = {}
        
        for node in plan.nodes:
            task_name = node.task_name
            model = self._task_performance_model.get(task_name, {})
            success_rate = model.get('success_rate', 1.0)
            
            # Add retry strategy for unreliable tasks
            if success_rate < 0.9:
                recovery_strategies[task_name] = {
                    'type': 'retry',
                    'max_retries': 3,
                    'backoff_factor': 2.0,
                    'fallback_task': None
                }
            
            # Add timeout strategy for slow tasks
            avg_time = model.get('avg_execution_time', 30.0)
            if avg_time > 60.0:
                if task_name not in recovery_strategies:
                    recovery_strategies[task_name] = {}
                
                recovery_strategies[task_name]['timeout'] = avg_time * 2.0
        
        plan.metadata['recovery_strategies'] = recovery_strategies
        plan.metadata['error_recovery_added'] = True
        
        return plan
    
    def _add_performance_predictions(self, 
                                   plan: ExecutionPlan,
                                   tasks: List[TaskWrapper]) -> None:
        """Add performance predictions to plan."""
        predictions = {}
        
        for task in tasks:
            task_name = task.name
            
            # Predict execution time
            predicted_time = self.predict_execution_time(task_name, {})
            
            # Predict success probability
            predicted_success = self.predict_success_probability(task_name, {})
            
            predictions[task_name] = {
                'estimated_execution_time': predicted_time,
                'estimated_success_probability': predicted_success,
                'confidence': self._calculate_prediction_confidence(task_name)
            }
        
        plan.metadata['performance_predictions'] = predictions
    
    def _calculate_prediction_confidence(self, task_name: str) -> float:
        """Calculate confidence in predictions based on historical data."""
        model = self._task_performance_model.get(task_name, {})
        execution_count = model.get('execution_count', 0)
        
        # Confidence increases with more historical data
        if execution_count == 0:
            return 0.1  # Very low confidence
        elif execution_count < 5:
            return 0.5  # Medium confidence
        elif execution_count < 20:
            return 0.8  # High confidence
        else:
            return 0.95  # Very high confidence
    
    def _should_replan(self, 
                      current_plan: ExecutionPlan,
                      execution_feedback: Dict[str, Any]) -> bool:
        """Determine if replanning is needed based on execution feedback."""
        # Check success rate threshold
        success_rate = execution_feedback.get('success_rate', 1.0)
        if success_rate < self.config.replan_threshold:
            return True
        
        # Check for significant performance degradation
        expected_time = current_plan.metadata.get('estimated_execution_time', 0.0)
        actual_time = execution_feedback.get('actual_execution_time', 0.0)
        
        if actual_time > expected_time * 1.5:  # 50% slower than expected
            return True
        
        # Check for resource constraint violations
        resource_violations = execution_feedback.get('resource_violations', [])
        if resource_violations:
            return True
        
        # Check for critical task failures
        failed_tasks = execution_feedback.get('failed_tasks', [])
        critical_path = current_plan.metadata.get('critical_path', [])
        
        for failed_task in failed_tasks:
            if failed_task in critical_path:
                return True
        
        return False
    
    def _get_remaining_tasks(self, 
                           current_plan: ExecutionPlan,
                           execution_feedback: Dict[str, Any]) -> List[TaskWrapper]:
        """Get remaining tasks that need to be executed."""
        completed_tasks = set(execution_feedback.get('completed_tasks', []))
        failed_tasks = set(execution_feedback.get('failed_tasks', []))
        
        remaining_tasks = []
        
        for node in current_plan.nodes:
            task_name = node.task_name
            
            # Skip completed tasks
            if task_name in completed_tasks:
                continue
            
            # Include failed tasks for retry
            if task_name in failed_tasks:
                # Reset task state for retry
                if hasattr(node, 'task_wrapper'):
                    node.task_wrapper.reset()
                remaining_tasks.append(node.task_wrapper)
            
            # Include pending tasks
            elif task_name not in completed_tasks:
                if hasattr(node, 'task_wrapper'):
                    remaining_tasks.append(node.task_wrapper)
        
        return remaining_tasks
    
    def _build_dependency_graph(self, tasks: List[TaskWrapper]) -> Dict[str, List[str]]:
        """Build dependency graph from tasks."""
        graph = {}
        
        for task in tasks:
            graph[task.name] = task.dependencies.copy()
        
        return graph
    
    def _calculate_task_priorities(self, 
                                 tasks: List[TaskWrapper],
                                 constraints: Optional[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate task priorities for optimal ordering."""
        priorities = {}
        
        for task in tasks:
            priority = 1  # Base priority
            
            # Higher priority for tasks with more dependents
            dependents = sum(
                1 for t in tasks 
                if task.name in t.dependencies
            )
            priority += dependents * 2
            
            # Higher priority for critical path tasks
            model = self._task_performance_model.get(task.name, {})
            avg_time = model.get('avg_execution_time', 30.0)
            if avg_time > 60.0:  # Long-running tasks get higher priority
                priority += 3
            
            # Lower priority for unreliable tasks
            success_rate = model.get('success_rate', 1.0)
            if success_rate < 0.8:
                priority -= 2
            
            priorities[task.name] = max(1, priority)  # Minimum priority of 1
        
        return priorities
    
    def _topological_sort_with_priorities(self, 
                                         dependency_graph: Dict[str, List[str]],
                                         priorities: Dict[str, int]) -> List[str]:
        """Perform topological sort with priority consideration."""
        # Standard topological sort with priority queue
        in_degree = {task: 0 for task in dependency_graph}
        
        # Calculate in-degrees
        for task, deps in dependency_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Priority queue (higher priority first)
        import heapq
        queue = []
        
        for task, degree in in_degree.items():
            if degree == 0:
                # Negative priority for max heap behavior
                heapq.heappush(queue, (-priorities.get(task, 1), task))
        
        result = []
        
        while queue:
            _, task = heapq.heappop(queue)
            result.append(task)
            
            # Update in-degrees for dependent tasks
            for dependent in dependency_graph:
                if task in dependency_graph[dependent]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        heapq.heappush(queue, (-priorities.get(dependent, 1), dependent))
        
        return result
    
    def _find_critical_path(self, tasks: List[TaskWrapper]) -> List[str]:
        """Find critical path in task dependency graph."""
        # Build dependency graph with execution times
        graph = {}
        execution_times = {}
        
        for task in tasks:
            graph[task.name] = task.dependencies.copy()
            execution_times[task.name] = self.predict_execution_time(task.name, {})
        
        # Calculate longest path (critical path)
        def calculate_longest_path(task_name, memo=None):
            if memo is None:
                memo = {}
            
            if task_name in memo:
                return memo[task_name]
            
            dependencies = graph.get(task_name, [])
            if not dependencies:
                memo[task_name] = (execution_times[task_name], [task_name])
                return memo[task_name]
            
            max_time = 0.0
            max_path = []
            
            for dep in dependencies:
                if dep in graph:
                    dep_time, dep_path = calculate_longest_path(dep, memo)
                    if dep_time > max_time:
                        max_time = dep_time
                        max_path = dep_path
            
            total_time = max_time + execution_times[task_name]
            total_path = max_path + [task_name]
            
            memo[task_name] = (total_time, total_path)
            return memo[task_name]
        
        # Find task with longest path
        longest_time = 0.0
        critical_path = []
        
        for task in tasks:
            time, path = calculate_longest_path(task.name)
            if time > longest_time:
                longest_time = time
                critical_path = path
        
        return critical_path
    
    def _find_long_dependency_chains(self, plan: ExecutionPlan) -> List[List[str]]:
        """Find long dependency chains in the plan."""
        chains = []
        
        def find_chain(node_name, current_chain, visited):
            if node_name in visited:
                return
            
            visited.add(node_name)
            current_chain.append(node_name)
            
            # Find dependent nodes
            dependents = []
            for edge in plan.edges:
                if edge.from_node == node_name:
                    dependents.append(edge.to_node)
            
            if not dependents:
                # End of chain
                if len(current_chain) > 3:  # Consider chains of 4+ tasks as long
                    chains.append(current_chain.copy())
            else:
                for dependent in dependents:
                    find_chain(dependent, current_chain, visited.copy())
            
            current_chain.pop()
        
        # Start from root nodes (nodes with no incoming edges)
        root_nodes = []
        all_to_nodes = set(edge.to_node for edge in plan.edges)
        
        for node in plan.nodes:
            if node.task_name not in all_to_nodes:
                root_nodes.append(node.task_name)
        
        for root in root_nodes:
            find_chain(root, [], set())
        
        return chains
    
    def shutdown(self):
        """Shutdown planner and cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
        
        logger.info("DynamicPlanner shutdown completed")
    
    def __repr__(self) -> str:
        return (f"DynamicPlanner(strategy='{self.config.strategy}', "
                f"mode={self.config.mode.value}, "
                f"active_plans={len(self._active_plans)})")
