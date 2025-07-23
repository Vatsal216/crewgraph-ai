"""
Adaptive Planner for CrewGraph AI Intelligence Layer

This module provides dynamic workflow planning and optimization based on
real-time feedback and execution patterns.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from ..utils.logging import get_logger
from ..types import WorkflowId, TaskStatus

logger = get_logger(__name__)


class PlanningStrategy(Enum):
    """Available planning strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    RESOURCE_OPTIMIZED = "resource_optimized"
    TIME_OPTIMIZED = "time_optimized"


@dataclass
class PlanningRecommendation:
    """Recommendation for workflow planning optimization."""
    strategy: PlanningStrategy
    confidence: float
    expected_improvement: float
    implementation_steps: List[str]
    estimated_execution_time: float
    resource_requirements: Dict[str, float]


class AdaptivePlanner:
    """
    Adaptive workflow planner that optimizes execution strategies based on
    real-time feedback, resource availability, and historical performance.
    """
    
    def __init__(self):
        """Initialize the adaptive planner."""
        self.planning_history: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[str, List[float]] = {}
        
        logger.info("AdaptivePlanner initialized")
    
    def recommend_strategy(self, 
                         workflow_definition: Dict[str, Any],
                         current_resources: Dict[str, float],
                         constraints: Optional[Dict[str, Any]] = None) -> PlanningRecommendation:
        """
        Recommend optimal planning strategy for a workflow.
        
        Args:
            workflow_definition: Workflow structure and requirements
            current_resources: Available system resources
            constraints: Execution constraints (time, cost, etc.)
            
        Returns:
            Planning recommendation with strategy and implementation steps
        """
        task_count = len(workflow_definition.get("tasks", []))
        dependencies = workflow_definition.get("dependencies", [])
        
        # Analyze workflow characteristics
        has_dependencies = len(dependencies) > 0
        cpu_available = current_resources.get("cpu_percent", 50) < 70
        memory_available = current_resources.get("memory_percent", 50) < 80
        
        # Default to sequential for safety
        strategy = PlanningStrategy.SEQUENTIAL
        confidence = 0.7
        expected_improvement = 10.0
        
        implementation_steps = ["Execute tasks in sequential order"]
        estimated_time = task_count * 5.0  # 5 seconds per task baseline
        
        # Decision logic for strategy selection
        if not has_dependencies and task_count > 2 and cpu_available and memory_available:
            strategy = PlanningStrategy.PARALLEL
            confidence = 0.8
            expected_improvement = min(50.0, task_count * 15.0)
            implementation_steps = [
                "Execute independent tasks in parallel",
                f"Use up to {min(task_count, 4)} concurrent workers",
                "Monitor resource usage during execution"
            ]
            estimated_time = max(5.0, task_count * 5.0 / min(task_count, 4))
            
        elif constraints and constraints.get("optimize_for") == "time":
            strategy = PlanningStrategy.TIME_OPTIMIZED
            confidence = 0.75
            expected_improvement = 30.0
            implementation_steps = [
                "Prioritize time-critical tasks",
                "Use aggressive parallelization where safe",
                "Pre-allocate resources"
            ]
            estimated_time = task_count * 3.0
            
        elif constraints and constraints.get("optimize_for") == "resources":
            strategy = PlanningStrategy.RESOURCE_OPTIMIZED
            confidence = 0.8
            expected_improvement = 20.0
            implementation_steps = [
                "Minimize resource usage",
                "Use task batching to reduce overhead",
                "Implement resource pooling"
            ]
            estimated_time = task_count * 7.0
            
        elif self._should_use_adaptive_strategy(workflow_definition):
            strategy = PlanningStrategy.ADAPTIVE
            confidence = 0.85
            expected_improvement = 25.0
            implementation_steps = [
                "Start with conservative resource allocation",
                "Monitor performance and adjust strategy dynamically",
                "Scale parallelism based on actual performance"
            ]
            estimated_time = task_count * 4.0
        
        resource_requirements = self._estimate_resource_requirements(
            task_count, strategy, workflow_definition
        )
        
        recommendation = PlanningRecommendation(
            strategy=strategy,
            confidence=confidence,
            expected_improvement=expected_improvement,
            implementation_steps=implementation_steps,
            estimated_execution_time=estimated_time,
            resource_requirements=resource_requirements
        )
        
        logger.info(f"Recommended strategy: {strategy.value} with {confidence:.2f} confidence")
        return recommendation
    
    def adapt_during_execution(self, 
                             current_strategy: PlanningStrategy,
                             execution_metrics: Dict[str, Any],
                             remaining_tasks: int) -> Optional[PlanningRecommendation]:
        """
        Adapt planning strategy during workflow execution based on performance.
        
        Args:
            current_strategy: Currently executing strategy
            execution_metrics: Real-time execution metrics
            remaining_tasks: Number of tasks remaining
            
        Returns:
            New recommendation if adaptation is needed, None otherwise
        """
        if remaining_tasks <= 1:
            return None  # Too late to adapt
        
        # Analyze current performance
        cpu_usage = execution_metrics.get("cpu_usage", 50)
        memory_usage = execution_metrics.get("memory_usage", 50)
        execution_rate = execution_metrics.get("tasks_per_second", 0.2)
        error_rate = execution_metrics.get("error_rate", 0.0)
        
        should_adapt = False
        new_strategy = current_strategy
        adaptation_reason = ""
        
        # Check for performance issues requiring adaptation
        if cpu_usage < 30 and current_strategy == PlanningStrategy.SEQUENTIAL:
            should_adapt = True
            new_strategy = PlanningStrategy.PARALLEL
            adaptation_reason = "Low CPU usage suggests parallel execution is safe"
            
        elif cpu_usage > 90 and current_strategy == PlanningStrategy.PARALLEL:
            should_adapt = True
            new_strategy = PlanningStrategy.SEQUENTIAL
            adaptation_reason = "High CPU usage requires throttling to sequential execution"
            
        elif memory_usage > 85:
            should_adapt = True
            new_strategy = PlanningStrategy.RESOURCE_OPTIMIZED
            adaptation_reason = "High memory usage requires resource optimization"
            
        elif error_rate > 0.1:
            should_adapt = True
            new_strategy = PlanningStrategy.SEQUENTIAL
            adaptation_reason = "High error rate suggests need for conservative execution"
            
        elif execution_rate < 0.1:
            should_adapt = True
            new_strategy = PlanningStrategy.TIME_OPTIMIZED
            adaptation_reason = "Slow execution rate requires time optimization"
        
        if should_adapt:
            logger.info(f"Adapting strategy from {current_strategy.value} to {new_strategy.value}: {adaptation_reason}")
            
            # Create new recommendation
            return PlanningRecommendation(
                strategy=new_strategy,
                confidence=0.7,
                expected_improvement=15.0,
                implementation_steps=[
                    f"Switch to {new_strategy.value} strategy",
                    adaptation_reason,
                    "Monitor performance after adaptation"
                ],
                estimated_execution_time=remaining_tasks * 4.0,
                resource_requirements=self._estimate_resource_requirements(
                    remaining_tasks, new_strategy, {}
                )
            )
        
        return None
    
    def analyze_execution_patterns(self, 
                                 execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze historical execution patterns to improve future planning.
        
        Args:
            execution_history: List of historical execution records
            
        Returns:
            Pattern analysis with insights and recommendations
        """
        if not execution_history:
            return {"insight": "Insufficient data for pattern analysis"}
        
        # Group executions by strategy
        strategy_groups = {}
        for execution in execution_history:
            strategy = execution.get("strategy", "unknown")
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(execution)
        
        # Analyze performance by strategy
        strategy_performance = {}
        for strategy, executions in strategy_groups.items():
            if executions:
                avg_time = sum(e.get("execution_time", 0) for e in executions) / len(executions)
                avg_success_rate = sum(e.get("success_rate", 0) for e in executions) / len(executions)
                
                strategy_performance[strategy] = {
                    "average_execution_time": avg_time,
                    "average_success_rate": avg_success_rate,
                    "total_executions": len(executions),
                    "performance_score": avg_success_rate / max(avg_time, 1.0)
                }
        
        # Find best performing strategy
        best_strategy = None
        best_score = 0
        for strategy, perf in strategy_performance.items():
            if perf["performance_score"] > best_score:
                best_score = perf["performance_score"]
                best_strategy = strategy
        
        # Generate insights
        insights = []
        if best_strategy:
            insights.append(f"Best performing strategy: {best_strategy}")
        
        # Check for patterns
        recent_executions = execution_history[-10:]  # Last 10 executions
        if recent_executions:
            recent_avg_time = sum(e.get("execution_time", 0) for e in recent_executions) / len(recent_executions)
            all_avg_time = sum(e.get("execution_time", 0) for e in execution_history) / len(execution_history)
            
            if recent_avg_time > all_avg_time * 1.2:
                insights.append("Recent executions are slower than historical average")
            elif recent_avg_time < all_avg_time * 0.8:
                insights.append("Recent executions are faster than historical average")
        
        return {
            "strategy_performance": strategy_performance,
            "best_strategy": best_strategy,
            "insights": insights,
            "total_executions_analyzed": len(execution_history),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _should_use_adaptive_strategy(self, workflow_definition: Dict[str, Any]) -> bool:
        """Determine if adaptive strategy would be beneficial."""
        task_count = len(workflow_definition.get("tasks", []))
        has_complex_dependencies = len(workflow_definition.get("dependencies", [])) > task_count // 2
        
        # Adaptive strategy is good for:
        # - Medium to large workflows (5+ tasks)
        # - Complex dependency patterns
        # - Unknown or variable resource requirements
        return (task_count >= 5 or has_complex_dependencies or 
                workflow_definition.get("resource_requirements") is None)
    
    def _estimate_resource_requirements(self, 
                                      task_count: int,
                                      strategy: PlanningStrategy,
                                      workflow_definition: Dict[str, Any]) -> Dict[str, float]:
        """Estimate resource requirements for a given strategy."""
        base_memory = task_count * 50  # 50MB per task
        base_cpu = min(task_count * 10, 80)  # 10% per task, max 80%
        
        if strategy == PlanningStrategy.PARALLEL:
            # Parallel execution uses more resources
            return {
                "memory_mb": base_memory * 1.5,
                "cpu_percent": min(base_cpu * 1.3, 90),
                "concurrent_tasks": min(task_count, 4)
            }
        elif strategy == PlanningStrategy.RESOURCE_OPTIMIZED:
            # Resource optimized uses minimal resources
            return {
                "memory_mb": base_memory * 0.8,
                "cpu_percent": base_cpu * 0.7,
                "concurrent_tasks": 1
            }
        elif strategy == PlanningStrategy.TIME_OPTIMIZED:
            # Time optimized may use more resources for speed
            return {
                "memory_mb": base_memory * 1.2,
                "cpu_percent": min(base_cpu * 1.5, 95),
                "concurrent_tasks": min(task_count, 6)
            }
        else:
            # Sequential or adaptive - moderate resource usage
            return {
                "memory_mb": base_memory,
                "cpu_percent": base_cpu,
                "concurrent_tasks": 1 if strategy == PlanningStrategy.SEQUENTIAL else 2
            }