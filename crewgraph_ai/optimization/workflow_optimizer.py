"""
Workflow Optimizer for CrewGraph AI

Provides comprehensive workflow optimization capabilities including
structure optimization, task scheduling, and performance tuning.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import copy

from ..utils.logging import get_logger
from ..types import WorkflowId

logger = get_logger(__name__)


class OptimizationType(Enum):
    """Types of optimizations available."""
    STRUCTURE = "structure"
    PARALLELIZATION = "parallelization"
    RESOURCE_ALLOCATION = "resource_allocation"
    SCHEDULING = "scheduling"
    CACHING = "caching"


@dataclass
class OptimizationResult:
    """Result of workflow optimization."""
    optimization_type: OptimizationType
    original_workflow: Dict[str, Any]
    optimized_workflow: Dict[str, Any]
    performance_improvement: float  # Percentage improvement
    implementation_steps: List[str]
    estimated_savings: Dict[str, float]
    confidence_score: float


class WorkflowOptimizer:
    """
    Comprehensive workflow optimizer that analyzes workflow structure
    and applies various optimization techniques to improve performance.
    """
    
    def __init__(self):
        """Initialize the workflow optimizer."""
        self.optimization_history: List[OptimizationResult] = []
        logger.info("WorkflowOptimizer initialized")
    
    def optimize_workflow(self, 
                         workflow_definition: Dict[str, Any],
                         optimization_goals: Optional[List[str]] = None,
                         constraints: Optional[Dict[str, Any]] = None) -> List[OptimizationResult]:
        """
        Optimize a workflow based on goals and constraints.
        
        Args:
            workflow_definition: Original workflow definition
            optimization_goals: List of optimization objectives
            constraints: Optimization constraints (time, cost, resources)
            
        Returns:
            List of optimization results
        """
        if optimization_goals is None:
            optimization_goals = ["performance", "resource_efficiency"]
        
        optimizations = []
        
        # Apply different optimization techniques
        if "performance" in optimization_goals:
            perf_opt = self._optimize_for_performance(workflow_definition, constraints)
            if perf_opt:
                optimizations.append(perf_opt)
        
        if "resource_efficiency" in optimization_goals:
            resource_opt = self._optimize_for_resources(workflow_definition, constraints)
            if resource_opt:
                optimizations.append(resource_opt)
        
        if "parallelization" in optimization_goals:
            parallel_opt = self._optimize_parallelization(workflow_definition)
            if parallel_opt:
                optimizations.append(parallel_opt)
        
        # Store optimization history
        self.optimization_history.extend(optimizations)
        
        logger.info(f"Applied {len(optimizations)} optimizations to workflow")
        return optimizations
    
    def suggest_optimizations(self, 
                            workflow_definition: Dict[str, Any],
                            execution_history: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Suggest potential optimizations based on workflow analysis.
        
        Args:
            workflow_definition: Workflow to analyze
            execution_history: Historical execution data
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])
        
        # Analyze structure for optimization opportunities
        if len(tasks) > 5:
            parallel_tasks = self._identify_parallelizable_tasks(tasks, dependencies)
            if len(parallel_tasks) > 1:
                suggestions.append({
                    "type": "parallelization",
                    "description": f"Parallelize {len(parallel_tasks)} independent tasks",
                    "potential_improvement": min(50.0, len(parallel_tasks) * 15.0),
                    "complexity": "medium"
                })
        
        # Check for caching opportunities
        repetitive_tasks = self._identify_repetitive_tasks(tasks)
        if repetitive_tasks:
            suggestions.append({
                "type": "caching",
                "description": f"Cache results of {len(repetitive_tasks)} repetitive tasks",
                "potential_improvement": 30.0,
                "complexity": "low"
            })
        
        # Resource optimization suggestions
        resource_intensive_tasks = [t for t in tasks 
                                  if t.get("estimated_duration", 0) > 20]
        if len(resource_intensive_tasks) > 2:
            suggestions.append({
                "type": "resource_scheduling",
                "description": "Optimize scheduling of resource-intensive tasks",
                "potential_improvement": 25.0,
                "complexity": "medium"
            })
        
        # Historical performance analysis
        if execution_history:
            history_suggestions = self._analyze_execution_history(execution_history)
            suggestions.extend(history_suggestions)
        
        return suggestions
    
    def apply_optimization(self, 
                          workflow_definition: Dict[str, Any],
                          optimization_result: OptimizationResult) -> Dict[str, Any]:
        """
        Apply an optimization result to a workflow.
        
        Args:
            workflow_definition: Original workflow
            optimization_result: Optimization to apply
            
        Returns:
            Optimized workflow definition
        """
        optimized_workflow = copy.deepcopy(workflow_definition)
        
        if optimization_result.optimization_type == OptimizationType.PARALLELIZATION:
            optimized_workflow = self._apply_parallelization(optimized_workflow, optimization_result)
        
        elif optimization_result.optimization_type == OptimizationType.RESOURCE_ALLOCATION:
            optimized_workflow = self._apply_resource_optimization(optimized_workflow, optimization_result)
        
        elif optimization_result.optimization_type == OptimizationType.STRUCTURE:
            optimized_workflow = self._apply_structure_optimization(optimized_workflow, optimization_result)
        
        elif optimization_result.optimization_type == OptimizationType.CACHING:
            optimized_workflow = self._apply_caching_optimization(optimized_workflow, optimization_result)
        
        logger.info(f"Applied {optimization_result.optimization_type.value} optimization")
        return optimized_workflow
    
    def get_optimization_impact(self, 
                              original_workflow: Dict[str, Any],
                              optimized_workflow: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate the impact of optimization changes.
        
        Args:
            original_workflow: Original workflow definition
            optimized_workflow: Optimized workflow definition
            
        Returns:
            Dictionary of impact metrics
        """
        impact = {}
        
        # Task count comparison
        orig_tasks = len(original_workflow.get("tasks", []))
        opt_tasks = len(optimized_workflow.get("tasks", []))
        impact["task_count_change"] = ((opt_tasks - orig_tasks) / max(orig_tasks, 1)) * 100
        
        # Dependency changes
        orig_deps = len(original_workflow.get("dependencies", []))
        opt_deps = len(optimized_workflow.get("dependencies", []))
        impact["dependency_change"] = ((opt_deps - orig_deps) / max(orig_deps, 1)) * 100
        
        # Estimated execution time
        orig_time = self._estimate_execution_time(original_workflow)
        opt_time = self._estimate_execution_time(optimized_workflow)
        impact["execution_time_improvement"] = ((orig_time - opt_time) / max(orig_time, 1)) * 100
        
        # Parallelization potential
        orig_parallel = self._calculate_parallelization_score(original_workflow)
        opt_parallel = self._calculate_parallelization_score(optimized_workflow)
        impact["parallelization_improvement"] = opt_parallel - orig_parallel
        
        return impact
    
    def _optimize_for_performance(self, 
                                workflow_definition: Dict[str, Any],
                                constraints: Optional[Dict[str, Any]]) -> Optional[OptimizationResult]:
        """Optimize workflow for performance."""
        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])
        
        # Find performance bottlenecks
        bottlenecks = self._identify_bottlenecks(tasks, dependencies)
        
        if not bottlenecks:
            return None
        
        optimized_workflow = copy.deepcopy(workflow_definition)
        
        # Optimize identified bottlenecks
        implementation_steps = []
        performance_improvement = 0.0
        
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "sequential_chain":
                # Break up long sequential chains
                implementation_steps.append(f"Parallelize tasks in chain: {bottleneck['tasks']}")
                performance_improvement += 20.0
            
            elif bottleneck["type"] == "resource_contention":
                # Add resource scheduling
                implementation_steps.append("Implement resource scheduling for intensive tasks")
                performance_improvement += 15.0
        
        return OptimizationResult(
            optimization_type=OptimizationType.STRUCTURE,
            original_workflow=workflow_definition,
            optimized_workflow=optimized_workflow,
            performance_improvement=min(performance_improvement, 60.0),
            implementation_steps=implementation_steps,
            estimated_savings={"execution_time_seconds": performance_improvement * 2},
            confidence_score=0.8
        )
    
    def _optimize_for_resources(self, 
                              workflow_definition: Dict[str, Any],
                              constraints: Optional[Dict[str, Any]]) -> Optional[OptimizationResult]:
        """Optimize workflow for resource efficiency."""
        tasks = workflow_definition.get("tasks", [])
        
        # Identify resource-intensive tasks
        resource_intensive = [t for t in tasks 
                            if t.get("estimated_duration", 0) > 15 or 
                               t.get("type") in ["data_processing", "ml"]]
        
        if not resource_intensive:
            return None
        
        optimized_workflow = copy.deepcopy(workflow_definition)
        
        # Add resource optimization metadata
        for task in optimized_workflow.get("tasks", []):
            if task in resource_intensive:
                task["resource_optimization"] = {
                    "priority": "low" if len(resource_intensive) > 3 else "normal",
                    "batch_processing": True,
                    "memory_limit": "moderate"
                }
        
        implementation_steps = [
            "Implement batched processing for resource-intensive tasks",
            "Add memory and CPU limits to prevent resource exhaustion",
            "Schedule resource-intensive tasks during off-peak hours"
        ]
        
        return OptimizationResult(
            optimization_type=OptimizationType.RESOURCE_ALLOCATION,
            original_workflow=workflow_definition,
            optimized_workflow=optimized_workflow,
            performance_improvement=25.0,
            implementation_steps=implementation_steps,
            estimated_savings={"resource_cost": 30.0, "memory_mb": 200.0},
            confidence_score=0.75
        )
    
    def _optimize_parallelization(self, workflow_definition: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Optimize workflow for parallel execution."""
        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])
        
        parallel_groups = self._identify_parallel_groups(tasks, dependencies)
        
        if len(parallel_groups) <= 1:
            return None
        
        optimized_workflow = copy.deepcopy(workflow_definition)
        
        # Add parallelization metadata
        for i, group in enumerate(parallel_groups):
            for task_id in group:
                # Find task and add parallel group info
                for task in optimized_workflow.get("tasks", []):
                    if task.get("id") == task_id:
                        task["parallel_group"] = f"group_{i}"
                        task["can_parallel"] = True
        
        implementation_steps = [
            f"Create {len(parallel_groups)} parallel execution groups",
            "Implement concurrent task execution within groups",
            "Add synchronization points between groups"
        ]
        
        improvement = min(40.0, len(parallel_groups) * 15.0)
        
        return OptimizationResult(
            optimization_type=OptimizationType.PARALLELIZATION,
            original_workflow=workflow_definition,
            optimized_workflow=optimized_workflow,
            performance_improvement=improvement,
            implementation_steps=implementation_steps,
            estimated_savings={"execution_time_seconds": improvement * 3},
            confidence_score=0.85
        )
    
    def _identify_parallelizable_tasks(self, 
                                     tasks: List[Dict[str, Any]], 
                                     dependencies: List[Dict[str, str]]) -> List[str]:
        """Identify tasks that can be executed in parallel."""
        # Build dependency graph
        dependent_tasks = set()
        for dep in dependencies:
            dependent_tasks.add(dep.get("target"))
        
        # Tasks without dependencies can potentially run in parallel
        independent_tasks = []
        for task in tasks:
            task_id = task.get("id")
            if task_id and task_id not in dependent_tasks:
                independent_tasks.append(task_id)
        
        return independent_tasks
    
    def _identify_repetitive_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify tasks that might benefit from caching."""
        repetitive = []
        
        # Look for tasks with similar descriptions or types
        task_signatures = {}
        for task in tasks:
            signature = f"{task.get('type', 'unknown')}_{len(task.get('description', ''))}"
            if signature in task_signatures:
                repetitive.append(task)
            else:
                task_signatures[signature] = task
        
        return repetitive
    
    def _identify_bottlenecks(self, 
                            tasks: List[Dict[str, Any]], 
                            dependencies: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in the workflow."""
        bottlenecks = []
        
        # Long sequential chains
        chains = self._find_sequential_chains(tasks, dependencies)
        for chain in chains:
            if len(chain) > 4:
                bottlenecks.append({
                    "type": "sequential_chain",
                    "severity": "high" if len(chain) > 8 else "medium",
                    "tasks": chain,
                    "description": f"Sequential chain of {len(chain)} tasks"
                })
        
        # Resource contention
        resource_intensive = [t for t in tasks if t.get("estimated_duration", 0) > 20]
        if len(resource_intensive) > 3:
            bottlenecks.append({
                "type": "resource_contention",
                "severity": "medium",
                "tasks": [t.get("id") for t in resource_intensive],
                "description": f"{len(resource_intensive)} resource-intensive tasks"
            })
        
        return bottlenecks
    
    def _find_sequential_chains(self, 
                              tasks: List[Dict[str, Any]], 
                              dependencies: List[Dict[str, str]]) -> List[List[str]]:
        """Find sequential chains of tasks."""
        # Simple implementation - find linear dependency chains
        chains = []
        
        # Build adjacency list
        graph = {}
        for task in tasks:
            graph[task.get("id")] = []
        
        for dep in dependencies:
            source = dep.get("source")
            target = dep.get("target")
            if source in graph:
                graph[source].append(target)
        
        # Find chains (linear paths)
        visited = set()
        for task_id in graph:
            if task_id not in visited:
                chain = self._build_chain(task_id, graph, visited)
                if len(chain) > 1:
                    chains.append(chain)
        
        return chains
    
    def _build_chain(self, start: str, graph: Dict[str, List[str]], visited: set) -> List[str]:
        """Build a chain starting from a task."""
        chain = []
        current = start
        
        while current and current not in visited:
            chain.append(current)
            visited.add(current)
            
            # Continue if there's exactly one successor
            successors = graph.get(current, [])
            if len(successors) == 1:
                current = successors[0]
            else:
                break
        
        return chain
    
    def _identify_parallel_groups(self, 
                                tasks: List[Dict[str, Any]], 
                                dependencies: List[Dict[str, str]]) -> List[List[str]]:
        """Identify groups of tasks that can run in parallel."""
        # Simple grouping - tasks at the same dependency level
        groups = []
        
        # Build reverse dependency map
        predecessors = {}
        for task in tasks:
            predecessors[task.get("id")] = []
        
        for dep in dependencies:
            target = dep.get("target")
            source = dep.get("source")
            if target in predecessors:
                predecessors[target].append(source)
        
        # Group by dependency depth
        depth_groups = {}
        for task_id, preds in predecessors.items():
            depth = len(preds)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(task_id)
        
        # Return groups with more than one task
        for depth, task_ids in depth_groups.items():
            if len(task_ids) > 1:
                groups.append(task_ids)
        
        return groups
    
    def _estimate_execution_time(self, workflow_definition: Dict[str, Any]) -> float:
        """Estimate total execution time for a workflow."""
        tasks = workflow_definition.get("tasks", [])
        
        # Simple estimation based on task durations
        total_time = sum(task.get("estimated_duration", 10) for task in tasks)
        
        # Adjust for potential parallelization
        parallel_groups = self._identify_parallel_groups(
            tasks, workflow_definition.get("dependencies", [])
        )
        
        if parallel_groups:
            # Assume some parallelization benefit
            parallel_reduction = len(parallel_groups) * 0.3
            total_time *= (1.0 - min(parallel_reduction, 0.5))
        
        return total_time
    
    def _calculate_parallelization_score(self, workflow_definition: Dict[str, Any]) -> float:
        """Calculate how well parallelized a workflow is."""
        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])
        
        if not tasks:
            return 0.0
        
        parallel_groups = self._identify_parallel_groups(tasks, dependencies)
        parallelizable_tasks = sum(len(group) for group in parallel_groups if len(group) > 1)
        
        return parallelizable_tasks / len(tasks)
    
    def _analyze_execution_history(self, execution_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze execution history for optimization suggestions."""
        suggestions = []
        
        if not execution_history:
            return suggestions
        
        # Analyze execution times
        execution_times = [h.get("execution_time", 0) for h in execution_history]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            
            if max_time > avg_time * 2:
                suggestions.append({
                    "type": "performance_stabilization",
                    "description": "High variance in execution times detected",
                    "potential_improvement": 20.0,
                    "complexity": "medium"
                })
        
        # Analyze failure patterns
        failure_rates = [1.0 - h.get("success_rate", 1.0) for h in execution_history]
        if failure_rates and sum(failure_rates) / len(failure_rates) > 0.1:
            suggestions.append({
                "type": "reliability_improvement",
                "description": "Implement enhanced error handling and retry logic",
                "potential_improvement": 15.0,
                "complexity": "low"
            })
        
        return suggestions
    
    def _apply_parallelization(self, workflow: Dict[str, Any], optimization: OptimizationResult) -> Dict[str, Any]:
        """Apply parallelization optimization to workflow."""
        # Implementation would modify the workflow structure for parallel execution
        # For now, just add metadata
        workflow["optimization_applied"] = "parallelization"
        workflow["parallel_execution"] = True
        return workflow
    
    def _apply_resource_optimization(self, workflow: Dict[str, Any], optimization: OptimizationResult) -> Dict[str, Any]:
        """Apply resource optimization to workflow."""
        workflow["optimization_applied"] = "resource_allocation"
        workflow["resource_optimized"] = True
        return workflow
    
    def _apply_structure_optimization(self, workflow: Dict[str, Any], optimization: OptimizationResult) -> Dict[str, Any]:
        """Apply structure optimization to workflow."""
        workflow["optimization_applied"] = "structure"
        workflow["structure_optimized"] = True
        return workflow
    
    def _apply_caching_optimization(self, workflow: Dict[str, Any], optimization: OptimizationResult) -> Dict[str, Any]:
        """Apply caching optimization to workflow."""
        workflow["optimization_applied"] = "caching"
        workflow["caching_enabled"] = True
        return workflow