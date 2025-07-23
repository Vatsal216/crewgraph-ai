"""
AI-Driven Workflow Optimizer for CrewGraph AI

Intelligent workflow optimization using machine learning and heuristics.

Author: Vatsal216
Created: 2025-07-23 06:16:00 UTC
"""

import time
import copy
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
import threading

from .predictor import PerformancePredictor, ResourcePredictor
from .ml_models import MLModelManager, ModelType
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class OptimizationStrategy(Enum):
    """Optimization strategies available"""
    PERFORMANCE = "performance"  # Optimize for speed
    RESOURCE = "resource"        # Optimize for resource usage
    COST = "cost"               # Optimize for cost
    RELIABILITY = "reliability"  # Optimize for reliability
    BALANCED = "balanced"       # Balanced optimization

"""
Resource Optimizer for CrewGraph AI Intelligence Layer

This module provides intelligent resource allocation and optimization
for workflow execution, including CPU, memory, and task scheduling.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

import asyncio
import psutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..utils.logging import get_logger
from ..types import WorkflowId, TaskStatus

logger = get_logger(__name__)


class OptimizationType(Enum):
    """Types of optimizations available."""
    MEMORY = "memory"
    CPU = "cpu"
    CONCURRENCY = "concurrency"
    SCHEDULING = "scheduling"
    RESOURCE_ALLOCATION = "resource_allocation"


@dataclass
class OptimizationResult:
    """Results of workflow optimization"""
    original_workflow: Dict[str, Any]
    optimized_workflow: Dict[str, Any]
    optimization_strategy: OptimizationStrategy
    improvements: Dict[str, float]
    applied_optimizations: List[str]
    performance_gain: float
    resource_savings: float
    estimated_cost_reduction: float
    confidence_score: float
    execution_plan: Dict[str, Any]
    created_by: str = "Vatsal216"
    created_at: str = "2025-07-23 06:16:00"


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    optimization_type: str
    description: str
    impact_score: float
    difficulty: str  # "easy", "medium", "hard"
    estimated_improvement: Dict[str, float]
    implementation_steps: List[str]
    risks: List[str]
    dependencies: List[str]


class WorkflowOptimizer:
    """
    AI-driven workflow optimization engine.
    
    Uses machine learning models and heuristics to optimize workflows
    for performance, resource usage, cost, and reliability.
    
    Created by: Vatsal216
    Date: 2025-07-23 06:16:00 UTC
    """
    
    def __init__(self, 
                 performance_predictor: Optional[PerformancePredictor] = None,
                 resource_predictor: Optional[ResourcePredictor] = None):
        """
        Initialize workflow optimizer.
        
        Args:
            performance_predictor: Performance prediction component
            resource_predictor: Resource prediction component
        """
        self.performance_predictor = performance_predictor or PerformancePredictor()
        self.resource_predictor = resource_predictor or ResourcePredictor()
        
        self._optimization_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        logger.info("WorkflowOptimizer initialized")
        logger.info("User: Vatsal216, Time: 2025-07-23 06:16:00")
    
    def optimize_workflow(self, 
                         workflow_data: Dict[str, Any],
                         strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                         max_iterations: int = 5) -> OptimizationResult:
        """
        Optimize workflow using specified strategy.
        
        Args:
            workflow_data: Original workflow definition
            strategy: Optimization strategy to apply
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization results
        """
        with self._lock:
            start_time = time.time()
            
            # Create a copy for optimization
            optimized_workflow = copy.deepcopy(workflow_data)
            original_workflow = copy.deepcopy(workflow_data)
            
            # Get baseline predictions
            baseline_performance = self.performance_predictor.predict_performance(original_workflow)
            baseline_resources = self.resource_predictor.predict_resources(original_workflow)
            
            applied_optimizations = []
            best_workflow = optimized_workflow
            best_score = self._calculate_optimization_score(
                baseline_performance, baseline_resources, strategy
            )
            
            # Iterative optimization
            for iteration in range(max_iterations):
                logger.info(f"Optimization iteration {iteration + 1}/{max_iterations}")
                
                # Generate optimization recommendations
                recommendations = self._generate_optimization_recommendations(
                    optimized_workflow, strategy
                )
                
                # Apply best recommendation
                if recommendations:
                    best_recommendation = max(recommendations, key=lambda r: r.impact_score)
                    
                    # Apply optimization
                    candidate_workflow = self._apply_optimization(
                        optimized_workflow, best_recommendation
                    )
                    
                    # Evaluate candidate
                    candidate_performance = self.performance_predictor.predict_performance(candidate_workflow)
                    candidate_resources = self.resource_predictor.predict_resources(candidate_workflow)
                    candidate_score = self._calculate_optimization_score(
                        candidate_performance, candidate_resources, strategy
                    )
                    
                    # Accept if improvement
                    if candidate_score > best_score:
                        best_workflow = candidate_workflow
                        best_score = candidate_score
                        optimized_workflow = candidate_workflow
                        applied_optimizations.append(best_recommendation.optimization_type)
                        
                        logger.info(f"Applied optimization: {best_recommendation.optimization_type}")
                    else:
                        # No improvement, stop iterating
                        break
                else:
                    # No more recommendations
                    break
            
            # Calculate final improvements
            final_performance = self.performance_predictor.predict_performance(best_workflow)
            final_resources = self.resource_predictor.predict_resources(best_workflow)
            
            improvements = self._calculate_improvements(
                baseline_performance, baseline_resources,
                final_performance, final_resources
            )
            
            # Generate execution plan
            execution_plan = self._generate_execution_plan(best_workflow, strategy)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                original_workflow, best_workflow, applied_optimizations
            )
            
            result = OptimizationResult(
                original_workflow=original_workflow,
                optimized_workflow=best_workflow,
                optimization_strategy=strategy,
                improvements=improvements,
                applied_optimizations=applied_optimizations,
                performance_gain=improvements.get('execution_time_improvement', 0.0),
                resource_savings=improvements.get('resource_usage_improvement', 0.0),
                estimated_cost_reduction=improvements.get('cost_improvement', 0.0),
                confidence_score=confidence_score,
                execution_plan=execution_plan
            )
            
            # Record optimization
            optimization_time = time.time() - start_time
            self._record_optimization(result, optimization_time)
            
            metrics.record_metric("workflow_optimizations_total", 1.0)
            metrics.record_metric("optimization_time_seconds", optimization_time)
            
            logger.info(f"Workflow optimization completed in {optimization_time:.3f}s")
            logger.info(f"Applied {len(applied_optimizations)} optimizations")
            
            return result
    
    def _generate_optimization_recommendations(self, 
                                             workflow_data: Dict[str, Any],
                                             strategy: OptimizationStrategy) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on strategy"""
        recommendations = []
        
        # Analyze workflow structure
        tasks = workflow_data.get('tasks', [])
        agents = workflow_data.get('agents', [])
        edges = workflow_data.get('edges', [])
        
        # Task-level optimizations
        recommendations.extend(self._recommend_task_optimizations(tasks, strategy))
        
        # Agent-level optimizations
        recommendations.extend(self._recommend_agent_optimizations(agents, strategy))
        
        # Workflow structure optimizations
        recommendations.extend(self._recommend_structure_optimizations(tasks, edges, strategy))
        
        # Resource optimizations
        recommendations.extend(self._recommend_resource_optimizations(workflow_data, strategy))
        
        # Sort by impact score
        recommendations.sort(key=lambda r: r.impact_score, reverse=True)
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def _recommend_task_optimizations(self, 
                                    tasks: List[Dict[str, Any]], 
                                    strategy: OptimizationStrategy) -> List[OptimizationRecommendation]:
        """Recommend task-level optimizations"""
        recommendations = []
        
        # Find complex tasks that can be optimized
        complex_tasks = [task for task in tasks if task.get('complexity', 1.0) > 3.0]
        
        if complex_tasks and strategy in [OptimizationStrategy.PERFORMANCE, OptimizationStrategy.BALANCED]:
            recommendations.append(OptimizationRecommendation(
                optimization_type="task_decomposition",
                description="Break down complex tasks into smaller, parallel subtasks",
                impact_score=0.8,
                difficulty="medium",
                estimated_improvement={"execution_time": 0.3, "reliability": 0.2},
                implementation_steps=[
                    "Identify task boundaries",
                    "Create subtask definitions",
                    "Update dependencies",
                    "Test decomposed workflow"
                ],
                risks=["Increased coordination overhead", "Potential dependency issues"],
                dependencies=["Task analysis", "Dependency mapping"]
            ))
        
        # Find tasks with high resource usage
        resource_heavy_tasks = [
            task for task in tasks 
            if task.get('memory_mb', 0) > 1024 or task.get('cpu_cores', 0) > 2
        ]
        
        if resource_heavy_tasks and strategy in [OptimizationStrategy.RESOURCE, OptimizationStrategy.COST]:
            recommendations.append(OptimizationRecommendation(
                optimization_type="resource_optimization",
                description="Optimize resource usage for heavy tasks",
                impact_score=0.7,
                difficulty="easy",
                estimated_improvement={"resource_usage": 0.25, "cost": 0.2},
                implementation_steps=[
                    "Profile task resource usage",
                    "Implement memory optimization",
                    "Optimize algorithms",
                    "Add resource monitoring"
                ],
                risks=["Potential performance trade-offs"],
                dependencies=["Resource profiling tools"]
            ))
        
        return recommendations
    
    def _recommend_agent_optimizations(self, 
                                     agents: List[Dict[str, Any]], 
                                     strategy: OptimizationStrategy) -> List[OptimizationRecommendation]:
        """Recommend agent-level optimizations"""
        recommendations = []
        
        # Agent pooling optimization
        if len(agents) > 5 and strategy in [OptimizationStrategy.RESOURCE, OptimizationStrategy.COST]:
            recommendations.append(OptimizationRecommendation(
                optimization_type="agent_pooling",
                description="Implement agent pooling to reduce resource overhead",
                impact_score=0.6,
                difficulty="medium",
                estimated_improvement={"resource_usage": 0.3, "cost": 0.25},
                implementation_steps=[
                    "Design agent pool architecture",
                    "Implement pool management",
                    "Update agent lifecycle",
                    "Add pool monitoring"
                ],
                risks=["Increased complexity", "Pool contention"],
                dependencies=["Agent management system"]
            ))
        
        # Agent specialization
        if len(agents) > 3 and strategy in [OptimizationStrategy.PERFORMANCE, OptimizationStrategy.BALANCED]:
            recommendations.append(OptimizationRecommendation(
                optimization_type="agent_specialization",
                description="Specialize agents for specific task types",
                impact_score=0.7,
                difficulty="hard",
                estimated_improvement={"execution_time": 0.2, "reliability": 0.15},
                implementation_steps=[
                    "Analyze task patterns",
                    "Design specialized agents",
                    "Implement agent selection logic",
                    "Test specialized workflows"
                ],
                risks=["Reduced flexibility", "Maintenance overhead"],
                dependencies=["Task analysis", "Agent framework updates"]
            ))
        
        return recommendations
    
    def _recommend_structure_optimizations(self, 
                                         tasks: List[Dict[str, Any]], 
                                         edges: List[Dict[str, Any]], 
                                         strategy: OptimizationStrategy) -> List[OptimizationRecommendation]:
        """Recommend workflow structure optimizations"""
        recommendations = []
        
        # Parallelization opportunities
        sequential_chains = self._find_sequential_chains(tasks, edges)
        if sequential_chains and strategy in [OptimizationStrategy.PERFORMANCE, OptimizationStrategy.BALANCED]:
            recommendations.append(OptimizationRecommendation(
                optimization_type="parallelization",
                description="Parallelize independent task sequences",
                impact_score=0.9,
                difficulty="medium",
                estimated_improvement={"execution_time": 0.4, "throughput": 0.5},
                implementation_steps=[
                    "Identify independent task groups",
                    "Redesign workflow structure",
                    "Update dependency management",
                    "Test parallel execution"
                ],
                risks=["Resource contention", "Synchronization complexity"],
                dependencies=["Dependency analysis", "Parallel execution framework"]
            ))
        
        # Dependency optimization
        if len(edges) > len(tasks) * 1.5:  # High dependency ratio
            recommendations.append(OptimizationRecommendation(
                optimization_type="dependency_optimization",
                description="Reduce unnecessary dependencies",
                impact_score=0.6,
                difficulty="easy",
                estimated_improvement={"execution_time": 0.15, "complexity": 0.3},
                implementation_steps=[
                    "Analyze dependency graph",
                    "Identify redundant dependencies",
                    "Simplify workflow structure",
                    "Validate workflow correctness"
                ],
                risks=["Potential logic errors"],
                dependencies=["Dependency analysis tools"]
            ))
        
        return recommendations
    
    def _recommend_resource_optimizations(self, 
                                        workflow_data: Dict[str, Any], 
                                        strategy: OptimizationStrategy) -> List[OptimizationRecommendation]:
        """Recommend resource-level optimizations"""
        recommendations = []
        
        # Memory optimization
        if strategy in [OptimizationStrategy.RESOURCE, OptimizationStrategy.COST, OptimizationStrategy.BALANCED]:
            recommendations.append(OptimizationRecommendation(
                optimization_type="memory_optimization",
                description="Implement intelligent memory management",
                impact_score=0.7,
                difficulty="medium",
                estimated_improvement={"memory_usage": 0.3, "cost": 0.2},
                implementation_steps=[
                    "Implement data streaming",
                    "Add memory pooling",
                    "Optimize data structures",
                    "Add garbage collection tuning"
                ],
                risks=["Implementation complexity"],
                dependencies=["Memory profiling", "Data structure optimization"]
            ))
        
        # Caching optimization
        recommendations.append(OptimizationRecommendation(
            optimization_type="intelligent_caching",
            description="Implement smart caching strategies",
            impact_score=0.8,
            difficulty="easy",
            estimated_improvement={"execution_time": 0.25, "resource_usage": 0.15},
            implementation_steps=[
                "Identify cacheable operations",
                "Implement cache strategy",
                "Add cache invalidation",
                "Monitor cache performance"
            ],
            risks=["Cache consistency issues", "Memory overhead"],
            dependencies=["Cache infrastructure"]
        ))
        
        return recommendations
    
    def _apply_optimization(self, 
                          workflow_data: Dict[str, Any], 
                          recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """Apply optimization recommendation to workflow"""
        optimized_workflow = copy.deepcopy(workflow_data)
        
        if recommendation.optimization_type == "task_decomposition":
            optimized_workflow = self._apply_task_decomposition(optimized_workflow)
        elif recommendation.optimization_type == "parallelization":
            optimized_workflow = self._apply_parallelization(optimized_workflow)
        elif recommendation.optimization_type == "resource_optimization":
            optimized_workflow = self._apply_resource_optimization(optimized_workflow)
        elif recommendation.optimization_type == "dependency_optimization":
            optimized_workflow = self._apply_dependency_optimization(optimized_workflow)
        elif recommendation.optimization_type == "agent_pooling":
            optimized_workflow = self._apply_agent_pooling(optimized_workflow)
        elif recommendation.optimization_type == "intelligent_caching":
            optimized_workflow = self._apply_intelligent_caching(optimized_workflow)
        
        # Add optimization metadata
        if 'optimizations' not in optimized_workflow:
            optimized_workflow['optimizations'] = []
        
        optimized_workflow['optimizations'].append({
            "type": recommendation.optimization_type,
            "applied_at": time.time(),
            "created_by": "Vatsal216"
        })
        
        return optimized_workflow
    
    def _apply_task_decomposition(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply task decomposition optimization"""
        tasks = workflow_data.get('tasks', [])
        edges = workflow_data.get('edges', [])
        
        new_tasks = []
        new_edges = []
        task_id_counter = len(tasks)
        
        for task in tasks:
            if task.get('complexity', 1.0) > 3.0:
                # Decompose complex task into subtasks
                subtask_count = min(int(task.get('complexity', 1.0)), 4)
                subtasks = []
                
                for i in range(subtask_count):
                    subtask = copy.deepcopy(task)
                    subtask['id'] = f"{task.get('id', task_id_counter)}_sub_{i}"
                    subtask['complexity'] = task.get('complexity', 1.0) / subtask_count
                    subtask['memory_mb'] = task.get('memory_mb', 128) // subtask_count
                    subtask['parent_task'] = task.get('id', task_id_counter)
                    subtasks.append(subtask)
                    task_id_counter += 1
                
                new_tasks.extend(subtasks)
                
                # Create dependencies between subtasks (sequential)
                for i in range(len(subtasks) - 1):
                    new_edges.append({
                        'source': subtasks[i]['id'],
                        'target': subtasks[i + 1]['id'],
                        'type': 'decomposition'
                    })
                
                # Update existing edges
                for edge in edges:
                    if edge.get('source') == task.get('id'):
                        edge['source'] = subtasks[-1]['id']
                    if edge.get('target') == task.get('id'):
                        edge['target'] = subtasks[0]['id']
            else:
                new_tasks.append(task)
        
        workflow_data['tasks'] = new_tasks
        workflow_data['edges'] = edges + new_edges
        
        return workflow_data
    
    def _apply_parallelization(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parallelization optimization"""
        tasks = workflow_data.get('tasks', [])
        edges = workflow_data.get('edges', [])
        
        # Find independent task groups
        independent_groups = self._find_independent_task_groups(tasks, edges)
        
        # Add parallel execution hints
        for group in independent_groups:
            if len(group) > 1:
                for task_id in group:
                    for task in tasks:
                        if task.get('id') == task_id:
                            task['parallel_group'] = f"group_{hash(tuple(sorted(group))) % 1000}"
                            task['can_parallelize'] = True
        
        return workflow_data
    
    def _apply_resource_optimization(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply resource optimization"""
        tasks = workflow_data.get('tasks', [])
        
        for task in tasks:
            # Reduce memory requirements by 20%
            if 'memory_mb' in task:
                task['memory_mb'] = int(task['memory_mb'] * 0.8)
            
            # Optimize CPU usage
            if 'cpu_cores' in task:
                task['cpu_cores'] = max(0.25, task['cpu_cores'] * 0.9)
            
            # Add resource optimization flags
            task['resource_optimized'] = True
            task['optimization_level'] = 'standard'
        
        return workflow_data
    
    def _apply_dependency_optimization(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply dependency optimization"""
        edges = workflow_data.get('edges', [])
        
        # Remove redundant dependencies (simplified logic)
        optimized_edges = []
        dependency_map = {}
        
        # Build dependency map
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            
            if source not in dependency_map:
                dependency_map[source] = set()
            dependency_map[source].add(target)
        
        # Keep only direct dependencies
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            
            # Check if this is a direct dependency
            is_direct = True
            for intermediate in dependency_map.get(source, set()):
                if intermediate != target and target in dependency_map.get(intermediate, set()):
                    is_direct = False
                    break
            
            if is_direct:
                optimized_edges.append(edge)
        
        workflow_data['edges'] = optimized_edges
        
        return workflow_data
    
    def _apply_agent_pooling(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply agent pooling optimization"""
        agents = workflow_data.get('agents', [])
        
        # Group similar agents
        agent_types = {}
        for agent in agents:
            agent_type = agent.get('type', 'default')
            if agent_type not in agent_types:
                agent_types[agent_type] = []
            agent_types[agent_type].append(agent)
        
        # Create agent pools
        pooled_agents = []
        for agent_type, agent_list in agent_types.items():
            if len(agent_list) > 2:
                # Create a pool for this agent type
                pool_agent = {
                    'id': f"{agent_type}_pool",
                    'type': f"{agent_type}_pool",
                    'pool_size': len(agent_list),
                    'pool_members': [agent.get('id') for agent in agent_list],
                    'pooled': True
                }
                pooled_agents.append(pool_agent)
            else:
                pooled_agents.extend(agent_list)
        
        workflow_data['agents'] = pooled_agents
        
        return workflow_data
    
    def _apply_intelligent_caching(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply intelligent caching optimization"""
        tasks = workflow_data.get('tasks', [])
        
        for task in tasks:
            # Add caching for expensive operations
            if task.get('complexity', 1.0) > 2.0 or 'data' in task.get('type', '').lower():
                task['caching_enabled'] = True
                task['cache_strategy'] = 'lru'
                task['cache_ttl'] = 3600  # 1 hour
                task['cache_key_fields'] = ['input', 'parameters']
        
        # Add global cache configuration
        workflow_data['cache_config'] = {
            'enabled': True,
            'backend': 'redis',
            'max_size_mb': 1024,
            'compression': True
        }
        
        return workflow_data
    
    def _find_sequential_chains(self, tasks: List[Dict], edges: List[Dict]) -> List[List[str]]:
        """Find sequences of tasks that could be parallelized"""
        chains = []
        # Simplified implementation - would need more sophisticated analysis
        return chains
    
    def _find_independent_task_groups(self, tasks: List[Dict], edges: List[Dict]) -> List[Set[str]]:
        """Find groups of tasks that can run in parallel"""
        # Build dependency graph
        dependencies = {}
        dependents = {}
        
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            
            if target not in dependencies:
                dependencies[target] = set()
            dependencies[target].add(source)
            
            if source not in dependents:
                dependents[source] = set()
            dependents[source].add(target)
        
        # Find independent groups (simplified)
        task_ids = {task.get('id', i) for i, task in enumerate(tasks)}
        independent_tasks = task_ids - set(dependencies.keys()) - set(dependents.keys())
        
        # Group independent tasks
        groups = []
        if len(independent_tasks) > 1:
            groups.append(independent_tasks)
        
        return groups
    
    def _calculate_optimization_score(self, 
                                    performance_prediction,
                                    resource_prediction,
                                    strategy: OptimizationStrategy) -> float:
        """Calculate optimization score based on strategy"""
        
        if strategy == OptimizationStrategy.PERFORMANCE:
            # Focus on execution time and throughput
            time_score = 1000.0 / max(performance_prediction.execution_time, 1.0)
            return time_score
        
        elif strategy == OptimizationStrategy.RESOURCE:
            # Focus on resource efficiency
            memory_score = 1000.0 / max(resource_prediction.peak_memory_mb, 1.0)
            cpu_score = 100.0 / max(resource_prediction.peak_cpu_percent, 1.0)
            return (memory_score + cpu_score) / 2.0
        
        elif strategy == OptimizationStrategy.COST:
            # Focus on cost efficiency
            cost_score = 100.0 / max(performance_prediction.estimated_cost, 0.001)
            return cost_score
        
        elif strategy == OptimizationStrategy.RELIABILITY:
            # Focus on success probability
            reliability_score = performance_prediction.success_probability * 100.0
            return reliability_score
        
        else:  # BALANCED
            # Balanced scoring
            time_score = 100.0 / max(performance_prediction.execution_time, 1.0)
            memory_score = 100.0 / max(resource_prediction.peak_memory_mb, 1.0)
            cost_score = 10.0 / max(performance_prediction.estimated_cost, 0.001)
            reliability_score = performance_prediction.success_probability * 10.0
            
            return (time_score + memory_score + cost_score + reliability_score) / 4.0
    
    def _calculate_improvements(self, 
                              baseline_performance,
                              baseline_resources,
                              final_performance,
                              final_resources) -> Dict[str, float]:
        """Calculate improvements from optimization"""
        improvements = {}
        
        # Execution time improvement
        time_improvement = (
            baseline_performance.execution_time - final_performance.execution_time
        ) / baseline_performance.execution_time
        improvements['execution_time_improvement'] = max(time_improvement, 0.0)
        
        # Memory usage improvement
        memory_improvement = (
            baseline_resources.peak_memory_mb - final_resources.peak_memory_mb
        ) / baseline_resources.peak_memory_mb
        improvements['memory_usage_improvement'] = max(memory_improvement, 0.0)
        
        # CPU usage improvement
        cpu_improvement = (
            baseline_resources.peak_cpu_percent - final_resources.peak_cpu_percent
        ) / baseline_resources.peak_cpu_percent
        improvements['cpu_usage_improvement'] = max(cpu_improvement, 0.0)
        
        # Cost improvement
        cost_improvement = (
            baseline_performance.estimated_cost - final_performance.estimated_cost
        ) / baseline_performance.estimated_cost
        improvements['cost_improvement'] = max(cost_improvement, 0.0)
        
        # Overall resource improvement
        improvements['resource_usage_improvement'] = (
            improvements['memory_usage_improvement'] + 
            improvements['cpu_usage_improvement']
        ) / 2.0
        
        return improvements
    
    def _generate_execution_plan(self, 
                               workflow_data: Dict[str, Any], 
                               strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Generate optimized execution plan"""
        tasks = workflow_data.get('tasks', [])
        
        # Group tasks by parallel groups
        parallel_groups = {}
        sequential_tasks = []
        
        for task in tasks:
            if task.get('can_parallelize') and 'parallel_group' in task:
                group_id = task['parallel_group']
                if group_id not in parallel_groups:
                    parallel_groups[group_id] = []
                parallel_groups[group_id].append(task)
            else:
                sequential_tasks.append(task)
        
        execution_plan = {
            "strategy": strategy.value,
            "parallel_groups": parallel_groups,
            "sequential_tasks": sequential_tasks,
            "estimated_stages": len(parallel_groups) + len(sequential_tasks),
            "parallelization_factor": len(parallel_groups) / max(len(tasks), 1),
            "optimization_metadata": {
                "created_by": "Vatsal216",
                "created_at": "2025-07-23 06:16:00",
                "optimizer_version": "1.0.0"
            }
        }
        
        return execution_plan
    
    def _calculate_confidence_score(self, 
                                  original_workflow: Dict[str, Any],
                                  optimized_workflow: Dict[str, Any],
                                  applied_optimizations: List[str]) -> float:
        """Calculate confidence in optimization results"""
        
        # Base confidence
        base_confidence = 0.8
        
        # Reduce confidence for many changes
        change_penalty = min(len(applied_optimizations) * 0.05, 0.3)
        
        # Reduce confidence for complex workflows
        original_complexity = len(original_workflow.get('tasks', []))
        complexity_penalty = min(original_complexity / 50.0, 0.2)
        
        # Increase confidence for well-known optimizations
        known_optimizations = ['intelligent_caching', 'resource_optimization', 'dependency_optimization']
        known_bonus = sum(0.05 for opt in applied_optimizations if opt in known_optimizations)
        
        confidence = base_confidence - change_penalty - complexity_penalty + known_bonus
        
        return max(min(confidence, 1.0), 0.1)  # Clamp between 0.1 and 1.0
    
    def _record_optimization(self, result: OptimizationResult, optimization_time: float):
        """Record optimization for analysis and improvement"""
        record = {
            "timestamp": time.time(),
            "result": result.__dict__,
            "optimization_time": optimization_time,
            "created_by": "Vatsal216"
        }
        
        self._optimization_history.append(record)
        
        # Keep only last 1000 optimizations
        if len(self._optimization_history) > 1000:
            self._optimization_history = self._optimization_history[-1000:]
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics about optimization performance"""
        if not self._optimization_history:
            return {"total_optimizations": 0}
        
        optimization_times = [o["optimization_time"] for o in self._optimization_history]
        performance_gains = [
            o["result"]["performance_gain"] for o in self._optimization_history
        ]
        
        return {
            "total_optimizations": len(self._optimization_history),
            "avg_optimization_time": sum(optimization_times) / len(optimization_times),
            "avg_performance_gain": sum(performance_gains) / len(performance_gains),
            "success_rate": len([g for g in performance_gains if g > 0]) / len(performance_gains),
            "created_by": "Vatsal216",
            "timestamp": time.time(),
            """Result of resource optimization analysis."""
            "optimization_type": OptimizationType,
            "current_utilization": float,
            "recommended_utilization": float,
            "potential_improvement": float,
            "implementation_steps": List[str],
            "confidence_score": float,
            "estimated_savings": Dict[str, float]  # time, memory, cpu savings
            }


class ResourceOptimizer:
    """
    Intelligent resource optimizer for CrewGraph AI workflows.
    
    Provides real-time resource monitoring, allocation optimization,
    and performance tuning recommendations.
    """
    
    def __init__(self, max_cpu_threshold: float = 80.0, max_memory_threshold: float = 85.0):
        """
        Initialize the resource optimizer.
        
        Args:
            max_cpu_threshold: Maximum CPU usage threshold (percentage)
            max_memory_threshold: Maximum memory usage threshold (percentage)
        """
        self.max_cpu_threshold = max_cpu_threshold
        self.max_memory_threshold = max_memory_threshold
        self.optimization_history: List[OptimizationResult] = []
        self.resource_snapshots: List[Dict[str, float]] = []
        
        logger.info(f"ResourceOptimizer initialized with CPU threshold: {max_cpu_threshold}%, "
                   f"Memory threshold: {max_memory_threshold}%")
    
    def get_current_resource_usage(self) -> Dict[str, float]:
        """
        Get current system resource usage.
        
        Returns:
            Dictionary containing current resource utilization
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            usage = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "timestamp": datetime.now().timestamp()
            }
            
            # Store snapshot for trend analysis
            self.resource_snapshots.append(usage)
            if len(self.resource_snapshots) > 100:  # Keep last 100 snapshots
                self.resource_snapshots.pop(0)
            
            return usage
            
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {"error": str(e)}
    
    def optimize_memory_usage(self, 
                            current_workflows: List[WorkflowId]) -> OptimizationResult:
        """
        Analyze and optimize memory usage for active workflows.
        
        Args:
            current_workflows: List of currently active workflow IDs
            
        Returns:
            Optimization result with memory recommendations
        """
        current_usage = self.get_current_resource_usage()
        memory_percent = current_usage.get("memory_percent", 0)
        
        recommendations = []
        potential_improvement = 0.0
        confidence = 0.8
        
        if memory_percent > self.max_memory_threshold:
            recommendations.extend([
                "Implement workflow batching to reduce concurrent memory usage",
                "Add memory pooling for task execution",
                "Consider increasing available system memory",
                "Implement lazy loading for large datasets"
            ])
            potential_improvement = min(30.0, memory_percent - self.max_memory_threshold)
            
        elif memory_percent > 70.0:
            recommendations.extend([
                "Monitor memory usage trends",
                "Implement memory optimization in data-heavy tasks",
                "Consider workflow scheduling adjustments"
            ])
            potential_improvement = 10.0
            confidence = 0.6
            
        else:
            recommendations.append("Memory usage is optimal")
            potential_improvement = 5.0
            confidence = 0.9
        
        result = OptimizationResult(
            optimization_type=OptimizationType.MEMORY,
            current_utilization=memory_percent,
            recommended_utilization=max(50.0, memory_percent - potential_improvement),
            potential_improvement=potential_improvement,
            implementation_steps=recommendations,
            confidence_score=confidence,
            estimated_savings={
                "memory_mb": potential_improvement * 10,  # Rough estimate
                "execution_time_seconds": potential_improvement * 0.5
            }
        )
        
        self.optimization_history.append(result)
        logger.info(f"Memory optimization analysis complete: {potential_improvement:.1f}% potential improvement")
        
        return result
    
    def optimize_cpu_usage(self, 
                         task_count: int, 
                         parallel_execution: bool = True) -> OptimizationResult:
        """
        Analyze and optimize CPU usage for workflow execution.
        
        Args:
            task_count: Number of tasks to execute
            parallel_execution: Whether parallel execution is enabled
            
        Returns:
            Optimization result with CPU recommendations
        """
        current_usage = self.get_current_resource_usage()
        cpu_percent = current_usage.get("cpu_percent", 0)
        
        # Get CPU core count for optimization calculations
        cpu_cores = psutil.cpu_count(logical=True)
        optimal_concurrency = min(task_count, cpu_cores)
        
        recommendations = []
        potential_improvement = 0.0
        confidence = 0.7
        
        if cpu_percent > self.max_cpu_threshold:
            recommendations.extend([
                f"Reduce task concurrency to {max(1, optimal_concurrency // 2)} parallel tasks",
                "Implement task queuing to prevent CPU overload",
                "Consider distributing workload across multiple machines",
                "Add CPU usage monitoring and throttling"
            ])
            potential_improvement = min(25.0, cpu_percent - self.max_cpu_threshold)
            
        elif not parallel_execution and task_count > 1:
            recommendations.extend([
                f"Enable parallel execution with {optimal_concurrency} concurrent tasks",
                "Implement async task execution where possible",
                "Use multiprocessing for CPU-intensive tasks"
            ])
            potential_improvement = min(40.0, (100.0 - cpu_percent) * 0.3)
            confidence = 0.8
            
        else:
            recommendations.append("CPU usage is within optimal range")
            potential_improvement = 5.0
            confidence = 0.9
        
        result = OptimizationResult(
            optimization_type=OptimizationType.CPU,
            current_utilization=cpu_percent,
            recommended_utilization=max(30.0, cpu_percent - potential_improvement),
            potential_improvement=potential_improvement,
            implementation_steps=recommendations,
            confidence_score=confidence,
            estimated_savings={
                "cpu_percent": potential_improvement,
                "execution_time_seconds": potential_improvement * 0.8
            }
        )
        
        self.optimization_history.append(result)
        logger.info(f"CPU optimization analysis complete: {potential_improvement:.1f}% potential improvement")
        
        return result
    
    def optimize_task_scheduling(self, 
                               tasks: List[Dict[str, Any]]) -> OptimizationResult:
        """
        Optimize task scheduling based on resource requirements and dependencies.
        
        Args:
            tasks: List of task definitions with resource requirements
            
        Returns:
            Optimization result with scheduling recommendations
        """
        if not tasks:
            return OptimizationResult(
                optimization_type=OptimizationType.SCHEDULING,
                current_utilization=0.0,
                recommended_utilization=0.0,
                potential_improvement=0.0,
                implementation_steps=["No tasks to optimize"],
                confidence_score=1.0,
                estimated_savings={}
            )
        
        # Analyze task characteristics
        cpu_intensive_tasks = []
        memory_intensive_tasks = []
        io_intensive_tasks = []
        
        for task in tasks:
            task_type = task.get("type", "unknown")
            if "cpu" in task_type.lower() or "compute" in task_type.lower():
                cpu_intensive_tasks.append(task)
            elif "memory" in task_type.lower() or "data" in task_type.lower():
                memory_intensive_tasks.append(task)
            else:
                io_intensive_tasks.append(task)
        
        recommendations = []
        current_efficiency = len(tasks) / max(1, len(tasks))  # Base efficiency
        
        if len(cpu_intensive_tasks) > psutil.cpu_count():
            recommendations.append(f"Serialize CPU-intensive tasks to prevent overload "
                                 f"({len(cpu_intensive_tasks)} tasks, {psutil.cpu_count()} cores)")
            current_efficiency *= 0.8
        
        if len(memory_intensive_tasks) > 2:
            recommendations.append("Batch memory-intensive tasks to prevent memory exhaustion")
            current_efficiency *= 0.9
        
        if len(io_intensive_tasks) > 5:
            recommendations.append("Group I/O tasks to improve throughput")
        
        # Calculate potential improvement
        potential_improvement = (1.0 - current_efficiency) * 100
        
        if not recommendations:
            recommendations.append("Task scheduling is optimal")
            potential_improvement = 5.0
        
        result = OptimizationResult(
            optimization_type=OptimizationType.SCHEDULING,
            current_utilization=current_efficiency * 100,
            recommended_utilization=min(95.0, current_efficiency * 100 + potential_improvement),
            potential_improvement=potential_improvement,
            implementation_steps=recommendations,
            confidence_score=0.7,
            estimated_savings={
                "execution_time_seconds": potential_improvement * 2,
                "resource_efficiency_percent": potential_improvement
            }
        )
        
        self.optimization_history.append(result)
        logger.info(f"Task scheduling optimization complete: {potential_improvement:.1f}% potential improvement")
        
        return result
    
    def get_resource_allocation_strategy(self, 
                                       workflow_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate optimal resource allocation strategy for a workflow.
        
        Args:
            workflow_requirements: Workflow resource requirements and constraints
            
        Returns:
            Resource allocation strategy with recommendations
        """
        current_resources = self.get_current_resource_usage()
        
        # Extract requirements
        required_memory = workflow_requirements.get("memory_gb", 1.0)
        required_cpu_cores = workflow_requirements.get("cpu_cores", 1)
        estimated_duration = workflow_requirements.get("duration_minutes", 10)
        
        # Calculate available resources
        available_memory = current_resources.get("memory_available_gb", 4.0)
        available_cpu = 100 - current_resources.get("cpu_percent", 50)
        
        strategy = {
            "can_execute": True,
            "recommended_concurrency": 1,
            "memory_allocation_gb": required_memory,
            "cpu_allocation_percent": min(50, available_cpu * 0.8),
            "scheduling_priority": "normal",
            "optimization_recommendations": []
        }
        
        # Check resource constraints
        if required_memory > available_memory * 0.8:
            strategy["can_execute"] = False
            strategy["optimization_recommendations"].append(
                f"Insufficient memory: requires {required_memory:.1f}GB, "
                f"available {available_memory:.1f}GB"
            )
        
        if required_cpu_cores > psutil.cpu_count():
            strategy["optimization_recommendations"].append(
                f"CPU requirement ({required_cpu_cores} cores) exceeds available cores "
                f"({psutil.cpu_count()})"
            )
            strategy["recommended_concurrency"] = max(1, psutil.cpu_count() // 2)
        
        # Optimize based on current load
        if current_resources.get("cpu_percent", 0) > 70:
            strategy["scheduling_priority"] = "low"
            strategy["optimization_recommendations"].append("System under high load - consider delayed execution")
        
        if current_resources.get("memory_percent", 0) > 80:
            strategy["memory_allocation_gb"] = min(required_memory, available_memory * 0.5)
            strategy["optimization_recommendations"].append("Memory pressure detected - reducing allocation")
        
        return strategy
    
    async def monitor_resource_trends(self, 
                                    duration_minutes: int = 10) -> Dict[str, List[float]]:
        """
        Monitor resource usage trends over time.
        
        Args:
            duration_minutes: How long to monitor (in minutes)
            
        Returns:
            Dictionary containing resource usage trends
        """
        trends = {
            "cpu_usage": [],
            "memory_usage": [],
            "timestamps": []
        }
        
        interval_seconds = 30  # Sample every 30 seconds
        total_samples = duration_minutes * 60 // interval_seconds
        
        logger.info(f"Starting resource trend monitoring for {duration_minutes} minutes")
        
        for i in range(total_samples):
            usage = self.get_current_resource_usage()
            trends["cpu_usage"].append(usage.get("cpu_percent", 0))
            trends["memory_usage"].append(usage.get("memory_percent", 0))
            trends["timestamps"].append(usage.get("timestamp", 0))
            
            if i < total_samples - 1:  # Don't sleep after last sample
                await asyncio.sleep(interval_seconds)
        
        # Calculate trend statistics
        if trends["cpu_usage"]:
            cpu_avg = sum(trends["cpu_usage"]) / len(trends["cpu_usage"])
            cpu_max = max(trends["cpu_usage"])
            memory_avg = sum(trends["memory_usage"]) / len(trends["memory_usage"])
            memory_max = max(trends["memory_usage"])
            
            logger.info(f"Resource monitoring complete - CPU avg: {cpu_avg:.1f}%, "
                       f"max: {cpu_max:.1f}%, Memory avg: {memory_avg:.1f}%, max: {memory_max:.1f}%")
        
        return trends
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get history of optimization results."""
        return self.optimization_history.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics for the optimizer."""
        if not self.optimization_history:
            return {"total_optimizations": 0, "average_improvement": 0.0}
        
        improvements = [opt.potential_improvement for opt in self.optimization_history]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "average_improvement": sum(improvements) / len(improvements),
            "optimization_types": list(set(opt.optimization_type.value for opt in self.optimization_history)),
            "last_optimization": self.optimization_history[-1].optimization_type.value,
            "optimizer_version": "1.0.0"
        }