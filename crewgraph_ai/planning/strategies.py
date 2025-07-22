"""
Planning strategy implementations
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .base import PlanningStrategy, ExecutionPlan, PlanNode, PlanEdge, NodeType, EdgeType
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SequentialStrategy(PlanningStrategy):
    """Sequential execution strategy - tasks run one after another"""
    
    def create_plan(self,
                   tasks: List[Any],
                   state: Any,
                   task_analysis: Dict[str, Any],
                   resource_analysis: Dict[str, Any],
                   constraints: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """Create sequential execution plan"""
        
        plan = ExecutionPlan(
            name="sequential_plan",
            constraints=constraints,
            metadata={
                'strategy': 'sequential',
                'task_count': len(tasks),
                'created_by': 'SequentialStrategy'
            }
        )
        
        # Create nodes for each task
        prev_node_id = None
        total_duration = 0.0
        
        for i, task in enumerate(tasks):
            # Create plan node
            node = PlanNode(
                task_name=task.name,
                node_type=NodeType.TASK,
                priority=i + 1,
                estimated_duration=task_analysis.get('estimated_total_time', 30.0) / len(tasks),
                task_wrapper=task
            )
            
            plan.nodes.append(node)
            total_duration += node.estimated_duration
            
            # Create edge from previous task
            if prev_node_id:
                edge = PlanEdge(
                    from_node=prev_node_id,
                    to_node=node.id,
                    edge_type=EdgeType.SEQUENTIAL
                )
                plan.edges.append(edge)
            
            prev_node_id = node.id
        
        plan.estimated_total_duration = total_duration
        
        logger.info(f"Sequential plan created with {len(tasks)} tasks")
        return plan
    
    def get_strategy_name(self) -> str:
        return "sequential"


class ParallelStrategy(PlanningStrategy):
    """Parallel execution strategy - independent tasks run simultaneously"""
    
    def __init__(self, max_parallel: int = 5):
        self.max_parallel = max_parallel
    
    def create_plan(self,
                   tasks: List[Any],
                   state: Any,
                   task_analysis: Dict[str, Any],
                   resource_analysis: Dict[str, Any],
                   constraints: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """Create parallel execution plan"""
        
        plan = ExecutionPlan(
            name="parallel_plan",
            constraints=constraints,
            metadata={
                'strategy': 'parallel',
                'task_count': len(tasks),
                'max_parallel': self.max_parallel,
                'created_by': 'ParallelStrategy'
            }
        )
        
        # Separate tasks with and without dependencies
        independent_tasks = []
        dependent_tasks = []
        
        for task in tasks:
            if task.dependencies:
                dependent_tasks.append(task)
            else:
                independent_tasks.append(task)
        
        # Create parallel groups for independent tasks
        parallel_groups = []
        current_group = []
        
        for task in independent_tasks:
            current_group.append(task)
            
            if len(current_group) >= self.max_parallel:
                parallel_groups.append(current_group)
                current_group = []
        
        if current_group:
            parallel_groups.append(current_group)
        
        # Create nodes and edges
        prev_group_nodes = []
        max_group_duration = 0.0
        
        # Process parallel groups
        for group_idx, group in enumerate(parallel_groups):
            group_nodes = []
            group_duration = 0.0
            
            for task in group:
                node = PlanNode(
                    task_name=task.name,
                    node_type=NodeType.TASK,
                    priority=1,  # All parallel tasks have same priority
                    estimated_duration=task_analysis.get('estimated_total_time', 30.0) / len(tasks),
                    task_wrapper=task
                )
                
                plan.nodes.append(node)
                group_nodes.append(node)
                group_duration = max(group_duration, node.estimated_duration)
                
                # Connect to previous group
                if prev_group_nodes:
                    for prev_node in prev_group_nodes:
                        edge = PlanEdge(
                            from_node=prev_node.id,
                            to_node=node.id,
                            edge_type=EdgeType.PARALLEL
                        )
                        plan.edges.append(edge)
            
            prev_group_nodes = group_nodes
            max_group_duration += group_duration
        
        # Add dependent tasks sequentially after parallel groups
        for task in dependent_tasks:
            node = PlanNode(
                task_name=task.name,
                node_type=NodeType.TASK,
                priority=10,  # Higher priority for dependent tasks
                estimated_duration=task_analysis.get('estimated_total_time', 30.0) / len(tasks),
                task_wrapper=task
            )
            
            plan.nodes.append(node)
            
            # Connect to all previous group nodes
            if prev_group_nodes:
                for prev_node in prev_group_nodes:
                    edge = PlanEdge(
                        from_node=prev_node.id,
                        to_node=node.id,
                        edge_type=EdgeType.SEQUENTIAL
                    )
                    plan.edges.append(edge)
            
            prev_group_nodes = [node]
            max_group_duration += node.estimated_duration
        
        plan.estimated_total_duration = max_group_duration
        plan.metadata['parallel_groups'] = len(parallel_groups)
        plan.metadata['dependent_tasks'] = len(dependent_tasks)
        
        logger.info(f"Parallel plan created with {len(parallel_groups)} parallel groups")
        return plan
    
    def get_strategy_name(self) -> str:
        return "parallel"


class ConditionalStrategy(PlanningStrategy):
    """Conditional execution strategy - tasks execute based on conditions"""
    
    def create_plan(self,
                   tasks: List[Any],
                   state: Any,
                   task_analysis: Dict[str, Any],
                   resource_analysis: Dict[str, Any],
                   constraints: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """Create conditional execution plan"""
        
        plan = ExecutionPlan(
            name="conditional_plan",
            constraints=constraints,
            metadata={
                'strategy': 'conditional',
                'task_count': len(tasks),
                'created_by': 'ConditionalStrategy'
            }
        )
        
        # Analyze task reliability for conditional routing
        reliable_tasks = []
        unreliable_tasks = []
        
        for task in tasks:
            # This would typically come from historical data
            # For now, assume tasks with dependencies are more reliable
            if task.dependencies or len(task.tools) > 0:
                reliable_tasks.append(task)
            else:
                unreliable_tasks.append(task)
        
        # Create conditional flow
        total_duration = 0.0
        prev_node_id = None
        
        # Start with most reliable tasks
        for task in reliable_tasks:
            node = PlanNode(
                task_name=task.name,
                node_type=NodeType.TASK,
                priority=10,  # High priority for reliable tasks
                estimated_duration=task_analysis.get('estimated_total_time', 30.0) / len(tasks),
                task_wrapper=task
            )
            
            plan.nodes.append(node)
            total_duration += node.estimated_duration
            
            if prev_node_id:
                edge = PlanEdge(
                    from_node=prev_node_id,
                    to_node=node.id,
                    edge_type=EdgeType.CONDITIONAL,
                    condition=f"previous_task_success == True"
                )
                plan.edges.append(edge)
            
            prev_node_id = node.id
        
        # Add unreliable tasks with fallback conditions
        for task in unreliable_tasks:
            node = PlanNode(
                task_name=task.name,
                node_type=NodeType.TASK,
                priority=5,  # Lower priority for unreliable tasks
                estimated_duration=task_analysis.get('estimated_total_time', 30.0) / len(tasks),
                task_wrapper=task
            )
            
            plan.nodes.append(node)
            
            # Create fallback edge
            if prev_node_id:
                edge = PlanEdge(
                    from_node=prev_node_id,
                    to_node=node.id,
                    edge_type=EdgeType.FALLBACK,
                    condition=f"previous_task_failed == True"
                )
                plan.edges.append(edge)
        
        plan.estimated_total_duration = total_duration
        plan.metadata['reliable_tasks'] = len(reliable_tasks)
        plan.metadata['unreliable_tasks'] = len(unreliable_tasks)
        
        logger.info(f"Conditional plan created with {len(reliable_tasks)} reliable and {len(unreliable_tasks)} unreliable tasks")
        return plan
    
    def get_strategy_name(self) -> str:
        return "conditional"


class OptimalStrategy(PlanningStrategy):
    """Optimal execution strategy - uses ML and optimization for best performance"""
    
    def create_plan(self,
                   tasks: List[Any],
                   state: Any,
                   task_analysis: Dict[str, Any],
                   resource_analysis: Dict[str, Any],
                   constraints: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """Create optimal execution plan using advanced optimization"""
        
        plan = ExecutionPlan(
            name="optimal_plan",
            constraints=constraints,
            metadata={
                'strategy': 'optimal',
                'task_count': len(tasks),
                'optimization_applied': True,
                'created_by': 'OptimalStrategy'
            }
        )
        
        # Analyze task characteristics
        task_scores = self._calculate_task_scores(tasks, task_analysis)
        
        # Create dependency graph
        dependency_graph = self._build_dependency_graph(tasks)
        
        # Optimize task ordering
        optimal_order = self._optimize_task_order(tasks, task_scores, dependency_graph)
        
        # Create execution plan with optimal order
        total_duration = 0.0
        prev_node_id = None
        parallel_candidates = []
        
        for i, task in enumerate(optimal_order):
            node = PlanNode(
                task_name=task.name,
                node_type=NodeType.TASK,
                priority=task_scores[task.name]['priority'],
                estimated_duration=task_scores[task.name]['estimated_duration'],
                estimated_resources=task_scores[task.name]['resources'],
                task_wrapper=task
            )
            
            plan.nodes.append(node)
            
            # Determine if task can run in parallel
            if self._can_run_parallel(task, dependency_graph, parallel_candidates):
                parallel_candidates.append(task.name)
                
                # Create parallel edges
                if len(parallel_candidates) > 1:
                    for candidate_name in parallel_candidates[:-1]:
                        candidate_node = next(n for n in plan.nodes if n.task_name == candidate_name)
                        edge = PlanEdge(
                            from_node=candidate_node.id,
                            to_node=node.id,
                            edge_type=EdgeType.PARALLEL
                        )
                        plan.edges.append(edge)
                
                # Update duration for parallel execution
                max_parallel_duration = max(
                    task_scores[name]['estimated_duration'] 
                    for name in parallel_candidates
                )
                total_duration = max(total_duration, max_parallel_duration)
            
            else:
                # Sequential execution
                parallel_candidates = [task.name]
                
                if prev_node_id:
                    edge = PlanEdge(
                        from_node=prev_node_id,
                        to_node=node.id,
                        edge_type=EdgeType.SEQUENTIAL
                    )
                    plan.edges.append(edge)
                
                total_duration += node.estimated_duration
                prev_node_id = node.id
        
        plan.estimated_total_duration = total_duration
        
        # Add optimization metadata
        plan.metadata.update({
            'optimization_score': self._calculate_optimization_score(plan, task_analysis),
            'parallel_opportunities': len([n for n in plan.nodes if any(
                e.edge_type == EdgeType.PARALLEL for e in plan.edges if e.to_node == n.id
            )]),
            'critical_path_length': self._calculate_critical_path_length(plan),
            'resource_efficiency': self._calculate_resource_efficiency(plan, resource_analysis)
        })
        
        logger.info(f"Optimal plan created with optimization score: {plan.metadata['optimization_score']:.2f}")
        return plan
    
    def _calculate_task_scores(self, tasks: List[Any], task_analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate comprehensive scores for each task"""
        scores = {}
        
        for task in tasks:
            # Base metrics
            estimated_duration = 30.0  # Default duration
            dependency_count = len(task.dependencies)
            tool_count = len(task.tools)
            
            # Calculate priority based on multiple factors
            priority = 1
            priority += dependency_count * 2  # Higher priority for tasks with dependencies
            priority += tool_count  # Higher priority for tasks with tools
            
            # Estimate resources (simplified)
            resources = {
                'cpu': 1.0 + tool_count * 0.5,
                'memory': 100.0 + dependency_count * 50.0,
                'network': 0.1 * tool_count
            }
            
            scores[task.name] = {
                'priority': priority,
                'estimated_duration': estimated_duration,
                'dependency_count': dependency_count,
                'tool_count': tool_count,
                'resources': resources,
                'complexity_score': dependency_count + tool_count
            }
        
        return scores
    
    def _build_dependency_graph(self, tasks: List[Any]) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        graph = {}
        
        for task in tasks:
            graph[task.name] = task.dependencies.copy()
        
        return graph
    
    def _optimize_task_order(self, 
                           tasks: List[Any],
                           task_scores: Dict[str, Dict[str, Any]],
                           dependency_graph: Dict[str, List[str]]) -> List[Any]:
        """Optimize task execution order"""
        
        # Topological sort with priority consideration
        in_degree = {task.name: 0 for task in tasks}
        
        # Calculate in-degrees
        for task_name, deps in dependency_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Priority queue based on multiple factors
        import heapq
        queue = []
        
        for task in tasks:
            if in_degree[task.name] == 0:
                # Create composite score for prioritization
                score = task_scores[task.name]
                composite_priority = (
                    -score['priority'] * 10 +  # Higher priority first (negative for max heap)
                    score['estimated_duration'] +  # Shorter tasks first
                    -score['complexity_score'] * 5  # More complex tasks first
                )
                heapq.heappush(queue, (composite_priority, task))
        
        result = []
        task_map = {task.name: task for task in tasks}
        
        while queue:
            _, task = heapq.heappop(queue)
            result.append(task)
            
            # Update in-degrees for dependent tasks
            for other_task_name in dependency_graph:
                if task.name in dependency_graph[other_task_name]:
                    in_degree[other_task_name] -= 1
                    if in_degree[other_task_name] == 0:
                        dependent_task = task_map[other_task_name]
                        score = task_scores[other_task_name]
                        composite_priority = (
                            -score['priority'] * 10 +
                            score['estimated_duration'] +
                            -score['complexity_score'] * 5
                        )
                        heapq.heappush(queue, (composite_priority, dependent_task))
        
        return result
    
    def _can_run_parallel(self, 
                         task: Any,
                         dependency_graph: Dict[str, List[str]],
                         current_parallel: List[str]) -> bool:
        """Check if task can run in parallel with current candidates"""
        
        # Check if task has dependencies that conflict with parallel candidates
        for dep in task.dependencies:
            if dep in current_parallel:
                return False
        
        # Check if any parallel candidate depends on this task
        for candidate in current_parallel:
            if task.name in dependency_graph.get(candidate, []):
                return False
        
        return True
    
    def _calculate_optimization_score(self, 
                                    plan: ExecutionPlan,
                                    task_analysis: Dict[str, Any]) -> float:
        """Calculate optimization score for the plan"""
        
        # Factors for optimization score
        parallelization_factor = 0.0
        dependency_efficiency = 0.0
        resource_efficiency = 0.0
        
        # Parallelization score (higher is better)
        parallel_edges = sum(1 for edge in plan.edges if edge.edge_type == EdgeType.PARALLEL)
        total_edges = len(plan.edges)
        if total_edges > 0:
            parallelization_factor = parallel_edges / total_edges
        
        # Dependency efficiency (lower total duration is better)
        sequential_duration = sum(node.estimated_duration for node in plan.nodes)
        if sequential_duration > 0:
            dependency_efficiency = min(1.0, sequential_duration / plan.estimated_total_duration)
        
        # Resource efficiency (balanced resource usage is better)
        total_cpu = sum(node.estimated_resources.get('cpu', 1.0) for node in plan.nodes)
        total_memory = sum(node.estimated_resources.get('memory', 100.0) for node in plan.nodes)
        
        # Normalize and balance (simplified)
        if total_cpu > 0 and total_memory > 0:
            cpu_balance = min(1.0, 10.0 / total_cpu)  # Prefer lower CPU usage
            memory_balance = min(1.0, 1000.0 / total_memory)  # Prefer lower memory usage
            resource_efficiency = (cpu_balance + memory_balance) / 2
        
        # Composite score (0.0 to 1.0)
        optimization_score = (
            parallelization_factor * 0.4 +
            dependency_efficiency * 0.4 +
            resource_efficiency * 0.2
        )
        
        return optimization_score
    
    def _calculate_critical_path_length(self, plan: ExecutionPlan) -> int:
        """Calculate critical path length in the plan"""
        # Simplified critical path calculation
        # In a real implementation, this would use proper critical path algorithms
        
        sequential_nodes = 0
        for edge in plan.edges:
            if edge.edge_type == EdgeType.SEQUENTIAL:
                sequential_nodes += 1
        
        return sequential_nodes + 1  # +1 for the final node
    
    def _calculate_resource_efficiency(self, 
                                     plan: ExecutionPlan,
                                     resource_analysis: Dict[str, Any]) -> float:
        """Calculate resource utilization efficiency"""
        
        total_estimated_cpu = sum(
            node.estimated_resources.get('cpu', 1.0) 
            for node in plan.nodes
        )
        
        total_estimated_memory = sum(
            node.estimated_resources.get('memory', 100.0) 
            for node in plan.nodes
        )
        
        # Compare with available resources (if specified)
        available_cpu = resource_analysis.get('available_cpu', total_estimated_cpu)
        available_memory = resource_analysis.get('available_memory', total_estimated_memory)
        
        cpu_efficiency = min(1.0, total_estimated_cpu / available_cpu) if available_cpu > 0 else 1.0
        memory_efficiency = min(1.0, total_estimated_memory / available_memory) if available_memory > 0 else 1.0
        
        return (cpu_efficiency + memory_efficiency) / 2
    
    def get_strategy_name(self) -> str:
        return "optimal"