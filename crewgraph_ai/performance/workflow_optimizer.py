"""Performance optimization for large workflows."""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for workflows."""
    execution_time: float
    memory_usage: int
    cpu_usage: float
    agent_count: int
    task_count: int
    throughput: float  # tasks per second

class WorkflowOptimizer:
    """Optimizes workflow performance for large-scale operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        self._optimization_cache: Dict[str, Any] = {}
        self._performance_history: List[PerformanceMetrics] = []
        
    def optimize_graph_traversal(self, workflow_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize workflow graph for efficient traversal."""
        
        # Topological sort for optimal execution order
        def topological_sort(graph: Dict[str, List[str]]) -> List[str]:
            in_degree = {node: 0 for node in graph}
            for node in graph:
                for neighbor in graph[node]:
                    if neighbor in in_degree:
                        in_degree[neighbor] += 1
            
            queue = [node for node in in_degree if in_degree[node] == 0]
            result = []
            
            while queue:
                node = queue.pop(0)
                result.append(node)
                
                for neighbor in graph.get(node, []):
                    if neighbor in in_degree:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            queue.append(neighbor)
            
            return result
        
        # Build adjacency list from workflow
        adjacency = {}
        for node_id, node_data in workflow_graph.get('nodes', {}).items():
            adjacency[node_id] = node_data.get('dependencies', [])
        
        # Get optimal execution order
        execution_order = topological_sort(adjacency)
        
        # Identify parallel execution opportunities
        parallel_groups = self._identify_parallel_groups(adjacency, execution_order)
        
        return {
            'execution_order': execution_order,
            'parallel_groups': parallel_groups,
            'optimization_score': self._calculate_optimization_score(workflow_graph, parallel_groups)
        }
    
    def _identify_parallel_groups(self, adjacency: Dict[str, List[str]], 
                                 execution_order: List[str]) -> List[List[str]]:
        """Identify groups of tasks that can run in parallel."""
        parallel_groups = []
        remaining_tasks = set(execution_order)
        
        while remaining_tasks:
            # Find tasks with no dependencies in remaining set
            current_group = []
            for task in list(remaining_tasks):
                dependencies = set(adjacency.get(task, []))
                if not dependencies.intersection(remaining_tasks):
                    current_group.append(task)
            
            if current_group:
                parallel_groups.append(current_group)
                remaining_tasks -= set(current_group)
            else:
                # Break potential deadlock by taking first remaining task
                if remaining_tasks:
                    task = remaining_tasks.pop()
                    parallel_groups.append([task])
        
        return parallel_groups
    
    def _calculate_optimization_score(self, workflow_graph: Dict[str, Any], 
                                    parallel_groups: List[List[str]]) -> float:
        """Calculate optimization score (0-100)."""
        total_tasks = len(workflow_graph.get('nodes', {}))
        if total_tasks == 0:
            return 100.0
        
        # Calculate parallelization potential
        parallel_tasks = sum(len(group) for group in parallel_groups if len(group) > 1)
        parallelization_score = (parallel_tasks / total_tasks) * 50
        
        # Calculate efficiency score based on dependency depth
        max_depth = len(parallel_groups)
        depth_score = max(0, 50 - (max_depth * 2))  # Prefer shallower graphs
        
        return min(100.0, parallelization_score + depth_score)
    
    async def execute_parallel_workflow(self, workflow_groups: List[List[str]], 
                                      task_executor: Callable) -> Dict[str, Any]:
        """Execute workflow groups in parallel."""
        results = {}
        total_start_time = time.time()
        
        for group in workflow_groups:
            if len(group) == 1:
                # Single task - execute directly
                task_id = group[0]
                try:
                    results[task_id] = await task_executor(task_id)
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    results[task_id] = {'error': str(e)}
            else:
                # Multiple tasks - execute in parallel
                group_tasks = []
                for task_id in group:
                    task = asyncio.create_task(task_executor(task_id))
                    group_tasks.append((task_id, task))
                
                # Wait for all tasks in group to complete
                for task_id, task in group_tasks:
                    try:
                        results[task_id] = await task
                    except Exception as e:
                        logger.error(f"Parallel task {task_id} failed: {e}")
                        results[task_id] = {'error': str(e)}
        
        total_execution_time = time.time() - total_start_time
        
        return {
            'results': results,
            'execution_time': total_execution_time,
            'parallel_groups_executed': len(workflow_groups),
            'total_tasks': sum(len(group) for group in workflow_groups)
        }
    
    def benchmark_performance(self, workflow_data: Dict[str, Any]) -> PerformanceMetrics:
        """Benchmark workflow performance."""
        import psutil
        import gc
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_cpu = psutil.cpu_percent()
        
        # Force garbage collection for accurate memory measurement
        gc.collect()
        
        # Simulate workflow execution (replace with actual execution)
        agent_count = len(workflow_data.get('agents', []))
        task_count = len(workflow_data.get('tasks', []))
        
        # Performance simulation
        time.sleep(0.1 * task_count)  # Simulate processing time
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        end_cpu = psutil.cpu_percent()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = (start_cpu + end_cpu) / 2
        throughput = task_count / execution_time if execution_time > 0 else 0
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            agent_count=agent_count,
            task_count=task_count,
            throughput=throughput
        )
        
        self._performance_history.append(metrics)
        return metrics
    
    def get_optimization_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Get optimization recommendations based on performance metrics."""
        recommendations = []
        
        if metrics.execution_time > 60:  # More than 1 minute
            recommendations.append("Consider breaking down large workflows into smaller chunks")
        
        if metrics.memory_usage > 1024 * 1024 * 1024:  # More than 1GB
            recommendations.append("Implement lazy loading for large datasets")
            recommendations.append("Consider using streaming data processing")
        
        if metrics.cpu_usage > 80:
            recommendations.append("Increase parallel processing workers")
            recommendations.append("Optimize CPU-intensive tasks")
        
        if metrics.throughput < 10:  # Less than 10 tasks per second
            recommendations.append("Enable async processing for I/O bound tasks")
            recommendations.append("Implement task result caching")
        
        if metrics.agent_count > 50:
            recommendations.append("Consider agent pooling and reuse")
            recommendations.append("Implement agent load balancing")
        
        return recommendations
    
    def cleanup(self):
        """Cleanup resources."""
        self._thread_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)