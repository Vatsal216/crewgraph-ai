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
    """Result of resource optimization analysis."""
    optimization_type: OptimizationType
    current_utilization: float
    recommended_utilization: float
    potential_improvement: float
    implementation_steps: List[str]
    confidence_score: float
    estimated_savings: Dict[str, float]  # time, memory, cpu savings


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