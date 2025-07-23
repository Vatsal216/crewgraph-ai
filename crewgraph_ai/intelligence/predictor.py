"""
Performance Predictor for CrewGraph AI Intelligence Layer

This module provides ML-based performance prediction capabilities for workflows,
enabling proactive optimization and resource planning.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..utils.logging import get_logger
from ..types import WorkflowId, TaskResult

logger = get_logger(__name__)


@dataclass
class WorkflowMetrics:
    """Metrics for workflow performance analysis."""
    workflow_id: WorkflowId
    execution_time: float
    memory_usage: float
    cpu_usage: float
    task_count: int
    success_rate: float
    error_count: int
    timestamp: datetime
    resource_utilization: Dict[str, float]


class PerformancePredictor:
    """
    ML-based performance predictor for workflow optimization.
    
    Uses lightweight statistical models and historical data to predict
    workflow performance, execution time, and resource requirements.
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize the performance predictor.
        
        Args:
            history_size: Maximum number of historical metrics to keep
        """
        self.history_size = history_size
        self.metrics_history: List[WorkflowMetrics] = []
        self.workflow_patterns: Dict[str, List[WorkflowMetrics]] = {}
        
        logger.info(f"PerformancePredictor initialized with history_size={history_size}")
    
    def record_metrics(self, metrics: WorkflowMetrics) -> None:
        """
        Record workflow execution metrics for learning.
        
        Args:
            metrics: Workflow execution metrics to record
        """
        self.metrics_history.append(metrics)
        
        # Maintain history size limit
        if len(self.metrics_history) > self.history_size:
            self.metrics_history.pop(0)
        
        # Group by workflow pattern
        pattern_key = f"{metrics.task_count}_{metrics.workflow_id[:8]}"
        if pattern_key not in self.workflow_patterns:
            self.workflow_patterns[pattern_key] = []
        
        self.workflow_patterns[pattern_key].append(metrics)
        
        logger.debug(f"Recorded metrics for workflow {metrics.workflow_id}")
    
    def predict_execution_time(self, 
                             workflow_id: WorkflowId, 
                             task_count: int,
                             context: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """
        Predict workflow execution time based on historical data.
        
        Args:
            workflow_id: Workflow identifier
            task_count: Number of tasks in the workflow
            context: Additional context for prediction
            
        Returns:
            Tuple of (predicted_time, confidence_score)
        """
        if not self.metrics_history:
            # No historical data, return default estimate
            base_time = task_count * 2.0  # 2 seconds per task baseline
            return base_time, 0.5
        
        # Find similar workflows
        similar_metrics = self._find_similar_workflows(task_count, workflow_id)
        
        if not similar_metrics:
            # Fallback to general statistics
            execution_times = [m.execution_time for m in self.metrics_history[-50:]]
            avg_time = statistics.mean(execution_times)
            scaling_factor = task_count / 5.0  # Assume 5 tasks average
            predicted_time = avg_time * scaling_factor
            confidence = 0.6
        else:
            # Use similar workflow data
            execution_times = [m.execution_time for m in similar_metrics]
            predicted_time = statistics.mean(execution_times)
            
            # Calculate confidence based on data consistency
            if len(execution_times) > 1:
                std_dev = statistics.stdev(execution_times)
                confidence = max(0.1, 1.0 - (std_dev / predicted_time))
            else:
                confidence = 0.7
        
        logger.debug(f"Predicted execution time: {predicted_time:.2f}s (confidence: {confidence:.2f})")
        return predicted_time, confidence
    
    def predict_resource_usage(self, 
                             workflow_id: WorkflowId,
                             task_count: int) -> Dict[str, Tuple[float, float]]:
        """
        Predict resource usage for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            task_count: Number of tasks
            
        Returns:
            Dictionary of resource predictions with confidence scores
        """
        similar_metrics = self._find_similar_workflows(task_count, workflow_id)
        
        if not similar_metrics:
            # Default resource estimates
            return {
                "memory": (task_count * 50.0, 0.5),  # 50MB per task
                "cpu": (min(task_count * 10.0, 80.0), 0.5)  # 10% per task, max 80%
            }
        
        # Calculate resource predictions
        memory_usage = [m.memory_usage for m in similar_metrics]
        cpu_usage = [m.cpu_usage for m in similar_metrics]
        
        predictions = {}
        
        if memory_usage:
            avg_memory = statistics.mean(memory_usage)
            memory_confidence = self._calculate_confidence(memory_usage)
            predictions["memory"] = (avg_memory, memory_confidence)
        
        if cpu_usage:
            avg_cpu = statistics.mean(cpu_usage)
            cpu_confidence = self._calculate_confidence(cpu_usage)
            predictions["cpu"] = (avg_cpu, cpu_confidence)
        
        return predictions
    
    def detect_performance_anomalies(self, 
                                   metrics: WorkflowMetrics) -> List[str]:
        """
        Detect performance anomalies in workflow execution.
        
        Args:
            metrics: Current workflow metrics
            
        Returns:
            List of detected anomaly descriptions
        """
        anomalies = []
        
        if not self.metrics_history:
            return anomalies
        
        # Compare with historical averages
        recent_metrics = self.metrics_history[-20:]  # Last 20 executions
        
        if recent_metrics:
            avg_execution_time = statistics.mean([m.execution_time for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_usage for m in recent_metrics])
            avg_cpu = statistics.mean([m.cpu_usage for m in recent_metrics])
            
            # Check for significant deviations (2x threshold)
            if metrics.execution_time > avg_execution_time * 2:
                anomalies.append(f"Execution time {metrics.execution_time:.2f}s is 2x higher than average {avg_execution_time:.2f}s")
            
            if metrics.memory_usage > avg_memory * 2:
                anomalies.append(f"Memory usage {metrics.memory_usage:.1f}MB is 2x higher than average {avg_memory:.1f}MB")
            
            if metrics.cpu_usage > avg_cpu * 1.5:
                anomalies.append(f"CPU usage {metrics.cpu_usage:.1f}% is 1.5x higher than average {avg_cpu:.1f}%")
            
            if metrics.success_rate < 0.8:
                anomalies.append(f"Success rate {metrics.success_rate:.2f} is below acceptable threshold (0.8)")
        
        if anomalies:
            logger.warning(f"Detected {len(anomalies)} performance anomalies for workflow {metrics.workflow_id}")
        
        return anomalies
    
    def get_optimization_recommendations(self, 
                                       workflow_id: WorkflowId) -> List[str]:
        """
        Generate optimization recommendations based on historical data.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        if not self.metrics_history:
            return ["Insufficient historical data for recommendations"]
        
        # Analyze recent performance trends
        recent_metrics = [m for m in self.metrics_history[-50:] 
                         if workflow_id in m.workflow_id or m.workflow_id in workflow_id]
        
        if recent_metrics:
            avg_success_rate = statistics.mean([m.success_rate for m in recent_metrics])
            avg_execution_time = statistics.mean([m.execution_time for m in recent_metrics])
            
            if avg_success_rate < 0.9:
                recommendations.append("Consider implementing retry mechanisms for improved reliability")
            
            if avg_execution_time > 30.0:
                recommendations.append("Workflow execution time is high - consider parallel task execution")
            
            # Check for memory issues
            high_memory_executions = [m for m in recent_metrics if m.memory_usage > 500.0]
            if len(high_memory_executions) > len(recent_metrics) * 0.3:
                recommendations.append("Frequent high memory usage detected - implement memory optimization")
            
            # Check for CPU bottlenecks
            high_cpu_executions = [m for m in recent_metrics if m.cpu_usage > 80.0]
            if len(high_cpu_executions) > len(recent_metrics) * 0.2:
                recommendations.append("CPU usage often high - consider task distribution or resource scaling")
        
        return recommendations
    
    def _find_similar_workflows(self, 
                              task_count: int, 
                              workflow_id: WorkflowId,
                              tolerance: int = 2) -> List[WorkflowMetrics]:
        """Find workflows with similar characteristics."""
        similar = []
        
        for metrics in self.metrics_history:
            # Match by task count (within tolerance)
            if abs(metrics.task_count - task_count) <= tolerance:
                similar.append(metrics)
        
        # If not enough similar workflows, broaden the search
        if len(similar) < 3:
            similar = [m for m in self.metrics_history 
                      if abs(m.task_count - task_count) <= tolerance * 2]
        
        return similar[-10:]  # Return most recent similar workflows
    
    def _calculate_confidence(self, values: List[float]) -> float:
        """Calculate confidence score based on data consistency."""
        if len(values) <= 1:
            return 0.5
        
        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        # Confidence is higher when standard deviation is lower relative to mean
        if mean_val == 0:
            return 0.5
        
        coefficient_of_variation = std_dev / mean_val
        confidence = max(0.1, 1.0 - coefficient_of_variation)
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance prediction capabilities."""
        return {
            "total_workflows_analyzed": len(self.metrics_history),
            "unique_patterns": len(self.workflow_patterns),
            "average_accuracy": self._estimate_prediction_accuracy(),
            "last_analysis": datetime.now().isoformat(),
            "predictor_version": "1.0.0"
        }
    
    def _estimate_prediction_accuracy(self) -> float:
        """Estimate prediction accuracy based on historical data."""
        if len(self.metrics_history) < 10:
            return 0.6  # Default accuracy for insufficient data
        
        # Simple accuracy estimation based on data consistency
        recent_times = [m.execution_time for m in self.metrics_history[-20:]]
        if len(recent_times) > 5:
            coefficient_of_variation = statistics.stdev(recent_times) / statistics.mean(recent_times)
            accuracy = max(0.5, 1.0 - coefficient_of_variation)
            return min(accuracy, 0.9)
        
        return 0.7