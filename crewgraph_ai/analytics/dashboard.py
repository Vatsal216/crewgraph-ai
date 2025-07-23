"""
Performance Dashboard for CrewGraph AI Analytics Module

Provides real-time performance monitoring and visualization capabilities
for workflow execution and system metrics.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict

from ..utils.logging import get_logger
from ..types import WorkflowId

logger = get_logger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the performance dashboard."""
    refresh_interval: int = 5  # seconds
    max_data_points: int = 100
    enable_real_time: bool = True
    metrics_to_track: List[str] = None
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics_to_track is None:
            self.metrics_to_track = ["execution_time", "success_rate", "cpu_usage", "memory_usage"]
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "execution_time": 300.0,  # 5 minutes
                "success_rate": 0.9,
                "cpu_usage": 80.0,
                "memory_usage": 85.0
            }


@dataclass
class MetricSnapshot:
    """Snapshot of system metrics at a point in time."""
    timestamp: datetime
    workflow_id: Optional[WorkflowId]
    execution_time: float
    success_rate: float
    cpu_usage: float
    memory_usage: float
    active_tasks: int
    completed_tasks: int
    error_count: int
    throughput: float  # tasks per minute


class PerformanceDashboard:
    """
    Real-time performance dashboard for monitoring workflow execution
    and system performance with alerting and visualization capabilities.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        Initialize the performance dashboard.
        
        Args:
            config: Dashboard configuration settings
        """
        self.config = config or DashboardConfig()
        self.metrics_history: deque = deque(maxlen=self.config.max_data_points)
        self.workflow_metrics: Dict[WorkflowId, List[MetricSnapshot]] = defaultdict(list)
        self.active_workflows: Dict[WorkflowId, Dict[str, Any]] = {}
        self.alert_handlers: List[Callable] = []
        self.is_monitoring = False
        
        logger.info(f"PerformanceDashboard initialized with {len(self.config.metrics_to_track)} metrics")
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        self.is_monitoring = True
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.is_monitoring = False
        logger.info("Performance monitoring stopped")
    
    def record_metrics(self, 
                      workflow_id: Optional[WorkflowId] = None,
                      metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Record performance metrics for monitoring.
        
        Args:
            workflow_id: ID of the workflow being monitored
            metrics: Dictionary of metric values
        """
        if not self.is_monitoring:
            return
        
        # Default metrics if not provided
        if metrics is None:
            metrics = self._collect_system_metrics()
        
        # Create metric snapshot
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            workflow_id=workflow_id,
            execution_time=metrics.get("execution_time", 0.0),
            success_rate=metrics.get("success_rate", 1.0),
            cpu_usage=metrics.get("cpu_usage", 0.0),
            memory_usage=metrics.get("memory_usage", 0.0),
            active_tasks=metrics.get("active_tasks", 0),
            completed_tasks=metrics.get("completed_tasks", 0),
            error_count=metrics.get("error_count", 0),
            throughput=metrics.get("throughput", 0.0)
        )
        
        # Store metrics
        self.metrics_history.append(snapshot)
        
        if workflow_id:
            self.workflow_metrics[workflow_id].append(snapshot)
            # Keep workflow metrics list bounded
            if len(self.workflow_metrics[workflow_id]) > self.config.max_data_points:
                self.workflow_metrics[workflow_id].pop(0)
        
        # Check for alerts
        self._check_alerts(snapshot)
        
        logger.debug(f"Recorded metrics for workflow {workflow_id}")
    
    def get_current_dashboard_data(self) -> Dict[str, Any]:
        """
        Get current dashboard data for visualization.
        
        Returns:
            Dashboard data including metrics, charts, and status
        """
        if not self.metrics_history:
            return {
                "status": "no_data",
                "message": "No metrics data available",
                "timestamp": datetime.now().isoformat()
            }
        
        latest_snapshot = self.metrics_history[-1]
        
        # Calculate summary statistics
        recent_snapshots = list(self.metrics_history)[-10:]  # Last 10 data points
        
        dashboard_data = {
            "status": "active" if self.is_monitoring else "stopped",
            "timestamp": datetime.now().isoformat(),
            "current_metrics": {
                "execution_time": latest_snapshot.execution_time,
                "success_rate": latest_snapshot.success_rate,
                "cpu_usage": latest_snapshot.cpu_usage,
                "memory_usage": latest_snapshot.memory_usage,
                "active_tasks": latest_snapshot.active_tasks,
                "completed_tasks": latest_snapshot.completed_tasks,
                "throughput": latest_snapshot.throughput
            },
            "summary_stats": self._calculate_summary_stats(recent_snapshots),
            "active_workflows": len(self.active_workflows),
            "total_workflows_tracked": len(self.workflow_metrics),
            "alerts": self._get_active_alerts(),
            "charts_data": self._generate_charts_data()
        }
        
        return dashboard_data
    
    def get_workflow_dashboard(self, workflow_id: WorkflowId) -> Dict[str, Any]:
        """
        Get dashboard data specific to a workflow.
        
        Args:
            workflow_id: Workflow to get dashboard for
            
        Returns:
            Workflow-specific dashboard data
        """
        if workflow_id not in self.workflow_metrics:
            return {
                "error": f"No metrics found for workflow {workflow_id}",
                "workflow_id": workflow_id
            }
        
        workflow_snapshots = self.workflow_metrics[workflow_id]
        latest_snapshot = workflow_snapshots[-1] if workflow_snapshots else None
        
        dashboard_data = {
            "workflow_id": workflow_id,
            "status": "active" if workflow_id in self.active_workflows else "completed",
            "timestamp": datetime.now().isoformat(),
            "total_snapshots": len(workflow_snapshots),
            "execution_summary": self._generate_workflow_summary(workflow_snapshots),
            "performance_trends": self._analyze_workflow_trends(workflow_snapshots),
            "current_state": asdict(latest_snapshot) if latest_snapshot else None
        }
        
        return dashboard_data
    
    def generate_performance_report(self, 
                                  time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            time_range: Time range for the report (default: last 24 hours)
            
        Returns:
            Performance report with insights and recommendations
        """
        if time_range is None:
            time_range = timedelta(hours=24)
        
        cutoff_time = datetime.now() - time_range
        
        # Filter metrics within time range
        relevant_snapshots = [
            snapshot for snapshot in self.metrics_history
            if snapshot.timestamp >= cutoff_time
        ]
        
        if not relevant_snapshots:
            return {
                "error": "No data available for the specified time range",
                "time_range": str(time_range)
            }
        
        report = {
            "report_title": f"Performance Report - Last {time_range}",
            "generation_time": datetime.now().isoformat(),
            "data_points_analyzed": len(relevant_snapshots),
            "time_range": {
                "start": relevant_snapshots[0].timestamp.isoformat(),
                "end": relevant_snapshots[-1].timestamp.isoformat()
            },
            "performance_summary": self._analyze_performance_summary(relevant_snapshots),
            "bottleneck_analysis": self._analyze_bottlenecks(relevant_snapshots),
            "optimization_recommendations": self._generate_optimization_recommendations(relevant_snapshots),
            "workflow_insights": self._analyze_workflow_patterns(relevant_snapshots)
        }
        
        return report
    
    def add_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Add a custom alert handler.
        
        Args:
            handler: Function to call when alerts are triggered
        """
        self.alert_handlers.append(handler)
        logger.info("Added custom alert handler")
    
    def register_workflow(self, workflow_id: WorkflowId, metadata: Dict[str, Any]) -> None:
        """
        Register a workflow for monitoring.
        
        Args:
            workflow_id: Workflow identifier
            metadata: Workflow metadata and configuration
        """
        self.active_workflows[workflow_id] = {
            "start_time": datetime.now(),
            "metadata": metadata,
            "status": "active"
        }
        logger.info(f"Registered workflow {workflow_id} for monitoring")
    
    def unregister_workflow(self, workflow_id: WorkflowId) -> None:
        """
        Unregister a workflow from active monitoring.
        
        Args:
            workflow_id: Workflow identifier
        """
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["status"] = "completed"
            self.active_workflows[workflow_id]["end_time"] = datetime.now()
            del self.active_workflows[workflow_id]
            logger.info(f"Unregistered workflow {workflow_id} from monitoring")
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        try:
            import psutil
            
            return {
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "memory_usage": psutil.virtual_memory().percent,
                "execution_time": 0.0,  # Will be set by calling code
                "success_rate": 1.0,    # Will be set by calling code
                "active_tasks": len(self.active_workflows),
                "completed_tasks": 0,   # Will be calculated
                "error_count": 0,       # Will be calculated
                "throughput": self._calculate_current_throughput()
            }
        except ImportError:
            # Fallback if psutil not available
            return {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "execution_time": 0.0,
                "success_rate": 1.0,
                "active_tasks": len(self.active_workflows),
                "completed_tasks": 0,
                "error_count": 0,
                "throughput": 0.0
            }
    
    def _calculate_current_throughput(self) -> float:
        """Calculate current throughput in tasks per minute."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Calculate throughput from recent snapshots
        recent_snapshots = list(self.metrics_history)[-5:]  # Last 5 snapshots
        
        if len(recent_snapshots) < 2:
            return 0.0
        
        time_diff = (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).total_seconds()
        if time_diff <= 0:
            return 0.0
        
        task_diff = recent_snapshots[-1].completed_tasks - recent_snapshots[0].completed_tasks
        
        return (task_diff / time_diff) * 60  # tasks per minute
    
    def _check_alerts(self, snapshot: MetricSnapshot) -> None:
        """Check if any metrics exceed alert thresholds."""
        alerts = []
        
        for metric, threshold in self.config.alert_thresholds.items():
            value = getattr(snapshot, metric, None)
            if value is None:
                continue
            
            # Check threshold based on metric type
            if metric in ["execution_time", "cpu_usage", "memory_usage", "error_count"]:
                if value > threshold:
                    alerts.append({
                        "type": "threshold_exceeded",
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "timestamp": snapshot.timestamp.isoformat(),
                        "workflow_id": snapshot.workflow_id
                    })
            elif metric == "success_rate":
                if value < threshold:
                    alerts.append({
                        "type": "threshold_below",
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "timestamp": snapshot.timestamp.isoformat(),
                        "workflow_id": snapshot.workflow_id
                    })
        
        # Trigger alert handlers
        for alert in alerts:
            for handler in self.alert_handlers:
                try:
                    handler("metric_alert", alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
    
    def _calculate_summary_stats(self, snapshots: List[MetricSnapshot]) -> Dict[str, float]:
        """Calculate summary statistics for a set of snapshots."""
        if not snapshots:
            return {}
        
        stats = {}
        
        # Calculate averages for numeric metrics
        numeric_metrics = ["execution_time", "success_rate", "cpu_usage", "memory_usage", "throughput"]
        
        for metric in numeric_metrics:
            values = [getattr(snapshot, metric) for snapshot in snapshots]
            if values:
                stats[f"avg_{metric}"] = sum(values) / len(values)
                stats[f"max_{metric}"] = max(values)
                stats[f"min_{metric}"] = min(values)
        
        return stats
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        # For now, return placeholder alerts
        # In a real implementation, this would track active alert states
        return []
    
    def _generate_charts_data(self) -> Dict[str, List[Any]]:
        """Generate data for dashboard charts."""
        if not self.metrics_history:
            return {}
        
        # Prepare time series data
        timestamps = [snapshot.timestamp.isoformat() for snapshot in self.metrics_history]
        
        charts_data = {
            "timestamps": timestamps,
            "execution_time": [snapshot.execution_time for snapshot in self.metrics_history],
            "success_rate": [snapshot.success_rate for snapshot in self.metrics_history],
            "cpu_usage": [snapshot.cpu_usage for snapshot in self.metrics_history],
            "memory_usage": [snapshot.memory_usage for snapshot in self.metrics_history],
            "throughput": [snapshot.throughput for snapshot in self.metrics_history]
        }
        
        return charts_data
    
    def _generate_workflow_summary(self, snapshots: List[MetricSnapshot]) -> Dict[str, Any]:
        """Generate summary for a specific workflow."""
        if not snapshots:
            return {}
        
        return {
            "total_executions": len(snapshots),
            "average_execution_time": sum(s.execution_time for s in snapshots) / len(snapshots),
            "average_success_rate": sum(s.success_rate for s in snapshots) / len(snapshots),
            "first_execution": snapshots[0].timestamp.isoformat(),
            "last_execution": snapshots[-1].timestamp.isoformat(),
            "total_tasks_completed": sum(s.completed_tasks for s in snapshots),
            "total_errors": sum(s.error_count for s in snapshots)
        }
    
    def _analyze_workflow_trends(self, snapshots: List[MetricSnapshot]) -> Dict[str, str]:
        """Analyze trends in workflow performance."""
        if len(snapshots) < 2:
            return {"trend": "insufficient_data"}
        
        # Simple trend analysis
        recent_performance = sum(s.success_rate for s in snapshots[-3:]) / min(3, len(snapshots))
        overall_performance = sum(s.success_rate for s in snapshots) / len(snapshots)
        
        if recent_performance > overall_performance * 1.1:
            trend = "improving"
        elif recent_performance < overall_performance * 0.9:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "performance_trend": trend,
            "recent_performance": recent_performance,
            "overall_performance": overall_performance
        }
    
    def _analyze_performance_summary(self, snapshots: List[MetricSnapshot]) -> Dict[str, Any]:
        """Analyze overall performance from snapshots."""
        summary = self._calculate_summary_stats(snapshots)
        
        # Add additional analysis
        summary["total_workflows"] = len(set(s.workflow_id for s in snapshots if s.workflow_id))
        summary["peak_cpu_time"] = max((s.timestamp for s in snapshots), 
                                     key=lambda t: next(s.cpu_usage for s in snapshots if s.timestamp == t))
        
        return summary
    
    def _analyze_bottlenecks(self, snapshots: List[MetricSnapshot]) -> List[Dict[str, Any]]:
        """Analyze potential bottlenecks from performance data."""
        bottlenecks = []
        
        # CPU bottleneck analysis
        high_cpu_snapshots = [s for s in snapshots if s.cpu_usage > 80]
        if len(high_cpu_snapshots) > len(snapshots) * 0.2:
            bottlenecks.append({
                "type": "cpu_bottleneck",
                "severity": "high",
                "description": f"CPU usage exceeded 80% in {len(high_cpu_snapshots)} of {len(snapshots)} measurements",
                "recommendation": "Consider reducing concurrent tasks or scaling compute resources"
            })
        
        # Memory bottleneck analysis
        high_memory_snapshots = [s for s in snapshots if s.memory_usage > 85]
        if len(high_memory_snapshots) > len(snapshots) * 0.15:
            bottlenecks.append({
                "type": "memory_bottleneck", 
                "severity": "medium",
                "description": f"Memory usage exceeded 85% in {len(high_memory_snapshots)} measurements",
                "recommendation": "Implement memory optimization or increase available memory"
            })
        
        return bottlenecks
    
    def _generate_optimization_recommendations(self, snapshots: List[MetricSnapshot]) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        if not snapshots:
            return ["Insufficient data for recommendations"]
        
        # Analyze average metrics
        avg_execution_time = sum(s.execution_time for s in snapshots) / len(snapshots)
        avg_success_rate = sum(s.success_rate for s in snapshots) / len(snapshots)
        avg_throughput = sum(s.throughput for s in snapshots) / len(snapshots)
        
        if avg_execution_time > 60:
            recommendations.append("Consider parallelizing tasks to reduce execution time")
        
        if avg_success_rate < 0.9:
            recommendations.append("Implement better error handling and retry mechanisms")
        
        if avg_throughput < 1.0:
            recommendations.append("Optimize workflow structure to improve task throughput")
        
        return recommendations
    
    def _analyze_workflow_patterns(self, snapshots: List[MetricSnapshot]) -> Dict[str, Any]:
        """Analyze patterns across different workflows."""
        workflow_groups = defaultdict(list)
        
        for snapshot in snapshots:
            if snapshot.workflow_id:
                workflow_groups[snapshot.workflow_id].append(snapshot)
        
        patterns = {
            "unique_workflows": len(workflow_groups),
            "most_active_workflow": None,
            "best_performing_workflow": None
        }
        
        if workflow_groups:
            # Find most active workflow
            most_active = max(workflow_groups.keys(), key=lambda w: len(workflow_groups[w]))
            patterns["most_active_workflow"] = {
                "workflow_id": most_active,
                "execution_count": len(workflow_groups[most_active])
            }
            
            # Find best performing workflow
            workflow_performance = {}
            for workflow_id, workflow_snapshots in workflow_groups.items():
                avg_success = sum(s.success_rate for s in workflow_snapshots) / len(workflow_snapshots)
                workflow_performance[workflow_id] = avg_success
            
            if workflow_performance:
                best_workflow = max(workflow_performance.keys(), key=lambda w: workflow_performance[w])
                patterns["best_performing_workflow"] = {
                    "workflow_id": best_workflow,
                    "success_rate": workflow_performance[best_workflow]
                }
        
        return patterns