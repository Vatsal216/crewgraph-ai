"""
Comprehensive metrics collection and monitoring
"""

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricSample:
    """Individual metric sample"""

    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Metric summary statistics"""

    count: int = 0
    sum: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    avg: float = 0.0
    last_value: float = 0.0
    last_timestamp: float = 0.0


class MetricsCollector:
    """Production-grade metrics collection system"""

    def __init__(self, max_samples: int = 10000):
        """
        Initialize metrics collector.

        Args:
            max_samples: Maximum number of samples to keep per metric
        """
        self.max_samples = max_samples
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self._summaries: Dict[str, MetricSummary] = defaultdict(MetricSummary)
        self._lock = threading.RLock()

        # System metrics collection
        self._system_metrics_enabled = True
        self._system_metrics_interval = 30  # seconds
        self._system_metrics_thread = None

        self._start_system_metrics_collection()

        logger.info(f"MetricsCollector initialized with max_samples={max_samples}")

    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for the metric
            timestamp: Optional timestamp (defaults to current time)
        """
        timestamp = timestamp or time.time()
        labels = labels or {}

        sample = MetricSample(timestamp=timestamp, value=value, labels=labels)

        with self._lock:
            # Add sample
            self._metrics[name].append(sample)

            # Update summary
            summary = self._summaries[name]
            summary.count += 1
            summary.sum += value
            summary.min = min(summary.min, value)
            summary.max = max(summary.max, value)
            summary.avg = summary.sum / summary.count
            summary.last_value = value
            summary.last_timestamp = timestamp

        logger.debug(f"Recorded metric '{name}': {value} at {timestamp}")

    def increment_counter(
        self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric"""
        current_value = self.get_current_value(name)
        self.record_metric(name, current_value + amount, labels)

    def record_duration(
        self, name: str, duration: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a duration metric in seconds"""
        self.record_metric(f"{name}_duration_seconds", duration, labels)

    def record_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a gauge metric (current value)"""
        self.record_metric(f"{name}_gauge", value, labels)

    def record_histogram(
        self,
        name: str,
        value: float,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram metric"""
        buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]

        # Record the value itself
        self.record_metric(f"{name}_histogram", value, labels)

        # Record bucket counts
        for bucket in buckets:
            bucket_name = f"{name}_histogram_bucket_le_{bucket}"
            if value <= bucket:
                self.increment_counter(bucket_name, labels=labels)

    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """Get summary statistics for a metric"""
        with self._lock:
            return self._summaries.get(name)

    def get_current_value(self, name: str) -> float:
        """Get current (last recorded) value for a metric"""
        with self._lock:
            if name in self._metrics and self._metrics[name]:
                return self._metrics[name][-1].value
            return 0.0

    def get_metric_samples(
        self, name: str, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> List[MetricSample]:
        """Get metric samples within time range"""
        with self._lock:
            samples = list(self._metrics.get(name, []))

            if start_time is not None:
                samples = [s for s in samples if s.timestamp >= start_time]

            if end_time is not None:
                samples = [s for s in samples if s.timestamp <= end_time]

            return samples

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics and summaries"""
        with self._lock:
            return {
                "summaries": dict(self._summaries),
                "current_values": {
                    name: self.get_current_value(name) for name in self._metrics.keys()
                },
                "sample_counts": {name: len(samples) for name, samples in self._metrics.items()},
            }

    def get_summary(self) -> Dict[str, Any]:
        """Get overall metrics summary"""
        with self._lock:
            total_metrics = len(self._metrics)
            total_samples = sum(len(samples) for samples in self._metrics.values())

            return {
                "total_metrics": total_metrics,
                "total_samples": total_samples,
                "max_samples_per_metric": self.max_samples,
                "collection_timestamp": time.time(),
                "system_metrics_enabled": self._system_metrics_enabled,
            }

    def record_workflow_execution(
        self, workflow_name: str, execution_time: float, success: bool, task_count: int
    ) -> None:
        """Record workflow execution metrics"""
        labels = {"workflow": workflow_name, "status": "success" if success else "failure"}

        self.record_duration("workflow_execution", execution_time, labels)
        self.record_gauge("workflow_task_count", task_count, labels)
        self.increment_counter("workflow_executions_total", labels=labels)

        if success:
            self.increment_counter("workflow_successes_total", labels={"workflow": workflow_name})
        else:
            self.increment_counter("workflow_failures_total", labels={"workflow": workflow_name})

    def record_task_execution(
        self, task_name: str, agent_name: str, execution_time: float, success: bool
    ) -> None:
        """Record task execution metrics"""
        labels = {
            "task": task_name,
            "agent": agent_name,
            "status": "success" if success else "failure",
        }

        self.record_duration("task_execution", execution_time, labels)
        self.increment_counter("task_executions_total", labels=labels)

    def record_memory_usage(self, backend: str, operation: str, duration: float) -> None:
        """Record memory operation metrics"""
        labels = {"backend": backend, "operation": operation}

        self.record_duration("memory_operation", duration, labels)
        self.increment_counter("memory_operations_total", labels=labels)

    def record_tool_usage(self, tool_name: str, execution_time: float, success: bool) -> None:
        """Record tool usage metrics"""
        labels = {"tool": tool_name, "status": "success" if success else "failure"}

        self.record_duration("tool_execution", execution_time, labels)
        self.increment_counter("tool_executions_total", labels=labels)

    def record_health_check(self, health_status: Dict[str, Any]) -> None:
        """Record health check metrics"""
        overall_status = health_status.get("status", "unknown")
        self.record_gauge("health_status", 1.0 if overall_status == "healthy" else 0.0)

        # Record individual check statuses
        for check_name, check_data in health_status.get("checks", {}).items():
            check_status = check_data.get("status", "unknown")
            self.record_gauge(
                f"health_check_{check_name}",
                1.0 if check_status == "ok" else 0.0,
                labels={"check": check_name},
            )

    def _start_system_metrics_collection(self) -> None:
        """Start background system metrics collection"""
        if not self._system_metrics_enabled:
            return

        def collect_system_metrics():
            while self._system_metrics_enabled:
                try:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.record_gauge("system_cpu_percent", cpu_percent)

                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.record_gauge("system_memory_percent", memory.percent)
                    self.record_gauge("system_memory_used_bytes", memory.used)
                    self.record_gauge("system_memory_available_bytes", memory.available)

                    # Disk metrics
                    disk = psutil.disk_usage("/")
                    self.record_gauge("system_disk_percent", (disk.used / disk.total) * 100)
                    self.record_gauge("system_disk_used_bytes", disk.used)
                    self.record_gauge("system_disk_free_bytes", disk.free)

                    # Network metrics
                    network = psutil.net_io_counters()
                    self.record_gauge("system_network_bytes_sent", network.bytes_sent)
                    self.record_gauge("system_network_bytes_recv", network.bytes_recv)

                    # Process metrics
                    process = psutil.Process()
                    self.record_gauge("process_cpu_percent", process.cpu_percent())
                    self.record_gauge("process_memory_percent", process.memory_percent())
                    self.record_gauge("process_num_threads", process.num_threads())

                    time.sleep(self._system_metrics_interval)

                except Exception as e:
                    logger.error(f"System metrics collection failed: {e}")
                    time.sleep(self._system_metrics_interval)

        self._system_metrics_thread = threading.Thread(
            target=collect_system_metrics, daemon=True, name="SystemMetricsCollector"
        )
        self._system_metrics_thread.start()

        logger.info("System metrics collection started")

    def stop_system_metrics_collection(self) -> None:
        """Stop system metrics collection"""
        self._system_metrics_enabled = False
        if self._system_metrics_thread:
            self._system_metrics_thread.join(timeout=5)

        logger.info("System metrics collection stopped")

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        with self._lock:
            for name, summary in self._summaries.items():
                # Help and type
                lines.append(f"# HELP {name} {name} metric")
                lines.append(f"# TYPE {name} gauge")

                # Current value
                lines.append(f"{name} {summary.last_value} {int(summary.last_timestamp * 1000)}")

                # Summary metrics
                lines.append(f"{name}_count {summary.count}")
                lines.append(f"{name}_sum {summary.sum}")
                lines.append(f"{name}_min {summary.min}")
                lines.append(f"{name}_max {summary.max}")
                lines.append(f"{name}_avg {summary.avg}")

        return "\n".join(lines)

    def clear_metrics(self) -> None:
        """Clear all collected metrics"""
        with self._lock:
            self._metrics.clear()
            self._summaries.clear()

        logger.info("All metrics cleared")


class PerformanceMonitor:
    """Monitor performance of operations and workflows"""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize performance monitor.

        Args:
            metrics_collector: Optional metrics collector to use
        """
        self.metrics = metrics_collector or MetricsCollector()
        self._active_operations: Dict[str, float] = {}
        self._lock = threading.RLock()

        logger.info("PerformanceMonitor initialized")

    @contextmanager
    def track_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to track operation performance"""
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        start_time = time.time()

        with self._lock:
            self._active_operations[operation_id] = start_time

        try:
            yield operation_id

            # Record successful completion
            duration = time.time() - start_time
            success_labels = (labels or {}).copy()
            success_labels["status"] = "success"

            self.metrics.record_duration(f"{operation_name}_operation", duration, success_labels)
            self.metrics.increment_counter(
                f"{operation_name}_operations_total", labels=success_labels
            )

        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            failure_labels = (labels or {}).copy()
            failure_labels["status"] = "failure"
            failure_labels["error_type"] = type(e).__name__

            self.metrics.record_duration(f"{operation_name}_operation", duration, failure_labels)
            self.metrics.increment_counter(
                f"{operation_name}_operations_total", labels=failure_labels
            )

            raise

        finally:
            with self._lock:
                self._active_operations.pop(operation_id, None)

    @contextmanager
    def track_execution(self, execution_id: str):
        """Track workflow/task execution"""
        with self.track_operation("execution", {"execution_id": execution_id}):
            yield

    def get_active_operations(self) -> Dict[str, float]:
        """Get currently active operations and their start times"""
        with self._lock:
            return self._active_operations.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            active_count = len(self._active_operations)
            oldest_operation = None

            if self._active_operations:
                oldest_start = min(self._active_operations.values())
                oldest_operation = time.time() - oldest_start

            return {
                "active_operations": active_count,
                "oldest_operation_duration": oldest_operation,
                "metrics_summary": self.metrics.get_summary(),
            }


# Global metrics collector instance
_global_metrics: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _global_metrics

    if _global_metrics is None:
        _global_metrics = MetricsCollector()

    return _global_metrics
