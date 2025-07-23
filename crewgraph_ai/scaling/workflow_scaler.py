"""
CrewGraph AI Workflow Auto-Scaling System
Intelligent scaling based on workload patterns and resource utilization

Author: Vatsal216
Created: 2025-07-22 13:17:52 UTC
"""

import asyncio
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil

from ..core.graph import CrewGraph
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class ScalingTrigger(Enum):
    """Auto-scaling trigger types"""

    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    CUSTOM_METRIC = "custom_metric"
    PREDICTIVE = "predictive"


class ScalingDirection(Enum):
    """Scaling directions"""

    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration"""

    name: str
    trigger: ScalingTrigger
    threshold_up: float
    threshold_down: float
    cooldown_period: int = 300  # 5 minutes
    min_instances: int = 1
    max_instances: int = 10
    scale_up_count: int = 1
    scale_down_count: int = 1
    enabled: bool = True

    def __post_init__(self):
        if self.threshold_up <= self.threshold_down:
            raise ValueError("threshold_up must be greater than threshold_down")


@dataclass
class WorkflowInstance:
    """Running workflow instance"""

    instance_id: str
    workflow: CrewGraph
    start_time: float
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    task_count: int = 0
    status: str = "running"
    last_activity: float = field(default_factory=time.time)


class WorkflowAutoScaler:
    """
    Intelligent auto-scaling system for CrewGraph AI workflows.

    Features:
    - Multi-metric scaling decisions (CPU, Memory, Queue, Response Time)
    - Predictive scaling based on historical patterns
    - Cost optimization with intelligent instance management
    - Health monitoring and automatic recovery
    - Custom scaling policies and rules
    - Real-time monitoring and alerting

    Created by: Vatsal216
    Date: 2025-07-22 13:17:52 UTC
    """

    def __init__(
        self,
        workflow_factory: Callable[[], CrewGraph],
        scaling_rules: Optional[List[ScalingRule]] = None,
        monitoring_interval: float = 30.0,
        prediction_window: int = 300,
        enable_predictive: bool = True,
    ):
        """
        Initialize workflow auto-scaler.

        Args:
            workflow_factory: Factory function to create new workflow instances
            scaling_rules: List of scaling rules
            monitoring_interval: Monitoring interval in seconds
            prediction_window: Prediction window in seconds
            enable_predictive: Enable predictive scaling
        """
        self.workflow_factory = workflow_factory
        self.scaling_rules = scaling_rules or self._default_scaling_rules()
        self.monitoring_interval = monitoring_interval
        self.prediction_window = prediction_window
        self.enable_predictive = enable_predictive

        # Instance management
        self.instances: Dict[str, WorkflowInstance] = {}
        self.request_queue = queue.Queue()
        self.load_balancer = WorkflowLoadBalancer()

        # Monitoring and metrics
        self.metrics_history = deque(maxlen=1000)
        self.scaling_history = deque(maxlen=100)
        self.last_scale_time = {}

        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._predictor_thread: Optional[threading.Thread] = None
        self._running = False

        # Scaling statistics
        self.scaling_stats = {
            "total_scale_ups": 0,
            "total_scale_downs": 0,
            "current_instances": 0,
            "avg_cpu_usage": 0.0,
            "avg_memory_usage": 0.0,
            "queue_length": 0,
            "avg_response_time": 0.0,
        }

        logger.info("WorkflowAutoScaler initialized")
        logger.info(f"Rules: {len(self.scaling_rules)}, Predictive: {enable_predictive}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 13:17:52")

    def _default_scaling_rules(self) -> List[ScalingRule]:
        """Create default auto-scaling rules"""
        return [
            ScalingRule(
                name="cpu_scaling",
                trigger=ScalingTrigger.CPU_THRESHOLD,
                threshold_up=80.0,  # Scale up at 80% CPU
                threshold_down=30.0,  # Scale down at 30% CPU
                cooldown_period=300,  # 5 minute cooldown
                max_instances=20,
            ),
            ScalingRule(
                name="memory_scaling",
                trigger=ScalingTrigger.MEMORY_THRESHOLD,
                threshold_up=85.0,  # Scale up at 85% memory
                threshold_down=40.0,  # Scale down at 40% memory
                cooldown_period=300,
                max_instances=15,
            ),
            ScalingRule(
                name="queue_scaling",
                trigger=ScalingTrigger.QUEUE_LENGTH,
                threshold_up=10.0,  # Scale up if queue > 10 items
                threshold_down=2.0,  # Scale down if queue < 2 items
                cooldown_period=60,  # 1 minute cooldown for queue
                scale_up_count=2,  # Scale up by 2 for queue pressure
                max_instances=50,
            ),
            ScalingRule(
                name="response_time_scaling",
                trigger=ScalingTrigger.RESPONSE_TIME,
                threshold_up=5000.0,  # Scale up if response time > 5s
                threshold_down=1000.0,  # Scale down if response time < 1s
                cooldown_period=180,  # 3 minute cooldown
                max_instances=30,
            ),
        ]

    def start(self) -> None:
        """Start the auto-scaling system"""
        if self._running:
            logger.warning("Auto-scaler is already running")
            return

        self._running = True

        # Start initial instances
        self._ensure_minimum_instances()

        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, name="WorkflowAutoScaler-Monitor", daemon=True
        )
        self._monitor_thread.start()

        # Start predictive scaling if enabled
        if self.enable_predictive:
            self._predictor_thread = threading.Thread(
                target=self._predictive_loop, name="WorkflowAutoScaler-Predictor", daemon=True
            )
            self._predictor_thread.start()

        # Record startup metrics
        metrics.increment_counter(
            "crewgraph_autoscaler_started_total", labels={"user": "Vatsal216"}
        )

        logger.info("WorkflowAutoScaler started successfully")
        logger.info(f"Initial instances: {len(self.instances)}")

    def stop(self) -> None:
        """Stop the auto-scaling system"""
        logger.info("Stopping WorkflowAutoScaler...")

        self._running = False

        # Wait for threads to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        if self._predictor_thread and self._predictor_thread.is_alive():
            self._predictor_thread.join(timeout=5.0)

        # Cleanup instances
        self._cleanup_all_instances()

        logger.info("WorkflowAutoScaler stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._running:
            try:
                # Collect current metrics
                current_metrics = self._collect_metrics()
                self.metrics_history.append(current_metrics)

                # Update statistics
                self._update_statistics(current_metrics)

                # Evaluate scaling rules
                scaling_decision = self._evaluate_scaling_rules(current_metrics)

                # Execute scaling if needed
                if scaling_decision != ScalingDirection.STABLE:
                    self._execute_scaling(scaling_decision, current_metrics)

                # Cleanup unhealthy instances
                self._cleanup_unhealthy_instances()

                # Record monitoring metrics
                self._record_monitoring_metrics(current_metrics)

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _predictive_loop(self) -> None:
        """Predictive scaling loop"""
        while self._running:
            try:
                if len(self.metrics_history) >= 10:  # Need some history
                    prediction = self._predict_scaling_need()

                    if prediction != ScalingDirection.STABLE:
                        logger.info(f"Predictive scaling suggests: {prediction.value}")

                        # Execute predictive scaling with lower threshold
                        current_metrics = self._collect_metrics()
                        self._execute_scaling(prediction, current_metrics, predictive=True)

                time.sleep(self.prediction_window)

            except Exception as e:
                logger.error(f"Error in predictive loop: {e}")
                time.sleep(self.prediction_window)

    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Instance metrics
        total_instances = len(self.instances)
        active_instances = len([i for i in self.instances.values() if i.status == "running"])

        # Queue metrics
        queue_size = self.request_queue.qsize()

        # Response time metrics (simplified)
        avg_response_time = self._calculate_avg_response_time()

        # Custom workflow metrics
        total_tasks = sum(i.task_count for i in self.instances.values())

        metrics_data = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "total_instances": total_instances,
            "active_instances": active_instances,
            "queue_size": queue_size,
            "avg_response_time": avg_response_time,
            "total_tasks": total_tasks,
            "system_load": psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0,
        }

        logger.debug(
            f"Collected metrics: CPU={cpu_percent}%, Memory={memory_percent}%, "
            f"Instances={total_instances}, Queue={queue_size}"
        )

        return metrics_data

    def _evaluate_scaling_rules(self, metrics: Dict[str, float]) -> ScalingDirection:
        """Evaluate all scaling rules to determine scaling decision"""
        scale_up_votes = 0
        scale_down_votes = 0

        current_time = time.time()

        for rule in self.scaling_rules:
            if not rule.enabled:
                continue

            # Check cooldown period
            last_scale = self.last_scale_time.get(rule.name, 0)
            if current_time - last_scale < rule.cooldown_period:
                continue

            # Get metric value for this rule
            metric_value = self._get_metric_for_rule(rule, metrics)
            if metric_value is None:
                continue

            # Evaluate rule
            current_instances = len(self.instances)

            if metric_value >= rule.threshold_up and current_instances < rule.max_instances:
                scale_up_votes += 1
                logger.debug(
                    f"Rule '{rule.name}': Scale UP vote (value={metric_value}, threshold={rule.threshold_up})"
                )

            elif metric_value <= rule.threshold_down and current_instances > rule.min_instances:
                scale_down_votes += 1
                logger.debug(
                    f"Rule '{rule.name}': Scale DOWN vote (value={metric_value}, threshold={rule.threshold_down})"
                )

        # Make scaling decision
        if scale_up_votes > scale_down_votes:
            return ScalingDirection.UP
        elif scale_down_votes > scale_up_votes:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE

    def _get_metric_for_rule(self, rule: ScalingRule, metrics: Dict[str, float]) -> Optional[float]:
        """Get metric value for specific rule"""
        if rule.trigger == ScalingTrigger.CPU_THRESHOLD:
            return metrics.get("cpu_percent")
        elif rule.trigger == ScalingTrigger.MEMORY_THRESHOLD:
            return metrics.get("memory_percent")
        elif rule.trigger == ScalingTrigger.QUEUE_LENGTH:
            return metrics.get("queue_size")
        elif rule.trigger == ScalingTrigger.RESPONSE_TIME:
            return metrics.get("avg_response_time")
        else:
            return None

    def _execute_scaling(
        self, direction: ScalingDirection, metrics: Dict[str, float], predictive: bool = False
    ) -> None:
        """Execute scaling decision"""
        if direction == ScalingDirection.UP:
            self._scale_up(metrics, predictive)
        elif direction == ScalingDirection.DOWN:
            self._scale_down(metrics, predictive)

        # Update last scale times
        current_time = time.time()
        for rule in self.scaling_rules:
            if rule.enabled:
                self.last_scale_time[rule.name] = current_time

    def _scale_up(self, metrics: Dict[str, float], predictive: bool = False) -> None:
        """Scale up workflow instances"""
        # Determine how many instances to add
        scale_count = self._calculate_scale_count(ScalingDirection.UP, metrics)

        success_count = 0
        for _ in range(scale_count):
            try:
                instance = self._create_instance()
                if instance:
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to create instance during scale-up: {e}")

        if success_count > 0:
            self.scaling_stats["total_scale_ups"] += success_count

            # Record scaling event
            scaling_event = {
                "timestamp": time.time(),
                "direction": "up",
                "count": success_count,
                "trigger_metrics": metrics,
                "predictive": predictive,
                "total_instances": len(self.instances),
            }
            self.scaling_history.append(scaling_event)

            # Record metrics
            metrics.increment_counter(
                "crewgraph_workflow_scale_events_total",
                success_count,
                labels={"direction": "up", "predictive": str(predictive), "user": "Vatsal216"},
            )

            logger.info(
                f"Scaled UP by {success_count} instances "
                f"({'predictive' if predictive else 'reactive'}). "
                f"Total instances: {len(self.instances)}"
            )

    def _scale_down(self, metrics: Dict[str, float], predictive: bool = False) -> None:
        """Scale down workflow instances"""
        # Determine how many instances to remove
        scale_count = self._calculate_scale_count(ScalingDirection.DOWN, metrics)

        # Select instances to terminate (least utilized first)
        instances_to_remove = self._select_instances_for_removal(scale_count)

        success_count = 0
        for instance_id in instances_to_remove:
            try:
                if self._remove_instance(instance_id):
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to remove instance {instance_id}: {e}")

        if success_count > 0:
            self.scaling_stats["total_scale_downs"] += success_count

            # Record scaling event
            scaling_event = {
                "timestamp": time.time(),
                "direction": "down",
                "count": success_count,
                "trigger_metrics": metrics,
                "predictive": predictive,
                "total_instances": len(self.instances),
            }
            self.scaling_history.append(scaling_event)

            # Record metrics
            metrics.increment_counter(
                "crewgraph_workflow_scale_events_total",
                success_count,
                labels={"direction": "down", "predictive": str(predictive), "user": "Vatsal216"},
            )

            logger.info(
                f"Scaled DOWN by {success_count} instances "
                f"({'predictive' if predictive else 'reactive'}). "
                f"Total instances: {len(self.instances)}"
            )

    def _create_instance(self) -> Optional[WorkflowInstance]:
        """Create a new workflow instance"""
        try:
            instance_id = f"workflow_{int(time.time())}_{len(self.instances)}"
            workflow = self.workflow_factory()

            instance = WorkflowInstance(
                instance_id=instance_id, workflow=workflow, start_time=time.time()
            )

            self.instances[instance_id] = instance
            self.scaling_stats["current_instances"] = len(self.instances)

            logger.debug(f"Created workflow instance: {instance_id}")
            return instance

        except Exception as e:
            logger.error(f"Failed to create workflow instance: {e}")
            return None

    def _remove_instance(self, instance_id: str) -> bool:
        """Remove a workflow instance"""
        try:
            if instance_id not in self.instances:
                return False

            instance = self.instances[instance_id]

            # Graceful shutdown
            try:
                if hasattr(instance.workflow, "stop"):
                    instance.workflow.stop()
            except Exception as e:
                logger.warning(f"Error stopping workflow {instance_id}: {e}")

            # Remove from tracking
            del self.instances[instance_id]
            self.scaling_stats["current_instances"] = len(self.instances)

            logger.debug(f"Removed workflow instance: {instance_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove workflow instance {instance_id}: {e}")
            return False

    def _predict_scaling_need(self) -> ScalingDirection:
        """Predict future scaling needs based on historical data"""
        if len(self.metrics_history) < 10:
            return ScalingDirection.STABLE

        try:
            # Simple trend analysis (can be enhanced with ML)
            recent_metrics = list(self.metrics_history)[-10:]

            # Calculate trends
            cpu_trend = self._calculate_trend([m["cpu_percent"] for m in recent_metrics])
            memory_trend = self._calculate_trend([m["memory_percent"] for m in recent_metrics])
            queue_trend = self._calculate_trend([m["queue_size"] for m in recent_metrics])

            # Predict scaling need
            if cpu_trend > 2.0 or memory_trend > 2.0 or queue_trend > 1.0:
                return ScalingDirection.UP
            elif cpu_trend < -2.0 and memory_trend < -2.0 and queue_trend < -0.5:
                return ScalingDirection.DOWN
            else:
                return ScalingDirection.STABLE

        except Exception as e:
            logger.error(f"Error in predictive scaling: {e}")
            return ScalingDirection.STABLE

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend"""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x_squared_sum = sum(i * i for i in range(n))

        # Linear regression slope
        slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum * x_sum)
        return slope

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling system status"""
        return {
            "running": self._running,
            "total_instances": len(self.instances),
            "active_instances": len([i for i in self.instances.values() if i.status == "running"]),
            "queue_size": self.request_queue.qsize(),
            "scaling_rules": len([r for r in self.scaling_rules if r.enabled]),
            "predictive_enabled": self.enable_predictive,
            "statistics": self.scaling_stats.copy(),
            "last_scaling_events": list(self.scaling_history)[-5:],  # Last 5 events
            "created_by": "Vatsal216",
            "status_time": "2025-07-22 13:17:52",
        }


class WorkflowLoadBalancer:
    """Load balancer for workflow instances"""

    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.current_index = 0

    def select_instance(
        self, instances: Dict[str, WorkflowInstance], request_info: Optional[Dict] = None
    ) -> Optional[str]:
        """Select best instance for request"""
        if not instances:
            return None

        available_instances = [
            instance_id
            for instance_id, instance in instances.items()
            if instance.status == "running"
        ]

        if not available_instances:
            return None

        if self.strategy == "round_robin":
            selected = available_instances[self.current_index % len(available_instances)]
            self.current_index += 1
            return selected

        elif self.strategy == "least_loaded":
            # Select instance with lowest task count
            return min(available_instances, key=lambda id: instances[id].task_count)

        else:
            # Default to first available
            return available_instances[0]
