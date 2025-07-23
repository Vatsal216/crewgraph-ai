"""
Auto-Scaling Capabilities for CrewGraph AI
Provides intelligent auto-scaling with resource monitoring and dynamic adjustment.

Author: Vatsal216
Created: 2025-07-23
"""

import asyncio
import threading
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError
from ..config.enterprise_config import get_enterprise_config

logger = get_logger(__name__)


class ScalingDirection(Enum):
    """Auto-scaling directions"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(Enum):
    """Resource types for monitoring"""
    CPU = "cpu"
    MEMORY = "memory"
    WORKFLOW_QUEUE = "workflow_queue"
    AGENT_UTILIZATION = "agent_utilization"
    TASK_LATENCY = "task_latency"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    workflow_queue_size: int
    active_workflows: int
    active_agents: int
    agent_utilization: float
    avg_task_latency: float
    request_rate: float


@dataclass
class ScalingRule:
    """Auto-scaling rule definition"""
    name: str
    resource_type: ResourceType
    threshold_up: float
    threshold_down: float
    cooldown_seconds: int = 300  # 5 minutes default
    min_instances: int = 1
    max_instances: int = 10
    scale_factor: float = 1.5
    evaluation_window: int = 60  # seconds
    enabled: bool = True


@dataclass
class ScalingEvent:
    """Auto-scaling event record"""
    timestamp: float
    direction: ScalingDirection
    rule_name: str
    trigger_value: float
    threshold: float
    current_instances: int
    target_instances: int
    reason: str
    success: bool = False
    error: Optional[str] = None


class AutoScaler:
    """
    Intelligent auto-scaling system for CrewGraph AI workflows.
    
    Features:
    - CPU and memory monitoring
    - Workflow queue depth monitoring
    - Agent utilization tracking
    - Customizable scaling rules
    - Cooldown periods to prevent oscillation
    - Scaling event logging and analysis
    - Integration with distributed processing
    """
    
    def __init__(self, config=None):
        self.config = config or get_enterprise_config().scaling
        self.scaling_rules: List[ScalingRule] = []
        self.metrics_history: List[ResourceMetrics] = []
        self.scaling_events: List[ScalingEvent] = []
        self.current_instances = self.config.auto_scaling_min_instances
        self.target_instances = self.current_instances
        
        # Runtime state
        self._running = False
        self._monitor_task = None
        self._lock = threading.Lock()
        
        # Metrics collection
        self.metrics_callbacks: List[Callable[[], Dict[str, float]]] = []
        self.scaling_callbacks: List[Callable[[int, int], bool]] = []
        
        # Statistics
        self.stats = {
            "total_scale_events": 0,
            "scale_up_events": 0,
            "scale_down_events": 0,
            "failed_scale_events": 0,
            "start_time": time.time()
        }
        
        # Initialize default scaling rules
        self._initialize_default_rules()
        
        logger.info("AutoScaler initialized with default rules")
    
    def _initialize_default_rules(self):
        """Initialize default auto-scaling rules"""
        # CPU-based scaling
        self.add_scaling_rule(ScalingRule(
            name="cpu_scaling",
            resource_type=ResourceType.CPU,
            threshold_up=self.config.auto_scaling_target_cpu,
            threshold_down=self.config.auto_scaling_target_cpu * 0.5,
            cooldown_seconds=300,
            min_instances=self.config.auto_scaling_min_instances,
            max_instances=self.config.auto_scaling_max_instances,
            scale_factor=1.5
        ))
        
        # Memory-based scaling
        self.add_scaling_rule(ScalingRule(
            name="memory_scaling",
            resource_type=ResourceType.MEMORY,
            threshold_up=80.0,
            threshold_down=40.0,
            cooldown_seconds=300,
            min_instances=self.config.auto_scaling_min_instances,
            max_instances=self.config.auto_scaling_max_instances,
            scale_factor=1.3
        ))
        
        # Workflow queue-based scaling
        self.add_scaling_rule(ScalingRule(
            name="queue_scaling",
            resource_type=ResourceType.WORKFLOW_QUEUE,
            threshold_up=float(self.config.max_concurrent_workflows * 0.8),
            threshold_down=float(self.config.max_concurrent_workflows * 0.3),
            cooldown_seconds=180,  # Shorter cooldown for queue-based scaling
            min_instances=self.config.auto_scaling_min_instances,
            max_instances=self.config.auto_scaling_max_instances,
            scale_factor=2.0
        ))
        
        # Agent utilization scaling
        self.add_scaling_rule(ScalingRule(
            name="agent_utilization_scaling",
            resource_type=ResourceType.AGENT_UTILIZATION,
            threshold_up=85.0,
            threshold_down=30.0,
            cooldown_seconds=240,
            min_instances=self.config.auto_scaling_min_instances,
            max_instances=self.config.auto_scaling_max_instances,
            scale_factor=1.4
        ))
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule"""
        with self._lock:
            self.scaling_rules.append(rule)
            logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove a scaling rule"""
        with self._lock:
            self.scaling_rules = [rule for rule in self.scaling_rules if rule.name != rule_name]
            logger.info(f"Removed scaling rule: {rule_name}")
    
    def add_metrics_callback(self, callback: Callable[[], Dict[str, float]]):
        """Add callback for custom metrics collection"""
        self.metrics_callbacks.append(callback)
    
    def add_scaling_callback(self, callback: Callable[[int, int], bool]):
        """Add callback for custom scaling actions"""
        self.scaling_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start auto-scaling monitoring"""
        if self._running:
            logger.warning("Auto-scaling monitoring already running")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started auto-scaling monitoring")
    
    async def stop_monitoring(self):
        """Stop auto-scaling monitoring"""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped auto-scaling monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring and scaling loop"""
        while self._running:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Store metrics history
                with self._lock:
                    self.metrics_history.append(metrics)
                    # Keep only last hour of metrics
                    cutoff_time = time.time() - 3600
                    self.metrics_history = [
                        m for m in self.metrics_history 
                        if m.timestamp > cutoff_time
                    ]
                
                # Evaluate scaling rules
                scaling_decision = await self._evaluate_scaling_rules(metrics)
                
                # Execute scaling if needed
                if scaling_decision:
                    await self._execute_scaling(scaling_decision)
                
                # Wait before next evaluation
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-scaling monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / (1024 * 1024)
        
        # Default workflow metrics (would be replaced with actual metrics in real implementation)
        workflow_queue_size = 0
        active_workflows = self.current_instances
        active_agents = self.current_instances * 5  # Assume 5 agents per instance
        agent_utilization = min(100.0, cpu_percent * 1.2)  # Rough approximation
        avg_task_latency = 100.0  # ms - default value
        request_rate = 10.0  # req/sec - default value
        
        # Collect custom metrics from callbacks
        for callback in self.metrics_callbacks:
            try:
                custom_metrics = callback()
                workflow_queue_size = custom_metrics.get("workflow_queue_size", workflow_queue_size)
                active_workflows = custom_metrics.get("active_workflows", active_workflows)
                active_agents = custom_metrics.get("active_agents", active_agents)
                agent_utilization = custom_metrics.get("agent_utilization", agent_utilization)
                avg_task_latency = custom_metrics.get("avg_task_latency", avg_task_latency)
                request_rate = custom_metrics.get("request_rate", request_rate)
            except Exception as e:
                logger.warning(f"Error collecting custom metrics: {e}")
        
        return ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            workflow_queue_size=workflow_queue_size,
            active_workflows=active_workflows,
            active_agents=active_agents,
            agent_utilization=agent_utilization,
            avg_task_latency=avg_task_latency,
            request_rate=request_rate
        )
    
    async def _evaluate_scaling_rules(self, metrics: ResourceMetrics) -> Optional[ScalingEvent]:
        """Evaluate all scaling rules and return scaling decision"""
        current_time = time.time()
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if self._is_in_cooldown(rule, current_time):
                continue
            
            # Get current value for the resource type
            current_value = self._get_metric_value(metrics, rule.resource_type)
            
            # Determine scaling direction
            direction = None
            threshold = None
            
            if current_value > rule.threshold_up and self.current_instances < rule.max_instances:
                direction = ScalingDirection.UP
                threshold = rule.threshold_up
            elif current_value < rule.threshold_down and self.current_instances > rule.min_instances:
                direction = ScalingDirection.DOWN
                threshold = rule.threshold_down
            
            if direction:
                # Calculate target instances
                if direction == ScalingDirection.UP:
                    target_instances = min(
                        rule.max_instances,
                        max(self.current_instances + 1, int(self.current_instances * rule.scale_factor))
                    )
                else:  # ScalingDirection.DOWN
                    target_instances = max(
                        rule.min_instances,
                        int(self.current_instances / rule.scale_factor)
                    )
                
                return ScalingEvent(
                    timestamp=current_time,
                    direction=direction,
                    rule_name=rule.name,
                    trigger_value=current_value,
                    threshold=threshold,
                    current_instances=self.current_instances,
                    target_instances=target_instances,
                    reason=f"{rule.resource_type.value} {direction.value}: {current_value:.1f} vs {threshold:.1f}"
                )
        
        return None
    
    def _get_metric_value(self, metrics: ResourceMetrics, resource_type: ResourceType) -> float:
        """Extract metric value based on resource type"""
        if resource_type == ResourceType.CPU:
            return metrics.cpu_percent
        elif resource_type == ResourceType.MEMORY:
            return metrics.memory_percent
        elif resource_type == ResourceType.WORKFLOW_QUEUE:
            return float(metrics.workflow_queue_size)
        elif resource_type == ResourceType.AGENT_UTILIZATION:
            return metrics.agent_utilization
        elif resource_type == ResourceType.TASK_LATENCY:
            return metrics.avg_task_latency
        else:
            return 0.0
    
    def _is_in_cooldown(self, rule: ScalingRule, current_time: float) -> bool:
        """Check if rule is in cooldown period"""
        # Find the last scaling event for this rule
        for event in reversed(self.scaling_events):
            if event.rule_name == rule.name and event.success:
                time_since_last = current_time - event.timestamp
                return time_since_last < rule.cooldown_seconds
        
        return False
    
    async def _execute_scaling(self, scaling_event: ScalingEvent):
        """Execute the scaling decision"""
        logger.info(f"Executing scaling: {scaling_event.reason}")
        
        try:
            # Execute custom scaling callbacks
            success = True
            for callback in self.scaling_callbacks:
                try:
                    result = callback(self.current_instances, scaling_event.target_instances)
                    if not result:
                        success = False
                        break
                except Exception as e:
                    logger.error(f"Scaling callback failed: {e}")
                    success = False
                    break
            
            if success:
                # Update current instances
                old_instances = self.current_instances
                self.current_instances = scaling_event.target_instances
                self.target_instances = scaling_event.target_instances
                
                scaling_event.success = True
                
                # Update statistics
                with self._lock:
                    self.stats["total_scale_events"] += 1
                    if scaling_event.direction == ScalingDirection.UP:
                        self.stats["scale_up_events"] += 1
                    else:
                        self.stats["scale_down_events"] += 1
                
                logger.info(f"Scaling successful: {old_instances} -> {self.current_instances} instances")
            
            else:
                scaling_event.success = False
                scaling_event.error = "Scaling callback failed"
                self.stats["failed_scale_events"] += 1
                logger.error("Scaling failed: callback returned False")
        
        except Exception as e:
            scaling_event.success = False
            scaling_event.error = str(e)
            self.stats["failed_scale_events"] += 1
            logger.error(f"Scaling failed with exception: {e}")
        
        # Store scaling event
        with self._lock:
            self.scaling_events.append(scaling_event)
            # Keep only last 100 events
            if len(self.scaling_events) > 100:
                self.scaling_events = self.scaling_events[-100:]
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get the most recent metrics"""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, duration_seconds: int = 3600) -> List[ResourceMetrics]:
        """Get metrics history for specified duration"""
        cutoff_time = time.time() - duration_seconds
        
        with self._lock:
            return [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
    
    def get_scaling_events(self, duration_seconds: int = 3600) -> List[ScalingEvent]:
        """Get scaling events for specified duration"""
        cutoff_time = time.time() - duration_seconds
        
        with self._lock:
            return [
                e for e in self.scaling_events
                if e.timestamp > cutoff_time
            ]
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics"""
        uptime = time.time() - self.stats["start_time"]
        
        with self._lock:
            recent_events = self.get_scaling_events(3600)  # Last hour
            
            return {
                "current_state": {
                    "current_instances": self.current_instances,
                    "target_instances": self.target_instances,
                    "monitoring_enabled": self._running
                },
                "statistics": {
                    "uptime_seconds": uptime,
                    "total_scale_events": self.stats["total_scale_events"],
                    "scale_up_events": self.stats["scale_up_events"],
                    "scale_down_events": self.stats["scale_down_events"],
                    "failed_scale_events": self.stats["failed_scale_events"],
                    "success_rate": (
                        (self.stats["total_scale_events"] - self.stats["failed_scale_events"]) /
                        max(1, self.stats["total_scale_events"]) * 100
                    )
                },
                "recent_activity": {
                    "events_last_hour": len(recent_events),
                    "last_scale_event": recent_events[-1].__dict__ if recent_events else None
                },
                "configuration": {
                    "min_instances": self.config.auto_scaling_min_instances,
                    "max_instances": self.config.auto_scaling_max_instances,
                    "target_cpu": self.config.auto_scaling_target_cpu,
                    "auto_scaling_enabled": self.config.auto_scaling_enabled
                },
                "active_rules": [
                    {
                        "name": rule.name,
                        "resource_type": rule.resource_type.value,
                        "threshold_up": rule.threshold_up,
                        "threshold_down": rule.threshold_down,
                        "enabled": rule.enabled
                    }
                    for rule in self.scaling_rules
                ]
            }
    
    def get_resource_recommendations(self) -> List[Dict[str, Any]]:
        """Get resource optimization recommendations"""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        # Analyze recent metrics
        recent_metrics = self.get_metrics_history(1800)  # Last 30 minutes
        
        if len(recent_metrics) < 5:
            return recommendations
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_queue = sum(m.workflow_queue_size for m in recent_metrics) / len(recent_metrics)
        
        # CPU recommendations
        if avg_cpu < 30:
            recommendations.append({
                "type": "optimization",
                "resource": "cpu",
                "current_avg": avg_cpu,
                "recommendation": "Consider reducing instances or increasing workload",
                "priority": "low"
            })
        elif avg_cpu > 80:
            recommendations.append({
                "type": "scaling",
                "resource": "cpu", 
                "current_avg": avg_cpu,
                "recommendation": "Consider increasing instances or optimizing CPU-intensive tasks",
                "priority": "high"
            })
        
        # Memory recommendations
        if avg_memory < 40:
            recommendations.append({
                "type": "optimization",
                "resource": "memory",
                "current_avg": avg_memory,
                "recommendation": "Memory utilization is low, consider memory-optimized instances",
                "priority": "low"
            })
        elif avg_memory > 85:
            recommendations.append({
                "type": "scaling",
                "resource": "memory",
                "current_avg": avg_memory,
                "recommendation": "High memory usage detected, consider scaling up",
                "priority": "high"
            })
        
        # Queue recommendations
        if avg_queue > self.config.max_concurrent_workflows * 0.7:
            recommendations.append({
                "type": "scaling",
                "resource": "workflow_queue",
                "current_avg": avg_queue,
                "recommendation": "Queue backlog detected, consider increasing parallel processing",
                "priority": "medium"
            })
        
        return recommendations
    
    async def force_scale(self, target_instances: int, reason: str = "Manual scaling") -> bool:
        """Force scaling to specific instance count"""
        if target_instances < 1:
            raise ValueError("Target instances must be at least 1")
        
        scaling_event = ScalingEvent(
            timestamp=time.time(),
            direction=ScalingDirection.UP if target_instances > self.current_instances else ScalingDirection.DOWN,
            rule_name="manual_scaling",
            trigger_value=float(target_instances),
            threshold=float(self.current_instances),
            current_instances=self.current_instances,
            target_instances=target_instances,
            reason=reason
        )
        
        await self._execute_scaling(scaling_event)
        return scaling_event.success


# Global auto-scaler instance
_global_auto_scaler: Optional[AutoScaler] = None
_scaler_lock = threading.Lock()


def get_auto_scaler() -> AutoScaler:
    """Get global auto-scaler instance"""
    global _global_auto_scaler
    
    with _scaler_lock:
        if _global_auto_scaler is None:
            _global_auto_scaler = AutoScaler()
        
        return _global_auto_scaler


# Convenience functions
async def start_auto_scaling():
    """Start auto-scaling monitoring"""
    scaler = get_auto_scaler()
    await scaler.start_monitoring()


async def stop_auto_scaling():
    """Stop auto-scaling monitoring"""
    scaler = get_auto_scaler()
    await scaler.stop_monitoring()


def get_scaling_status() -> Dict[str, Any]:
    """Get current auto-scaling status"""
    scaler = get_auto_scaler()
    return scaler.get_scaling_stats()


async def manual_scale(target_instances: int, reason: str = "Manual scaling") -> bool:
    """Manually scale to target instance count"""
    scaler = get_auto_scaler()
    return await scaler.force_scale(target_instances, reason)