"""
Auto-Scaling Algorithms for CrewGraph AI

Implements intelligent auto-scaling with cost optimization, predictive scaling,
and performance vs. cost trade-off analysis.

Author: Vatsal216
Created: 2025-07-23 17:35:00 UTC
"""

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple, Callable

from .resource_manager import ResourceManager, ResourceMetrics, ScalingRecommendation, ResourceType, ScalingDirection
from ..types import WorkflowId
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ScalingPolicy(Enum):
    """Auto-scaling policy types."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    ADAPTIVE = "adaptive"


class ScalingTrigger(Enum):
    """Auto-scaling trigger conditions."""
    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    COST_THRESHOLD = "cost_threshold"
    SCHEDULED_TIME = "scheduled_time"


@dataclass
class ScalingRule:
    """Auto-scaling rule definition."""
    
    name: str
    trigger_type: ScalingTrigger
    threshold_value: float
    scaling_direction: ScalingDirection
    scaling_amount: float  # Factor or absolute amount
    cooldown_minutes: int
    enabled: bool = True
    priority: int = 5
    
    def evaluate(self, current_value: float) -> bool:
        """Evaluate if scaling rule should trigger."""
        if not self.enabled:
            return False
            
        if self.scaling_direction == ScalingDirection.UP:
            return current_value >= self.threshold_value
        elif self.scaling_direction == ScalingDirection.DOWN:
            return current_value <= self.threshold_value
        
        return False


@dataclass
class ScalingEvent:
    """Auto-scaling event record."""
    
    timestamp: datetime
    trigger: ScalingTrigger
    resource_type: ResourceType
    scaling_direction: ScalingDirection
    scaling_factor: float
    original_capacity: float
    new_capacity: float
    cost_impact: float
    success: bool
    reason: str


@dataclass
class CostOptimizationSettings:
    """Cost optimization configuration."""
    
    max_hourly_cost: float
    cost_efficiency_target: float  # Cost per unit of work
    enable_spot_instances: bool
    enable_time_shifting: bool
    off_peak_hours: List[int]
    weekend_discount_factor: float


@dataclass
class ScalingMetrics:
    """Scaling performance metrics."""
    
    total_scaling_events: int
    successful_scaling_events: int
    average_scaling_latency: float
    cost_savings_percentage: float
    performance_improvement_percentage: float
    stability_score: float


class AutoScaler:
    """
    Intelligent auto-scaling system with ML-driven predictions and cost optimization.
    
    Provides reactive and predictive scaling based on multiple metrics,
    with sophisticated cost optimization and performance trade-off analysis.
    """
    
    def __init__(
        self,
        resource_manager: ResourceManager,
        cost_settings: Optional[CostOptimizationSettings] = None
    ):
        """Initialize auto-scaler."""
        self.resource_manager = resource_manager
        self.cost_settings = cost_settings or CostOptimizationSettings(
            max_hourly_cost=10.0,
            cost_efficiency_target=0.05,
            enable_spot_instances=True,
            enable_time_shifting=True,
            off_peak_hours=[0, 1, 2, 3, 4, 5, 6, 22, 23],
            weekend_discount_factor=0.8
        )
        
        # Scaling rules and policies
        self.scaling_rules: List[ScalingRule] = []
        self.active_policy = ScalingPolicy.ADAPTIVE
        
        # Event tracking
        self.scaling_events: deque = deque(maxlen=1000)
        self.cooldown_tracker: Dict[str, datetime] = {}
        
        # Thread safety
        self._lock = Lock()
        self._running = False
        self._monitor_thread: Optional[Thread] = None
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=100)
        self.cost_history: deque = deque(maxlen=100)
        
        # Predictive models (simplified)
        self.usage_predictions: Dict[str, List[float]] = defaultdict(list)
        self.scaling_effectiveness: Dict[str, float] = defaultdict(lambda: 0.8)
        
        self._initialize_default_rules()
        logger.info("Auto-scaler initialized")
    
    def _initialize_default_rules(self):
        """Initialize default scaling rules."""
        self.scaling_rules = [
            # CPU scaling rules
            ScalingRule(
                name="cpu_scale_up",
                trigger_type=ScalingTrigger.CPU_THRESHOLD,
                threshold_value=0.85,
                scaling_direction=ScalingDirection.UP,
                scaling_amount=1.5,
                cooldown_minutes=5,
                priority=9
            ),
            ScalingRule(
                name="cpu_scale_down",
                trigger_type=ScalingTrigger.CPU_THRESHOLD,
                threshold_value=0.30,
                scaling_direction=ScalingDirection.DOWN,
                scaling_amount=0.7,
                cooldown_minutes=10,
                priority=3
            ),
            
            # Memory scaling rules
            ScalingRule(
                name="memory_scale_up",
                trigger_type=ScalingTrigger.MEMORY_THRESHOLD,
                threshold_value=0.90,
                scaling_direction=ScalingDirection.UP,
                scaling_amount=1.3,
                cooldown_minutes=3,
                priority=10
            ),
            ScalingRule(
                name="memory_scale_down",
                trigger_type=ScalingTrigger.MEMORY_THRESHOLD,
                threshold_value=0.40,
                scaling_direction=ScalingDirection.DOWN,
                scaling_amount=0.8,
                cooldown_minutes=15,
                priority=2
            ),
            
            # Cost-based scaling
            ScalingRule(
                name="cost_scale_down",
                trigger_type=ScalingTrigger.COST_THRESHOLD,
                threshold_value=self.cost_settings.max_hourly_cost,
                scaling_direction=ScalingDirection.DOWN,
                scaling_amount=0.8,
                cooldown_minutes=5,
                priority=8
            )
        ]
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring and auto-scaling."""
        if self._running:
            logger.warning("Auto-scaler monitoring already running")
            return
        
        self._running = True
        self._monitor_thread = Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Auto-scaler monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Auto-scaler monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop for auto-scaling."""
        while self._running:
            try:
                self._evaluate_scaling_conditions()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _evaluate_scaling_conditions(self):
        """Evaluate all scaling conditions and trigger scaling if needed."""
        current_metrics = self.resource_manager.get_current_metrics()
        
        # Store metrics for analysis
        self.metrics_history.append(current_metrics)
        
        # Evaluate each scaling rule
        triggered_rules = []
        
        for rule in self.scaling_rules:
            if self._is_in_cooldown(rule.name):
                continue
            
            current_value = self._get_metric_value(current_metrics, rule.trigger_type)
            
            if rule.evaluate(current_value):
                triggered_rules.append((rule, current_value))
        
        # Sort by priority and execute
        triggered_rules.sort(key=lambda x: x[0].priority, reverse=True)
        
        for rule, current_value in triggered_rules[:3]:  # Limit to top 3 rules
            self._execute_scaling_action(rule, current_value, current_metrics)
    
    def _get_metric_value(self, metrics: ResourceMetrics, trigger_type: ScalingTrigger) -> float:
        """Get metric value for trigger evaluation."""
        if trigger_type == ScalingTrigger.CPU_THRESHOLD:
            return metrics.cpu_usage
        elif trigger_type == ScalingTrigger.MEMORY_THRESHOLD:
            return metrics.memory_usage
        elif trigger_type == ScalingTrigger.COST_THRESHOLD:
            return self.resource_manager._calculate_current_cost()
        else:
            return 0.0
    
    def _execute_scaling_action(
        self,
        rule: ScalingRule,
        current_value: float,
        metrics: ResourceMetrics
    ):
        """Execute a scaling action based on triggered rule."""
        with self._lock:
            # Determine resource type from trigger
            if rule.trigger_type in [ScalingTrigger.CPU_THRESHOLD]:
                resource_type = ResourceType.CPU
            elif rule.trigger_type in [ScalingTrigger.MEMORY_THRESHOLD]:
                resource_type = ResourceType.MEMORY
            else:
                resource_type = ResourceType.CPU  # Default
            
            # Apply cost optimization considerations
            if self._should_apply_cost_optimization(rule, current_value):
                rule = self._optimize_scaling_for_cost(rule)
            
            # Execute scaling
            success = self._perform_scaling(resource_type, rule.scaling_amount, rule.scaling_direction)
            
            # Record scaling event
            event = ScalingEvent(
                timestamp=datetime.now(),
                trigger=rule.trigger_type,
                resource_type=resource_type,
                scaling_direction=rule.scaling_direction,
                scaling_factor=rule.scaling_amount,
                original_capacity=current_value,
                new_capacity=current_value * rule.scaling_amount if success else current_value,
                cost_impact=self._calculate_scaling_cost_impact(rule),
                success=success,
                reason=f"Rule '{rule.name}' triggered at {current_value:.2f}"
            )
            
            self.scaling_events.append(event)
            
            # Set cooldown
            self.cooldown_tracker[rule.name] = datetime.now() + timedelta(minutes=rule.cooldown_minutes)
            
            logger.info(f"Scaling action executed: {rule.name} - Success: {success}")
    
    def _perform_scaling(
        self,
        resource_type: ResourceType,
        scaling_factor: float,
        direction: ScalingDirection
    ) -> bool:
        """Perform actual scaling operation."""
        try:
            # In a real implementation, this would interface with cloud APIs
            # or container orchestrators to actually scale resources
            
            # For now, we'll simulate scaling by adjusting internal limits
            if resource_type == ResourceType.CPU:
                current_max = self.resource_manager.max_cpu_cores
                if direction == ScalingDirection.UP:
                    new_max = current_max * scaling_factor
                else:
                    new_max = current_max * scaling_factor
                
                self.resource_manager.max_cpu_cores = max(1, int(new_max))
                
            elif resource_type == ResourceType.MEMORY:
                current_max = self.resource_manager.max_memory_gb
                if direction == ScalingDirection.UP:
                    new_max = current_max * scaling_factor
                else:
                    new_max = current_max * scaling_factor
                
                self.resource_manager.max_memory_gb = max(1.0, new_max)
            
            return True
            
        except Exception as e:
            logger.error(f"Scaling operation failed: {e}")
            return False
    
    def _should_apply_cost_optimization(self, rule: ScalingRule, current_value: float) -> bool:
        """Determine if cost optimization should be applied to scaling decision."""
        current_cost = self.resource_manager._calculate_current_cost()
        
        # Apply cost optimization if:
        # 1. Scaling up and near cost limit
        # 2. Cost efficiency is below target
        # 3. Rule is not high priority
        
        if rule.scaling_direction == ScalingDirection.UP:
            cost_ratio = current_cost / self.cost_settings.max_hourly_cost
            if cost_ratio > 0.8 and rule.priority < 8:
                return True
        
        return False
    
    def _optimize_scaling_for_cost(self, rule: ScalingRule) -> ScalingRule:
        """Optimize scaling rule for cost efficiency."""
        # Reduce scaling amount to minimize cost impact
        optimized_rule = ScalingRule(
            name=f"{rule.name}_cost_optimized",
            trigger_type=rule.trigger_type,
            threshold_value=rule.threshold_value,
            scaling_direction=rule.scaling_direction,
            scaling_amount=rule.scaling_amount * 0.7,  # Reduce scaling by 30%
            cooldown_minutes=rule.cooldown_minutes,
            enabled=rule.enabled,
            priority=rule.priority
        )
        
        logger.info(f"Applied cost optimization to rule {rule.name}")
        return optimized_rule
    
    def _calculate_scaling_cost_impact(self, rule: ScalingRule) -> float:
        """Calculate cost impact of scaling action."""
        current_cost = self.resource_manager._calculate_current_cost()
        
        if rule.scaling_direction == ScalingDirection.UP:
            cost_impact = current_cost * (rule.scaling_amount - 1.0)
        else:
            cost_impact = current_cost * (1.0 - rule.scaling_amount)
        
        return cost_impact
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if scaling rule is in cooldown period."""
        if rule_name not in self.cooldown_tracker:
            return False
        
        return datetime.now() < self.cooldown_tracker[rule_name]
    
    def predict_scaling_needs(self, horizon_minutes: int = 60) -> List[Dict[str, Any]]:
        """Predict future scaling needs using ML models."""
        # Get resource usage forecast
        forecast = self.resource_manager.predict_future_usage(horizon_minutes)
        
        scaling_predictions = []
        
        # Analyze CPU predictions
        cpu_predictions = forecast.predicted_cpu
        max_cpu = max(cpu_predictions)
        max_cpu_time = cpu_predictions.index(max_cpu)
        
        if max_cpu > 0.85:
            scaling_predictions.append({
                "resource_type": "cpu",
                "predicted_trigger_time": datetime.now() + timedelta(minutes=max_cpu_time),
                "predicted_value": max_cpu,
                "recommended_action": "scale_up",
                "confidence": forecast.forecast_accuracy,
                "proactive_scaling_factor": 1.3
            })
        
        # Analyze memory predictions
        memory_predictions = forecast.predicted_memory
        max_memory = max(memory_predictions)
        max_memory_time = memory_predictions.index(max_memory)
        
        if max_memory > 0.90:
            scaling_predictions.append({
                "resource_type": "memory",
                "predicted_trigger_time": datetime.now() + timedelta(minutes=max_memory_time),
                "predicted_value": max_memory,
                "recommended_action": "scale_up",
                "confidence": forecast.forecast_accuracy,
                "proactive_scaling_factor": 1.2
            })
        
        # Check for scale-down opportunities
        min_cpu = min(cpu_predictions[30:])  # Check latter half of prediction window
        if min_cpu < 0.30:
            scaling_predictions.append({
                "resource_type": "cpu",
                "predicted_trigger_time": datetime.now() + timedelta(minutes=30),
                "predicted_value": min_cpu,
                "recommended_action": "scale_down",
                "confidence": forecast.forecast_accuracy * 0.8,
                "proactive_scaling_factor": 0.8
            })
        
        return scaling_predictions
    
    def analyze_cost_vs_performance(self) -> Dict[str, Any]:
        """Analyze cost vs performance trade-offs for scaling decisions."""
        if len(self.scaling_events) < 5:
            return {
                "analysis_available": False,
                "reason": "Insufficient scaling history"
            }
        
        recent_events = list(self.scaling_events)[-20:]
        
        # Analyze scaling effectiveness
        scale_up_events = [e for e in recent_events if e.scaling_direction == ScalingDirection.UP]
        scale_down_events = [e for e in recent_events if e.scaling_direction == ScalingDirection.DOWN]
        
        # Calculate metrics
        total_cost_impact = sum(e.cost_impact for e in recent_events)
        successful_events = len([e for e in recent_events if e.success])
        success_rate = successful_events / len(recent_events)
        
        # Performance impact analysis (simplified)
        avg_scale_up_factor = (
            sum(e.scaling_factor for e in scale_up_events) / len(scale_up_events)
            if scale_up_events else 1.0
        )
        
        avg_scale_down_factor = (
            sum(e.scaling_factor for e in scale_down_events) / len(scale_down_events) 
            if scale_down_events else 1.0
        )
        
        # Cost efficiency calculation
        current_cost = self.resource_manager._calculate_current_cost()
        cost_efficiency = self._calculate_cost_efficiency()
        
        analysis = {
            "analysis_available": True,
            "time_period": "last_20_events",
            "scaling_summary": {
                "total_events": len(recent_events),
                "scale_up_events": len(scale_up_events),
                "scale_down_events": len(scale_down_events),
                "success_rate": success_rate
            },
            "cost_analysis": {
                "total_cost_impact": total_cost_impact,
                "current_hourly_cost": current_cost,
                "cost_efficiency": cost_efficiency,
                "cost_vs_target": current_cost / self.cost_settings.max_hourly_cost
            },
            "performance_analysis": {
                "avg_scale_up_factor": avg_scale_up_factor,
                "avg_scale_down_factor": avg_scale_down_factor,
                "scaling_responsiveness": self._calculate_scaling_responsiveness()
            },
            "recommendations": self._generate_optimization_recommendations(
                current_cost, cost_efficiency, success_rate
            )
        }
        
        return analysis
    
    def _calculate_cost_efficiency(self) -> float:
        """Calculate current cost efficiency metric."""
        current_metrics = self.resource_manager.get_current_metrics()
        current_cost = self.resource_manager._calculate_current_cost()
        
        # Simple efficiency: performance per dollar
        # Higher utilization with lower cost is more efficient
        performance_score = (current_metrics.cpu_usage + current_metrics.memory_usage) / 2.0
        efficiency = performance_score / max(current_cost, 0.01)
        
        return efficiency
    
    def _calculate_scaling_responsiveness(self) -> float:
        """Calculate how responsive scaling has been to load changes."""
        if len(self.scaling_events) < 3:
            return 0.5
        
        # Analyze time between trigger and scaling action
        recent_events = list(self.scaling_events)[-10:]
        successful_events = [e for e in recent_events if e.success]
        
        if not successful_events:
            return 0.2
        
        # In a real implementation, this would measure actual response times
        # For now, return a simulated responsiveness score
        return 0.8
    
    def _generate_optimization_recommendations(
        self,
        current_cost: float,
        cost_efficiency: float,
        success_rate: float
    ) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Cost optimization recommendations
        if current_cost > self.cost_settings.max_hourly_cost * 0.9:
            recommendations.append(
                "Consider enabling more aggressive scale-down policies to reduce costs"
            )
        
        if cost_efficiency < self.cost_settings.cost_efficiency_target:
            recommendations.append(
                "Cost efficiency below target - review resource allocation strategies"
            )
        
        # Performance optimization recommendations
        if success_rate < 0.8:
            recommendations.append(
                "Scaling success rate is low - review scaling rules and thresholds"
            )
        
        # Time-based recommendations
        current_hour = datetime.now().hour
        if current_hour in self.cost_settings.off_peak_hours:
            recommendations.append(
                "Consider scaling up during off-peak hours for better cost efficiency"
            )
        
        return recommendations
    
    def add_custom_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules.append(rule)
        logger.info(f"Added custom scaling rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a scaling rule by name."""
        self.scaling_rules = [r for r in self.scaling_rules if r.name != rule_name]
        logger.info(f"Removed scaling rule: {rule_name}")
    
    def update_cost_settings(self, new_settings: CostOptimizationSettings):
        """Update cost optimization settings."""
        self.cost_settings = new_settings
        logger.info("Updated cost optimization settings")
    
    def get_scaling_metrics(self) -> ScalingMetrics:
        """Get comprehensive scaling metrics."""
        total_events = len(self.scaling_events)
        successful_events = len([e for e in self.scaling_events if e.success])
        
        # Calculate average latency (simplified)
        avg_latency = 2.5  # Simulated average scaling latency in seconds
        
        # Cost savings calculation
        cost_savings = self._calculate_cost_savings_percentage()
        
        # Performance improvement (simplified)
        performance_improvement = self._calculate_performance_improvement()
        
        # Stability score based on scaling frequency
        stability_score = self._calculate_stability_score()
        
        return ScalingMetrics(
            total_scaling_events=total_events,
            successful_scaling_events=successful_events,
            average_scaling_latency=avg_latency,
            cost_savings_percentage=cost_savings,
            performance_improvement_percentage=performance_improvement,
            stability_score=stability_score
        )
    
    def _calculate_cost_savings_percentage(self) -> float:
        """Calculate cost savings from auto-scaling."""
        if len(self.scaling_events) < 5:
            return 0.0
        
        # Compare current cost to baseline without scaling
        current_cost = self.resource_manager._calculate_current_cost()
        
        # Estimate baseline cost (what cost would be without scaling)
        baseline_cost = current_cost * 1.3  # Assume 30% higher without scaling
        
        savings_percentage = ((baseline_cost - current_cost) / baseline_cost) * 100
        return max(0.0, savings_percentage)
    
    def _calculate_performance_improvement(self) -> float:
        """Calculate performance improvement from auto-scaling."""
        # Simplified calculation based on successful scale-up events
        recent_events = list(self.scaling_events)[-10:]
        scale_up_events = [e for e in recent_events if e.scaling_direction == ScalingDirection.UP and e.success]
        
        if not scale_up_events:
            return 0.0
        
        # Estimate performance improvement from scaling up
        avg_improvement = sum(
            (e.scaling_factor - 1.0) * 100 for e in scale_up_events
        ) / len(scale_up_events)
        
        return min(50.0, avg_improvement)  # Cap at 50%
    
    def _calculate_stability_score(self) -> float:
        """Calculate system stability score."""
        if len(self.scaling_events) < 5:
            return 1.0
        
        # Lower scaling frequency indicates higher stability
        recent_events = list(self.scaling_events)[-20:]
        time_span = (recent_events[-1].timestamp - recent_events[0].timestamp).total_seconds() / 3600  # hours
        
        scaling_frequency = len(recent_events) / max(time_span, 1.0)  # events per hour
        
        # Lower frequency = higher stability (capped at 1.0)
        stability = max(0.0, 1.0 - (scaling_frequency / 10.0))
        
        return stability
    
    def export_scaling_history(self, filepath: str):
        """Export scaling history to JSON file."""
        history_data = {
            "export_timestamp": datetime.now().isoformat(),
            "scaling_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "trigger": event.trigger.value,
                    "resource_type": event.resource_type.value,
                    "scaling_direction": event.scaling_direction.value,
                    "scaling_factor": event.scaling_factor,
                    "cost_impact": event.cost_impact,
                    "success": event.success,
                    "reason": event.reason
                }
                for event in self.scaling_events
            ],
            "metrics": self.get_scaling_metrics().__dict__
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2)
            logger.info(f"Scaling history exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export scaling history: {e}")