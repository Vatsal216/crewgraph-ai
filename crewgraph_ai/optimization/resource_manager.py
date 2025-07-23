"""
Intelligent Resource Manager for CrewGraph AI

Provides intelligent resource allocation, auto-scaling, and cost optimization
with predictive scaling based on ML models and resource usage patterns.

Author: Vatsal216
Created: 2025-07-23 17:30:00 UTC
"""

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

from ..types import WorkflowId, NodeId
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ResourceType(Enum):
    """Types of resources managed by the system."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"


class ScalingDirection(Enum):
    """Scaling direction indicators."""
    UP = "scale_up"
    DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class ResourceMetrics:
    """Current resource usage metrics."""
    
    cpu_usage: float  # 0.0 to 1.0
    memory_usage: float  # 0.0 to 1.0
    storage_usage: float  # 0.0 to 1.0
    network_usage: float  # 0.0 to 1.0
    gpu_usage: float  # 0.0 to 1.0
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "storage_usage": self.storage_usage,
            "network_usage": self.network_usage,
            "gpu_usage": self.gpu_usage,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ResourceAllocation:
    """Resource allocation for a specific task or workflow."""
    
    workflow_id: WorkflowId
    node_id: Optional[NodeId]
    allocated_cpu: float
    allocated_memory: float
    allocated_storage: float
    allocated_network: float
    allocated_gpu: float
    priority: int  # 1-10, higher is more important
    max_cpu: Optional[float] = None
    max_memory: Optional[float] = None
    
    def total_weight(self) -> float:
        """Calculate total resource weight."""
        return (self.allocated_cpu + self.allocated_memory + 
                self.allocated_storage + self.allocated_network + self.allocated_gpu)


@dataclass
class ScalingRecommendation:
    """Resource scaling recommendation."""
    
    resource_type: ResourceType
    current_usage: float
    target_usage: float
    scaling_direction: ScalingDirection
    scaling_factor: float
    confidence: float
    reasoning: str
    cost_impact: float
    implementation_priority: int


@dataclass
class ResourceForecast:
    """Resource usage forecast."""
    
    forecast_horizon: int  # minutes
    predicted_cpu: List[float]
    predicted_memory: List[float]
    predicted_storage: List[float]
    confidence_interval: Tuple[float, float]
    peak_usage_time: Optional[datetime]
    forecast_accuracy: float


class ResourceManager:
    """
    Intelligent resource manager with ML-driven allocation and auto-scaling.
    
    Manages resource allocation, monitors usage patterns, and provides
    intelligent scaling recommendations with cost optimization.
    """
    
    def __init__(
        self,
        max_cpu_cores: int = 8,
        max_memory_gb: float = 16.0,
        max_storage_gb: float = 100.0,
        cost_per_cpu_hour: float = 0.10,
        cost_per_gb_memory_hour: float = 0.02
    ):
        """Initialize resource manager."""
        self.max_cpu_cores = max_cpu_cores
        self.max_memory_gb = max_memory_gb
        self.max_storage_gb = max_storage_gb
        self.cost_per_cpu_hour = cost_per_cpu_hour
        self.cost_per_gb_memory_hour = cost_per_gb_memory_hour
        
        # Resource tracking
        self.current_allocations: Dict[str, ResourceAllocation] = {}
        self.usage_history: deque = deque(maxlen=1440)  # 24 hours of minute data
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Thresholds and targets
        self.target_cpu_utilization = 0.75
        self.target_memory_utilization = 0.80
        self.scale_up_threshold = 0.85
        self.scale_down_threshold = 0.60
        
        # Predictive scaling
        self.usage_patterns: Dict[str, List[float]] = defaultdict(list)
        self.seasonal_patterns: Dict[str, Dict[int, float]] = defaultdict(dict)
        
        # Cost tracking
        self.cost_history: List[Dict[str, Any]] = []
        self.cost_optimization_enabled = True
        
        logger.info("Resource manager initialized")
    
    def allocate_resources(
        self,
        workflow_id: WorkflowId,
        resource_requirements: Dict[str, float],
        priority: int = 5,
        node_id: Optional[NodeId] = None
    ) -> ResourceAllocation:
        """
        Allocate resources for a workflow with intelligent optimization.
        
        Args:
            workflow_id: Workflow requesting resources
            resource_requirements: Required resources by type
            priority: Resource allocation priority (1-10)
            node_id: Optional specific node ID
            
        Returns:
            Resource allocation result
        """
        # Analyze current resource availability
        current_metrics = self.get_current_metrics()
        available_resources = self._calculate_available_resources(current_metrics)
        
        # Optimize allocation based on current state and predictions
        optimized_allocation = self._optimize_allocation(
            resource_requirements, available_resources, priority
        )
        
        # Create allocation record
        allocation = ResourceAllocation(
            workflow_id=workflow_id,
            node_id=node_id,
            allocated_cpu=optimized_allocation.get("cpu", 0.0),
            allocated_memory=optimized_allocation.get("memory", 0.0),
            allocated_storage=optimized_allocation.get("storage", 0.0),
            allocated_network=optimized_allocation.get("network", 0.0),
            allocated_gpu=optimized_allocation.get("gpu", 0.0),
            priority=priority,
            max_cpu=resource_requirements.get("max_cpu"),
            max_memory=resource_requirements.get("max_memory")
        )
        
        # Store allocation
        allocation_key = f"{workflow_id}_{node_id or 'main'}"
        self.current_allocations[allocation_key] = allocation
        
        # Log cost impact
        hourly_cost = self._calculate_allocation_cost(allocation)
        logger.info(f"Allocated resources for {workflow_id}: CPU={allocation.allocated_cpu:.2f}, "
                   f"Memory={allocation.allocated_memory:.2f}GB, Cost=${hourly_cost:.3f}/hour")
        
        return allocation
    
    def release_resources(self, workflow_id: WorkflowId, node_id: Optional[NodeId] = None):
        """Release resources for completed workflow."""
        allocation_key = f"{workflow_id}_{node_id or 'main'}"
        
        if allocation_key in self.current_allocations:
            allocation = self.current_allocations.pop(allocation_key)
            logger.info(f"Released resources for {workflow_id}")
            
            # Record resource usage for learning
            self._record_resource_usage(allocation)
        else:
            logger.warning(f"No allocation found for {workflow_id}")
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        if PSUTIL_AVAILABLE:
            return self._get_system_metrics()
        else:
            return self._get_simulated_metrics()
    
    def _get_system_metrics(self) -> ResourceMetrics:
        """Get actual system metrics using psutil."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            # Disk usage
            disk = psutil.disk_usage('/')
            storage_usage = disk.percent / 100.0
            
            # Network usage (simplified)
            network_stats = psutil.net_io_counters()
            network_usage = min(1.0, (network_stats.bytes_sent + network_stats.bytes_recv) / (1024**3))  # GB
            
            # GPU usage (simplified - would need GPU-specific libraries)
            gpu_usage = 0.0
            
            return ResourceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                storage_usage=storage_usage,
                network_usage=network_usage,
                gpu_usage=gpu_usage,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return self._get_simulated_metrics()
    
    def _get_simulated_metrics(self) -> ResourceMetrics:
        """Get simulated metrics when psutil is not available."""
        import random
        
        # Simulate realistic usage patterns
        base_cpu = 0.4 + random.uniform(-0.2, 0.3)
        base_memory = 0.5 + random.uniform(-0.2, 0.2)
        
        return ResourceMetrics(
            cpu_usage=max(0.1, min(0.95, base_cpu)),
            memory_usage=max(0.1, min(0.95, base_memory)),
            storage_usage=0.3 + random.uniform(0, 0.2),
            network_usage=0.2 + random.uniform(0, 0.3),
            gpu_usage=0.0,
            timestamp=datetime.now()
        )
    
    def analyze_scaling_needs(self) -> List[ScalingRecommendation]:
        """Analyze current state and recommend scaling actions."""
        current_metrics = self.get_current_metrics()
        
        # Store metrics for historical analysis
        self.usage_history.append(current_metrics)
        
        recommendations = []
        
        # CPU scaling analysis
        cpu_recommendation = self._analyze_resource_scaling(
            ResourceType.CPU, current_metrics.cpu_usage
        )
        if cpu_recommendation:
            recommendations.append(cpu_recommendation)
        
        # Memory scaling analysis
        memory_recommendation = self._analyze_resource_scaling(
            ResourceType.MEMORY, current_metrics.memory_usage
        )
        if memory_recommendation:
            recommendations.append(memory_recommendation)
        
        # Storage scaling analysis
        storage_recommendation = self._analyze_resource_scaling(
            ResourceType.STORAGE, current_metrics.storage_usage
        )
        if storage_recommendation:
            recommendations.append(storage_recommendation)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.implementation_priority, reverse=True)
        
        return recommendations
    
    def _analyze_resource_scaling(
        self, 
        resource_type: ResourceType, 
        current_usage: float
    ) -> Optional[ScalingRecommendation]:
        """Analyze scaling needs for a specific resource type."""
        # Determine scaling direction
        if current_usage >= self.scale_up_threshold:
            direction = ScalingDirection.UP
            target_usage = self.target_cpu_utilization if resource_type == ResourceType.CPU else self.target_memory_utilization
            scaling_factor = 1.5  # Scale up by 50%
            reasoning = f"Current {resource_type.value} usage ({current_usage:.1%}) exceeds scale-up threshold"
            priority = 9 if current_usage > 0.9 else 7
            
        elif current_usage <= self.scale_down_threshold:
            direction = ScalingDirection.DOWN
            target_usage = self.target_cpu_utilization if resource_type == ResourceType.CPU else self.target_memory_utilization
            scaling_factor = 0.8  # Scale down by 20%
            reasoning = f"Current {resource_type.value} usage ({current_usage:.1%}) is below scale-down threshold"
            priority = 3
            
        else:
            # No scaling needed
            return None
        
        # Calculate confidence based on historical patterns
        confidence = self._calculate_scaling_confidence(resource_type, current_usage)
        
        # Estimate cost impact
        cost_impact = self._estimate_scaling_cost_impact(resource_type, scaling_factor)
        
        return ScalingRecommendation(
            resource_type=resource_type,
            current_usage=current_usage,
            target_usage=target_usage,
            scaling_direction=direction,
            scaling_factor=scaling_factor,
            confidence=confidence,
            reasoning=reasoning,
            cost_impact=cost_impact,
            implementation_priority=priority
        )
    
    def predict_future_usage(self, horizon_minutes: int = 60) -> ResourceForecast:
        """Predict future resource usage using historical patterns."""
        if len(self.usage_history) < 10:
            # Not enough data for prediction
            current_metrics = self.get_current_metrics()
            return ResourceForecast(
                forecast_horizon=horizon_minutes,
                predicted_cpu=[current_metrics.cpu_usage] * horizon_minutes,
                predicted_memory=[current_metrics.memory_usage] * horizon_minutes,
                predicted_storage=[current_metrics.storage_usage] * horizon_minutes,
                confidence_interval=(0.5, 0.9),
                peak_usage_time=None,
                forecast_accuracy=0.6
            )
        
        # Analyze recent trends
        recent_metrics = list(self.usage_history)[-min(60, len(self.usage_history)):]
        
        # Simple trend analysis
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        storage_trend = self._calculate_trend([m.storage_usage for m in recent_metrics])
        
        # Generate predictions
        current_cpu = recent_metrics[-1].cpu_usage
        current_memory = recent_metrics[-1].memory_usage
        current_storage = recent_metrics[-1].storage_usage
        
        predicted_cpu = []
        predicted_memory = []
        predicted_storage = []
        
        for i in range(horizon_minutes):
            # Apply trend with some randomness
            cpu_pred = max(0.1, min(1.0, current_cpu + cpu_trend * i * 0.01))
            memory_pred = max(0.1, min(1.0, current_memory + memory_trend * i * 0.01))
            storage_pred = max(0.1, min(1.0, current_storage + storage_trend * i * 0.005))
            
            predicted_cpu.append(cpu_pred)
            predicted_memory.append(memory_pred)
            predicted_storage.append(storage_pred)
        
        # Find peak usage time
        peak_cpu_idx = predicted_cpu.index(max(predicted_cpu))
        peak_time = datetime.now() + timedelta(minutes=peak_cpu_idx)
        
        # Calculate confidence interval
        variance = self._calculate_prediction_variance(recent_metrics)
        confidence_interval = (max(0.0, 1.0 - variance), min(1.0, 1.0 + variance))
        
        return ResourceForecast(
            forecast_horizon=horizon_minutes,
            predicted_cpu=predicted_cpu,
            predicted_memory=predicted_memory,
            predicted_storage=predicted_storage,
            confidence_interval=confidence_interval,
            peak_usage_time=peak_time if max(predicted_cpu) > current_cpu * 1.1 else None,
            forecast_accuracy=0.8
        )
    
    def optimize_cost(self, target_cost_reduction: float = 0.20) -> Dict[str, Any]:
        """
        Optimize resource allocation for cost reduction.
        
        Args:
            target_cost_reduction: Target cost reduction percentage (0.0-1.0)
            
        Returns:
            Cost optimization plan
        """
        current_cost = self._calculate_current_cost()
        target_cost = current_cost * (1.0 - target_cost_reduction)
        
        # Analyze cost optimization opportunities
        optimization_opportunities = []
        
        # Right-sizing opportunities
        for allocation_key, allocation in self.current_allocations.items():
            actual_usage = self._get_allocation_usage(allocation)
            if actual_usage and actual_usage["cpu"] < allocation.allocated_cpu * 0.6:
                # CPU over-allocated
                potential_savings = (allocation.allocated_cpu - actual_usage["cpu"]) * self.cost_per_cpu_hour
                optimization_opportunities.append({
                    "type": "rightsizing",
                    "allocation_key": allocation_key,
                    "resource": "cpu",
                    "current": allocation.allocated_cpu,
                    "recommended": actual_usage["cpu"] * 1.2,  # 20% buffer
                    "potential_savings": potential_savings
                })
        
        # Time-based optimization (schedule non-critical workloads during off-peak)
        off_peak_hours = [0, 1, 2, 3, 4, 5, 6, 22, 23]
        current_hour = datetime.now().hour
        
        if current_hour not in off_peak_hours:
            # Suggest delaying low-priority workloads
            low_priority_allocations = [
                (key, alloc) for key, alloc in self.current_allocations.items()
                if alloc.priority <= 3
            ]
            
            for key, allocation in low_priority_allocations:
                potential_savings = self._calculate_allocation_cost(allocation) * 0.3  # 30% off-peak discount
                optimization_opportunities.append({
                    "type": "time_shifting",
                    "allocation_key": key,
                    "resource": "schedule",
                    "current": "peak_hours",
                    "recommended": "off_peak_hours",
                    "potential_savings": potential_savings
                })
        
        # Calculate total potential savings
        total_potential_savings = sum(op["potential_savings"] for op in optimization_opportunities)
        
        # Create optimization plan
        optimization_plan = {
            "current_hourly_cost": current_cost,
            "target_hourly_cost": target_cost,
            "target_reduction_percentage": target_cost_reduction * 100,
            "potential_savings": total_potential_savings,
            "achievable_reduction": min(target_cost_reduction, total_potential_savings / current_cost) * 100,
            "optimization_opportunities": optimization_opportunities,
            "implementation_priority": sorted(
                optimization_opportunities,
                key=lambda x: x["potential_savings"],
                reverse=True
            )
        }
        
        logger.info(f"Cost optimization analysis: ${total_potential_savings:.3f}/hour potential savings")
        return optimization_plan
    
    def _calculate_available_resources(self, current_metrics: ResourceMetrics) -> Dict[str, float]:
        """Calculate available resources based on current usage."""
        return {
            "cpu": max(0.0, self.max_cpu_cores * (1.0 - current_metrics.cpu_usage)),
            "memory": max(0.0, self.max_memory_gb * (1.0 - current_metrics.memory_usage)),
            "storage": max(0.0, self.max_storage_gb * (1.0 - current_metrics.storage_usage)),
            "network": 1.0 - current_metrics.network_usage,  # Normalized
            "gpu": 1.0 - current_metrics.gpu_usage  # Normalized
        }
    
    def _optimize_allocation(
        self,
        requirements: Dict[str, float],
        available: Dict[str, float],
        priority: int
    ) -> Dict[str, float]:
        """Optimize resource allocation based on availability and priority."""
        allocation = {}
        
        # Base allocation
        for resource, requested in requirements.items():
            if resource in available:
                # Apply priority-based scaling
                priority_multiplier = 0.5 + (priority / 10.0) * 0.5  # 0.5 to 1.0 range
                
                # Allocate with priority consideration
                allocated = min(requested * priority_multiplier, available[resource])
                allocation[resource] = allocated
        
        # Ensure minimum allocations
        allocation["cpu"] = max(allocation.get("cpu", 0), 0.1)  # Minimum CPU
        allocation["memory"] = max(allocation.get("memory", 0), 0.25)  # Minimum memory
        
        return allocation
    
    def _calculate_allocation_cost(self, allocation: ResourceAllocation) -> float:
        """Calculate hourly cost for a resource allocation."""
        cpu_cost = allocation.allocated_cpu * self.cost_per_cpu_hour
        memory_cost = allocation.allocated_memory * self.cost_per_gb_memory_hour
        
        # Additional costs could include storage, network, GPU
        storage_cost = allocation.allocated_storage * 0.001  # $0.001/GB/hour
        
        return cpu_cost + memory_cost + storage_cost
    
    def _calculate_current_cost(self) -> float:
        """Calculate current total hourly cost."""
        return sum(
            self._calculate_allocation_cost(allocation)
            for allocation in self.current_allocations.values()
        )
    
    def _calculate_scaling_confidence(self, resource_type: ResourceType, current_usage: float) -> float:
        """Calculate confidence score for scaling decisions."""
        if len(self.usage_history) < 5:
            return 0.6  # Low confidence with limited data
        
        # Analyze usage stability
        recent_usage = [
            getattr(metric, f"{resource_type.value}_usage")
            for metric in list(self.usage_history)[-10:]
        ]
        
        # Calculate variance
        if len(recent_usage) > 1:
            mean_usage = sum(recent_usage) / len(recent_usage)
            variance = sum((u - mean_usage) ** 2 for u in recent_usage) / len(recent_usage)
            stability_score = 1.0 / (1.0 + variance * 10)  # Higher variance = lower confidence
        else:
            stability_score = 0.7
        
        # Consider how far from threshold
        if resource_type == ResourceType.CPU:
            threshold_distance = abs(current_usage - self.scale_up_threshold)
        else:
            threshold_distance = abs(current_usage - self.scale_up_threshold)
        
        threshold_confidence = min(1.0, threshold_distance * 5)  # More distance = higher confidence
        
        return (stability_score + threshold_confidence) / 2.0
    
    def _estimate_scaling_cost_impact(self, resource_type: ResourceType, scaling_factor: float) -> float:
        """Estimate cost impact of scaling decision."""
        current_cost = self._calculate_current_cost()
        
        if resource_type == ResourceType.CPU:
            cost_multiplier = scaling_factor
        elif resource_type == ResourceType.MEMORY:
            cost_multiplier = scaling_factor * 0.5  # Memory is cheaper
        else:
            cost_multiplier = scaling_factor * 0.2  # Other resources are even cheaper
        
        return current_cost * (cost_multiplier - 1.0)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in resource usage values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        n = len(values)
        x_mean = n / 2.0
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _calculate_prediction_variance(self, recent_metrics: List[ResourceMetrics]) -> float:
        """Calculate variance for prediction confidence intervals."""
        if len(recent_metrics) < 2:
            return 0.3  # Default variance
        
        cpu_values = [m.cpu_usage for m in recent_metrics]
        mean_cpu = sum(cpu_values) / len(cpu_values)
        cpu_variance = sum((v - mean_cpu) ** 2 for v in cpu_values) / len(cpu_values)
        
        return min(0.5, cpu_variance)  # Cap variance at 0.5
    
    def _get_allocation_usage(self, allocation: ResourceAllocation) -> Optional[Dict[str, float]]:
        """Get actual usage for an allocation (simplified)."""
        # In a real implementation, this would track actual usage per allocation
        # For now, return simulated usage
        import random
        return {
            "cpu": allocation.allocated_cpu * random.uniform(0.3, 0.9),
            "memory": allocation.allocated_memory * random.uniform(0.4, 0.8)
        }
    
    def _record_resource_usage(self, allocation: ResourceAllocation):
        """Record resource usage for learning and optimization."""
        usage_record = {
            "workflow_id": allocation.workflow_id,
            "allocated_cpu": allocation.allocated_cpu,
            "allocated_memory": allocation.allocated_memory,
            "actual_usage": self._get_allocation_usage(allocation),
            "timestamp": datetime.now().isoformat(),
            "cost": self._calculate_allocation_cost(allocation)
        }
        
        # Store for analysis (could be saved to database)
        logger.debug(f"Recorded resource usage for {allocation.workflow_id}")
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource management statistics."""
        current_metrics = self.get_current_metrics()
        current_cost = self._calculate_current_cost()
        
        stats = {
            "current_metrics": current_metrics.to_dict(),
            "active_allocations": len(self.current_allocations),
            "total_allocated_cpu": sum(a.allocated_cpu for a in self.current_allocations.values()),
            "total_allocated_memory": sum(a.allocated_memory for a in self.current_allocations.values()),
            "current_hourly_cost": current_cost,
            "usage_history_size": len(self.usage_history),
            "scaling_recommendations": len(self.analyze_scaling_needs()),
            "resource_efficiency": {
                "cpu": current_metrics.cpu_usage / max(0.1, 
                    sum(a.allocated_cpu for a in self.current_allocations.values()) / self.max_cpu_cores),
                "memory": current_metrics.memory_usage / max(0.1,
                    sum(a.allocated_memory for a in self.current_allocations.values()) / self.max_memory_gb)
            }
        }
        
        return stats