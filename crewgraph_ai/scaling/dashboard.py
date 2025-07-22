"""
CrewGraph AI Auto-Scaling Dashboard
Real-time monitoring and control for auto-scaling systems

Author: Vatsal216
Created: 2025-07-22 13:17:52 UTC  
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


@dataclass
class ScalingMetrics:
    """Real-time scaling metrics"""
    timestamp: str
    workflow_instances: int
    agent_pool_size: int
    memory_usage_percent: float
    cpu_usage_percent: float
    queue_length: int
    response_time_ms: float
    scaling_events_24h: int
    cost_estimate_hourly: float
    efficiency_score: float


class ScalingDashboard:
    """
    Comprehensive auto-scaling dashboard.
    
    Provides real-time monitoring, control, and optimization
    for all scaling systems in CrewGraph AI.
    
    Created by: Vatsal216
    Date: 2025-07-22 13:17:52 UTC
    """
    
    def __init__(self, 
                 workflow_scaler,
                 agent_scaler, 
                 memory_scaler):
        """Initialize scaling dashboard"""
        self.workflow_scaler = workflow_scaler
        self.agent_scaler = agent_scaler
        self.memory_scaler = memory_scaler
        
        self.metrics_history: List[ScalingMetrics] = []
        
        logger.info("ScalingDashboard initialized")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 13:17:52")
    
    def get_real_time_metrics(self) -> ScalingMetrics:
        """Get current real-time scaling metrics"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        
        # Collect metrics from all scalers
        workflow_status = self.workflow_scaler.get_scaling_status()
        agent_status = self.agent_scaler.get_pool_status()  
        memory_status = self.memory_scaler.get_scaling_status()
        
        # Calculate derived metrics
        efficiency_score = self._calculate_efficiency_score(
            workflow_status, agent_status, memory_status
        )
        
        cost_estimate = self._calculate_hourly_cost(
            workflow_status, agent_status, memory_status
        )
        
        metrics_snapshot = ScalingMetrics(
            timestamp=current_time,
            workflow_instances=workflow_status.get('total_instances', 0),
            agent_pool_size=agent_status.get('total_agents', 0),
            memory_usage_percent=self._get_avg_memory_usage(memory_status),
            cpu_usage_percent=workflow_status.get('statistics', {}).get('avg_cpu_usage', 0),
            queue_length=workflow_status.get('queue_size', 0),
            response_time_ms=workflow_status.get('statistics', {}).get('avg_response_time', 0),
            scaling_events_24h=self._count_recent_scaling_events(workflow_status),
            cost_estimate_hourly=cost_estimate,
            efficiency_score=efficiency_score
        )
        
        # Store for history
        self.metrics_history.append(metrics_snapshot)
        if len(self.metrics_history) > 1440:  # Keep 24 hours (1 minute intervals)
            self.metrics_history.pop(0)
        
        return metrics_snapshot
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        current_metrics = self.get_real_time_metrics()
        
        return {
            'current_metrics': asdict(current_metrics),
            'workflow_scaling': self.workflow_scaler.get_scaling_status(),
            'agent_scaling': self.agent_scaler.get_pool_status(),
            'memory_scaling': self.memory_scaler.get_scaling_status(),
            'optimization_recommendations': self._generate_recommendations(),
            'cost_analysis': self._generate_cost_analysis(),
            'performance_trends': self._analyze_performance_trends(),
            'dashboard_metadata': {
                'created_by': 'Vatsal216',
                'generated_at': '2025-07-22 13:17:52',
                'version': '1.0.0'
            }
        }
    
    def _calculate_efficiency_score(self, workflow_status, agent_status, memory_status) -> float:
        """Calculate overall system efficiency score (0-100)"""
        # Simplified efficiency calculation
        utilization_score = 50.0  # Base score
        
        # Factor in resource utilization
        cpu_usage = workflow_status.get('statistics', {}).get('avg_cpu_usage', 0)
        if 50 <= cpu_usage <= 80:  # Optimal range
            utilization_score += 20
        elif cpu_usage < 50:
            utilization_score += cpu_usage * 0.4
        else:
            utilization_score += max(0, 100 - cpu_usage)
        
        # Factor in response times
        response_time = workflow_status.get('statistics', {}).get('avg_response_time', 0)
        if response_time < 1000:  # Under 1 second
            utilization_score += 20
        elif response_time < 5000:  # Under 5 seconds
            utilization_score += 15
        
        # Factor in scaling stability (fewer scaling events = better)
        scaling_events = self._count_recent_scaling_events(workflow_status)
        if scaling_events < 5:
            utilization_score += 10
        
        return min(100.0, max(0.0, utilization_score))
    
    def _calculate_hourly_cost(self, workflow_status, agent_status, memory_status) -> float:
        """Calculate estimated hourly cost"""
        # Simplified cost calculation (would integrate with actual pricing)
        workflow_cost = workflow_status.get('total_instances', 0) * 0.10  # $0.10 per instance
        agent_cost = agent_status.get('total_agents', 0) * 0.05           # $0.05 per agent
        memory_cost = len(memory_status.get('backends', {})) * 0.02       # $0.02 per backend
        
        return workflow_cost + agent_cost + memory_cost
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if len(self.metrics_history) < 10:
            return recommendations
        
        # Analyze recent metrics
        recent_metrics = self.metrics_history[-10:]
        avg_efficiency = sum(m.efficiency_score for m in recent_metrics) / len(recent_metrics)
        avg_cost = sum(m.cost_estimate_hourly for m in recent_metrics) / len(recent_metrics)
        
        # Generate recommendations based on patterns
        if avg_efficiency < 70:
            recommendations.append({
                'type': 'efficiency',
                'priority': 'high',
                'title': 'Low Efficiency Detected',
                'description': f'System efficiency is {avg_efficiency:.1f}%. Consider optimizing resource allocation.',
                'action': 'Review scaling thresholds and instance utilization'
            })
        
        if avg_cost > 5.0:  # $5/hour threshold
            recommendations.append({
                'type': 'cost',
                'priority': 'medium', 
                'title': 'High Cost Alert',
                'description': f'Estimated cost is ${avg_cost:.2f}/hour. Review scaling policies.',
                'action': 'Consider more aggressive scale-down thresholds'
            })
        
        # Add more sophisticated recommendations
        recommendations.extend(self._analyze_scaling_patterns())
        
        return recommendations
    
    def _analyze_scaling_patterns(self) -> List[Dict[str, str]]:
        """Analyze scaling patterns for recommendations"""
        patterns = []
        
        # This would include more sophisticated analysis
        # For now, return basic pattern analysis
        
        return patterns
    
    def export_metrics(self, format_type: str = 'json', hours: int = 24) -> str:
        """Export metrics data"""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        filtered_metrics = [
            m for m in self.metrics_history 
            if time.mktime(time.strptime(m.timestamp, "%Y-%m-%d %H:%M:%S")) >= start_time
        ]
        
        export_data = {
            'export_info': {
                'generated_by': 'Vatsal216',
                'generated_at': '2025-07-22 13:17:52',
                'time_range_hours': hours,
                'total_records': len(filtered_metrics)
            },
            'metrics': [asdict(m) for m in filtered_metrics]
        }
        
        if format_type.lower() == 'json':
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")