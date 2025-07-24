"""Production-ready Prometheus metrics backend."""

import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging

logger = logging.getLogger(__name__)

class PrometheusMetricsBackend:
    """Production Prometheus metrics backend."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self._setup_metrics()
        
    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        self.workflow_executions = Counter(
            'crewgraph_workflow_executions_total',
            'Total workflow executions',
            ['status', 'workflow_type']
        )
        
        self.workflow_duration = Histogram(
            'crewgraph_workflow_duration_seconds',
            'Workflow execution duration',
            ['workflow_type']
        )
        
        self.agent_tasks = Counter(
            'crewgraph_agent_tasks_total',
            'Total agent tasks executed',
            ['agent_id', 'status']
        )
        
        self.memory_usage = Gauge(
            'crewgraph_memory_usage_bytes',
            'Memory usage by component',
            ['component']
        )
        
        self.error_count = Counter(
            'crewgraph_errors_total',
            'Total errors by type',
            ['error_type', 'component']
        )
    
    def start_server(self):
        """Start Prometheus metrics server."""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def record_workflow_execution(self, workflow_type: str, status: str, duration: float):
        """Record workflow execution metrics."""
        self.workflow_executions.labels(status=status, workflow_type=workflow_type).inc()
        self.workflow_duration.labels(workflow_type=workflow_type).observe(duration)
    
    def record_agent_task(self, agent_id: str, status: str):
        """Record agent task metrics."""
        self.agent_tasks.labels(agent_id=agent_id, status=status).inc()
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics."""
        self.error_count.labels(error_type=error_type, component=component).inc()
    
    def update_memory_usage(self, component: str, bytes_used: int):
        """Update memory usage metrics."""
        self.memory_usage.labels(component=component).set(bytes_used)