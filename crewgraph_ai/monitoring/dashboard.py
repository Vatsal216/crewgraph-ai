"""
Basic Monitoring Dashboard for CrewGraph AI
Provides real-time web-based monitoring of workflow execution with performance analytics.
"""

import json
import time
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import os

from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


@dataclass
class WorkflowStatus:
    """Workflow status information"""
    workflow_id: str
    name: str
    status: str
    start_time: Optional[float]
    end_time: Optional[float]
    nodes_total: int
    nodes_completed: int
    nodes_failed: int
    current_node: Optional[str]
    error_count: int
    last_update: float


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    active_workflows: int
    total_workflows: int
    error_rate: float
    avg_execution_time: float


class MonitoringDashboard:
    """
    Basic monitoring dashboard for CrewGraph AI workflows.
    
    Features:
    - Real-time workflow monitoring
    - Performance metrics visualization
    - Error tracking and analytics
    - System health monitoring
    - Simple web interface
    """
    
    def __init__(
        self,
        update_interval: float = 5.0,
        max_history_size: int = 1000,
        enable_web_interface: bool = True,
        web_port: int = 8080
    ):
        self.update_interval = update_interval
        self.max_history_size = max_history_size
        self.enable_web_interface = enable_web_interface
        self.web_port = web_port
        
        # Data storage
        self.active_workflows: Dict[str, WorkflowStatus] = {}
        self.completed_workflows: List[WorkflowStatus] = []
        self.system_metrics_history: List[SystemMetrics] = []
        
        # Statistics
        self.total_workflows = 0
        self.total_errors = 0
        self.total_execution_time = 0.0
        
        # Threading
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._data_lock = threading.Lock()
        
        # Web server
        self._web_server = None
        
        logger.info("MonitoringDashboard initialized")
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        if self.enable_web_interface:
            self._start_web_server()
        
        logger.info("Monitoring dashboard started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self._stop_monitoring.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        if self._web_server:
            self._stop_web_server()
        
        logger.info("Monitoring dashboard stopped")
    
    def register_workflow(
        self,
        workflow_id: str,
        name: str,
        nodes_total: int = 0
    ):
        """Register a new workflow for monitoring"""
        with self._data_lock:
            status = WorkflowStatus(
                workflow_id=workflow_id,
                name=name,
                status="starting",
                start_time=time.time(),
                end_time=None,
                nodes_total=nodes_total,
                nodes_completed=0,
                nodes_failed=0,
                current_node=None,
                error_count=0,
                last_update=time.time()
            )
            self.active_workflows[workflow_id] = status
            self.total_workflows += 1
        
        logger.info(f"Registered workflow for monitoring: {workflow_id}")
    
    def update_workflow_status(
        self,
        workflow_id: str,
        status: str = None,
        current_node: str = None,
        nodes_completed: int = None,
        nodes_failed: int = None,
        error_count: int = None,
        **kwargs
    ):
        """Update workflow status"""
        with self._data_lock:
            if workflow_id not in self.active_workflows:
                logger.warning(f"Unknown workflow: {workflow_id}")
                return
            
            workflow_status = self.active_workflows[workflow_id]
            
            if status:
                workflow_status.status = status
            if current_node:
                workflow_status.current_node = current_node
            if nodes_completed is not None:
                workflow_status.nodes_completed = nodes_completed
            if nodes_failed is not None:
                workflow_status.nodes_failed = nodes_failed
            if error_count is not None:
                workflow_status.error_count = error_count
            
            workflow_status.last_update = time.time()
            
            # Move to completed if finished
            if status in ["completed", "failed", "cancelled"]:
                workflow_status.end_time = time.time()
                self.completed_workflows.append(workflow_status)
                del self.active_workflows[workflow_id]
                
                # Update statistics
                if workflow_status.end_time and workflow_status.start_time:
                    execution_time = workflow_status.end_time - workflow_status.start_time
                    self.total_execution_time += execution_time
                
                if workflow_status.error_count > 0:
                    self.total_errors += workflow_status.error_count
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._stop_monitoring.wait(self.update_interval):
            try:
                self._collect_system_metrics()
                self._cleanup_old_data()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            import psutil
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Calculate workflow metrics
            with self._data_lock:
                active_count = len(self.active_workflows)
                
                # Calculate error rate
                total_workflows = max(self.total_workflows, 1)
                error_rate = (self.total_errors / total_workflows) * 100
                
                # Calculate average execution time
                completed_count = len(self.completed_workflows)
                avg_execution_time = (
                    self.total_execution_time / completed_count
                    if completed_count > 0 else 0.0
                )
            
            # Create metrics object
            system_metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                active_workflows=active_count,
                total_workflows=self.total_workflows,
                error_rate=error_rate,
                avg_execution_time=avg_execution_time
            )
            
            # Add to history
            with self._data_lock:
                self.system_metrics_history.append(system_metrics)
                
                # Maintain size limit
                if len(self.system_metrics_history) > self.max_history_size:
                    self.system_metrics_history = self.system_metrics_history[-self.max_history_size:]
            
            # Record metrics
            metrics.record_metric("system_cpu_usage", cpu_usage)
            metrics.record_metric("system_memory_usage", memory_usage)
            metrics.record_metric("active_workflows", active_count)
            
        except ImportError:
            logger.warning("psutil not available, system metrics disabled")
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        cutoff_time = time.time() - 86400  # 24 hours
        
        with self._data_lock:
            # Clean up completed workflows older than 24 hours
            self.completed_workflows = [
                wf for wf in self.completed_workflows
                if wf.end_time and wf.end_time > cutoff_time
            ]
            
            # Clean up old system metrics
            self.system_metrics_history = [
                metrics for metrics in self.system_metrics_history
                if metrics.timestamp > cutoff_time
            ]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        with self._data_lock:
            # Current status
            current_time = time.time()
            
            # Recent metrics (last hour)
            recent_metrics = [
                m for m in self.system_metrics_history
                if current_time - m.timestamp < 3600
            ]
            
            # Recent completed workflows
            recent_completed = [
                wf for wf in self.completed_workflows
                if wf.end_time and current_time - wf.end_time < 3600
            ]
            
            return {
                "timestamp": current_time,
                "active_workflows": [asdict(wf) for wf in self.active_workflows.values()],
                "recent_completed": [asdict(wf) for wf in recent_completed],
                "system_metrics": [asdict(m) for m in recent_metrics],
                "statistics": {
                    "total_workflows": self.total_workflows,
                    "active_count": len(self.active_workflows),
                    "completed_today": len(recent_completed),
                    "total_errors": self.total_errors,
                    "avg_execution_time": (
                        self.total_execution_time / len(self.completed_workflows)
                        if self.completed_workflows else 0.0
                    ),
                    "error_rate": (
                        (self.total_errors / self.total_workflows) * 100
                        if self.total_workflows > 0 else 0.0
                    )
                }
            }
    
    def _start_web_server(self):
        """Start simple web server for dashboard"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import json
            
            dashboard_instance = self
            
            class DashboardHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == "/" or self.path == "/dashboard":
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(self._get_dashboard_html().encode())
                    elif self.path == "/api/data":
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        data = dashboard_instance.get_dashboard_data()
                        self.wfile.write(json.dumps(data, default=str).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                def _get_dashboard_html(self):
                    return """
<!DOCTYPE html>
<html>
<head>
    <title>CrewGraph AI Monitoring Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .workflows { background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .workflow-item { border-bottom: 1px solid #eee; padding: 10px 0; }
        .status-running { color: #27ae60; }
        .status-failed { color: #e74c3c; }
        .status-completed { color: #3498db; }
        .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 3px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 CrewGraph AI Monitoring Dashboard</h1>
            <p>Real-time workflow monitoring and performance analytics</p>
            <button class="refresh-btn" onclick="loadData()">Refresh Data</button>
        </div>
        
        <div class="stats" id="stats">
            <!-- Statistics will be loaded here -->
        </div>
        
        <div class="workflows">
            <h2>Active Workflows</h2>
            <div id="active-workflows">
                <!-- Active workflows will be loaded here -->
            </div>
        </div>
        
        <div class="workflows">
            <h2>Recent Completed Workflows</h2>
            <div id="completed-workflows">
                <!-- Completed workflows will be loaded here -->
            </div>
        </div>
    </div>

    <script>
        async function loadData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Failed to load data:', error);
            }
        }
        
        function updateDashboard(data) {
            // Update statistics
            const stats = data.statistics;
            document.getElementById('stats').innerHTML = `
                <div class="stat-card">
                    <h3>Total Workflows</h3>
                    <p style="font-size: 2em; margin: 0;">${stats.total_workflows}</p>
                </div>
                <div class="stat-card">
                    <h3>Active Now</h3>
                    <p style="font-size: 2em; margin: 0; color: #27ae60;">${stats.active_count}</p>
                </div>
                <div class="stat-card">
                    <h3>Error Rate</h3>
                    <p style="font-size: 2em; margin: 0; color: #e74c3c;">${stats.error_rate.toFixed(1)}%</p>
                </div>
                <div class="stat-card">
                    <h3>Avg Execution</h3>
                    <p style="font-size: 2em; margin: 0;">${stats.avg_execution_time.toFixed(1)}s</p>
                </div>
            `;
            
            // Update active workflows
            const activeHtml = data.active_workflows.map(wf => `
                <div class="workflow-item">
                    <strong>${wf.name}</strong> (${wf.workflow_id})
                    <span class="status-${wf.status}">${wf.status}</span>
                    <br>
                    Progress: ${wf.nodes_completed}/${wf.nodes_total} nodes
                    ${wf.current_node ? `• Current: ${wf.current_node}` : ''}
                </div>
            `).join('');
            document.getElementById('active-workflows').innerHTML = activeHtml || '<p>No active workflows</p>';
            
            // Update completed workflows
            const completedHtml = data.recent_completed.map(wf => `
                <div class="workflow-item">
                    <strong>${wf.name}</strong> (${wf.workflow_id})
                    <span class="status-${wf.status}">${wf.status}</span>
                    <br>
                    Duration: ${((wf.end_time - wf.start_time) || 0).toFixed(1)}s
                    • Errors: ${wf.error_count}
                </div>
            `).join('');
            document.getElementById('completed-workflows').innerHTML = completedHtml || '<p>No recent completed workflows</p>';
        }
        
        // Load data initially and set up auto-refresh
        loadData();
        setInterval(loadData, 5000); // Refresh every 5 seconds
    </script>
</body>
</html>
                    """
                
                def log_message(self, format, *args):
                    # Suppress HTTP server logs
                    pass
            
            server = HTTPServer(('localhost', self.web_port), DashboardHandler)
            self._web_server = server
            
            # Start server in background thread
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            
            logger.info(f"Monitoring dashboard available at http://localhost:{self.web_port}")
            
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
    
    def _stop_web_server(self):
        """Stop web server"""
        if self._web_server:
            self._web_server.shutdown()
            self._web_server = None


# Global monitoring instance
_global_dashboard: Optional[MonitoringDashboard] = None


def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get global monitoring dashboard instance"""
    global _global_dashboard
    if _global_dashboard is None:
        _global_dashboard = MonitoringDashboard()
    return _global_dashboard


def start_monitoring(port: int = 8080):
    """Start global monitoring dashboard"""
    dashboard = get_monitoring_dashboard()
    dashboard.web_port = port
    dashboard.start_monitoring()
    return dashboard


def stop_monitoring():
    """Stop global monitoring dashboard"""
    global _global_dashboard
    if _global_dashboard:
        _global_dashboard.stop_monitoring()
        _global_dashboard = None