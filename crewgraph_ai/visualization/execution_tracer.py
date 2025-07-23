"""
Execution Tracer - Real-time execution tracking and monitoring

Implements execution tracing with real-time tracking, step-by-step workflow progress,
performance metrics visualization, error tracking, and timeline visualization.
"""

import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from pathlib import Path
import threading

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    # Create mock objects when dependencies are not available
    go = None
    pyo = None
    make_subplots = None
    pd = None
    VISUALIZATION_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError

logger = get_logger(__name__)


@dataclass
class ExecutionEvent:
    """Represents a single execution event in the workflow."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    node_id: str = ""
    event_type: str = ""  # start, complete, error, pause, resume
    message: str = ""
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeExecutionTrace:
    """Tracks execution history for a single node."""
    node_id: str
    node_name: str = ""
    events: List[ExecutionEvent] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"
    total_duration: float = 0.0
    error_count: int = 0
    retry_count: int = 0


class ExecutionTracer:
    """
    Real-time execution tracking and monitoring for CrewGraph workflows.
    
    Provides comprehensive execution tracing with performance metrics,
    error tracking, and visualization capabilities.
    """
    
    def __init__(self, 
                 workflow_name: str = "default",
                 max_events: int = 10000,
                 enable_performance_tracking: bool = True,
                 output_dir: str = "execution_traces"):
        """
        Initialize the ExecutionTracer.
        
        Args:
            workflow_name: Name of the workflow being traced
            max_events: Maximum number of events to keep in memory
            enable_performance_tracking: Whether to track performance metrics
            output_dir: Directory to save trace outputs
            
        Raises:
            CrewGraphError: If visualization dependencies are not available
        """
        if not VISUALIZATION_AVAILABLE:
            raise CrewGraphError(
                "Visualization dependencies not available. "
                "Install with: pip install crewgraph-ai[visualization]"
            )
        
        self.workflow_name = workflow_name
        self.max_events = max_events
        self.enable_performance_tracking = enable_performance_tracking
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Execution tracking data
        self.node_traces: Dict[str, NodeExecutionTrace] = {}
        self.global_events: deque = deque(maxlen=max_events)
        self.active_nodes: Set[str] = set()
        self.workflow_start_time: Optional[datetime] = None
        self.workflow_end_time: Optional[datetime] = None
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.error_log: List[ExecutionEvent] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Event callbacks
        self.event_callbacks: List[Callable[[ExecutionEvent], None]] = []
        
        logger.info(f"ExecutionTracer initialized for workflow: {workflow_name}")
    
    def start_workflow_trace(self) -> str:
        """
        Start tracing a workflow execution.
        
        Returns:
            Trace session ID
        """
        with self._lock:
            self.workflow_start_time = datetime.now()
            session_id = str(uuid.uuid4())
            
            event = ExecutionEvent(
                event_id=session_id,
                node_id="__workflow__",
                event_type="workflow_start",
                message=f"Started tracing workflow: {self.workflow_name}"
            )
            
            self._add_event(event)
            logger.info(f"Started workflow trace with session ID: {session_id}")
            return session_id
    
    def end_workflow_trace(self, session_id: str) -> Dict[str, Any]:
        """
        End workflow tracing and generate summary.
        
        Args:
            session_id: Trace session ID from start_workflow_trace
            
        Returns:
            Workflow execution summary
        """
        with self._lock:
            self.workflow_end_time = datetime.now()
            
            event = ExecutionEvent(
                event_id=str(uuid.uuid4()),
                node_id="__workflow__",
                event_type="workflow_end",
                message=f"Ended tracing workflow: {self.workflow_name}",
                duration=self._calculate_workflow_duration()
            )
            
            self._add_event(event)
            summary = self._generate_execution_summary()
            
            logger.info(f"Ended workflow trace. Total duration: {summary['total_duration']:.2f}s")
            return summary
    
    def trace_node_start(self, node_id: str, node_name: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record the start of a node execution.
        
        Args:
            node_id: Unique identifier for the node
            node_name: Human-readable name for the node
            metadata: Additional metadata about the node
        """
        with self._lock:
            # Initialize node trace if not exists
            if node_id not in self.node_traces:
                self.node_traces[node_id] = NodeExecutionTrace(
                    node_id=node_id,
                    node_name=node_name or node_id
                )
            
            trace = self.node_traces[node_id]
            trace.start_time = datetime.now()
            trace.status = "running"
            self.active_nodes.add(node_id)
            
            event = ExecutionEvent(
                node_id=node_id,
                event_type="node_start",
                message=f"Started execution of node: {node_name or node_id}",
                metadata=metadata or {}
            )
            
            trace.events.append(event)
            self._add_event(event)
            
            if self.enable_performance_tracking:
                self._record_performance_metric("nodes_started", 1)
    
    def trace_node_complete(self, 
                           node_id: str, 
                           result: Optional[Any] = None,
                           performance_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Record the completion of a node execution.
        
        Args:
            node_id: Unique identifier for the node
            result: Execution result (will be serialized if not string)
            performance_metrics: Performance metrics for this execution
        """
        with self._lock:
            if node_id not in self.node_traces:
                logger.warning(f"Node {node_id} completed but was never started")
                return
            
            trace = self.node_traces[node_id]
            trace.end_time = datetime.now()
            trace.status = "completed"
            
            if trace.start_time:
                duration = (trace.end_time - trace.start_time).total_seconds()
                trace.total_duration = duration
            
            self.active_nodes.discard(node_id)
            
            # Prepare result for storage
            result_summary = str(result)[:500] if result else "No result"
            
            event = ExecutionEvent(
                node_id=node_id,
                event_type="node_complete",
                message=f"Completed execution of node: {trace.node_name}",
                duration=trace.total_duration,
                metadata={"result_summary": result_summary},
                performance_metrics=performance_metrics or {}
            )
            
            trace.events.append(event)
            self._add_event(event)
            
            if self.enable_performance_tracking:
                self._record_performance_metric("nodes_completed", 1)
                if trace.total_duration > 0:
                    self._record_performance_metric("node_duration", trace.total_duration)
    
    def trace_node_error(self, 
                        node_id: str, 
                        error: Exception,
                        retry_attempt: int = 0) -> None:
        """
        Record an error during node execution.
        
        Args:
            node_id: Unique identifier for the node
            error: Exception that occurred
            retry_attempt: Current retry attempt number
        """
        with self._lock:
            if node_id not in self.node_traces:
                logger.warning(f"Node {node_id} errored but was never started")
                return
            
            trace = self.node_traces[node_id]
            trace.error_count += 1
            trace.retry_count = retry_attempt
            trace.status = "failed" if retry_attempt == 0 else "retrying"
            
            event = ExecutionEvent(
                node_id=node_id,
                event_type="node_error",
                message=f"Error in node {trace.node_name}: {str(error)}",
                metadata={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retry_attempt": retry_attempt
                }
            )
            
            trace.events.append(event)
            self._add_event(event)
            self.error_log.append(event)
            
            if self.enable_performance_tracking:
                self._record_performance_metric("node_errors", 1)
    
    def add_custom_event(self, 
                        node_id: str,
                        event_type: str,
                        message: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a custom event to the trace.
        
        Args:
            node_id: Node ID associated with the event
            event_type: Type of the custom event
            message: Event message
            metadata: Additional event metadata
        """
        with self._lock:
            event = ExecutionEvent(
                node_id=node_id,
                event_type=event_type,
                message=message,
                metadata=metadata or {}
            )
            
            self._add_event(event)
            
            if node_id in self.node_traces:
                self.node_traces[node_id].events.append(event)
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """
        Get current real-time status of the workflow execution.
        
        Returns:
            Dictionary containing current execution status
        """
        with self._lock:
            return {
                "workflow_name": self.workflow_name,
                "workflow_running": bool(self.active_nodes),
                "active_nodes": list(self.active_nodes),
                "total_nodes": len(self.node_traces),
                "completed_nodes": len([t for t in self.node_traces.values() if t.status == "completed"]),
                "failed_nodes": len([t for t in self.node_traces.values() if t.status == "failed"]),
                "total_events": len(self.global_events),
                "error_count": len(self.error_log),
                "current_time": datetime.now().isoformat(),
                "workflow_duration": self._calculate_workflow_duration()
            }
    
    def export_trace_data(self, format: str = "json") -> str:
        """
        Export complete trace data to file.
        
        Args:
            format: Export format ('json', 'csv')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            return self._export_json_trace(timestamp)
        elif format.lower() == "csv":
            return self._export_csv_trace(timestamp)
        else:
            raise CrewGraphError(f"Unsupported export format: {format}")
    
    def _export_json_trace(self, timestamp: str) -> str:
        """Export trace data as JSON."""
        trace_data = {
            "workflow_name": self.workflow_name,
            "workflow_start_time": self.workflow_start_time.isoformat() if self.workflow_start_time else None,
            "workflow_end_time": self.workflow_end_time.isoformat() if self.workflow_end_time else None,
            "total_duration": self._calculate_workflow_duration(),
            "node_traces": {},
            "global_events": [],
            "performance_metrics": dict(self.performance_metrics),
            "error_summary": {
                "total_errors": len(self.error_log),
                "errors_by_node": {}
            }
        }
        
        # Add node traces
        for node_id, trace in self.node_traces.items():
            trace_data["node_traces"][node_id] = {
                "node_name": trace.node_name,
                "status": trace.status,
                "start_time": trace.start_time.isoformat() if trace.start_time else None,
                "end_time": trace.end_time.isoformat() if trace.end_time else None,
                "total_duration": trace.total_duration,
                "error_count": trace.error_count,
                "retry_count": trace.retry_count,
                "events": [
                    {
                        "event_id": event.event_id,
                        "timestamp": event.timestamp.isoformat(),
                        "event_type": event.event_type,
                        "message": event.message,
                        "duration": event.duration,
                        "metadata": event.metadata,
                        "performance_metrics": event.performance_metrics
                    }
                    for event in trace.events
                ]
            }
        
        # Add global events
        for event in self.global_events:
            trace_data["global_events"].append({
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "node_id": event.node_id,
                "event_type": event.event_type,
                "message": event.message,
                "duration": event.duration,
                "metadata": event.metadata,
                "performance_metrics": event.performance_metrics
            })
        
        # Error summary by node
        for event in self.error_log:
            node_id = event.node_id
            if node_id not in trace_data["error_summary"]["errors_by_node"]:
                trace_data["error_summary"]["errors_by_node"][node_id] = 0
            trace_data["error_summary"]["errors_by_node"][node_id] += 1
        
        filename = f"execution_trace_{self.workflow_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(trace_data, f, indent=2, default=str)
        
        logger.info(f"Execution trace exported to: {filepath}")
        return str(filepath)
    
    def _export_csv_trace(self, timestamp: str) -> str:
        """Export trace events as CSV."""
        events_data = []
        
        for event in self.global_events:
            events_data.append({
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "node_id": event.node_id,
                "event_type": event.event_type,
                "message": event.message,
                "duration": event.duration or 0,
                "metadata": json.dumps(event.metadata),
                "performance_metrics": json.dumps(event.performance_metrics)
            })
        
        df = pd.DataFrame(events_data)
        
        filename = f"execution_events_{self.workflow_name}_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        df.to_csv(filepath, index=False)
        
        logger.info(f"Execution events exported to CSV: {filepath}")
        return str(filepath)
    
    def visualize_execution_timeline(self, 
                                   show_performance_metrics: bool = True,
                                   title: Optional[str] = None) -> str:
        """
        Create interactive timeline visualization of execution.
        
        Args:
            show_performance_metrics: Whether to include performance metrics
            title: Custom title for the visualization
            
        Returns:
            Path to generated HTML file
        """
        if not title:
            title = f"Execution Timeline - {self.workflow_name}"
        
        # Create subplots
        fig = make_subplots(
            rows=3 if show_performance_metrics else 2,
            cols=1,
            subplot_titles=["Node Execution Timeline", "Error Events"] + 
                          (["Performance Metrics"] if show_performance_metrics else []),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]] + 
                  ([[{"secondary_y": True}]] if show_performance_metrics else [])
        )
        
        # Add node execution timeline
        self._add_timeline_traces(fig, row=1)
        
        # Add error events
        self._add_error_traces(fig, row=2)
        
        # Add performance metrics if requested
        if show_performance_metrics:
            self._add_performance_traces(fig, row=3)
        
        fig.update_layout(
            title=title,
            height=800 if show_performance_metrics else 600,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Save timeline
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"execution_timeline_{self.workflow_name}_{timestamp}.html"
        filepath = self.output_dir / filename
        
        pyo.plot(fig, filename=str(filepath), auto_open=False)
        
        logger.info(f"Execution timeline visualization saved to: {filepath}")
        return str(filepath)
    
    def _add_timeline_traces(self, fig: Any, row: int) -> None:
        """Add node execution timeline traces to the figure."""
        for node_id, trace in self.node_traces.items():
            if not trace.start_time:
                continue
            
            # Create timeline for this node
            y_position = list(self.node_traces.keys()).index(node_id)
            
            # Node execution bar
            end_time = trace.end_time or datetime.now()
            
            fig.add_trace(
                go.Scatter(
                    x=[trace.start_time, end_time],
                    y=[y_position, y_position],
                    mode='lines',
                    name=trace.node_name,
                    line=dict(width=10),
                    hovertemplate=f"<b>{trace.node_name}</b><br>" +
                                 f"Status: {trace.status}<br>" +
                                 f"Duration: {trace.total_duration:.2f}s<br>" +
                                 f"Errors: {trace.error_count}<extra></extra>"
                ),
                row=row, col=1
            )
        
        # Update y-axis labels
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(len(self.node_traces))),
            ticktext=[trace.node_name for trace in self.node_traces.values()],
            row=row, col=1
        )
    
    def _add_error_traces(self, fig: Any, row: int) -> None:
        """Add error event traces to the figure."""
        if not self.error_log:
            # Add placeholder if no errors
            fig.add_trace(
                go.Scatter(
                    x=[datetime.now()],
                    y=[0],
                    mode='markers',
                    name="No Errors",
                    marker=dict(size=1, opacity=0),
                    showlegend=False
                ),
                row=row, col=1
            )
            return
        
        error_times = [event.timestamp for event in self.error_log]
        error_messages = [event.message for event in self.error_log]
        error_nodes = [event.node_id for event in self.error_log]
        
        fig.add_trace(
            go.Scatter(
                x=error_times,
                y=[0] * len(error_times),
                mode='markers',
                name="Errors",
                marker=dict(size=10, color='red', symbol='x'),
                hovertemplate="<b>Error</b><br>" +
                             "Time: %{x}<br>" +
                             "Node: %{customdata}<br>" +
                             "Message: %{text}<extra></extra>",
                text=error_messages,
                customdata=error_nodes
            ),
            row=row, col=1
        )
    
    def _add_performance_traces(self, fig: Any, row: int) -> None:
        """Add performance metrics traces to the figure."""
        if not self.performance_metrics:
            return
        
        # Add performance metrics over time
        for metric_name, values in self.performance_metrics.items():
            if metric_name == "node_duration" and values:
                # Show duration trend
                timestamps = [datetime.now() - timedelta(seconds=(len(values)-i)*10) for i in range(len(values))]
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=values,
                        mode='lines+markers',
                        name=f"Duration (s)",
                        yaxis="y3"
                    ),
                    row=row, col=1
                )
    
    def add_event_callback(self, callback: Callable[[ExecutionEvent], None]) -> None:
        """
        Add a callback function to be called on each event.
        
        Args:
            callback: Function to call with each ExecutionEvent
        """
        self.event_callbacks.append(callback)
    
    def _add_event(self, event: ExecutionEvent) -> None:
        """Add an event to the global event log and trigger callbacks."""
        self.global_events.append(event)
        
        # Trigger callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    def _record_performance_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric."""
        self.performance_metrics[metric_name].append(value)
    
    def _calculate_workflow_duration(self) -> float:
        """Calculate total workflow duration in seconds."""
        if not self.workflow_start_time:
            return 0.0
        
        end_time = self.workflow_end_time or datetime.now()
        return (end_time - self.workflow_start_time).total_seconds()
    
    def _generate_execution_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive execution summary."""
        total_nodes = len(self.node_traces)
        completed_nodes = len([t for t in self.node_traces.values() if t.status == "completed"])
        failed_nodes = len([t for t in self.node_traces.values() if t.status == "failed"])
        
        return {
            "workflow_name": self.workflow_name,
            "total_duration": self._calculate_workflow_duration(),
            "total_nodes": total_nodes,
            "completed_nodes": completed_nodes,
            "failed_nodes": failed_nodes,
            "success_rate": (completed_nodes / total_nodes * 100) if total_nodes > 0 else 0,
            "total_events": len(self.global_events),
            "total_errors": len(self.error_log),
            "average_node_duration": sum(t.total_duration for t in self.node_traces.values()) / total_nodes if total_nodes > 0 else 0,
            "performance_metrics": dict(self.performance_metrics),
            "start_time": self.workflow_start_time.isoformat() if self.workflow_start_time else None,
            "end_time": self.workflow_end_time.isoformat() if self.workflow_end_time else None
        }