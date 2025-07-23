"""
Memory Inspector - Memory state inspection and analytics

Provides memory backend visualization, state dumps, usage analytics,
cache hit/miss visualization, and memory leak detection.
"""

import gc
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots

    VISUALIZATION_AVAILABLE = True
except ImportError:
    # Create mock objects when dependencies are not available
    go = None
    pyo = None
    make_subplots = None
    pd = None
    plt = None
    VISUALIZATION_AVAILABLE = False

from ..memory.base import BaseMemory
from ..utils.exceptions import CrewGraphError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Represents a snapshot of memory state at a point in time."""

    timestamp: datetime = field(default_factory=datetime.now)
    total_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    memory_percent: float = 0.0
    python_memory_mb: float = 0.0
    object_count: int = 0
    gc_stats: Dict[str, int] = field(default_factory=dict)
    backend_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    average_access_time: float = 0.0


class MemoryInspector:
    """
    Memory state inspection and analytics for CrewGraph AI workflows.

    Provides comprehensive memory monitoring, visualization, and leak detection
    capabilities for different memory backends.
    """

    def __init__(
        self,
        memory_backend: Optional[BaseMemory] = None,
        monitoring_interval: float = 5.0,
        max_snapshots: int = 1000,
        output_dir: str = "memory_analysis",
    ):
        """
        Initialize the MemoryInspector.

        Args:
            memory_backend: Memory backend to inspect
            monitoring_interval: Interval between memory snapshots (seconds)
            max_snapshots: Maximum number of snapshots to keep
            output_dir: Directory to save analysis outputs

        Raises:
            CrewGraphError: If visualization dependencies are not available
        """
        if not VISUALIZATION_AVAILABLE:
            raise CrewGraphError(
                "Visualization dependencies not available. "
                "Install with: pip install crewgraph-ai[visualization]"
            )

        self.memory_backend = memory_backend
        self.monitoring_interval = monitoring_interval
        self.max_snapshots = max_snapshots
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Memory tracking data
        self.snapshots: List[MemorySnapshot] = []
        self.cache_metrics: Dict[str, CacheMetrics] = {}
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Memory leak detection
        self.leak_threshold_mb = 100.0  # MB
        self.leak_detection_enabled = True
        self.baseline_memory: Optional[float] = None

        # Performance tracking
        self.access_times: List[float] = []

        logger.info("MemoryInspector initialized")

    def start_monitoring(self) -> None:
        """
        Start continuous memory monitoring.

        Creates a background thread that takes memory snapshots at regular intervals.
        """
        if self.monitoring_active:
            logger.warning("Memory monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info(f"Started memory monitoring with {self.monitoring_interval}s interval")

    def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        logger.info("Stopped memory monitoring")

    def take_snapshot(self) -> MemorySnapshot:
        """
        Take a manual memory snapshot.

        Returns:
            MemorySnapshot containing current memory state
        """
        # System memory stats
        memory = psutil.virtual_memory()

        # Python-specific memory stats
        python_memory = self._get_python_memory_usage()

        # Garbage collector stats
        gc_stats = self._get_gc_stats()

        # Backend-specific stats
        backend_stats = self._get_backend_stats()

        snapshot = MemorySnapshot(
            total_memory_mb=memory.total / (1024 * 1024),
            available_memory_mb=memory.available / (1024 * 1024),
            memory_percent=memory.percent,
            python_memory_mb=python_memory,
            object_count=len(gc.get_objects()),
            gc_stats=gc_stats,
            backend_stats=backend_stats,
        )

        # Add to snapshot history
        self.snapshots.append(snapshot)

        # Maintain max snapshots limit
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

        # Check for memory leaks
        if self.leak_detection_enabled:
            self._check_memory_leak(snapshot)

        return snapshot

    def get_memory_usage_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive memory usage report.

        Returns:
            Dictionary containing detailed memory analysis
        """
        if not self.snapshots:
            self.take_snapshot()

        recent_snapshot = self.snapshots[-1]

        # Calculate trends if we have multiple snapshots
        trend_data = {}
        if len(self.snapshots) > 1:
            trend_data = self._calculate_memory_trends()

        # Cache performance analysis
        cache_analysis = self._analyze_cache_performance()

        report = {
            "timestamp": recent_snapshot.timestamp.isoformat(),
            "current_memory": {
                "total_system_mb": recent_snapshot.total_memory_mb,
                "available_mb": recent_snapshot.available_memory_mb,
                "used_percent": recent_snapshot.memory_percent,
                "python_process_mb": recent_snapshot.python_memory_mb,
                "object_count": recent_snapshot.object_count,
            },
            "gc_statistics": recent_snapshot.gc_stats,
            "backend_statistics": recent_snapshot.backend_stats,
            "trends": trend_data,
            "cache_performance": cache_analysis,
            "snapshots_collected": len(self.snapshots),
            "monitoring_active": self.monitoring_active,
        }

        # Add leak detection results
        if self.baseline_memory:
            current_usage = recent_snapshot.python_memory_mb
            growth = current_usage - self.baseline_memory
            report["leak_detection"] = {
                "baseline_memory_mb": self.baseline_memory,
                "current_memory_mb": current_usage,
                "growth_mb": growth,
                "potential_leak": growth > self.leak_threshold_mb,
            }

        return report

    def dump_memory_state(
        self, include_backend_details: bool = True, include_gc_objects: bool = False
    ) -> Dict[str, Any]:
        """
        Dump detailed memory state for debugging.

        Args:
            include_backend_details: Whether to include detailed backend state
            include_gc_objects: Whether to include garbage collector object details

        Returns:
            Comprehensive memory state dump
        """
        state_dump = {
            "dump_timestamp": datetime.now().isoformat(),
            "memory_snapshots": [
                {
                    "timestamp": snapshot.timestamp.isoformat(),
                    "total_memory_mb": snapshot.total_memory_mb,
                    "available_memory_mb": snapshot.available_memory_mb,
                    "memory_percent": snapshot.memory_percent,
                    "python_memory_mb": snapshot.python_memory_mb,
                    "object_count": snapshot.object_count,
                    "gc_stats": snapshot.gc_stats,
                    "backend_stats": snapshot.backend_stats,
                }
                for snapshot in self.snapshots[-50:]  # Last 50 snapshots
            ],
        }

        # Include backend details if requested
        if include_backend_details and self.memory_backend:
            try:
                backend_state = self._get_detailed_backend_state()
                state_dump["backend_details"] = backend_state
            except Exception as e:
                state_dump["backend_details"] = {"error": str(e)}

        # Include GC object details if requested (can be very large)
        if include_gc_objects:
            try:
                object_analysis = self._analyze_gc_objects()
                state_dump["gc_object_analysis"] = object_analysis
            except Exception as e:
                state_dump["gc_object_analysis"] = {"error": str(e)}

        # Cache metrics
        state_dump["cache_metrics"] = {
            name: {
                "hits": metrics.hits,
                "misses": metrics.misses,
                "evictions": metrics.evictions,
                "total_requests": metrics.total_requests,
                "hit_rate": metrics.hit_rate,
                "average_access_time": metrics.average_access_time,
            }
            for name, metrics in self.cache_metrics.items()
        }

        return state_dump

    def visualize_memory_usage(
        self, time_range_hours: Optional[float] = None, title: str = "Memory Usage Analysis"
    ) -> str:
        """
        Create visualization of memory usage over time.

        Args:
            time_range_hours: Show data for last N hours (None for all data)
            title: Title for the visualization

        Returns:
            Path to generated HTML file
        """
        if not self.snapshots:
            raise CrewGraphError("No memory snapshots available for visualization")

        # Filter snapshots by time range if specified
        snapshots = self.snapshots
        if time_range_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]

        if not snapshots:
            raise CrewGraphError(f"No snapshots found in the last {time_range_hours} hours")

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "System Memory Usage",
                "Python Process Memory",
                "Object Count",
                "GC Collections",
                "Cache Hit Rate",
                "Memory Growth Rate",
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Prepare data
        timestamps = [s.timestamp for s in snapshots]
        memory_percents = [s.memory_percent for s in snapshots]
        python_memory = [s.python_memory_mb for s in snapshots]
        object_counts = [s.object_count for s in snapshots]

        # System memory usage
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=memory_percents,
                mode="lines+markers",
                name="Memory %",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        # Python process memory
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=python_memory,
                mode="lines+markers",
                name="Python Memory (MB)",
                line=dict(color="green"),
            ),
            row=1,
            col=2,
        )

        # Object count
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=object_counts,
                mode="lines+markers",
                name="Object Count",
                line=dict(color="orange"),
            ),
            row=2,
            col=1,
        )

        # GC collections
        if snapshots and snapshots[0].gc_stats:
            gc_totals = [sum(s.gc_stats.values()) for s in snapshots]
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=gc_totals,
                    mode="lines+markers",
                    name="GC Collections",
                    line=dict(color="red"),
                ),
                row=2,
                col=2,
            )

        # Cache hit rate (if available)
        if self.cache_metrics:
            for cache_name, metrics in self.cache_metrics.items():
                # Create synthetic time series for demo
                cache_timestamps = timestamps[-10:] if len(timestamps) >= 10 else timestamps
                hit_rates = [metrics.hit_rate] * len(cache_timestamps)

                fig.add_trace(
                    go.Scatter(
                        x=cache_timestamps,
                        y=hit_rates,
                        mode="lines+markers",
                        name=f"{cache_name} Hit Rate",
                        line=dict(dash="dash"),
                    ),
                    row=3,
                    col=1,
                )

        # Memory growth rate
        if len(python_memory) > 1:
            growth_rates = []
            for i in range(1, len(python_memory)):
                time_diff = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600  # hours
                memory_diff = python_memory[i] - python_memory[i - 1]
                growth_rate = memory_diff / time_diff if time_diff > 0 else 0
                growth_rates.append(growth_rate)

            fig.add_trace(
                go.Scatter(
                    x=timestamps[1:],
                    y=growth_rates,
                    mode="lines+markers",
                    name="Memory Growth (MB/h)",
                    line=dict(color="purple"),
                ),
                row=3,
                col=2,
            )

        fig.update_layout(title=title, height=800, showlegend=True, hovermode="x unified")

        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_analysis_{timestamp}.html"
        filepath = self.output_dir / filename

        pyo.plot(fig, filename=str(filepath), auto_open=False)

        logger.info(f"Memory usage visualization saved to: {filepath}")
        return str(filepath)

    def detect_memory_leaks(self, threshold_mb: float = 100.0) -> Dict[str, Any]:
        """
        Analyze memory snapshots for potential memory leaks.

        Args:
            threshold_mb: Memory growth threshold to consider as potential leak

        Returns:
            Memory leak analysis results
        """
        if len(self.snapshots) < 10:
            return {
                "status": "insufficient_data",
                "message": "Need at least 10 snapshots for leak detection",
                "snapshots_available": len(self.snapshots),
            }

        # Analyze memory growth trend
        recent_snapshots = self.snapshots[-10:]
        memory_values = [s.python_memory_mb for s in recent_snapshots]

        # Calculate linear regression for trend
        timestamps = [
            (s.timestamp - recent_snapshots[0].timestamp).total_seconds() for s in recent_snapshots
        ]

        # Simple linear regression
        n = len(memory_values)
        sum_x = sum(timestamps)
        sum_y = sum(memory_values)
        sum_xy = sum(x * y for x, y in zip(timestamps, memory_values))
        sum_x2 = sum(x * x for x in timestamps)

        slope = (
            (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            if (n * sum_x2 - sum_x * sum_x) != 0
            else 0
        )

        # Convert slope to MB per hour
        slope_mb_per_hour = slope * 3600

        # Object count analysis
        object_counts = [s.object_count for s in recent_snapshots]
        object_growth = object_counts[-1] - object_counts[0] if len(object_counts) > 1 else 0

        # GC analysis
        gc_efficiency = self._analyze_gc_efficiency()

        analysis = {
            "memory_growth_rate_mb_per_hour": slope_mb_per_hour,
            "total_memory_growth_mb": memory_values[-1] - memory_values[0],
            "object_count_growth": object_growth,
            "gc_efficiency": gc_efficiency,
            "potential_leak_detected": abs(slope_mb_per_hour)
            > threshold_mb / 24,  # Per day threshold
            "analysis_period_hours": (
                recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp
            ).total_seconds()
            / 3600,
            "recommendations": [],
        }

        # Generate recommendations
        if analysis["potential_leak_detected"]:
            analysis["recommendations"].append(
                "Memory growth detected - investigate object lifecycle"
            )

        if object_growth > 10000:
            analysis["recommendations"].append(
                "High object count growth - check for unreferenced objects"
            )

        if gc_efficiency["efficiency_score"] < 0.5:
            analysis["recommendations"].append(
                "Low GC efficiency - consider manual garbage collection"
            )

        return analysis

    def export_analysis_report(self, format: str = "json") -> str:
        """
        Export comprehensive memory analysis report.

        Args:
            format: Export format ('json', 'html')

        Returns:
            Path to exported report file
        """
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "memory_usage_report": self.get_memory_usage_report(),
            "memory_state_dump": self.dump_memory_state(include_gc_objects=False),
            "leak_detection_analysis": self.detect_memory_leaks(),
            "configuration": {
                "monitoring_interval": self.monitoring_interval,
                "max_snapshots": self.max_snapshots,
                "leak_threshold_mb": self.leak_threshold_mb,
                "memory_backend_type": (
                    type(self.memory_backend).__name__ if self.memory_backend else None
                ),
            },
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format.lower() == "json":
            filename = f"memory_analysis_report_{timestamp}.json"
            filepath = self.output_dir / filename

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)

        elif format.lower() == "html":
            filename = f"memory_analysis_report_{timestamp}.html"
            filepath = self.output_dir / filename

            self._generate_html_report(report, filepath)
        else:
            raise CrewGraphError(f"Unsupported export format: {format}")

        logger.info(f"Memory analysis report exported to: {filepath}")
        return str(filepath)

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                self.take_snapshot()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _get_python_memory_usage(self) -> float:
        """Get current Python process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def _get_gc_stats(self) -> Dict[str, int]:
        """Get garbage collector statistics."""
        return {f"generation_{i}": count for i, count in enumerate(gc.get_stats())}

    def _get_backend_stats(self) -> Dict[str, Any]:
        """Get memory backend specific statistics."""
        if not self.memory_backend:
            return {}

        try:
            # Try to get backend-specific stats
            stats = {}

            # Common stats that might be available
            if hasattr(self.memory_backend, "get_stats"):
                stats.update(self.memory_backend.get_stats())

            if hasattr(self.memory_backend, "size"):
                stats["backend_size"] = self.memory_backend.size()

            # For specific backend types
            backend_type = type(self.memory_backend).__name__
            stats["backend_type"] = backend_type

            if "Redis" in backend_type:
                stats.update(self._get_redis_stats())
            elif "FAISS" in backend_type:
                stats.update(self._get_faiss_stats())
            elif "Dict" in backend_type:
                stats.update(self._get_dict_stats())

            return stats

        except Exception as e:
            return {"error": str(e), "backend_type": type(self.memory_backend).__name__}

    def _get_redis_stats(self) -> Dict[str, Any]:
        """Get Redis-specific statistics."""
        try:
            if hasattr(self.memory_backend, "client"):
                info = self.memory_backend.client.info("memory")
                return {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "mem_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0),
                }
        except Exception:
            pass
        return {}

    def _get_faiss_stats(self) -> Dict[str, Any]:
        """Get FAISS-specific statistics."""
        try:
            if hasattr(self.memory_backend, "index"):
                return {
                    "index_size": (
                        self.memory_backend.index.ntotal
                        if hasattr(self.memory_backend.index, "ntotal")
                        else 0
                    ),
                    "is_trained": (
                        self.memory_backend.index.is_trained
                        if hasattr(self.memory_backend.index, "is_trained")
                        else False
                    ),
                }
        except Exception:
            pass
        return {}

    def _get_dict_stats(self) -> Dict[str, Any]:
        """Get dictionary memory backend statistics."""
        try:
            if hasattr(self.memory_backend, "_data"):
                data = self.memory_backend._data
                return {"key_count": len(data), "estimated_size_bytes": sys.getsizeof(data)}
        except Exception:
            pass
        return {}

    def _check_memory_leak(self, snapshot: MemorySnapshot) -> None:
        """Check for potential memory leaks."""
        if self.baseline_memory is None:
            self.baseline_memory = snapshot.python_memory_mb
            return

        growth = snapshot.python_memory_mb - self.baseline_memory
        if growth > self.leak_threshold_mb:
            logger.warning(f"Potential memory leak detected: {growth:.2f}MB growth from baseline")

    def _calculate_memory_trends(self) -> Dict[str, Any]:
        """Calculate memory usage trends."""
        if len(self.snapshots) < 2:
            return {}

        recent_snapshots = self.snapshots[-min(50, len(self.snapshots)) :]

        memory_values = [s.python_memory_mb for s in recent_snapshots]
        object_counts = [s.object_count for s in recent_snapshots]

        return {
            "memory_trend": "increasing" if memory_values[-1] > memory_values[0] else "decreasing",
            "memory_change_mb": memory_values[-1] - memory_values[0],
            "object_count_change": object_counts[-1] - object_counts[0],
            "average_memory_mb": sum(memory_values) / len(memory_values),
            "max_memory_mb": max(memory_values),
            "min_memory_mb": min(memory_values),
        }

    def _analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze cache performance metrics."""
        if not self.cache_metrics:
            return {"status": "no_cache_data"}

        total_requests = sum(m.total_requests for m in self.cache_metrics.values())
        total_hits = sum(m.hits for m in self.cache_metrics.values())

        return {
            "overall_hit_rate": (total_hits / total_requests * 100) if total_requests > 0 else 0,
            "total_requests": total_requests,
            "cache_backends": len(self.cache_metrics),
            "individual_caches": {
                name: {
                    "hit_rate": metrics.hit_rate,
                    "total_requests": metrics.total_requests,
                    "average_access_time": metrics.average_access_time,
                }
                for name, metrics in self.cache_metrics.items()
            },
        }

    def _get_detailed_backend_state(self) -> Dict[str, Any]:
        """Get detailed backend state information."""
        if not self.memory_backend:
            return {}

        state = {
            "backend_type": type(self.memory_backend).__name__,
            "backend_id": id(self.memory_backend),
        }

        # Try to get all backend data (be careful with large backends)
        try:
            if hasattr(self.memory_backend, "keys"):
                keys = list(self.memory_backend.keys())
                state["key_count"] = len(keys)
                state["sample_keys"] = keys[:10]  # First 10 keys as sample

            if hasattr(self.memory_backend, "__len__"):
                state["length"] = len(self.memory_backend)

        except Exception as e:
            state["error"] = str(e)

        return state

    def _analyze_gc_objects(self) -> Dict[str, Any]:
        """Analyze garbage collector objects."""
        objects = gc.get_objects()
        type_counts = {}

        for obj in objects[:1000]:  # Limit to first 1000 objects
            obj_type = type(obj).__name__
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

        return {
            "total_objects": len(objects),
            "analyzed_objects": min(1000, len(objects)),
            "type_distribution": dict(
                sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            ),
        }

    def _analyze_gc_efficiency(self) -> Dict[str, Any]:
        """Analyze garbage collection efficiency."""
        if len(self.snapshots) < 2:
            return {"efficiency_score": 1.0, "status": "insufficient_data"}

        recent_snapshots = self.snapshots[-10:]

        # Calculate GC efficiency based on object count vs memory usage correlation
        object_counts = [s.object_count for s in recent_snapshots]
        memory_usage = [s.python_memory_mb for s in recent_snapshots]

        # Simple efficiency score: inverse correlation between GC runs and memory growth
        if len(set(object_counts)) > 1:  # Avoid division by zero
            efficiency_score = (
                1.0
                - (max(memory_usage) - min(memory_usage))
                / (max(object_counts) - min(object_counts))
                * 100
            )
            efficiency_score = max(0.0, min(1.0, efficiency_score))  # Clamp between 0 and 1
        else:
            efficiency_score = 1.0

        return {
            "efficiency_score": efficiency_score,
            "object_count_variance": max(object_counts) - min(object_counts),
            "memory_variance_mb": max(memory_usage) - min(memory_usage),
            "status": "analyzed",
        }

    def _generate_html_report(self, report: Dict[str, Any], filepath: Path) -> None:
        """Generate HTML report from analysis data."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CrewGraph AI Memory Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; }}
                .warning {{ color: red; font-weight: bold; }}
                .good {{ color: green; font-weight: bold; }}
                pre {{ background-color: #f5f5f5; padding: 10px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CrewGraph AI Memory Analysis Report</h1>
                <p>Generated: {report['analysis_timestamp']}</p>
            </div>
            
            <div class="section">
                <h2>Current Memory Usage</h2>
                <div class="metric">Python Process: {report['memory_usage_report']['current_memory']['python_process_mb']:.2f} MB</div>
                <div class="metric">System Memory Used: {report['memory_usage_report']['current_memory']['used_percent']:.1f}%</div>
                <div class="metric">Object Count: {report['memory_usage_report']['current_memory']['object_count']:,}</div>
            </div>
            
            <div class="section">
                <h2>Leak Detection</h2>
                {"<div class='warning'>Potential memory leak detected!</div>" if report['leak_detection_analysis'].get('potential_leak_detected') else "<div class='good'>No memory leaks detected</div>"}
                <div class="metric">Memory Growth Rate: {report['leak_detection_analysis']['memory_growth_rate_mb_per_hour']:.2f} MB/hour</div>
            </div>
            
            <div class="section">
                <h2>Detailed Data</h2>
                <pre>{json.dumps(report, indent=2, default=str)[:5000]}...</pre>
            </div>
        </body>
        </html>
        """

        with open(filepath, "w") as f:
            f.write(html_content)
