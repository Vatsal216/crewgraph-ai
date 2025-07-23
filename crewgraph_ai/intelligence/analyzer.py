"""
Bottleneck and Resource Analyzers for CrewGraph AI

AI-driven analysis of workflow bottlenecks and resource usage patterns.

Author: Vatsal216
Created: 2025-07-23 06:16:00 UTC
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector
from .ml_models import MLModelManager, ModelType

logger = get_logger(__name__)
metrics = get_metrics_collector()


@dataclass
class BottleneckDetection:
    """Bottleneck detection result"""

    bottleneck_type: str
    severity: str  # "low", "medium", "high", "critical"
    location: str
    description: str
    impact_score: float
    probability: float
    affected_components: List[str]
    suggested_solutions: List[str]
    estimated_delay: float
    detection_confidence: float
    created_by: str = "Vatsal216"
    created_at: str = "2025-07-23 06:16:00"


@dataclass
class ResourceAnalysis:
    """Resource usage analysis result"""

    resource_type: str
    current_usage: float
    peak_usage: float
    average_usage: float
    trend: str  # "increasing", "decreasing", "stable"
    utilization_efficiency: float
    waste_percentage: float
    recommendations: List[str]
    scaling_suggestions: List[str]
    cost_implications: Dict[str, float]
    created_by: str = "Vatsal216"
    created_at: str = "2025-07-23 06:16:00"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""

    timestamp: float
    metric_name: str
    value: float
    component: str
    metadata: Dict[str, Any]


class BottleneckAnalyzer:
    """
    AI-driven bottleneck detection and analysis.

    Identifies performance bottlenecks, resource constraints,
    and optimization opportunities in workflows.

    Created by: Vatsal216
    Date: 2025-07-23 06:16:00 UTC
    """

    def __init__(self, model_manager: Optional[MLModelManager] = None):
        """
        Initialize bottleneck analyzer.

        Args:
            model_manager: ML model manager instance
        """
        self.model_manager = model_manager or MLModelManager()

        # Performance monitoring
        self._metrics_buffer: deque = deque(maxlen=10000)
        self._bottleneck_history: List[BottleneckDetection] = []
        self._component_stats: Dict[str, Dict[str, float]] = defaultdict(dict)

        self._lock = threading.RLock()

        # Initialize bottleneck detection model
        self._initialize_models()

        logger.info("BottleneckAnalyzer initialized")
        logger.info("User: Vatsal216, Time: 2025-07-23 06:16:00")

    def _initialize_models(self):
        """Initialize and train bottleneck detection models"""
        try:
            # Create and train bottleneck detection model
            bottleneck_data = self.model_manager.generate_synthetic_training_data(
                ModelType.BOTTLENECK_DETECTOR, num_samples=1000
            )
            self.model_manager.train_model(ModelType.BOTTLENECK_DETECTOR, bottleneck_data)

            logger.info("Bottleneck detection models initialized")

        except Exception as e:
            logger.error(f"Failed to initialize bottleneck models: {e}")

    def add_performance_metric(
        self,
        metric_name: str,
        value: float,
        component: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add performance metric for analysis.

        Args:
            metric_name: Name of the metric
            value: Metric value
            component: Component being measured
            metadata: Additional metadata
        """
        with self._lock:
            metric = PerformanceMetric(
                timestamp=time.time(),
                metric_name=metric_name,
                value=value,
                component=component,
                metadata=metadata or {},
            )

            self._metrics_buffer.append(metric)

            # Update component statistics
            if component not in self._component_stats:
                self._component_stats[component] = {
                    "total_metrics": 0,
                    "avg_response_time": 0.0,
                    "error_rate": 0.0,
                    "throughput": 0.0,
                }

            self._update_component_stats(component, metric)

    def detect_bottlenecks(self, time_window_minutes: int = 10) -> List[BottleneckDetection]:
        """
        Detect bottlenecks in the specified time window.

        Args:
            time_window_minutes: Time window for analysis in minutes

        Returns:
            List of detected bottlenecks
        """
        with self._lock:
            start_time = time.time()

            # Get recent metrics
            cutoff_time = time.time() - (time_window_minutes * 60)
            recent_metrics = [m for m in self._metrics_buffer if m.timestamp >= cutoff_time]

            if not recent_metrics:
                logger.warning("No recent metrics available for bottleneck detection")
                return []

            bottlenecks = []

            # Analyze different types of bottlenecks
            bottlenecks.extend(self._detect_response_time_bottlenecks(recent_metrics))
            bottlenecks.extend(self._detect_throughput_bottlenecks(recent_metrics))
            bottlenecks.extend(self._detect_resource_bottlenecks(recent_metrics))
            bottlenecks.extend(self._detect_error_rate_bottlenecks(recent_metrics))
            bottlenecks.extend(self._detect_dependency_bottlenecks(recent_metrics))

            # Use ML model for additional detection
            ml_bottlenecks = self._ml_bottleneck_detection(recent_metrics)
            bottlenecks.extend(ml_bottlenecks)

            # Sort by severity and impact
            bottlenecks.sort(
                key=lambda b: (self._severity_score(b.severity), b.impact_score), reverse=True
            )

            # Record detections
            detection_time = time.time() - start_time
            for bottleneck in bottlenecks:
                self._bottleneck_history.append(bottleneck)

            metrics.record_metric("bottleneck_detections_total", len(bottlenecks))
            metrics.record_metric("bottleneck_detection_time_seconds", detection_time)

            logger.info(f"Detected {len(bottlenecks)} bottlenecks in {detection_time:.3f}s")

            return bottlenecks[:10]  # Return top 10 bottlenecks

    def _detect_response_time_bottlenecks(
        self, metrics: List[PerformanceMetric]
    ) -> List[BottleneckDetection]:
        """Detect response time bottlenecks"""
        bottlenecks = []

        # Group metrics by component
        component_response_times = defaultdict(list)
        for metric in metrics:
            if "response_time" in metric.metric_name.lower():
                component_response_times[metric.component].append(metric.value)

        for component, times in component_response_times.items():
            if not times:
                continue

            avg_response_time = np.mean(times)
            p95_response_time = np.percentile(times, 95)

            # Detect slow response times
            if avg_response_time > 1000:  # 1 second
                severity = "critical" if avg_response_time > 5000 else "high"

                bottlenecks.append(
                    BottleneckDetection(
                        bottleneck_type="response_time",
                        severity=severity,
                        location=component,
                        description=f"High response time detected in {component}: {avg_response_time:.0f}ms average",
                        impact_score=min(avg_response_time / 1000.0, 10.0),
                        probability=0.9,
                        affected_components=[component],
                        suggested_solutions=[
                            "Optimize database queries",
                            "Add caching layer",
                            "Scale component horizontally",
                            "Review algorithm efficiency",
                        ],
                        estimated_delay=avg_response_time / 1000.0,
                        detection_confidence=0.85,
                    )
                )

            # Detect response time variability
            if len(times) > 5:
                std_dev = np.std(times)
                coefficient_of_variation = (
                    std_dev / avg_response_time if avg_response_time > 0 else 0
                )

                if coefficient_of_variation > 0.5:  # High variability
                    bottlenecks.append(
                        BottleneckDetection(
                            bottleneck_type="response_time_variability",
                            severity="medium",
                            location=component,
                            description=f"High response time variability in {component}: CV={coefficient_of_variation:.2f}",
                            impact_score=coefficient_of_variation * 5.0,
                            probability=0.7,
                            affected_components=[component],
                            suggested_solutions=[
                                "Implement load balancing",
                                "Add request queuing",
                                "Optimize resource allocation",
                                "Review concurrency handling",
                            ],
                            estimated_delay=std_dev / 1000.0,
                            detection_confidence=0.75,
                        )
                    )

        return bottlenecks

    def _detect_throughput_bottlenecks(
        self, metrics: List[PerformanceMetric]
    ) -> List[BottleneckDetection]:
        """Detect throughput bottlenecks"""
        bottlenecks = []

        # Group metrics by component
        component_throughputs = defaultdict(list)
        for metric in metrics:
            if (
                "throughput" in metric.metric_name.lower()
                or "requests_per_second" in metric.metric_name.lower()
            ):
                component_throughputs[metric.component].append(metric.value)

        for component, throughputs in component_throughputs.items():
            if len(throughputs) < 3:  # Need enough data points
                continue

            # Detect declining throughput
            recent_throughput = np.mean(throughputs[-3:])
            earlier_throughput = np.mean(throughputs[:3])

            if recent_throughput < earlier_throughput * 0.8:  # 20% decline
                decline_percentage = (earlier_throughput - recent_throughput) / earlier_throughput

                bottlenecks.append(
                    BottleneckDetection(
                        bottleneck_type="throughput_decline",
                        severity="high" if decline_percentage > 0.5 else "medium",
                        location=component,
                        description=f"Throughput decline in {component}: {decline_percentage:.1%} decrease",
                        impact_score=decline_percentage * 10.0,
                        probability=0.8,
                        affected_components=[component],
                        suggested_solutions=[
                            "Scale resources up",
                            "Optimize processing logic",
                            "Clear resource leaks",
                            "Review system dependencies",
                        ],
                        estimated_delay=0.0,  # Ongoing impact
                        detection_confidence=0.8,
                    )
                )

            # Detect low absolute throughput
            avg_throughput = np.mean(throughputs)
            if avg_throughput < 10:  # Less than 10 requests per second
                bottlenecks.append(
                    BottleneckDetection(
                        bottleneck_type="low_throughput",
                        severity="medium",
                        location=component,
                        description=f"Low throughput in {component}: {avg_throughput:.1f} req/s",
                        impact_score=5.0 / max(avg_throughput, 0.1),
                        probability=0.7,
                        affected_components=[component],
                        suggested_solutions=[
                            "Optimize processing algorithms",
                            "Add parallel processing",
                            "Review resource allocation",
                            "Eliminate processing bottlenecks",
                        ],
                        estimated_delay=0.0,
                        detection_confidence=0.7,
                    )
                )

        return bottlenecks

    def _detect_resource_bottlenecks(
        self, metrics: List[PerformanceMetric]
    ) -> List[BottleneckDetection]:
        """Detect resource utilization bottlenecks"""
        bottlenecks = []

        # Group metrics by resource type
        resource_usage = defaultdict(lambda: defaultdict(list))
        for metric in metrics:
            metric_lower = metric.metric_name.lower()
            if any(resource in metric_lower for resource in ["cpu", "memory", "disk", "network"]):
                if "cpu" in metric_lower:
                    resource_type = "cpu"
                elif "memory" in metric_lower:
                    resource_type = "memory"
                elif "disk" in metric_lower:
                    resource_type = "disk"
                elif "network" in metric_lower:
                    resource_type = "network"
                else:
                    continue

                resource_usage[resource_type][metric.component].append(metric.value)

        for resource_type, components in resource_usage.items():
            for component, values in components.items():
                if not values:
                    continue

                avg_usage = np.mean(values)
                max_usage = np.max(values)

                # Detect high resource usage
                if resource_type == "cpu" and avg_usage > 80:
                    severity = "critical" if avg_usage > 95 else "high"
                elif resource_type == "memory" and avg_usage > 85:
                    severity = "critical" if avg_usage > 95 else "high"
                elif resource_type in ["disk", "network"] and avg_usage > 90:
                    severity = "high"
                else:
                    continue

                bottlenecks.append(
                    BottleneckDetection(
                        bottleneck_type=f"{resource_type}_saturation",
                        severity=severity,
                        location=component,
                        description=f"High {resource_type} usage in {component}: {avg_usage:.1f}% average",
                        impact_score=avg_usage / 10.0,
                        probability=0.9,
                        affected_components=[component],
                        suggested_solutions=self._get_resource_solutions(resource_type),
                        estimated_delay=0.0,
                        detection_confidence=0.9,
                    )
                )

        return bottlenecks

    def _detect_error_rate_bottlenecks(
        self, metrics: List[PerformanceMetric]
    ) -> List[BottleneckDetection]:
        """Detect error rate bottlenecks"""
        bottlenecks = []

        # Group metrics by component
        component_errors = defaultdict(list)
        for metric in metrics:
            if "error" in metric.metric_name.lower() or "failure" in metric.metric_name.lower():
                component_errors[metric.component].append(metric.value)

        for component, error_rates in component_errors.items():
            if not error_rates:
                continue

            avg_error_rate = np.mean(error_rates)

            if avg_error_rate > 5:  # More than 5% error rate
                severity = (
                    "critical"
                    if avg_error_rate > 20
                    else "high" if avg_error_rate > 10 else "medium"
                )

                bottlenecks.append(
                    BottleneckDetection(
                        bottleneck_type="high_error_rate",
                        severity=severity,
                        location=component,
                        description=f"High error rate in {component}: {avg_error_rate:.1f}%",
                        impact_score=avg_error_rate / 2.0,
                        probability=0.95,
                        affected_components=[component],
                        suggested_solutions=[
                            "Investigate error causes",
                            "Improve error handling",
                            "Add circuit breakers",
                            "Review input validation",
                            "Check external dependencies",
                        ],
                        estimated_delay=0.0,
                        detection_confidence=0.9,
                    )
                )

        return bottlenecks

    def _detect_dependency_bottlenecks(
        self, metrics: List[PerformanceMetric]
    ) -> List[BottleneckDetection]:
        """Detect dependency-related bottlenecks"""
        bottlenecks = []

        # Look for patterns indicating dependency issues
        component_patterns = defaultdict(list)
        for metric in metrics:
            if any(
                keyword in metric.metric_name.lower()
                for keyword in ["timeout", "connection", "dependency", "external"]
            ):
                component_patterns[metric.component].append(metric)

        for component, component_metrics in component_patterns.items():
            if len(component_metrics) < 3:
                continue

            # Check for timeout patterns
            timeout_metrics = [m for m in component_metrics if "timeout" in m.metric_name.lower()]
            if timeout_metrics:
                timeout_rate = np.mean([m.value for m in timeout_metrics])

                if timeout_rate > 2:  # More than 2% timeout rate
                    bottlenecks.append(
                        BottleneckDetection(
                            bottleneck_type="dependency_timeout",
                            severity="high" if timeout_rate > 10 else "medium",
                            location=component,
                            description=f"High timeout rate in {component}: {timeout_rate:.1f}%",
                            impact_score=timeout_rate / 2.0,
                            probability=0.85,
                            affected_components=[component],
                            suggested_solutions=[
                                "Increase timeout values",
                                "Implement retry logic",
                                "Add circuit breakers",
                                "Optimize dependency calls",
                                "Review network connectivity",
                            ],
                            estimated_delay=5.0,  # Typical timeout delay
                            detection_confidence=0.8,
                        )
                    )

        return bottlenecks

    def _ml_bottleneck_detection(
        self, metrics: List[PerformanceMetric]
    ) -> List[BottleneckDetection]:
        """Use ML model for additional bottleneck detection"""
        bottlenecks = []

        try:
            # Prepare features for ML model
            component_features = self._extract_ml_features(metrics)

            for component, features in component_features.items():
                if features.size == 0:
                    continue

                # Predict bottleneck probability
                predictions, probabilities = self.model_manager.predict(
                    ModelType.BOTTLENECK_DETECTOR, features.reshape(1, -1)
                )

                bottleneck_probability = probabilities[0]

                if bottleneck_probability > 0.7:  # High probability of bottleneck
                    severity = "high" if bottleneck_probability > 0.9 else "medium"

                    bottlenecks.append(
                        BottleneckDetection(
                            bottleneck_type="ml_detected_bottleneck",
                            severity=severity,
                            location=component,
                            description=f"ML model detected potential bottleneck in {component}",
                            impact_score=bottleneck_probability * 10.0,
                            probability=bottleneck_probability,
                            affected_components=[component],
                            suggested_solutions=[
                                "Investigate component performance",
                                "Review recent changes",
                                "Check resource allocation",
                                "Analyze usage patterns",
                            ],
                            estimated_delay=1.0,
                            detection_confidence=bottleneck_probability,
                        )
                    )

        except Exception as e:
            logger.warning(f"ML bottleneck detection failed: {e}")

        return bottlenecks

    def _extract_ml_features(self, metrics: List[PerformanceMetric]) -> Dict[str, np.ndarray]:
        """Extract features for ML bottleneck detection"""
        component_features = {}

        # Group metrics by component
        component_metrics = defaultdict(list)
        for metric in metrics:
            component_metrics[metric.component].append(metric)

        for component, component_metric_list in component_metrics.items():
            if len(component_metric_list) < 4:  # Need minimum data
                continue

            # Extract features: queue_length, response_time, error_rate, throughput
            response_times = [
                m.value for m in component_metric_list if "response_time" in m.metric_name.lower()
            ]
            error_rates = [
                m.value for m in component_metric_list if "error" in m.metric_name.lower()
            ]
            throughputs = [
                m.value for m in component_metric_list if "throughput" in m.metric_name.lower()
            ]

            # Calculate aggregate features
            queue_length = len(component_metric_list)  # Proxy for queue length
            avg_response_time = np.mean(response_times) if response_times else 0
            avg_error_rate = np.mean(error_rates) if error_rates else 0
            avg_throughput = np.mean(throughputs) if throughputs else 0

            features = np.array([queue_length, avg_response_time, avg_error_rate, avg_throughput])
            component_features[component] = features

        return component_features

    def _get_resource_solutions(self, resource_type: str) -> List[str]:
        """Get solutions for resource bottlenecks"""
        solutions = {
            "cpu": [
                "Scale to more CPU cores",
                "Optimize algorithms",
                "Implement parallel processing",
                "Cache computation results",
                "Profile and optimize hot paths",
            ],
            "memory": [
                "Increase memory allocation",
                "Implement memory pooling",
                "Optimize data structures",
                "Add garbage collection tuning",
                "Use streaming for large datasets",
            ],
            "disk": [
                "Upgrade to SSD storage",
                "Implement disk caching",
                "Optimize I/O operations",
                "Use asynchronous I/O",
                "Compress data",
            ],
            "network": [
                "Optimize network calls",
                "Implement connection pooling",
                "Use data compression",
                "Add CDN for static content",
                "Implement request batching",
            ],
        }

        return solutions.get(resource_type, ["Investigate resource usage patterns"])

    def _severity_score(self, severity: str) -> int:
        """Convert severity to numeric score for sorting"""
        severity_scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return severity_scores.get(severity, 0)

    def _update_component_stats(self, component: str, metric: PerformanceMetric):
        """Update component statistics with new metric"""
        stats = self._component_stats[component]
        stats["total_metrics"] += 1

        # Update rolling averages (simplified)
        if "response_time" in metric.metric_name.lower():
            current_avg = stats.get("avg_response_time", 0.0)
            stats["avg_response_time"] = current_avg * 0.9 + metric.value * 0.1

        elif "error" in metric.metric_name.lower():
            current_avg = stats.get("error_rate", 0.0)
            stats["error_rate"] = current_avg * 0.9 + metric.value * 0.1

        elif "throughput" in metric.metric_name.lower():
            current_avg = stats.get("throughput", 0.0)
            stats["throughput"] = current_avg * 0.9 + metric.value * 0.1

    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status for a specific component"""
        with self._lock:
            if component not in self._component_stats:
                return {"status": "unknown", "reason": "No metrics available"}

            stats = self._component_stats[component]

            # Simple health scoring
            health_score = 100.0
            issues = []

            if stats.get("avg_response_time", 0) > 1000:
                health_score -= 30
                issues.append("High response time")

            if stats.get("error_rate", 0) > 5:
                health_score -= 40
                issues.append("High error rate")

            if stats.get("throughput", 0) < 10:
                health_score -= 20
                issues.append("Low throughput")

            # Determine status
            if health_score >= 80:
                status = "healthy"
            elif health_score >= 60:
                status = "warning"
            elif health_score >= 40:
                status = "degraded"
            else:
                status = "critical"

            return {
                "status": status,
                "health_score": health_score,
                "issues": issues,
                "stats": stats,
                "created_by": "Vatsal216",
                "timestamp": time.time(),
            }

    def get_bottleneck_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected bottlenecks"""
        with self._lock:
            if not self._bottleneck_history:
                return {"total_detections": 0}

            # Group by type and severity
            by_type = defaultdict(int)
            by_severity = defaultdict(int)

            for bottleneck in self._bottleneck_history:
                by_type[bottleneck.bottleneck_type] += 1
                by_severity[bottleneck.severity] += 1

            return {
                "total_detections": len(self._bottleneck_history),
                "by_type": dict(by_type),
                "by_severity": dict(by_severity),
                "avg_impact_score": np.mean([b.impact_score for b in self._bottleneck_history]),
                "avg_confidence": np.mean(
                    [b.detection_confidence for b in self._bottleneck_history]
                ),
                "created_by": "Vatsal216",
                "timestamp": time.time(),
            }


class ResourceAnalyzer:
    """
    AI-driven resource usage analysis and optimization.

    Analyzes resource usage patterns, predicts scaling needs,
    and provides optimization recommendations.

    Created by: Vatsal216
    Date: 2025-07-23 06:16:00 UTC
    """

    def __init__(self, model_manager: Optional[MLModelManager] = None):
        """
        Initialize resource analyzer.

        Args:
            model_manager: ML model manager instance
        """
        self.model_manager = model_manager or MLModelManager()

        # Resource tracking
        self._resource_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._analysis_history: List[ResourceAnalysis] = []

        self._lock = threading.RLock()

        logger.info("ResourceAnalyzer initialized")
        logger.info("User: Vatsal216, Time: 2025-07-23 06:16:00")

    def add_resource_metric(
        self,
        resource_type: str,
        usage: float,
        component: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add resource usage metric for analysis.

        Args:
            resource_type: Type of resource (cpu, memory, disk, network)
            usage: Resource usage value
            component: Component using the resource
            metadata: Additional metadata
        """
        with self._lock:
            resource_key = f"{resource_type}:{component}"

            metric_data = {
                "timestamp": time.time(),
                "usage": usage,
                "component": component,
                "metadata": metadata or {},
            }

            self._resource_history[resource_key].append(metric_data)

    def analyze_resource_usage(
        self, resource_type: str, component: Optional[str] = None, time_window_hours: int = 1
    ) -> List[ResourceAnalysis]:
        """
        Analyze resource usage patterns.

        Args:
            resource_type: Type of resource to analyze
            component: Specific component (None for all)
            time_window_hours: Time window for analysis

        Returns:
            List of resource analyses
        """
        with self._lock:
            start_time = time.time()
            analyses = []

            # Filter resource data
            cutoff_time = time.time() - (time_window_hours * 3600)

            for resource_key, metrics in self._resource_history.items():
                key_resource_type, key_component = resource_key.split(":", 1)

                # Filter by resource type and component
                if key_resource_type != resource_type:
                    continue
                if component and key_component != component:
                    continue

                # Filter by time window
                recent_metrics = [m for m in metrics if m["timestamp"] >= cutoff_time]

                if len(recent_metrics) < 3:  # Need minimum data
                    continue

                # Perform analysis
                analysis = self._analyze_component_resource(
                    resource_type, key_component, recent_metrics
                )
                analyses.append(analysis)
                self._analysis_history.append(analysis)

            # Sort by utilization efficiency (lowest first)
            analyses.sort(key=lambda a: a.utilization_efficiency)

            analysis_time = time.time() - start_time
            metrics.record_metric("resource_analyses_total", len(analyses))
            metrics.record_metric("resource_analysis_time_seconds", analysis_time)

            logger.info(f"Analyzed {len(analyses)} resource components in {analysis_time:.3f}s")

            return analyses

    def _analyze_component_resource(
        self, resource_type: str, component: str, metrics: List[Dict[str, Any]]
    ) -> ResourceAnalysis:
        """Analyze resource usage for a specific component"""

        # Extract usage values
        usage_values = [m["usage"] for m in metrics]
        timestamps = [m["timestamp"] for m in metrics]

        # Calculate statistics
        current_usage = usage_values[-1]
        peak_usage = max(usage_values)
        average_usage = np.mean(usage_values)

        # Determine trend
        if len(usage_values) >= 5:
            recent_avg = np.mean(usage_values[-5:])
            earlier_avg = np.mean(usage_values[:5])

            if recent_avg > earlier_avg * 1.1:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Calculate efficiency and waste
        utilization_efficiency = self._calculate_utilization_efficiency(resource_type, usage_values)
        waste_percentage = self._calculate_waste_percentage(resource_type, usage_values)

        # Generate recommendations
        recommendations = self._generate_resource_recommendations(
            resource_type, component, average_usage, peak_usage, trend
        )

        # Generate scaling suggestions
        scaling_suggestions = self._generate_scaling_suggestions(
            resource_type, average_usage, peak_usage, trend
        )

        # Calculate cost implications
        cost_implications = self._calculate_cost_implications(
            resource_type, average_usage, peak_usage
        )

        return ResourceAnalysis(
            resource_type=resource_type,
            current_usage=current_usage,
            peak_usage=peak_usage,
            average_usage=average_usage,
            trend=trend,
            utilization_efficiency=utilization_efficiency,
            waste_percentage=waste_percentage,
            recommendations=recommendations,
            scaling_suggestions=scaling_suggestions,
            cost_implications=cost_implications,
        )

    def _calculate_utilization_efficiency(
        self, resource_type: str, usage_values: List[float]
    ) -> float:
        """Calculate resource utilization efficiency"""

        # Define optimal usage ranges by resource type
        optimal_ranges = {
            "cpu": (70, 85),  # 70-85% is optimal for CPU
            "memory": (75, 90),  # 75-90% is optimal for memory
            "disk": (60, 80),  # 60-80% is optimal for disk
            "network": (50, 70),  # 50-70% is optimal for network
        }

        optimal_min, optimal_max = optimal_ranges.get(resource_type, (70, 85))

        # Calculate how much time was spent in optimal range
        optimal_time = sum(1 for usage in usage_values if optimal_min <= usage <= optimal_max)

        efficiency = optimal_time / len(usage_values)
        return efficiency

    def _calculate_waste_percentage(self, resource_type: str, usage_values: List[float]) -> float:
        """Calculate resource waste percentage"""

        # Define waste thresholds by resource type
        waste_thresholds = {
            "cpu": 30,  # Below 30% is considered waste
            "memory": 40,  # Below 40% is considered waste
            "disk": 20,  # Below 20% is considered waste
            "network": 15,  # Below 15% is considered waste
        }

        waste_threshold = waste_thresholds.get(resource_type, 30)

        # Calculate percentage of time spent below waste threshold
        waste_time = sum(1 for usage in usage_values if usage < waste_threshold)

        waste_percentage = (waste_time / len(usage_values)) * 100
        return waste_percentage

    def _generate_resource_recommendations(
        self,
        resource_type: str,
        component: str,
        average_usage: float,
        peak_usage: float,
        trend: str,
    ) -> List[str]:
        """Generate resource optimization recommendations"""
        recommendations = []

        # Resource-specific recommendations
        if resource_type == "cpu":
            if average_usage < 30:
                recommendations.append("Consider reducing CPU allocation to save costs")
            elif average_usage > 85:
                recommendations.append("Consider increasing CPU allocation or optimizing code")

            if peak_usage > 95:
                recommendations.append("Add CPU burst capacity or implement load balancing")

        elif resource_type == "memory":
            if average_usage < 40:
                recommendations.append("Consider reducing memory allocation")
            elif average_usage > 90:
                recommendations.append(
                    "Increase memory allocation or implement memory optimization"
                )

            if peak_usage > 95:
                recommendations.append("Implement memory pooling or garbage collection tuning")

        elif resource_type == "disk":
            if average_usage < 20:
                recommendations.append("Consider smaller disk allocation")
            elif average_usage > 80:
                recommendations.append("Add disk capacity or implement data archiving")

            if peak_usage > 90:
                recommendations.append("Implement disk cleanup policies")

        elif resource_type == "network":
            if average_usage < 15:
                recommendations.append("Consider lower bandwidth allocation")
            elif average_usage > 70:
                recommendations.append("Optimize network usage or increase bandwidth")

        # Trend-based recommendations
        if trend == "increasing":
            recommendations.append(f"Monitor {resource_type} usage closely - upward trend detected")
            recommendations.append("Consider proactive scaling")
        elif trend == "decreasing":
            recommendations.append("Consider optimization or resource reallocation opportunities")

        return recommendations

    def _generate_scaling_suggestions(
        self, resource_type: str, average_usage: float, peak_usage: float, trend: str
    ) -> List[str]:
        """Generate scaling suggestions"""
        suggestions = []

        # Scale up suggestions
        if average_usage > 80 or peak_usage > 95:
            suggestions.append(f"Scale up {resource_type} allocation")

            if trend == "increasing":
                suggestions.append("Consider aggressive scaling due to upward trend")

        # Scale down suggestions
        if average_usage < 30 and peak_usage < 60:
            suggestions.append(f"Scale down {resource_type} allocation")

            if trend == "decreasing":
                suggestions.append("Consider significant scale-down due to downward trend")

        # Auto-scaling suggestions
        if peak_usage > average_usage * 2:
            suggestions.append("Implement auto-scaling for dynamic workloads")

        return suggestions

    def _calculate_cost_implications(
        self, resource_type: str, average_usage: float, peak_usage: float
    ) -> Dict[str, float]:
        """Calculate cost implications of resource usage"""

        # Simplified cost model (would be configurable in production)
        cost_per_unit = {
            "cpu": 0.05,  # $0.05 per CPU core per hour
            "memory": 0.01,  # $0.01 per GB per hour
            "disk": 0.001,  # $0.001 per GB per hour
            "network": 0.1,  # $0.1 per GB transferred
        }

        unit_cost = cost_per_unit.get(resource_type, 0.01)

        # Calculate current costs
        current_hourly_cost = (average_usage / 100.0) * unit_cost * 8  # Assume 8 units allocated

        # Calculate optimized costs
        optimal_allocation = max(peak_usage * 1.2, 50)  # 20% buffer above peak, minimum 50%
        optimized_hourly_cost = (optimal_allocation / 100.0) * unit_cost * 8

        # Calculate potential savings
        cost_savings = current_hourly_cost - optimized_hourly_cost

        return {
            "current_hourly_cost": current_hourly_cost,
            "optimized_hourly_cost": optimized_hourly_cost,
            "potential_monthly_savings": cost_savings * 24 * 30,
            "optimization_percentage": (
                (cost_savings / current_hourly_cost) * 100 if current_hourly_cost > 0 else 0
            ),
        }

    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get statistics about resource analysis"""
        with self._lock:
            if not self._analysis_history:
                return {"total_analyses": 0}

            # Group by resource type
            by_resource_type = defaultdict(list)
            for analysis in self._analysis_history:
                by_resource_type[analysis.resource_type].append(analysis)

            stats = {}
            for resource_type, analyses in by_resource_type.items():
                stats[resource_type] = {
                    "count": len(analyses),
                    "avg_efficiency": np.mean([a.utilization_efficiency for a in analyses]),
                    "avg_waste": np.mean([a.waste_percentage for a in analyses]),
                    "avg_usage": np.mean([a.average_usage for a in analyses]),
                }

            return {
                "total_analyses": len(self._analysis_history),
                "by_resource_type": stats,
                "created_by": "Vatsal216",
                "timestamp": time.time(),
            }


"""       
Pattern Analyzer for CrewGraph AI Intelligence Layer

This module provides workflow pattern recognition and analysis capabilities
to identify optimization opportunities and best practices.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ..types import WorkflowId
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WorkflowPattern:
    """Represents a recognized workflow pattern."""

    pattern_id: str
    pattern_type: str
    frequency: int
    average_performance: float
    optimization_potential: float
    characteristics: Dict[str, Any]
    examples: List[WorkflowId]
    recommendations: List[str]


class PatternAnalyzer:
    """
    Analyzes workflow execution patterns to identify optimization opportunities,
    best practices, and performance trends.
    """

    def __init__(self, min_pattern_frequency: int = 3):
        """
        Initialize the pattern analyzer.

        Args:
            min_pattern_frequency: Minimum frequency for a pattern to be significant
        """
        self.min_pattern_frequency = min_pattern_frequency
        self.discovered_patterns: Dict[str, WorkflowPattern] = {}
        self.workflow_fingerprints: Dict[WorkflowId, str] = {}
        self.execution_history: List[Dict[str, Any]] = []

        logger.info(f"PatternAnalyzer initialized with min_frequency={min_pattern_frequency}")

    def analyze_workflow_structure(self, workflow_definition: Dict[str, Any]) -> str:
        """
        Analyze workflow structure and generate a pattern fingerprint.

        Args:
            workflow_definition: Workflow structure definition

        Returns:
            Pattern fingerprint string
        """
        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])

        # Extract structural features
        features = {
            "task_count": len(tasks),
            "dependency_count": len(dependencies),
            "has_parallel_tasks": self._has_parallel_structure(tasks, dependencies),
            "has_loops": self._has_loops(dependencies),
            "task_types": self._extract_task_types(tasks),
            "complexity_score": self._calculate_complexity(tasks, dependencies),
        }

        # Create fingerprint from features
        fingerprint_data = f"{features['task_count']}-{features['dependency_count']}-"
        fingerprint_data += f"{'P' if features['has_parallel_tasks'] else 'S'}-"
        fingerprint_data += f"{'L' if features['has_loops'] else 'N'}-"
        fingerprint_data += f"{features['complexity_score']:.1f}-"
        fingerprint_data += "-".join(sorted(features["task_types"]))

        # Generate hash for compact representation
        fingerprint = hashlib.md5(fingerprint_data.encode()).hexdigest()[:12]

        logger.debug(f"Generated workflow fingerprint: {fingerprint}")
        return fingerprint

    def record_execution(
        self,
        workflow_id: WorkflowId,
        workflow_definition: Dict[str, Any],
        execution_metrics: Dict[str, Any],
    ) -> None:
        """
        Record workflow execution for pattern analysis.

        Args:
            workflow_id: Unique workflow identifier
            workflow_definition: Workflow structure
            execution_metrics: Execution performance metrics
        """
        fingerprint = self.analyze_workflow_structure(workflow_definition)
        self.workflow_fingerprints[workflow_id] = fingerprint

        execution_record = {
            "workflow_id": workflow_id,
            "fingerprint": fingerprint,
            "execution_time": execution_metrics.get("execution_time", 0),
            "success_rate": execution_metrics.get("success_rate", 1.0),
            "resource_usage": execution_metrics.get("resource_usage", {}),
            "error_count": execution_metrics.get("error_count", 0),
            "timestamp": datetime.now(),
            "workflow_definition": workflow_definition,
        }

        self.execution_history.append(execution_record)

        # Trigger pattern discovery if we have enough data
        if len(self.execution_history) % 10 == 0:
            self.discover_patterns()

        logger.debug(
            f"Recorded execution for workflow {workflow_id} with fingerprint {fingerprint}"
        )

    def discover_patterns(self) -> Dict[str, WorkflowPattern]:
        """
        Discover workflow patterns from execution history.

        Returns:
            Dictionary of discovered patterns
        """
        if len(self.execution_history) < self.min_pattern_frequency:
            return self.discovered_patterns

        # Group executions by fingerprint
        fingerprint_groups = defaultdict(list)
        for execution in self.execution_history:
            fingerprint = execution["fingerprint"]
            fingerprint_groups[fingerprint].append(execution)

        # Analyze each group for patterns
        new_patterns = {}
        for fingerprint, executions in fingerprint_groups.items():
            if len(executions) >= self.min_pattern_frequency:
                pattern = self._analyze_pattern_group(fingerprint, executions)
                if pattern:
                    new_patterns[pattern.pattern_id] = pattern

        self.discovered_patterns.update(new_patterns)

        logger.info(
            f"Discovered {len(new_patterns)} new patterns, total: {len(self.discovered_patterns)}"
        )
        return self.discovered_patterns

    def get_optimization_recommendations(self, workflow_definition: Dict[str, Any]) -> List[str]:
        """
        Get optimization recommendations based on recognized patterns.

        Args:
            workflow_definition: Workflow to analyze

        Returns:
            List of optimization recommendations
        """
        fingerprint = self.analyze_workflow_structure(workflow_definition)

        recommendations = []

        # Check if this workflow matches any known patterns
        matching_pattern = None
        for pattern in self.discovered_patterns.values():
            if self._patterns_match(fingerprint, pattern):
                matching_pattern = pattern
                break

        if matching_pattern:
            recommendations.extend(matching_pattern.recommendations)

            # Add pattern-specific recommendations
            if matching_pattern.optimization_potential > 30:
                recommendations.append(
                    f"High optimization potential detected ({matching_pattern.optimization_potential:.1f}%)"
                )

            if matching_pattern.average_performance < 0.7:
                recommendations.append(
                    "Similar workflows have shown performance issues - consider optimization"
                )

        else:
            # General recommendations for unknown patterns
            recommendations.extend(self._get_general_recommendations(workflow_definition))

        # Add structural recommendations
        recommendations.extend(self._get_structural_recommendations(workflow_definition))

        return list(set(recommendations))  # Remove duplicates

    def identify_performance_bottlenecks(
        self, workflow_definition: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify potential performance bottlenecks in workflow structure.

        Args:
            workflow_definition: Workflow to analyze

        Returns:
            List of identified bottlenecks with details
        """
        bottlenecks = []

        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])

        # Check for sequential bottlenecks
        sequential_chains = self._find_sequential_chains(tasks, dependencies)
        for chain in sequential_chains:
            if len(chain) > 5:
                bottlenecks.append(
                    {
                        "type": "sequential_chain",
                        "severity": "high" if len(chain) > 10 else "medium",
                        "description": f"Long sequential chain of {len(chain)} tasks",
                        "affected_tasks": chain,
                        "recommendation": "Consider parallelizing independent tasks in the chain",
                    }
                )

        # Check for resource-intensive task clusters
        resource_intensive_tasks = [
            task for task in tasks if task.get("resource_requirements", {}).get("memory", 0) > 1000
        ]

        if len(resource_intensive_tasks) > 2:
            bottlenecks.append(
                {
                    "type": "resource_bottleneck",
                    "severity": "medium",
                    "description": f"{len(resource_intensive_tasks)} resource-intensive tasks detected",
                    "affected_tasks": [
                        task.get("id", "unknown") for task in resource_intensive_tasks
                    ],
                    "recommendation": "Consider scheduling resource-intensive tasks separately",
                }
            )

        # Check for dependency bottlenecks
        dependency_counts = Counter()
        for dep in dependencies:
            dependency_counts[dep.get("target")] += 1

        high_dependency_tasks = [
            task_id for task_id, count in dependency_counts.items() if count > 3
        ]
        if high_dependency_tasks:
            bottlenecks.append(
                {
                    "type": "dependency_bottleneck",
                    "severity": "high",
                    "description": f"Tasks with high dependency count: {high_dependency_tasks}",
                    "affected_tasks": high_dependency_tasks,
                    "recommendation": "Review dependencies to reduce coupling",
                }
            )

        return bottlenecks

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of discovered patterns and analysis."""
        if not self.discovered_patterns:
            return {
                "total_patterns": 0,
                "total_executions_analyzed": len(self.execution_history),
                "message": "Insufficient data for pattern analysis",
            }

        # Calculate summary statistics
        total_optimizations = sum(
            1 for p in self.discovered_patterns.values() if p.optimization_potential > 20
        )

        avg_performance = sum(
            p.average_performance for p in self.discovered_patterns.values()
        ) / len(self.discovered_patterns)

        most_common_pattern = max(self.discovered_patterns.values(), key=lambda p: p.frequency)

        return {
            "total_patterns": len(self.discovered_patterns),
            "total_executions_analyzed": len(self.execution_history),
            "patterns_with_optimization_potential": total_optimizations,
            "average_pattern_performance": avg_performance,
            "most_common_pattern": {
                "type": most_common_pattern.pattern_type,
                "frequency": most_common_pattern.frequency,
                "performance": most_common_pattern.average_performance,
            },
            "analysis_coverage": len(set(self.workflow_fingerprints.values())),
            "last_analysis": datetime.now().isoformat(),
        }

    def _has_parallel_structure(self, tasks: List[Dict], dependencies: List[Dict]) -> bool:
        """Check if workflow has parallel task execution opportunities."""
        if len(tasks) < 2:
            return False

        # Build dependency graph
        dependent_tasks = set()
        for dep in dependencies:
            dependent_tasks.add(dep.get("target"))

        # If there are tasks without dependencies, there's potential for parallelism
        independent_tasks = [task for task in tasks if task.get("id") not in dependent_tasks]
        return len(independent_tasks) > 1

    def _has_loops(self, dependencies: List[Dict]) -> bool:
        """Check if workflow has loops (cycles) in dependencies."""
        # Simple cycle detection - build adjacency list and check for back edges
        graph = defaultdict(list)
        for dep in dependencies:
            source = dep.get("source")
            target = dep.get("target")
            if source and target:
                graph[source].append(target)

        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True

        return False

    def _extract_task_types(self, tasks: List[Dict]) -> List[str]:
        """Extract and categorize task types."""
        types = []
        for task in tasks:
            task_type = task.get("type", "unknown")
            # Normalize task type
            if "data" in task_type.lower():
                types.append("data")
            elif "compute" in task_type.lower() or "process" in task_type.lower():
                types.append("compute")
            elif "io" in task_type.lower() or "file" in task_type.lower():
                types.append("io")
            elif "ml" in task_type.lower() or "model" in task_type.lower():
                types.append("ml")
            else:
                types.append("general")

        return list(set(types))  # Return unique types

    def _calculate_complexity(self, tasks: List[Dict], dependencies: List[Dict]) -> float:
        """Calculate workflow complexity score."""
        task_count = len(tasks)
        dependency_count = len(dependencies)

        if task_count == 0:
            return 0.0

        # Base complexity from task and dependency counts
        base_complexity = task_count + (dependency_count * 0.5)

        # Normalize to 0-10 scale
        normalized = min(10.0, base_complexity / 2.0)

        return normalized

    def _analyze_pattern_group(
        self, fingerprint: str, executions: List[Dict]
    ) -> Optional[WorkflowPattern]:
        """Analyze a group of similar executions to extract patterns."""
        if not executions:
            return None

        # Calculate performance metrics
        execution_times = [e["execution_time"] for e in executions]
        success_rates = [e["success_rate"] for e in executions]

        avg_performance = sum(success_rates) / len(success_rates)
        avg_execution_time = sum(execution_times) / len(execution_times)

        # Calculate optimization potential
        time_variance = (
            max(execution_times) - min(execution_times) if len(execution_times) > 1 else 0
        )
        optimization_potential = (
            min(50.0, (time_variance / avg_execution_time) * 100) if avg_execution_time > 0 else 0
        )

        # Extract characteristics
        sample_workflow = executions[0]["workflow_definition"]
        characteristics = {
            "task_count": len(sample_workflow.get("tasks", [])),
            "dependency_count": len(sample_workflow.get("dependencies", [])),
            "average_execution_time": avg_execution_time,
            "execution_count": len(executions),
        }

        # Generate recommendations
        recommendations = []
        if optimization_potential > 25:
            recommendations.append(
                "High execution time variance suggests optimization opportunities"
            )
        if avg_performance < 0.8:
            recommendations.append("Success rate could be improved with error handling")
        if avg_execution_time > 60:
            recommendations.append("Long execution time - consider task parallelization")

        # Determine pattern type
        task_count = characteristics["task_count"]
        if task_count <= 3:
            pattern_type = "simple"
        elif task_count <= 10:
            pattern_type = "moderate"
        else:
            pattern_type = "complex"

        pattern = WorkflowPattern(
            pattern_id=fingerprint,
            pattern_type=pattern_type,
            frequency=len(executions),
            average_performance=avg_performance,
            optimization_potential=optimization_potential,
            characteristics=characteristics,
            examples=[e["workflow_id"] for e in executions[:5]],  # Store up to 5 examples
            recommendations=recommendations,
        )

        return pattern

    def _patterns_match(self, fingerprint: str, pattern: WorkflowPattern) -> bool:
        """Check if a fingerprint matches a discovered pattern."""
        return fingerprint == pattern.pattern_id

    def _get_general_recommendations(self, workflow_definition: Dict[str, Any]) -> List[str]:
        """Get general optimization recommendations."""
        recommendations = []

        task_count = len(workflow_definition.get("tasks", []))
        dependency_count = len(workflow_definition.get("dependencies", []))

        if task_count > 5 and dependency_count < task_count * 0.3:
            recommendations.append("Consider parallel execution for independent tasks")

        if dependency_count > task_count:
            recommendations.append("High dependency ratio - review workflow structure")

        return recommendations

    def _get_structural_recommendations(self, workflow_definition: Dict[str, Any]) -> List[str]:
        """Get recommendations based on workflow structure."""
        recommendations = []

        tasks = workflow_definition.get("tasks", [])

        # Check for resource imbalance
        memory_requirements = [
            task.get("resource_requirements", {}).get("memory", 0) for task in tasks
        ]
        if (
            memory_requirements
            and max(memory_requirements) > sum(memory_requirements) / len(memory_requirements) * 3
        ):
            recommendations.append(
                "Memory requirements are imbalanced - consider resource leveling"
            )

        return recommendations

    def _find_sequential_chains(
        self, tasks: List[Dict], dependencies: List[Dict]
    ) -> List[List[str]]:
        """Find sequential chains in the workflow."""
        # Build dependency graph
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)

        for dep in dependencies:
            source = dep.get("source")
            target = dep.get("target")
            if source and target:
                graph[source].append(target)
                reverse_graph[target].append(source)

        # Find chains (paths with no branching)
        chains = []
        visited = set()

        for task in tasks:
            task_id = task.get("id")
            if task_id and task_id not in visited:
                # Start a new chain if this task has at most one predecessor
                if len(reverse_graph[task_id]) <= 1:
                    chain = self._build_chain(task_id, graph, visited)
                    if len(chain) > 1:
                        chains.append(chain)

        return chains

    def _build_chain(self, start_task: str, graph: Dict, visited: Set[str]) -> List[str]:
        """Build a sequential chain starting from a task."""
        chain = []
        current = start_task

        while current and current not in visited and len(graph[current]) <= 1:
            chain.append(current)
            visited.add(current)

            # Move to next task if it exists and has only one dependency
            next_tasks = graph[current]
            if next_tasks:
                current = next_tasks[0]
            else:
                break

        return chain
