"""
Bottleneck and Resource Analyzers for CrewGraph AI

AI-driven analysis of workflow bottlenecks and resource usage patterns.

Author: Vatsal216
Created: 2025-07-23 06:16:00 UTC
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque

from .ml_models import MLModelManager, ModelType
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

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
    
    def add_performance_metric(self, 
                             metric_name: str,
                             value: float,
                             component: str,
                             metadata: Optional[Dict[str, Any]] = None):
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
                metadata=metadata or {}
            )
            
            self._metrics_buffer.append(metric)
            
            # Update component statistics
            if component not in self._component_stats:
                self._component_stats[component] = {
                    'total_metrics': 0,
                    'avg_response_time': 0.0,
                    'error_rate': 0.0,
                    'throughput': 0.0
                }
            
            self._update_component_stats(component, metric)
    
    def detect_bottlenecks(self, 
                          time_window_minutes: int = 10) -> List[BottleneckDetection]:
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
            recent_metrics = [
                m for m in self._metrics_buffer 
                if m.timestamp >= cutoff_time
            ]
            
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
            bottlenecks.sort(key=lambda b: (self._severity_score(b.severity), b.impact_score), reverse=True)
            
            # Record detections
            detection_time = time.time() - start_time
            for bottleneck in bottlenecks:
                self._bottleneck_history.append(bottleneck)
            
            metrics.record_metric("bottleneck_detections_total", len(bottlenecks))
            metrics.record_metric("bottleneck_detection_time_seconds", detection_time)
            
            logger.info(f"Detected {len(bottlenecks)} bottlenecks in {detection_time:.3f}s")
            
            return bottlenecks[:10]  # Return top 10 bottlenecks
    
    def _detect_response_time_bottlenecks(self, 
                                        metrics: List[PerformanceMetric]) -> List[BottleneckDetection]:
        """Detect response time bottlenecks"""
        bottlenecks = []
        
        # Group metrics by component
        component_response_times = defaultdict(list)
        for metric in metrics:
            if 'response_time' in metric.metric_name.lower():
                component_response_times[metric.component].append(metric.value)
        
        for component, times in component_response_times.items():
            if not times:
                continue
                
            avg_response_time = np.mean(times)
            p95_response_time = np.percentile(times, 95)
            
            # Detect slow response times
            if avg_response_time > 1000:  # 1 second
                severity = "critical" if avg_response_time > 5000 else "high"
                
                bottlenecks.append(BottleneckDetection(
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
                        "Review algorithm efficiency"
                    ],
                    estimated_delay=avg_response_time / 1000.0,
                    detection_confidence=0.85
                ))
            
            # Detect response time variability
            if len(times) > 5:
                std_dev = np.std(times)
                coefficient_of_variation = std_dev / avg_response_time if avg_response_time > 0 else 0
                
                if coefficient_of_variation > 0.5:  # High variability
                    bottlenecks.append(BottleneckDetection(
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
                            "Review concurrency handling"
                        ],
                        estimated_delay=std_dev / 1000.0,
                        detection_confidence=0.75
                    ))
        
        return bottlenecks
    
    def _detect_throughput_bottlenecks(self, 
                                     metrics: List[PerformanceMetric]) -> List[BottleneckDetection]:
        """Detect throughput bottlenecks"""
        bottlenecks = []
        
        # Group metrics by component
        component_throughputs = defaultdict(list)
        for metric in metrics:
            if 'throughput' in metric.metric_name.lower() or 'requests_per_second' in metric.metric_name.lower():
                component_throughputs[metric.component].append(metric.value)
        
        for component, throughputs in component_throughputs.items():
            if len(throughputs) < 3:  # Need enough data points
                continue
            
            # Detect declining throughput
            recent_throughput = np.mean(throughputs[-3:])
            earlier_throughput = np.mean(throughputs[:3])
            
            if recent_throughput < earlier_throughput * 0.8:  # 20% decline
                decline_percentage = (earlier_throughput - recent_throughput) / earlier_throughput
                
                bottlenecks.append(BottleneckDetection(
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
                        "Review system dependencies"
                    ],
                    estimated_delay=0.0,  # Ongoing impact
                    detection_confidence=0.8
                ))
            
            # Detect low absolute throughput
            avg_throughput = np.mean(throughputs)
            if avg_throughput < 10:  # Less than 10 requests per second
                bottlenecks.append(BottleneckDetection(
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
                        "Eliminate processing bottlenecks"
                    ],
                    estimated_delay=0.0,
                    detection_confidence=0.7
                ))
        
        return bottlenecks
    
    def _detect_resource_bottlenecks(self, 
                                   metrics: List[PerformanceMetric]) -> List[BottleneckDetection]:
        """Detect resource utilization bottlenecks"""
        bottlenecks = []
        
        # Group metrics by resource type
        resource_usage = defaultdict(lambda: defaultdict(list))
        for metric in metrics:
            metric_lower = metric.metric_name.lower()
            if any(resource in metric_lower for resource in ['cpu', 'memory', 'disk', 'network']):
                if 'cpu' in metric_lower:
                    resource_type = 'cpu'
                elif 'memory' in metric_lower:
                    resource_type = 'memory'
                elif 'disk' in metric_lower:
                    resource_type = 'disk'
                elif 'network' in metric_lower:
                    resource_type = 'network'
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
                if resource_type == 'cpu' and avg_usage > 80:
                    severity = "critical" if avg_usage > 95 else "high"
                elif resource_type == 'memory' and avg_usage > 85:
                    severity = "critical" if avg_usage > 95 else "high"
                elif resource_type in ['disk', 'network'] and avg_usage > 90:
                    severity = "high"
                else:
                    continue
                
                bottlenecks.append(BottleneckDetection(
                    bottleneck_type=f"{resource_type}_saturation",
                    severity=severity,
                    location=component,
                    description=f"High {resource_type} usage in {component}: {avg_usage:.1f}% average",
                    impact_score=avg_usage / 10.0,
                    probability=0.9,
                    affected_components=[component],
                    suggested_solutions=self._get_resource_solutions(resource_type),
                    estimated_delay=0.0,
                    detection_confidence=0.9
                ))
        
        return bottlenecks
    
    def _detect_error_rate_bottlenecks(self, 
                                     metrics: List[PerformanceMetric]) -> List[BottleneckDetection]:
        """Detect error rate bottlenecks"""
        bottlenecks = []
        
        # Group metrics by component
        component_errors = defaultdict(list)
        for metric in metrics:
            if 'error' in metric.metric_name.lower() or 'failure' in metric.metric_name.lower():
                component_errors[metric.component].append(metric.value)
        
        for component, error_rates in component_errors.items():
            if not error_rates:
                continue
            
            avg_error_rate = np.mean(error_rates)
            
            if avg_error_rate > 5:  # More than 5% error rate
                severity = "critical" if avg_error_rate > 20 else "high" if avg_error_rate > 10 else "medium"
                
                bottlenecks.append(BottleneckDetection(
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
                        "Check external dependencies"
                    ],
                    estimated_delay=0.0,
                    detection_confidence=0.9
                ))
        
        return bottlenecks
    
    def _detect_dependency_bottlenecks(self, 
                                     metrics: List[PerformanceMetric]) -> List[BottleneckDetection]:
        """Detect dependency-related bottlenecks"""
        bottlenecks = []
        
        # Look for patterns indicating dependency issues
        component_patterns = defaultdict(list)
        for metric in metrics:
            if any(keyword in metric.metric_name.lower() 
                   for keyword in ['timeout', 'connection', 'dependency', 'external']):
                component_patterns[metric.component].append(metric)
        
        for component, component_metrics in component_patterns.items():
            if len(component_metrics) < 3:
                continue
            
            # Check for timeout patterns
            timeout_metrics = [m for m in component_metrics if 'timeout' in m.metric_name.lower()]
            if timeout_metrics:
                timeout_rate = np.mean([m.value for m in timeout_metrics])
                
                if timeout_rate > 2:  # More than 2% timeout rate
                    bottlenecks.append(BottleneckDetection(
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
                            "Review network connectivity"
                        ],
                        estimated_delay=5.0,  # Typical timeout delay
                        detection_confidence=0.8
                    ))
        
        return bottlenecks
    
    def _ml_bottleneck_detection(self, 
                               metrics: List[PerformanceMetric]) -> List[BottleneckDetection]:
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
                    
                    bottlenecks.append(BottleneckDetection(
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
                            "Analyze usage patterns"
                        ],
                        estimated_delay=1.0,
                        detection_confidence=bottleneck_probability
                    ))
        
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
            response_times = [m.value for m in component_metric_list 
                            if 'response_time' in m.metric_name.lower()]
            error_rates = [m.value for m in component_metric_list 
                          if 'error' in m.metric_name.lower()]
            throughputs = [m.value for m in component_metric_list 
                          if 'throughput' in m.metric_name.lower()]
            
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
            'cpu': [
                "Scale to more CPU cores",
                "Optimize algorithms",
                "Implement parallel processing",
                "Cache computation results",
                "Profile and optimize hot paths"
            ],
            'memory': [
                "Increase memory allocation",
                "Implement memory pooling",
                "Optimize data structures",
                "Add garbage collection tuning",
                "Use streaming for large datasets"
            ],
            'disk': [
                "Upgrade to SSD storage",
                "Implement disk caching",
                "Optimize I/O operations",
                "Use asynchronous I/O",
                "Compress data"
            ],
            'network': [
                "Optimize network calls",
                "Implement connection pooling",
                "Use data compression",
                "Add CDN for static content",
                "Implement request batching"
            ]
        }
        
        return solutions.get(resource_type, ["Investigate resource usage patterns"])
    
    def _severity_score(self, severity: str) -> int:
        """Convert severity to numeric score for sorting"""
        severity_scores = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        return severity_scores.get(severity, 0)
    
    def _update_component_stats(self, component: str, metric: PerformanceMetric):
        """Update component statistics with new metric"""
        stats = self._component_stats[component]
        stats['total_metrics'] += 1
        
        # Update rolling averages (simplified)
        if 'response_time' in metric.metric_name.lower():
            current_avg = stats.get('avg_response_time', 0.0)
            stats['avg_response_time'] = (current_avg * 0.9 + metric.value * 0.1)
        
        elif 'error' in metric.metric_name.lower():
            current_avg = stats.get('error_rate', 0.0)
            stats['error_rate'] = (current_avg * 0.9 + metric.value * 0.1)
        
        elif 'throughput' in metric.metric_name.lower():
            current_avg = stats.get('throughput', 0.0)
            stats['throughput'] = (current_avg * 0.9 + metric.value * 0.1)
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status for a specific component"""
        with self._lock:
            if component not in self._component_stats:
                return {"status": "unknown", "reason": "No metrics available"}
            
            stats = self._component_stats[component]
            
            # Simple health scoring
            health_score = 100.0
            issues = []
            
            if stats.get('avg_response_time', 0) > 1000:
                health_score -= 30
                issues.append("High response time")
            
            if stats.get('error_rate', 0) > 5:
                health_score -= 40
                issues.append("High error rate")
            
            if stats.get('throughput', 0) < 10:
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
                "timestamp": time.time()
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
                "avg_confidence": np.mean([b.detection_confidence for b in self._bottleneck_history]),
                "created_by": "Vatsal216",
                "timestamp": time.time()
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
    
    def add_resource_metric(self, 
                          resource_type: str,
                          usage: float,
                          component: str,
                          metadata: Optional[Dict[str, Any]] = None):
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
                'timestamp': time.time(),
                'usage': usage,
                'component': component,
                'metadata': metadata or {}
            }
            
            self._resource_history[resource_key].append(metric_data)
    
    def analyze_resource_usage(self, 
                             resource_type: str,
                             component: Optional[str] = None,
                             time_window_hours: int = 1) -> List[ResourceAnalysis]:
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
                key_resource_type, key_component = resource_key.split(':', 1)
                
                # Filter by resource type and component
                if key_resource_type != resource_type:
                    continue
                if component and key_component != component:
                    continue
                
                # Filter by time window
                recent_metrics = [
                    m for m in metrics 
                    if m['timestamp'] >= cutoff_time
                ]
                
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
    
    def _analyze_component_resource(self, 
                                  resource_type: str,
                                  component: str,
                                  metrics: List[Dict[str, Any]]) -> ResourceAnalysis:
        """Analyze resource usage for a specific component"""
        
        # Extract usage values
        usage_values = [m['usage'] for m in metrics]
        timestamps = [m['timestamp'] for m in metrics]
        
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
        utilization_efficiency = self._calculate_utilization_efficiency(
            resource_type, usage_values
        )
        waste_percentage = self._calculate_waste_percentage(
            resource_type, usage_values
        )
        
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
            cost_implications=cost_implications
        )
    
    def _calculate_utilization_efficiency(self, 
                                        resource_type: str,
                                        usage_values: List[float]) -> float:
        """Calculate resource utilization efficiency"""
        
        # Define optimal usage ranges by resource type
        optimal_ranges = {
            'cpu': (70, 85),      # 70-85% is optimal for CPU
            'memory': (75, 90),   # 75-90% is optimal for memory
            'disk': (60, 80),     # 60-80% is optimal for disk
            'network': (50, 70)   # 50-70% is optimal for network
        }
        
        optimal_min, optimal_max = optimal_ranges.get(resource_type, (70, 85))
        
        # Calculate how much time was spent in optimal range
        optimal_time = sum(
            1 for usage in usage_values 
            if optimal_min <= usage <= optimal_max
        )
        
        efficiency = optimal_time / len(usage_values)
        return efficiency
    
    def _calculate_waste_percentage(self, 
                                  resource_type: str,
                                  usage_values: List[float]) -> float:
        """Calculate resource waste percentage"""
        
        # Define waste thresholds by resource type
        waste_thresholds = {
            'cpu': 30,      # Below 30% is considered waste
            'memory': 40,   # Below 40% is considered waste
            'disk': 20,     # Below 20% is considered waste
            'network': 15   # Below 15% is considered waste
        }
        
        waste_threshold = waste_thresholds.get(resource_type, 30)
        
        # Calculate percentage of time spent below waste threshold
        waste_time = sum(
            1 for usage in usage_values 
            if usage < waste_threshold
        )
        
        waste_percentage = (waste_time / len(usage_values)) * 100
        return waste_percentage
    
    def _generate_resource_recommendations(self, 
                                         resource_type: str,
                                         component: str,
                                         average_usage: float,
                                         peak_usage: float,
                                         trend: str) -> List[str]:
        """Generate resource optimization recommendations"""
        recommendations = []
        
        # Resource-specific recommendations
        if resource_type == 'cpu':
            if average_usage < 30:
                recommendations.append("Consider reducing CPU allocation to save costs")
            elif average_usage > 85:
                recommendations.append("Consider increasing CPU allocation or optimizing code")
            
            if peak_usage > 95:
                recommendations.append("Add CPU burst capacity or implement load balancing")
        
        elif resource_type == 'memory':
            if average_usage < 40:
                recommendations.append("Consider reducing memory allocation")
            elif average_usage > 90:
                recommendations.append("Increase memory allocation or implement memory optimization")
            
            if peak_usage > 95:
                recommendations.append("Implement memory pooling or garbage collection tuning")
        
        elif resource_type == 'disk':
            if average_usage < 20:
                recommendations.append("Consider smaller disk allocation")
            elif average_usage > 80:
                recommendations.append("Add disk capacity or implement data archiving")
            
            if peak_usage > 90:
                recommendations.append("Implement disk cleanup policies")
        
        elif resource_type == 'network':
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
    
    def _generate_scaling_suggestions(self, 
                                    resource_type: str,
                                    average_usage: float,
                                    peak_usage: float,
                                    trend: str) -> List[str]:
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
    
    def _calculate_cost_implications(self, 
                                   resource_type: str,
                                   average_usage: float,
                                   peak_usage: float) -> Dict[str, float]:
        """Calculate cost implications of resource usage"""
        
        # Simplified cost model (would be configurable in production)
        cost_per_unit = {
            'cpu': 0.05,      # $0.05 per CPU core per hour
            'memory': 0.01,   # $0.01 per GB per hour
            'disk': 0.001,    # $0.001 per GB per hour
            'network': 0.1    # $0.1 per GB transferred
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
            "optimization_percentage": (cost_savings / current_hourly_cost) * 100 if current_hourly_cost > 0 else 0
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
                    "avg_usage": np.mean([a.average_usage for a in analyses])
                }
            
            return {
                "total_analyses": len(self._analysis_history),
                "by_resource_type": stats,
                "created_by": "Vatsal216",
                "timestamp": time.time()
            }