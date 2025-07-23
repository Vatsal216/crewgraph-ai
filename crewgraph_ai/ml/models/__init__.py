"""
Core ML Models for CrewGraph AI

Implements the core machine learning models for workflow optimization:
- Workflow Pattern Learning
- Predictive Resource Scaling
- Intelligent Task Scheduling
- Performance Anomaly Detection
- Cost Prediction Models
- Auto-tuning Parameters

Author: Vatsal216
Created: 2025-01-27
"""

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
    ML_AVAILABLE = True
except ImportError:
    np = pd = KMeans = DBSCAN = IsolationForest = None
    RandomForestRegressor = GradientBoostingRegressor = LogisticRegression = None
    LinearRegression = MLPRegressor = MLPClassifier = None
    StandardScaler = MinMaxScaler = train_test_split = GridSearchCV = None
    mean_squared_error = accuracy_score = f1_score = None
    ML_AVAILABLE = False

from ..types import WorkflowId, ExecutionId
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WorkflowPattern:
    """Represents a learned workflow execution pattern."""
    
    pattern_id: str
    pattern_type: str  # sequential, parallel, hybrid, conditional
    frequency: int
    success_rate: float
    avg_duration: float
    resource_usage: Dict[str, float]
    task_sequence: List[str]
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for serialization."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "frequency": self.frequency,
            "success_rate": self.success_rate,
            "avg_duration": self.avg_duration,
            "resource_usage": self.resource_usage,
            "task_sequence": self.task_sequence,
            "conditions": self.conditions
        }


@dataclass 
class ResourcePrediction:
    """Resource scaling prediction result."""
    
    predicted_cpu: float
    predicted_memory: float
    predicted_storage: float
    predicted_instances: int
    confidence_score: float
    scaling_trigger: str
    recommendation: str


@dataclass
class SchedulingDecision:
    """Intelligent task scheduling decision."""
    
    task_id: str
    assigned_node: str
    priority_score: float
    estimated_start_time: float
    estimated_duration: float
    resource_allocation: Dict[str, float]
    dependencies_resolved: bool
    reasoning: str


@dataclass
class AnomalyAlert:
    """Performance anomaly detection alert."""
    
    anomaly_id: str
    timestamp: float
    severity: str  # low, medium, high, critical
    anomaly_type: str  # performance, resource, error_rate, cost
    description: str
    affected_components: List[str]
    suggested_actions: List[str]
    confidence_score: float


class WorkflowPatternLearner:
    """
    Learns patterns from workflow execution history using clustering
    and sequence analysis to identify optimal execution patterns.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize pattern learner."""
        self.model_dir = Path(model_dir) if model_dir else Path("models/patterns")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.patterns: Dict[str, WorkflowPattern] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # ML models for pattern detection
        self.sequence_clusterer = None
        self.pattern_classifier = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        
        if ML_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for pattern learning."""
        if not ML_AVAILABLE:
            return
            
        self.sequence_clusterer = KMeans(n_clusters=5, random_state=42)
        self.pattern_classifier = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        )
    
    def learn_from_execution(
        self,
        workflow_id: WorkflowId,
        execution_data: Dict[str, Any]
    ):
        """Learn patterns from workflow execution data."""
        # Store execution history
        execution_record = {
            "workflow_id": workflow_id,
            "timestamp": time.time(),
            **execution_data
        }
        self.execution_history.append(execution_record)
        
        # Update patterns if we have enough data
        if len(self.execution_history) >= 10:
            self._update_patterns()
        
        logger.info(f"Learned from execution: {workflow_id}")
    
    def _update_patterns(self):
        """Update workflow patterns based on execution history."""
        if not ML_AVAILABLE or len(self.execution_history) < 10:
            return
        
        try:
            # Extract features for clustering
            features = self._extract_pattern_features()
            
            if len(features) > 0:
                # Normalize features
                features_scaled = self.scaler.fit_transform(features)
                
                # Cluster similar execution patterns
                cluster_labels = self.sequence_clusterer.fit_predict(features_scaled)
                
                # Analyze each cluster to identify patterns
                self._analyze_clusters(cluster_labels)
                
                logger.info(f"Updated patterns: {len(self.patterns)} patterns identified")
                
        except Exception as e:
            logger.error(f"Pattern update failed: {e}")
    
    def _extract_pattern_features(self) -> List[List[float]]:
        """Extract features from execution history for pattern analysis."""
        features = []
        
        for execution in self.execution_history:
            # Extract timing features
            duration = execution.get("duration", 0)
            task_count = execution.get("task_count", 1)
            parallel_ratio = execution.get("parallel_ratio", 0)
            
            # Extract resource features
            cpu_usage = execution.get("cpu_usage", 0)
            memory_usage = execution.get("memory_usage", 0)
            
            # Extract success features
            success_rate = 1.0 if execution.get("success", True) else 0.0
            error_count = execution.get("error_count", 0)
            
            feature_vector = [
                duration, task_count, parallel_ratio,
                cpu_usage, memory_usage, success_rate, error_count
            ]
            features.append(feature_vector)
        
        return features
    
    def _analyze_clusters(self, cluster_labels: List[int]):
        """Analyze clusters to identify workflow patterns."""
        cluster_data = {}
        
        # Group executions by cluster
        for i, label in enumerate(cluster_labels):
            if label not in cluster_data:
                cluster_data[label] = []
            cluster_data[label].append(self.execution_history[i])
        
        # Analyze each cluster
        for cluster_id, executions in cluster_data.items():
            if len(executions) < 3:  # Need minimum executions for pattern
                continue
            
            pattern = self._create_pattern_from_cluster(cluster_id, executions)
            if pattern:
                self.patterns[pattern.pattern_id] = pattern
    
    def _create_pattern_from_cluster(
        self, 
        cluster_id: int, 
        executions: List[Dict[str, Any]]
    ) -> Optional[WorkflowPattern]:
        """Create a workflow pattern from cluster analysis."""
        if not executions:
            return None
        
        # Calculate pattern statistics
        durations = [e.get("duration", 0) for e in executions]
        success_rates = [1.0 if e.get("success", True) else 0.0 for e in executions]
        
        avg_duration = sum(durations) / len(durations)
        success_rate = sum(success_rates) / len(success_rates)
        
        # Determine pattern type based on characteristics
        parallel_ratios = [e.get("parallel_ratio", 0) for e in executions]
        avg_parallel_ratio = sum(parallel_ratios) / len(parallel_ratios)
        
        if avg_parallel_ratio > 0.7:
            pattern_type = "parallel"
        elif avg_parallel_ratio < 0.2:
            pattern_type = "sequential"
        else:
            pattern_type = "hybrid"
        
        # Extract common task sequence
        task_sequences = [e.get("task_sequence", []) for e in executions]
        common_sequence = self._find_common_sequence(task_sequences)
        
        # Calculate resource usage
        resource_usage = {
            "cpu": sum(e.get("cpu_usage", 0) for e in executions) / len(executions),
            "memory": sum(e.get("memory_usage", 0) for e in executions) / len(executions)
        }
        
        pattern = WorkflowPattern(
            pattern_id=f"pattern_{cluster_id}_{int(time.time())}",
            pattern_type=pattern_type,
            frequency=len(executions),
            success_rate=success_rate,
            avg_duration=avg_duration,
            resource_usage=resource_usage,
            task_sequence=common_sequence
        )
        
        return pattern
    
    def _find_common_sequence(self, sequences: List[List[str]]) -> List[str]:
        """Find common task sequence across executions."""
        if not sequences:
            return []
        
        # Simple approach: find most common starting sequence
        min_length = min(len(seq) for seq in sequences) if sequences else 0
        common_sequence = []
        
        for i in range(min_length):
            task_at_pos = [seq[i] for seq in sequences if len(seq) > i]
            if task_at_pos:
                # Find most common task at this position
                most_common = max(set(task_at_pos), key=task_at_pos.count)
                if task_at_pos.count(most_common) > len(sequences) * 0.5:
                    common_sequence.append(most_common)
                else:
                    break
        
        return common_sequence
    
    def get_similar_patterns(
        self, 
        workflow_features: Dict[str, Any]
    ) -> List[WorkflowPattern]:
        """Find patterns similar to current workflow features."""
        if not self.patterns:
            return []
        
        # Simple similarity matching (could be enhanced with ML)
        similar_patterns = []
        
        for pattern in self.patterns.values():
            similarity_score = self._calculate_similarity(workflow_features, pattern)
            if similarity_score > 0.7:  # Threshold for similarity
                similar_patterns.append(pattern)
        
        # Sort by similarity
        similar_patterns.sort(
            key=lambda p: self._calculate_similarity(workflow_features, p),
            reverse=True
        )
        
        return similar_patterns[:5]  # Return top 5 similar patterns
    
    def _calculate_similarity(
        self, 
        features: Dict[str, Any], 
        pattern: WorkflowPattern
    ) -> float:
        """Calculate similarity between workflow features and pattern."""
        # Simple similarity calculation
        score = 0.0
        total_weight = 0.0
        
        # Task count similarity
        feature_tasks = features.get("task_count", 1)
        pattern_tasks = len(pattern.task_sequence)
        if max(feature_tasks, pattern_tasks) > 0:
            task_similarity = 1.0 - abs(feature_tasks - pattern_tasks) / max(feature_tasks, pattern_tasks)
            score += task_similarity * 0.3
            total_weight += 0.3
        
        # Duration similarity
        feature_duration = features.get("estimated_duration", 0)
        if feature_duration > 0 and pattern.avg_duration > 0:
            duration_similarity = 1.0 - abs(feature_duration - pattern.avg_duration) / max(feature_duration, pattern.avg_duration)
            score += duration_similarity * 0.4
            total_weight += 0.4
        
        # Resource similarity
        feature_cpu = features.get("cpu_requirement", 1.0)
        pattern_cpu = pattern.resource_usage.get("cpu", 1.0)
        if max(feature_cpu, pattern_cpu) > 0:
            cpu_similarity = 1.0 - abs(feature_cpu - pattern_cpu) / max(feature_cpu, pattern_cpu)
            score += cpu_similarity * 0.3
            total_weight += 0.3
        
        return score / max(total_weight, 1.0)


class ResourceScalingPredictor:
    """
    Predicts optimal resource scaling based on workload patterns
    and historical performance data.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize resource scaling predictor."""
        self.model_dir = Path(model_dir) if model_dir else Path("models/scaling")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # ML models for different resource types
        self.cpu_predictor = None
        self.memory_predictor = None
        self.instance_predictor = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        
        # Training data
        self.training_data: List[Dict[str, Any]] = []
        
        if ML_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for resource prediction."""
        if not ML_AVAILABLE:
            return
        
        self.cpu_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.memory_predictor = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        self.instance_predictor = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        )
    
    def predict_resource_needs(
        self,
        workload_metrics: Dict[str, float],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> ResourcePrediction:
        """Predict optimal resource allocation for given workload."""
        if not ML_AVAILABLE or not self._models_trained():
            return self._fallback_prediction(workload_metrics)
        
        try:
            # Prepare features
            features = self._prepare_features(workload_metrics, historical_data)
            features_scaled = self.scaler.transform([features])
            
            # Make predictions
            cpu_pred = self.cpu_predictor.predict(features_scaled)[0]
            memory_pred = self.memory_predictor.predict(features_scaled)[0]
            instances_pred = int(self.instance_predictor.predict(features_scaled)[0])
            
            # Calculate confidence based on feature quality
            confidence = self._calculate_prediction_confidence(features)
            
            # Determine scaling trigger
            trigger = self._determine_scaling_trigger(workload_metrics)
            
            # Generate recommendation
            recommendation = self._generate_scaling_recommendation(
                cpu_pred, memory_pred, instances_pred, trigger
            )
            
            return ResourcePrediction(
                predicted_cpu=max(0.1, cpu_pred),
                predicted_memory=max(0.5, memory_pred),
                predicted_storage=workload_metrics.get("storage_usage", 10.0) * 1.2,
                predicted_instances=max(1, instances_pred),
                confidence_score=confidence,
                scaling_trigger=trigger,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Resource prediction failed: {e}")
            return self._fallback_prediction(workload_metrics)
    
    def _prepare_features(
        self,
        workload_metrics: Dict[str, float],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[float]:
        """Prepare features for ML models."""
        # Current workload features
        current_cpu = workload_metrics.get("cpu_usage", 0.5)
        current_memory = workload_metrics.get("memory_usage", 1.0)
        current_throughput = workload_metrics.get("throughput", 100.0)
        current_latency = workload_metrics.get("latency", 50.0)
        task_queue_size = workload_metrics.get("queue_size", 0)
        
        # Historical trend features
        if historical_data and len(historical_data) > 0:
            cpu_trend = self._calculate_trend([d.get("cpu_usage", 0) for d in historical_data[-10:]])
            memory_trend = self._calculate_trend([d.get("memory_usage", 0) for d in historical_data[-10:]])
            throughput_trend = self._calculate_trend([d.get("throughput", 0) for d in historical_data[-10:]])
        else:
            cpu_trend = memory_trend = throughput_trend = 0.0
        
        # Time-based features
        hour_of_day = time.localtime().tm_hour / 24.0
        day_of_week = time.localtime().tm_wday / 7.0
        
        return [
            current_cpu, current_memory, current_throughput, current_latency,
            task_queue_size, cpu_trend, memory_trend, throughput_trend,
            hour_of_day, day_of_week
        ]
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction from historical values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        y = values
        
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def _models_trained(self) -> bool:
        """Check if ML models are trained."""
        return (
            self.cpu_predictor is not None and 
            hasattr(self.cpu_predictor, 'feature_importances_')
        )
    
    def _calculate_prediction_confidence(self, features: List[float]) -> float:
        """Calculate confidence score for predictions."""
        # Simple confidence calculation based on feature quality
        base_confidence = 0.8
        
        # Adjust based on training data size
        data_factor = min(1.0, len(self.training_data) / 100.0)
        
        # Adjust based on feature completeness
        feature_completeness = sum(1 for f in features if f > 0) / len(features)
        
        return base_confidence * data_factor * feature_completeness
    
    def _determine_scaling_trigger(self, metrics: Dict[str, float]) -> str:
        """Determine what triggered the scaling need."""
        cpu_usage = metrics.get("cpu_usage", 0)
        memory_usage = metrics.get("memory_usage", 0)
        queue_size = metrics.get("queue_size", 0)
        latency = metrics.get("latency", 0)
        
        if cpu_usage > 0.8:
            return "high_cpu_usage"
        elif memory_usage > 0.8:
            return "high_memory_usage"
        elif queue_size > 100:
            return "high_queue_size"
        elif latency > 1000:
            return "high_latency"
        else:
            return "proactive_scaling"
    
    def _generate_scaling_recommendation(
        self,
        cpu: float,
        memory: float,
        instances: int,
        trigger: str
    ) -> str:
        """Generate human-readable scaling recommendation."""
        recommendations = []
        
        if cpu > 2.0:
            recommendations.append(f"Scale CPU to {cpu:.1f} cores")
        if memory > 4.0:
            recommendations.append(f"Scale memory to {memory:.1f} GB")
        if instances > 1:
            recommendations.append(f"Scale to {instances} instances")
        
        base_msg = f"Triggered by {trigger.replace('_', ' ')}"
        
        if recommendations:
            return f"{base_msg}. Recommend: {', '.join(recommendations)}"
        else:
            return f"{base_msg}. Current resources sufficient"
    
    def _fallback_prediction(self, metrics: Dict[str, float]) -> ResourcePrediction:
        """Fallback prediction when ML is not available."""
        cpu_usage = metrics.get("cpu_usage", 0.5)
        memory_usage = metrics.get("memory_usage", 1.0)
        queue_size = metrics.get("queue_size", 0)
        
        # Simple heuristic-based scaling
        cpu_pred = cpu_usage * 1.5 if cpu_usage > 0.7 else cpu_usage
        memory_pred = memory_usage * 1.5 if memory_usage > 0.7 else memory_usage
        instances_pred = 2 if queue_size > 50 else 1
        
        return ResourcePrediction(
            predicted_cpu=cpu_pred,
            predicted_memory=memory_pred,
            predicted_storage=10.0,
            predicted_instances=instances_pred,
            confidence_score=0.6,
            scaling_trigger="heuristic_based",
            recommendation="Basic scaling recommendation based on current usage"
        )
    
    def train_from_data(self, training_data: List[Dict[str, Any]]):
        """Train models from historical scaling data."""
        if not ML_AVAILABLE or len(training_data) < 10:
            return
        
        self.training_data.extend(training_data)
        
        try:
            # Prepare training data
            X = []
            y_cpu = []
            y_memory = []
            y_instances = []
            
            for record in training_data:
                features = self._prepare_features(
                    record.get("workload_metrics", {}),
                    record.get("historical_data", [])
                )
                X.append(features)
                y_cpu.append(record.get("actual_cpu", 1.0))
                y_memory.append(record.get("actual_memory", 2.0))
                y_instances.append(record.get("actual_instances", 1))
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.cpu_predictor.fit(X_scaled, y_cpu)
            self.memory_predictor.fit(X_scaled, y_memory)
            self.instance_predictor.fit(X_scaled, y_instances)
            
            logger.info(f"Trained resource scaling models with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")


class AnomalyDetector:
    """
    Detects performance anomalies using machine learning techniques
    including isolation forests and statistical analysis.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize anomaly detector."""
        self.model_dir = Path(model_dir) if model_dir else Path("models/anomaly")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # ML models for anomaly detection
        self.isolation_forest = None
        self.statistical_detector = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        
        # Baseline metrics for comparison
        self.baseline_metrics: Dict[str, float] = {}
        self.metric_history: List[Dict[str, float]] = []
        
        if ML_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for anomaly detection."""
        if not ML_AVAILABLE:
            return
        
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
    
    def detect_anomalies(
        self,
        current_metrics: Dict[str, float],
        context: Optional[Dict[str, Any]] = None
    ) -> List[AnomalyAlert]:
        """Detect anomalies in current performance metrics."""
        alerts = []
        
        # Store current metrics
        self.metric_history.append(current_metrics)
        
        # Keep only recent history
        if len(self.metric_history) > 1000:
            self.metric_history = self.metric_history[-1000:]
        
        # Update baseline if we have enough data
        if len(self.metric_history) >= 50:
            self._update_baseline()
        
        # ML-based anomaly detection
        if ML_AVAILABLE and self._detector_trained():
            ml_anomalies = self._detect_ml_anomalies(current_metrics)
            alerts.extend(ml_anomalies)
        
        # Statistical anomaly detection
        stat_anomalies = self._detect_statistical_anomalies(current_metrics)
        alerts.extend(stat_anomalies)
        
        # Rule-based anomaly detection
        rule_anomalies = self._detect_rule_based_anomalies(current_metrics)
        alerts.extend(rule_anomalies)
        
        return alerts
    
    def _update_baseline(self):
        """Update baseline metrics from recent history."""
        if not self.metric_history:
            return
        
        metrics_keys = set()
        for metrics in self.metric_history:
            metrics_keys.update(metrics.keys())
        
        for key in metrics_keys:
            values = [m.get(key, 0) for m in self.metric_history[-100:]]  # Last 100 samples
            if values:
                self.baseline_metrics[key] = sum(values) / len(values)
    
    def _detector_trained(self) -> bool:
        """Check if ML detector is trained."""
        return (
            self.isolation_forest is not None and 
            hasattr(self.isolation_forest, 'decision_function')
        )
    
    def _detect_ml_anomalies(
        self,
        current_metrics: Dict[str, float]
    ) -> List[AnomalyAlert]:
        """Detect anomalies using ML models."""
        if not ML_AVAILABLE or len(self.metric_history) < 50:
            return []
        
        alerts = []
        
        try:
            # Prepare data for ML detection
            metric_keys = list(current_metrics.keys())
            X = []
            
            # Historical data
            for metrics in self.metric_history[-100:]:
                features = [metrics.get(key, 0) for key in metric_keys]
                X.append(features)
            
            # Current metrics
            current_features = [current_metrics.get(key, 0) for key in metric_keys]
            
            if len(X) >= 20:  # Need minimum data for training
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                current_scaled = self.scaler.transform([current_features])
                
                # Train and predict with isolation forest
                self.isolation_forest.fit(X_scaled)
                anomaly_score = self.isolation_forest.decision_function(current_scaled)[0]
                is_anomaly = self.isolation_forest.predict(current_scaled)[0] == -1
                
                if is_anomaly:
                    # Identify which metrics are anomalous
                    anomalous_metrics = self._identify_anomalous_metrics(
                        current_metrics, metric_keys
                    )
                    
                    alert = AnomalyAlert(
                        anomaly_id=f"ml_anomaly_{int(time.time())}",
                        timestamp=time.time(),
                        severity=self._determine_severity(abs(anomaly_score)),
                        anomaly_type="performance",
                        description=f"ML-detected performance anomaly in metrics: {', '.join(anomalous_metrics)}",
                        affected_components=anomalous_metrics,
                        suggested_actions=[
                            "Check system resources",
                            "Review recent changes",
                            "Monitor for continued anomalous behavior"
                        ],
                        confidence_score=min(1.0, abs(anomaly_score))
                    )
                    alerts.append(alert)
                    
        except Exception as e:
            logger.error(f"ML anomaly detection failed: {e}")
        
        return alerts
    
    def _identify_anomalous_metrics(
        self,
        current_metrics: Dict[str, float],
        metric_keys: List[str]
    ) -> List[str]:
        """Identify which specific metrics are anomalous."""
        anomalous = []
        
        for key in metric_keys:
            current_value = current_metrics.get(key, 0)
            baseline_value = self.baseline_metrics.get(key, 0)
            
            if baseline_value > 0:
                deviation = abs(current_value - baseline_value) / baseline_value
                if deviation > 0.5:  # 50% deviation threshold
                    anomalous.append(key)
        
        return anomalous
    
    def _detect_statistical_anomalies(
        self,
        current_metrics: Dict[str, float]
    ) -> List[AnomalyAlert]:
        """Detect anomalies using statistical methods."""
        alerts = []
        
        if len(self.metric_history) < 20:
            return alerts
        
        for metric_name, current_value in current_metrics.items():
            # Get historical values for this metric
            historical_values = [
                m.get(metric_name, 0) for m in self.metric_history[-50:]
            ]
            
            if len(historical_values) < 10:
                continue
            
            # Calculate statistical measures
            mean_val = sum(historical_values) / len(historical_values)
            variance = sum((x - mean_val) ** 2 for x in historical_values) / len(historical_values)
            std_dev = variance ** 0.5
            
            if std_dev > 0:
                # Z-score based detection
                z_score = abs(current_value - mean_val) / std_dev
                
                if z_score > 3.0:  # 3-sigma rule
                    alert = AnomalyAlert(
                        anomaly_id=f"stat_anomaly_{metric_name}_{int(time.time())}",
                        timestamp=time.time(),
                        severity="high" if z_score > 4.0 else "medium",
                        anomaly_type="statistical",
                        description=f"Statistical anomaly in {metric_name}: {current_value:.2f} (z-score: {z_score:.2f})",
                        affected_components=[metric_name],
                        suggested_actions=[
                            f"Investigate {metric_name} spike",
                            "Check related system components",
                            "Review system logs"
                        ],
                        confidence_score=min(1.0, z_score / 5.0)
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _detect_rule_based_anomalies(
        self,
        current_metrics: Dict[str, float]
    ) -> List[AnomalyAlert]:
        """Detect anomalies using predefined rules."""
        alerts = []
        
        # CPU anomalies
        cpu_usage = current_metrics.get("cpu_usage", 0)
        if cpu_usage > 0.95:
            alerts.append(AnomalyAlert(
                anomaly_id=f"cpu_spike_{int(time.time())}",
                timestamp=time.time(),
                severity="critical",
                anomaly_type="resource",
                description=f"Critical CPU usage: {cpu_usage*100:.1f}%",
                affected_components=["cpu"],
                suggested_actions=[
                    "Scale up CPU resources",
                    "Identify CPU-intensive processes",
                    "Consider load balancing"
                ],
                confidence_score=1.0
            ))
        
        # Memory anomalies
        memory_usage = current_metrics.get("memory_usage", 0)
        if memory_usage > 0.90:
            alerts.append(AnomalyAlert(
                anomaly_id=f"memory_spike_{int(time.time())}",
                timestamp=time.time(),
                severity="high",
                anomaly_type="resource",
                description=f"High memory usage: {memory_usage*100:.1f}%",
                affected_components=["memory"],
                suggested_actions=[
                    "Scale up memory resources",
                    "Check for memory leaks",
                    "Optimize memory usage"
                ],
                confidence_score=1.0
            ))
        
        # Error rate anomalies
        error_rate = current_metrics.get("error_rate", 0)
        if error_rate > 0.05:  # 5% error rate
            alerts.append(AnomalyAlert(
                anomaly_id=f"error_spike_{int(time.time())}",
                timestamp=time.time(),
                severity="high",
                anomaly_type="error_rate",
                description=f"High error rate: {error_rate*100:.1f}%",
                affected_components=["application"],
                suggested_actions=[
                    "Review application logs",
                    "Check for recent deployments",
                    "Validate input data"
                ],
                confidence_score=1.0
            ))
        
        # Latency anomalies
        latency = current_metrics.get("latency", 0)
        if latency > 5000:  # 5 second latency
            alerts.append(AnomalyAlert(
                anomaly_id=f"latency_spike_{int(time.time())}",
                timestamp=time.time(),
                severity="medium",
                anomaly_type="performance",
                description=f"High latency: {latency:.0f}ms",
                affected_components=["network", "application"],
                suggested_actions=[
                    "Check network connectivity",
                    "Optimize database queries",
                    "Review application performance"
                ],
                confidence_score=1.0
            ))
        
        return alerts
    
    def _determine_severity(self, anomaly_score: float) -> str:
        """Determine severity level based on anomaly score."""
        if anomaly_score > 0.8:
            return "critical"
        elif anomaly_score > 0.6:
            return "high"
        elif anomaly_score > 0.4:
            return "medium"
        else:
            return "low"


# Export all model classes
__all__ = [
    "WorkflowPattern",
    "ResourcePrediction", 
    "SchedulingDecision",
    "AnomalyAlert",
    "WorkflowPatternLearner",
    "ResourceScalingPredictor",
    "AnomalyDetector"
]