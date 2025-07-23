"""
Performance Prediction Models for CrewGraph AI

Advanced performance prediction using historical data analysis,
machine learning models, and statistical forecasting techniques.

Author: Vatsal216
Created: 2025-07-23 17:25:00 UTC
"""

import json
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    np = None
    RandomForestRegressor = LinearRegression = None
    StandardScaler = mean_absolute_error = r2_score = None
    SKLEARN_AVAILABLE = False

from ..types import WorkflowId, ExecutionId, PerformanceMetrics
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformancePrediction:
    """Performance prediction result with confidence intervals."""
    
    predicted_execution_time: float
    predicted_resource_usage: Dict[str, float]
    confidence_interval: Tuple[float, float]
    prediction_accuracy: float
    model_used: str
    factors_analysis: Dict[str, float]


@dataclass
class HistoricalExecution:
    """Historical execution data for training prediction models."""
    
    workflow_id: WorkflowId
    execution_id: ExecutionId
    execution_time: float
    resource_usage: Dict[str, float]
    task_count: int
    dependency_count: int
    input_size: Optional[float]
    timestamp: datetime
    success: bool
    error_type: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class PredictionFeatures:
    """Features used for performance prediction."""
    
    task_count: int
    dependency_count: int
    avg_task_complexity: float
    parallel_potential: float
    resource_intensity: float
    input_data_size: float
    historical_avg_time: float
    time_of_day: float
    day_of_week: int
    system_load: float


@dataclass
class BottleneckPrediction:
    """Prediction of potential performance bottlenecks."""
    
    bottleneck_type: str
    location: str
    severity: float
    impact_on_performance: float
    recommended_mitigation: List[str]
    confidence: float


class PerformancePredictor:
    """
    Advanced performance prediction system using ML and statistical methods.
    
    Predicts workflow execution time, resource usage, and potential bottlenecks
    based on historical data and workflow characteristics.
    """
    
    def __init__(self, history_limit: int = 1000):
        """Initialize performance predictor."""
        self.history_limit = history_limit
        self.execution_history: deque = deque(maxlen=history_limit)
        
        # ML models
        self.time_predictor = None
        self.resource_predictor = None
        self.bottleneck_predictor = None
        self.feature_scaler = None
        
        # Statistical models
        self.workflow_baselines: Dict[WorkflowId, Dict[str, float]] = defaultdict(dict)
        self.temporal_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Performance tracking
        self.prediction_accuracy_history: List[float] = []
        self.model_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        if SKLEARN_AVAILABLE:
            self._initialize_ml_models()
        
        logger.info("Performance predictor initialized")
    
    def _initialize_ml_models(self):
        """Initialize machine learning models."""
        if not SKLEARN_AVAILABLE:
            return
        
        # Execution time prediction model
        self.time_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Resource usage prediction model
        self.resource_predictor = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        
        # Feature scaler
        self.feature_scaler = StandardScaler()
        
        logger.info("ML models initialized for performance prediction")
    
    def predict_performance(
        self,
        workflow_definition: Dict[str, Any],
        input_context: Optional[Dict[str, Any]] = None
    ) -> PerformancePrediction:
        """
        Predict workflow performance using multiple approaches.
        
        Args:
            workflow_definition: Workflow structure and configuration
            input_context: Additional context for prediction
            
        Returns:
            Performance prediction with confidence metrics
        """
        # Extract features for prediction
        features = self._extract_prediction_features(workflow_definition, input_context)
        
        # Use ML prediction if available, otherwise fall back to statistical
        if SKLEARN_AVAILABLE and self._models_trained():
            return self._ml_predict_performance(features, workflow_definition)
        else:
            return self._statistical_predict_performance(features, workflow_definition)
    
    def _extract_prediction_features(
        self,
        workflow_definition: Dict[str, Any],
        input_context: Optional[Dict[str, Any]] = None
    ) -> PredictionFeatures:
        """Extract features from workflow for prediction."""
        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])
        
        # Basic workflow characteristics
        task_count = len(tasks)
        dependency_count = len(dependencies)
        
        # Task complexity analysis
        complexity_scores = []
        for task in tasks:
            base_complexity = 1.0
            
            # Adjust for task type
            task_type = task.get("type", "standard")
            type_multipliers = {
                "data_processing": 2.0,
                "ml": 3.0,
                "compute_intensive": 2.5,
                "io_bound": 1.5,
                "standard": 1.0
            }
            base_complexity *= type_multipliers.get(task_type, 1.0)
            
            # Adjust for estimated duration
            duration = task.get("estimated_duration", 10)
            base_complexity *= (1 + duration / 30.0)
            
            complexity_scores.append(base_complexity)
        
        avg_task_complexity = statistics.mean(complexity_scores) if complexity_scores else 1.0
        
        # Parallel execution potential
        parallel_potential = self._calculate_parallel_potential(tasks, dependencies)
        
        # Resource intensity
        resource_intensity = self._calculate_resource_intensity(tasks)
        
        # Input data size (if available)
        input_data_size = 0.0
        if input_context:
            # Estimate data size from context
            input_data_size = self._estimate_input_size(input_context)
        
        # Historical performance for similar workflows
        historical_avg_time = self._get_historical_average(workflow_definition)
        
        # Temporal features
        now = datetime.now()
        time_of_day = now.hour + now.minute / 60.0
        day_of_week = now.weekday()
        
        # System load (simplified)
        system_load = self._estimate_current_system_load()
        
        return PredictionFeatures(
            task_count=task_count,
            dependency_count=dependency_count,
            avg_task_complexity=avg_task_complexity,
            parallel_potential=parallel_potential,
            resource_intensity=resource_intensity,
            input_data_size=input_data_size,
            historical_avg_time=historical_avg_time,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            system_load=system_load
        )
    
    def _ml_predict_performance(
        self,
        features: PredictionFeatures,
        workflow_definition: Dict[str, Any]
    ) -> PerformancePrediction:
        """ML-based performance prediction."""
        try:
            # Prepare feature vector
            feature_vector = self._features_to_vector(features)
            feature_vector_scaled = self.feature_scaler.transform([feature_vector])
            
            # Predict execution time
            predicted_time = self.time_predictor.predict(feature_vector_scaled)[0]
            predicted_time = max(0.1, predicted_time)  # Ensure positive
            
            # Predict resource usage
            predicted_resources = {"cpu": 0.5, "memory": 0.5}
            if self.resource_predictor:
                resource_pred = self.resource_predictor.predict(feature_vector_scaled)[0]
                predicted_resources = {
                    "cpu": max(0.1, min(1.0, resource_pred)),
                    "memory": max(0.1, min(1.0, resource_pred * 0.8))
                }
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(predicted_time)
            
            # Get feature importance for analysis
            factors_analysis = {}
            if hasattr(self.time_predictor, 'feature_importances_'):
                feature_names = [
                    "task_count", "dependency_count", "avg_task_complexity",
                    "parallel_potential", "resource_intensity", "input_data_size",
                    "historical_avg_time", "time_of_day", "day_of_week", "system_load"
                ]
                factors_analysis = dict(zip(
                    feature_names,
                    self.time_predictor.feature_importances_
                ))
            
            # Estimate prediction accuracy
            accuracy = self._estimate_prediction_accuracy()
            
            return PerformancePrediction(
                predicted_execution_time=predicted_time,
                predicted_resource_usage=predicted_resources,
                confidence_interval=confidence_interval,
                prediction_accuracy=accuracy,
                model_used="RandomForestRegressor",
                factors_analysis=factors_analysis
            )
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._statistical_predict_performance(features, workflow_definition)
    
    def _statistical_predict_performance(
        self,
        features: PredictionFeatures,
        workflow_definition: Dict[str, Any]
    ) -> PerformancePrediction:
        """Statistical/heuristic performance prediction."""
        # Base prediction from task analysis
        base_time = features.avg_task_complexity * features.task_count * 5.0  # 5 seconds per complexity unit
        
        # Adjust for parallelization
        parallel_factor = 1.0 - (features.parallel_potential * 0.4)
        base_time *= parallel_factor
        
        # Adjust for resource intensity
        resource_factor = 1.0 + (features.resource_intensity * 0.3)
        base_time *= resource_factor
        
        # Adjust for input data size
        data_factor = 1.0 + (features.input_data_size / 1000.0) * 0.2
        base_time *= data_factor
        
        # Use historical average if available and reliable
        if features.historical_avg_time > 0:
            # Weighted average with historical data
            historical_weight = 0.7 if len(self.execution_history) > 10 else 0.3
            base_time = (base_time * (1 - historical_weight) + 
                        features.historical_avg_time * historical_weight)
        
        # Adjust for temporal factors
        temporal_factor = self._get_temporal_adjustment(features.time_of_day, features.day_of_week)
        base_time *= temporal_factor
        
        # Adjust for system load
        system_load_factor = 1.0 + features.system_load * 0.2
        base_time *= system_load_factor
        
        # Resource usage prediction (simplified)
        predicted_resources = {
            "cpu": min(1.0, 0.3 + features.resource_intensity * 0.6),
            "memory": min(1.0, 0.2 + features.resource_intensity * 0.5)
        }
        
        # Confidence interval (wider for statistical prediction)
        confidence_interval = (base_time * 0.7, base_time * 1.5)
        
        # Factors analysis
        factors_analysis = {
            "task_complexity": 0.3,
            "parallelization": 0.25,
            "resource_intensity": 0.2,
            "historical_data": 0.15,
            "temporal_factors": 0.1
        }
        
        return PerformancePrediction(
            predicted_execution_time=base_time,
            predicted_resource_usage=predicted_resources,
            confidence_interval=confidence_interval,
            prediction_accuracy=0.75,  # Lower accuracy for statistical method
            model_used="statistical_heuristic",
            factors_analysis=factors_analysis
        )
    
    def predict_bottlenecks(
        self,
        workflow_definition: Dict[str, Any],
        predicted_performance: PerformancePrediction
    ) -> List[BottleneckPrediction]:
        """Predict potential performance bottlenecks."""
        bottlenecks = []
        
        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])
        
        # Resource bottleneck analysis
        cpu_usage = predicted_performance.predicted_resource_usage.get("cpu", 0.5)
        memory_usage = predicted_performance.predicted_resource_usage.get("memory", 0.5)
        
        if cpu_usage > 0.8:
            bottlenecks.append(BottleneckPrediction(
                bottleneck_type="cpu_contention",
                location="system_resources",
                severity=cpu_usage,
                impact_on_performance=cpu_usage * 30.0,  # % impact
                recommended_mitigation=[
                    "Optimize CPU-intensive tasks",
                    "Implement task parallelization",
                    "Scale up CPU resources"
                ],
                confidence=0.8
            ))
        
        if memory_usage > 0.85:
            bottlenecks.append(BottleneckPrediction(
                bottleneck_type="memory_pressure",
                location="system_resources",
                severity=memory_usage,
                impact_on_performance=memory_usage * 25.0,
                recommended_mitigation=[
                    "Optimize memory usage patterns",
                    "Implement data streaming",
                    "Scale up memory resources"
                ],
                confidence=0.75
            ))
        
        # Task dependency bottlenecks
        dependency_chains = self._analyze_dependency_chains(tasks, dependencies)
        longest_chain = max(dependency_chains, key=len) if dependency_chains else []
        
        if len(longest_chain) > 5:
            bottlenecks.append(BottleneckPrediction(
                bottleneck_type="sequential_dependency",
                location=f"task_chain_{longest_chain[0]}",
                severity=len(longest_chain) / 10.0,
                impact_on_performance=len(longest_chain) * 5.0,
                recommended_mitigation=[
                    "Break long sequential chains",
                    "Identify parallelizable tasks",
                    "Optimize critical path tasks"
                ],
                confidence=0.9
            ))
        
        # I/O bottleneck prediction
        io_intensive_tasks = [
            task for task in tasks 
            if task.get("type") in ["data_processing", "io_bound"]
        ]
        
        if len(io_intensive_tasks) > 3:
            bottlenecks.append(BottleneckPrediction(
                bottleneck_type="io_contention",
                location="data_access",
                severity=len(io_intensive_tasks) / 10.0,
                impact_on_performance=len(io_intensive_tasks) * 8.0,
                recommended_mitigation=[
                    "Implement I/O batching",
                    "Use asynchronous operations",
                    "Optimize data access patterns"
                ],
                confidence=0.7
            ))
        
        # Sort by severity
        bottlenecks.sort(key=lambda x: x.severity * x.confidence, reverse=True)
        
        return bottlenecks
    
    def record_execution(self, execution_data: HistoricalExecution):
        """Record actual execution data for model training."""
        self.execution_history.append(execution_data)
        
        # Update workflow baselines
        workflow_id = execution_data.workflow_id
        self.workflow_baselines[workflow_id]["avg_time"] = self._calculate_rolling_average(
            workflow_id, "execution_time", execution_data.execution_time
        )
        
        # Update temporal patterns
        hour_key = f"hour_{execution_data.timestamp.hour}"
        self.temporal_patterns[hour_key].append(execution_data.execution_time)
        
        day_key = f"day_{execution_data.timestamp.weekday()}"
        self.temporal_patterns[day_key].append(execution_data.execution_time)
        
        # Retrain models if enough data
        if len(self.execution_history) >= 50 and len(self.execution_history) % 20 == 0:
            self._retrain_models()
        
        logger.debug(f"Recorded execution data for workflow {workflow_id}")
    
    def validate_prediction(
        self,
        prediction: PerformancePrediction,
        actual_execution: HistoricalExecution
    ) -> float:
        """Validate prediction accuracy against actual execution."""
        predicted_time = prediction.predicted_execution_time
        actual_time = actual_execution.execution_time
        
        # Calculate percentage error
        error_percentage = abs(predicted_time - actual_time) / actual_time * 100
        accuracy = max(0, 100 - error_percentage) / 100  # Convert to 0-1 scale
        
        # Store accuracy for tracking
        self.prediction_accuracy_history.append(accuracy)
        
        # Update model performance tracking
        model_used = prediction.model_used
        if model_used not in self.model_performance:
            self.model_performance[model_used] = {"accuracies": [], "errors": []}
        
        self.model_performance[model_used]["accuracies"].append(accuracy)
        self.model_performance[model_used]["errors"].append(error_percentage)
        
        logger.debug(f"Prediction validation: {accuracy:.2f} accuracy for {model_used}")
        return accuracy
    
    def _features_to_vector(self, features: PredictionFeatures) -> List[float]:
        """Convert features to numerical vector for ML models."""
        return [
            features.task_count,
            features.dependency_count,
            features.avg_task_complexity,
            features.parallel_potential,
            features.resource_intensity,
            features.input_data_size,
            features.historical_avg_time,
            features.time_of_day,
            features.day_of_week,
            features.system_load
        ]
    
    def _calculate_parallel_potential(self, tasks: List[Dict], dependencies: List[Dict]) -> float:
        """Calculate potential for parallel execution."""
        if not tasks:
            return 0.0
        
        # Build dependency graph
        dependent_tasks = {dep.get("target") for dep in dependencies}
        independent_tasks = len([t for t in tasks if t.get("id") not in dependent_tasks])
        
        return min(1.0, independent_tasks / len(tasks))
    
    def _calculate_resource_intensity(self, tasks: List[Dict]) -> float:
        """Calculate overall resource intensity."""
        if not tasks:
            return 0.0
        
        intensity_scores = []
        for task in tasks:
            task_type = task.get("type", "standard")
            duration = task.get("estimated_duration", 10)
            
            type_weights = {
                "data_processing": 0.8,
                "ml": 0.9,
                "compute_intensive": 0.95,
                "io_bound": 0.4,
                "standard": 0.5
            }
            
            base_score = type_weights.get(task_type, 0.5)
            duration_factor = min(1.0, duration / 60.0)
            intensity_scores.append(base_score * (1 + duration_factor))
        
        return sum(intensity_scores) / len(intensity_scores)
    
    def _estimate_input_size(self, input_context: Dict[str, Any]) -> float:
        """Estimate input data size from context."""
        # Simple heuristic based on context
        size_estimate = 0.0
        
        for key, value in input_context.items():
            if isinstance(value, str):
                size_estimate += len(value) / 1000.0  # KB
            elif isinstance(value, (list, dict)):
                size_estimate += len(str(value)) / 1000.0  # Rough estimate
            elif isinstance(value, (int, float)):
                size_estimate += 0.1  # Small size for numbers
        
        return size_estimate
    
    def _get_historical_average(self, workflow_definition: Dict[str, Any]) -> float:
        """Get historical average execution time for similar workflows."""
        # Simple hash-based similarity matching
        workflow_signature = self._get_workflow_signature(workflow_definition)
        
        similar_executions = [
            exec_data.execution_time for exec_data in self.execution_history
            if self._get_workflow_signature({"tasks": [], "dependencies": []}) == workflow_signature
        ]
        
        if similar_executions:
            return statistics.mean(similar_executions)
        
        return 0.0
    
    def _get_workflow_signature(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate signature for workflow similarity matching."""
        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])
        
        # Simple signature based on structure
        task_types = sorted([task.get("type", "standard") for task in tasks])
        return f"{len(tasks)}_{len(dependencies)}_{hash(tuple(task_types))}"
    
    def _estimate_current_system_load(self) -> float:
        """Estimate current system load (simplified)."""
        # In a real implementation, this would check actual system metrics
        # For now, return a random value that simulates varying load
        import random
        return random.uniform(0.2, 0.8)
    
    def _get_temporal_adjustment(self, time_of_day: float, day_of_week: int) -> float:
        """Get temporal adjustment factor for predictions."""
        # Peak hours adjustment (9 AM - 5 PM)
        if 9 <= time_of_day <= 17:
            time_factor = 1.1  # Slightly slower during peak hours
        else:
            time_factor = 0.95  # Faster during off-peak
        
        # Weekday vs weekend adjustment
        if day_of_week < 5:  # Weekdays
            day_factor = 1.0
        else:  # Weekends
            day_factor = 0.9  # Faster on weekends
        
        return time_factor * day_factor
    
    def _calculate_confidence_interval(self, predicted_time: float) -> Tuple[float, float]:
        """Calculate confidence interval for prediction."""
        # Use historical accuracy to determine interval width
        if self.prediction_accuracy_history:
            avg_accuracy = statistics.mean(self.prediction_accuracy_history[-20:])
            interval_width = (1.0 - avg_accuracy) * predicted_time
        else:
            interval_width = 0.3 * predicted_time  # Default 30% interval
        
        return (
            max(0.1, predicted_time - interval_width),
            predicted_time + interval_width
        )
    
    def _estimate_prediction_accuracy(self) -> float:
        """Estimate current prediction accuracy."""
        if self.prediction_accuracy_history:
            # Use recent accuracy history
            return statistics.mean(self.prediction_accuracy_history[-10:])
        else:
            # Default accuracy estimate
            return 0.8
    
    def _models_trained(self) -> bool:
        """Check if ML models are trained."""
        return (self.time_predictor is not None and 
                hasattr(self.time_predictor, 'feature_importances_'))
    
    def _calculate_rolling_average(self, workflow_id: WorkflowId, metric: str, new_value: float) -> float:
        """Calculate rolling average for workflow baseline."""
        # Get recent executions for this workflow
        recent_values = [
            getattr(exec_data, metric) for exec_data in self.execution_history
            if exec_data.workflow_id == workflow_id
        ][-10:]  # Last 10 executions
        
        recent_values.append(new_value)
        return statistics.mean(recent_values)
    
    def _analyze_dependency_chains(self, tasks: List[Dict], dependencies: List[Dict]) -> List[List[str]]:
        """Analyze dependency chains in workflow."""
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for task in tasks:
            task_id = task.get("id")
            if task_id:
                in_degree[task_id] = 0
        
        for dep in dependencies:
            source = dep.get("source")
            target = dep.get("target")
            if source and target:
                graph[source].append(target)
                in_degree[target] += 1
        
        # Find all paths (simplified)
        chains = []
        for task_id in in_degree:
            if in_degree[task_id] == 0:  # Starting nodes
                chain = self._dfs_chain(task_id, graph, set())
                if len(chain) > 1:
                    chains.append(chain)
        
        return chains
    
    def _dfs_chain(self, node: str, graph: Dict[str, List[str]], visited: set) -> List[str]:
        """DFS to find longest chain from node."""
        if node in visited:
            return []
        
        visited.add(node)
        
        if not graph[node]:  # Leaf node
            visited.remove(node)
            return [node]
        
        longest_chain = [node]
        for neighbor in graph[node]:
            chain = self._dfs_chain(neighbor, graph, visited)
            if len(chain) + 1 > len(longest_chain):
                longest_chain = [node] + chain
        
        visited.remove(node)
        return longest_chain
    
    def _retrain_models(self):
        """Retrain ML models with accumulated data."""
        if not SKLEARN_AVAILABLE or len(self.execution_history) < 20:
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for exec_data in self.execution_history:
                # Create features from execution data (simplified)
                features = PredictionFeatures(
                    task_count=exec_data.task_count,
                    dependency_count=exec_data.dependency_count,
                    avg_task_complexity=1.0,  # Simplified
                    parallel_potential=0.5,  # Simplified
                    resource_intensity=exec_data.resource_usage.get("cpu", 0.5),
                    input_data_size=exec_data.input_size or 0.0,
                    historical_avg_time=exec_data.execution_time,  # Self-reference
                    time_of_day=exec_data.timestamp.hour,
                    day_of_week=exec_data.timestamp.weekday(),
                    system_load=0.5  # Simplified
                )
                
                X.append(self._features_to_vector(features))
                y.append(exec_data.execution_time)
            
            if len(X) >= 20:
                X = np.array(X)
                y = np.array(y)
                
                # Fit scaler and transform features
                self.feature_scaler.fit(X)
                X_scaled = self.feature_scaler.transform(X)
                
                # Train models
                self.time_predictor.fit(X_scaled, y)
                
                # Calculate training accuracy
                y_pred = self.time_predictor.predict(X_scaled)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                
                logger.info(f"Models retrained - R²: {r2:.3f}, MAE: {mae:.2f}")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance predictor statistics."""
        stats = {
            "execution_history_size": len(self.execution_history),
            "models_available": SKLEARN_AVAILABLE,
            "models_trained": self._models_trained(),
            "prediction_accuracy": {},
            "temporal_patterns_tracked": len(self.temporal_patterns),
            "workflow_baselines": len(self.workflow_baselines)
        }
        
        # Calculate accuracy statistics
        if self.prediction_accuracy_history:
            stats["prediction_accuracy"] = {
                "average": statistics.mean(self.prediction_accuracy_history),
                "recent_average": statistics.mean(self.prediction_accuracy_history[-10:]),
                "total_predictions": len(self.prediction_accuracy_history)
            }
        
        # Model performance by type
        for model_name, performance in self.model_performance.items():
            if performance["accuracies"]:
                stats[f"{model_name}_performance"] = {
                    "average_accuracy": statistics.mean(performance["accuracies"]),
                    "average_error": statistics.mean(performance["errors"]),
                    "prediction_count": len(performance["accuracies"])
                }
        
        return stats