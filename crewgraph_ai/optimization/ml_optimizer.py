"""
ML-Based Performance Optimization Engine for CrewGraph AI

Provides machine learning-driven optimization capabilities including
performance prediction, intelligent task scheduling, and resource optimization.

Author: Vatsal216
Created: 2025-07-23 17:15:00 UTC
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    # Graceful fallback when ML libraries not available
    np = None
    RandomForestRegressor = GradientBoostingRegressor = LinearRegression = None
    StandardScaler = train_test_split = None
    mean_absolute_error = mean_squared_error = None
    ML_AVAILABLE = False

from ..types import WorkflowId, ExecutionId
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MLPrediction:
    """ML prediction result with confidence metrics."""
    
    predicted_value: float
    confidence_score: float
    model_used: str
    feature_importance: Dict[str, float]
    prediction_interval: Tuple[float, float]


@dataclass
class PerformanceFeatures:
    """Features extracted from workflow for ML prediction."""
    
    task_count: int
    avg_task_duration: float
    max_task_duration: float
    dependency_count: int
    parallel_potential: float
    resource_intensity: float
    complexity_score: float
    historical_performance: float


@dataclass
class OptimizationRecommendation:
    """ML-driven optimization recommendation."""
    
    optimization_type: str
    expected_improvement: float
    confidence: float
    implementation_effort: str
    cost_benefit_ratio: float
    priority_score: float
    specific_actions: List[str]


class MLOptimizer:
    """
    Machine Learning-based workflow optimization engine.
    
    Uses ML models to predict performance, recommend optimizations,
    and learn from execution history to improve future predictions.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize ML optimizer with optional model directory."""
        self.model_dir = Path(model_dir) if model_dir else Path.cwd() / "models"
        self.model_dir.mkdir(exist_ok=True)
        
        # ML models
        self.performance_model = None
        self.resource_model = None
        self.cost_model = None
        self.scaler = None
        
        # Training data storage
        self.training_data: List[Dict[str, Any]] = []
        self.feature_columns = [
            "task_count", "avg_task_duration", "max_task_duration",
            "dependency_count", "parallel_potential", "resource_intensity",
            "complexity_score", "historical_performance"
        ]
        
        # Performance tracking
        self.prediction_history: List[Dict[str, Any]] = []
        
        if ML_AVAILABLE:
            self._initialize_models()
        else:
            logger.warning("ML libraries not available. Using fallback implementations.")
    
    def _initialize_models(self):
        """Initialize ML models with default configurations."""
        if not ML_AVAILABLE:
            return
            
        # Performance prediction model
        self.performance_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Resource usage prediction model
        self.resource_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        
        # Cost prediction model
        self.cost_model = LinearRegression()
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Try to load existing models
        self._load_models()
    
    def predict_performance(
        self, 
        workflow_features: PerformanceFeatures
    ) -> MLPrediction:
        """
        Predict workflow execution performance using ML.
        
        Args:
            workflow_features: Extracted workflow features
            
        Returns:
            ML prediction with confidence metrics
        """
        if not ML_AVAILABLE or self.performance_model is None:
            return self._fallback_performance_prediction(workflow_features)
        
        try:
            # Prepare features
            features = self._features_to_array(workflow_features)
            
            # Scale features if scaler is trained
            if hasattr(self.scaler, 'scale_'):
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = [features]
            
            # Make prediction
            prediction = self.performance_model.predict(features_scaled)[0]
            
            # Calculate confidence (based on model's prediction variance)
            confidence = self._calculate_prediction_confidence(features_scaled)
            
            # Get feature importance
            feature_importance = dict(zip(
                self.feature_columns,
                self.performance_model.feature_importances_
            ))
            
            # Calculate prediction interval (simple approach)
            std_error = 0.1 * prediction  # Simplified error estimation
            interval = (prediction - 1.96 * std_error, prediction + 1.96 * std_error)
            
            result = MLPrediction(
                predicted_value=max(0.1, prediction),  # Ensure positive prediction
                confidence_score=confidence,
                model_used="GradientBoostingRegressor",
                feature_importance=feature_importance,
                prediction_interval=interval
            )
            
            logger.info(f"ML performance prediction: {prediction:.2f}s (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._fallback_performance_prediction(workflow_features)
    
    def recommend_optimizations(
        self,
        workflow_definition: Dict[str, Any],
        current_performance: Optional[float] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationRecommendation]:
        """
        Generate ML-driven optimization recommendations.
        
        Args:
            workflow_definition: Current workflow structure
            current_performance: Current execution performance
            constraints: Optimization constraints
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Extract features
        features = self._extract_workflow_features(workflow_definition)
        
        # Predict baseline performance
        baseline_prediction = self.predict_performance(features)
        baseline_time = baseline_prediction.predicted_value
        
        # Generate optimization scenarios
        optimization_scenarios = self._generate_optimization_scenarios(features)
        
        for scenario in optimization_scenarios:
            # Predict performance after optimization
            optimized_prediction = self.predict_performance(scenario["features"])
            
            # Calculate improvement
            improvement = ((baseline_time - optimized_prediction.predicted_value) / baseline_time) * 100
            
            if improvement > 5.0:  # Only recommend if >5% improvement
                recommendation = OptimizationRecommendation(
                    optimization_type=scenario["type"],
                    expected_improvement=improvement,
                    confidence=optimized_prediction.confidence_score,
                    implementation_effort=scenario["effort"],
                    cost_benefit_ratio=improvement / scenario["cost_factor"],
                    priority_score=improvement * optimized_prediction.confidence_score,
                    specific_actions=scenario["actions"]
                )
                recommendations.append(recommendation)
        
        # Sort by priority score
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)
        
        logger.info(f"Generated {len(recommendations)} ML-driven optimization recommendations")
        return recommendations[:5]  # Return top 5 recommendations
    
    def learn_from_execution(
        self,
        workflow_id: WorkflowId,
        execution_id: ExecutionId,
        workflow_features: PerformanceFeatures,
        actual_performance: float,
        resource_usage: Dict[str, float],
        execution_cost: Optional[float] = None
    ):
        """
        Learn from actual execution results to improve future predictions.
        
        Args:
            workflow_id: Workflow identifier
            execution_id: Execution identifier
            workflow_features: Features used for prediction
            actual_performance: Actual execution time
            resource_usage: Actual resource consumption
            execution_cost: Actual execution cost
        """
        # Store training data
        training_record = {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "features": self._features_to_array(workflow_features),
            "actual_performance": actual_performance,
            "resource_usage": resource_usage,
            "execution_cost": execution_cost or 0.0,
            "timestamp": float(pd.Timestamp.now().timestamp()) if 'pd' in globals() else 0.0
        }
        
        self.training_data.append(training_record)
        
        # Retrain models if we have enough data
        if len(self.training_data) >= 20:
            self._retrain_models()
        
        logger.info(f"Learned from execution {execution_id} - actual performance: {actual_performance:.2f}s")
    
    def _extract_workflow_features(self, workflow_definition: Dict[str, Any]) -> PerformanceFeatures:
        """Extract ML features from workflow definition."""
        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])
        
        # Basic task metrics
        task_count = len(tasks)
        durations = [task.get("estimated_duration", 10) for task in tasks]
        avg_duration = sum(durations) / max(len(durations), 1)
        max_duration = max(durations) if durations else 10
        
        # Dependency analysis
        dependency_count = len(dependencies)
        
        # Parallel potential (simplified)
        parallel_potential = self._calculate_parallel_potential(tasks, dependencies)
        
        # Resource intensity
        resource_intensity = self._calculate_resource_intensity(tasks)
        
        # Complexity score
        complexity_score = self._calculate_complexity_score(workflow_definition)
        
        # Historical performance (simplified)
        historical_performance = avg_duration  # Placeholder
        
        return PerformanceFeatures(
            task_count=task_count,
            avg_task_duration=avg_duration,
            max_task_duration=max_duration,
            dependency_count=dependency_count,
            parallel_potential=parallel_potential,
            resource_intensity=resource_intensity,
            complexity_score=complexity_score,
            historical_performance=historical_performance
        )
    
    def _features_to_array(self, features: PerformanceFeatures) -> List[float]:
        """Convert feature object to array for ML models."""
        return [
            features.task_count,
            features.avg_task_duration,
            features.max_task_duration,
            features.dependency_count,
            features.parallel_potential,
            features.resource_intensity,
            features.complexity_score,
            features.historical_performance
        ]
    
    def _calculate_parallel_potential(self, tasks: List[Dict], dependencies: List[Dict]) -> float:
        """Calculate potential for parallel execution."""
        if not tasks:
            return 0.0
        
        # Simple heuristic: ratio of independent tasks
        dependent_tasks = {dep.get("target") for dep in dependencies}
        independent_tasks = len([t for t in tasks if t.get("id") not in dependent_tasks])
        
        return min(1.0, independent_tasks / len(tasks))
    
    def _calculate_resource_intensity(self, tasks: List[Dict]) -> float:
        """Calculate overall resource intensity of workflow."""
        if not tasks:
            return 0.0
        
        intensity_scores = []
        for task in tasks:
            # Score based on task type and estimated duration
            task_type = task.get("type", "standard")
            duration = task.get("estimated_duration", 10)
            
            type_weights = {
                "data_processing": 0.8,
                "ml": 0.9,
                "compute_intensive": 0.95,
                "io_bound": 0.3,
                "standard": 0.5
            }
            
            base_score = type_weights.get(task_type, 0.5)
            duration_factor = min(1.0, duration / 60.0)  # Normalize to hour
            
            intensity_scores.append(base_score * (1 + duration_factor))
        
        return sum(intensity_scores) / len(intensity_scores)
    
    def _calculate_complexity_score(self, workflow_definition: Dict[str, Any]) -> float:
        """Calculate overall complexity score of workflow."""
        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])
        
        if not tasks:
            return 0.0
        
        # Factors: task count, dependency density, estimated durations
        task_factor = min(1.0, len(tasks) / 20.0)  # Normalize to 20 tasks
        
        dependency_density = len(dependencies) / max(len(tasks), 1)
        dependency_factor = min(1.0, dependency_density)
        
        duration_variance = self._calculate_duration_variance(tasks)
        
        return (task_factor + dependency_factor + duration_variance) / 3.0
    
    def _calculate_duration_variance(self, tasks: List[Dict]) -> float:
        """Calculate variance in task durations (normalized)."""
        if len(tasks) < 2:
            return 0.0
        
        durations = [task.get("estimated_duration", 10) for task in tasks]
        mean_duration = sum(durations) / len(durations)
        
        if mean_duration == 0:
            return 0.0
        
        variance = sum((d - mean_duration) ** 2 for d in durations) / len(durations)
        normalized_variance = min(1.0, variance / (mean_duration ** 2))
        
        return normalized_variance
    
    def _generate_optimization_scenarios(self, baseline_features: PerformanceFeatures) -> List[Dict]:
        """Generate optimization scenarios for comparison."""
        scenarios = []
        
        # Parallelization scenario
        if baseline_features.parallel_potential < 0.8:
            parallel_features = PerformanceFeatures(
                task_count=baseline_features.task_count,
                avg_task_duration=baseline_features.avg_task_duration * 0.7,
                max_task_duration=baseline_features.max_task_duration,
                dependency_count=baseline_features.dependency_count,
                parallel_potential=min(1.0, baseline_features.parallel_potential + 0.3),
                resource_intensity=baseline_features.resource_intensity,
                complexity_score=baseline_features.complexity_score * 1.1,
                historical_performance=baseline_features.historical_performance
            )
            
            scenarios.append({
                "type": "parallelization",
                "features": parallel_features,
                "effort": "medium",
                "cost_factor": 2.0,
                "actions": [
                    "Identify independent tasks for parallel execution",
                    "Implement concurrent task execution",
                    "Add synchronization points"
                ]
            })
        
        # Resource optimization scenario
        if baseline_features.resource_intensity > 0.6:
            resource_features = PerformanceFeatures(
                task_count=baseline_features.task_count,
                avg_task_duration=baseline_features.avg_task_duration * 0.85,
                max_task_duration=baseline_features.max_task_duration * 0.8,
                dependency_count=baseline_features.dependency_count,
                parallel_potential=baseline_features.parallel_potential,
                resource_intensity=baseline_features.resource_intensity * 0.7,
                complexity_score=baseline_features.complexity_score,
                historical_performance=baseline_features.historical_performance
            )
            
            scenarios.append({
                "type": "resource_optimization",
                "features": resource_features,
                "effort": "low",
                "cost_factor": 1.5,
                "actions": [
                    "Implement resource pooling",
                    "Add memory and CPU limits",
                    "Enable resource-aware scheduling"
                ]
            })
        
        # Caching scenario
        if baseline_features.task_count > 5:
            caching_features = PerformanceFeatures(
                task_count=baseline_features.task_count,
                avg_task_duration=baseline_features.avg_task_duration * 0.6,
                max_task_duration=baseline_features.max_task_duration,
                dependency_count=baseline_features.dependency_count,
                parallel_potential=baseline_features.parallel_potential,
                resource_intensity=baseline_features.resource_intensity * 0.9,
                complexity_score=baseline_features.complexity_score * 1.05,
                historical_performance=baseline_features.historical_performance
            )
            
            scenarios.append({
                "type": "caching",
                "features": caching_features,
                "effort": "low",
                "cost_factor": 1.2,
                "actions": [
                    "Implement result caching for repetitive tasks",
                    "Add cache invalidation logic",
                    "Configure cache storage backend"
                ]
            })
        
        return scenarios
    
    def _calculate_prediction_confidence(self, features_scaled) -> float:
        """Calculate confidence score for predictions."""
        # Simplified confidence calculation
        # In practice, could use prediction intervals, cross-validation, etc.
        base_confidence = 0.75
        
        # Adjust based on training data size
        data_confidence_factor = min(1.0, len(self.training_data) / 100.0)
        
        return base_confidence * (0.5 + 0.5 * data_confidence_factor)
    
    def _retrain_models(self):
        """Retrain ML models with accumulated training data."""
        if not ML_AVAILABLE or len(self.training_data) < 20:
            return
        
        try:
            # Prepare training data
            X = np.array([record["features"] for record in self.training_data])
            y_performance = np.array([record["actual_performance"] for record in self.training_data])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_performance, test_size=0.2, random_state=42
            )
            
            # Fit scaler
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train performance model
            self.performance_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.performance_model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            
            logger.info(f"Retrained performance model - MAE: {mae:.2f}")
            
            # Save models
            self._save_models()
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def _save_models(self):
        """Save trained models to disk."""
        if not ML_AVAILABLE:
            return
        
        try:
            model_files = {
                "performance_model.pkl": self.performance_model,
                "resource_model.pkl": self.resource_model,
                "cost_model.pkl": self.cost_model,
                "scaler.pkl": self.scaler
            }
            
            for filename, model in model_files.items():
                if model is not None:
                    with open(self.model_dir / filename, 'wb') as f:
                        pickle.dump(model, f)
            
            # Save training data
            with open(self.model_dir / "training_data.json", 'w') as f:
                json.dump(self.training_data, f, indent=2)
            
            logger.info(f"Models saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
    
    def _load_models(self):
        """Load pre-trained models from disk."""
        if not ML_AVAILABLE:
            return
        
        try:
            model_files = {
                "performance_model.pkl": "performance_model",
                "resource_model.pkl": "resource_model", 
                "cost_model.pkl": "cost_model",
                "scaler.pkl": "scaler"
            }
            
            for filename, attr_name in model_files.items():
                model_path = self.model_dir / filename
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        setattr(self, attr_name, pickle.load(f))
            
            # Load training data
            training_data_path = self.model_dir / "training_data.json"
            if training_data_path.exists():
                with open(training_data_path, 'r') as f:
                    self.training_data = json.load(f)
            
            logger.info(f"Models loaded from {self.model_dir}")
            
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")
    
    def _fallback_performance_prediction(self, features: PerformanceFeatures) -> MLPrediction:
        """Fallback prediction when ML libraries are not available."""
        # Simple heuristic-based prediction
        base_time = features.avg_task_duration * features.task_count
        
        # Adjust for parallelization potential
        parallel_factor = 1.0 - (features.parallel_potential * 0.3)
        
        # Adjust for resource intensity
        resource_factor = 1.0 + (features.resource_intensity * 0.2)
        
        # Adjust for complexity
        complexity_factor = 1.0 + (features.complexity_score * 0.15)
        
        predicted_time = base_time * parallel_factor * resource_factor * complexity_factor
        
        return MLPrediction(
            predicted_value=predicted_time,
            confidence_score=0.6,  # Lower confidence for heuristic
            model_used="heuristic_fallback",
            feature_importance={col: 1.0/len(self.feature_columns) for col in self.feature_columns},
            prediction_interval=(predicted_time * 0.8, predicted_time * 1.2)
        )
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about trained models."""
        stats = {
            "ml_available": ML_AVAILABLE,
            "training_samples": len(self.training_data),
            "models_trained": {},
            "model_directory": str(self.model_dir)
        }
        
        if ML_AVAILABLE:
            models = {
                "performance_model": self.performance_model,
                "resource_model": self.resource_model,
                "cost_model": self.cost_model
            }
            
            for name, model in models.items():
                if model is not None and hasattr(model, 'n_estimators'):
                    stats["models_trained"][name] = {
                        "type": type(model).__name__,
                        "n_estimators": getattr(model, 'n_estimators', 'N/A'),
                        "is_trained": hasattr(model, 'feature_importances_')
                    }
        
        return stats