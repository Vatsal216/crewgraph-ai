"""
Machine Learning Models for CrewGraph AI Intelligence

Lightweight ML models for performance prediction and optimization.

Author: Vatsal216
Created: 2025-07-23 06:16:00 UTC
"""

import pickle
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.exceptions import CrewGraphError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ModelType(Enum):
    """Types of ML models available"""

    PERFORMANCE_PREDICTOR = "performance_predictor"
    RESOURCE_PREDICTOR = "resource_predictor"
    BOTTLENECK_DETECTOR = "bottleneck_detector"
    OPTIMIZATION_RECOMMENDER = "optimization_recommender"


@dataclass
class ModelMetrics:
    """Model performance metrics"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    training_time: float
    prediction_time: float
    model_size_mb: float
    created_by: str = "Vatsal216"
    created_at: str = "2025-07-23 06:16:00"


@dataclass
class TrainingData:
    """Training data structure"""

    features: np.ndarray
    targets: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    metadata: Dict[str, Any]


class SimpleLinearRegressor:
    """
    Lightweight linear regression model for performance prediction.

    Uses simple numpy operations to avoid heavy ML dependencies.
    """

    def __init__(self):
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        # Normalize features
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        X_norm = (X - self.feature_means) / self.feature_stds

        # Add bias term
        X_with_bias = np.column_stack([np.ones(X_norm.shape[0]), X_norm])

        # Solve normal equation: w = (X^T X)^-1 X^T y
        try:
            self.weights = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
            self.bias = self.weights[0]
            self.weights = self.weights[1:]
            self.is_trained = True
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            self.weights = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            self.bias = self.weights[0]
            self.weights = self.weights[1:]
            self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_norm = (X - self.feature_means) / self.feature_stds
        return X_norm @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class SimpleClassifier:
    """
    Lightweight classification model for bottleneck detection.

    Uses logistic regression with gradient descent.
    """

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.is_trained = False

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classifier"""
        # Normalize features
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.feature_means) / self.feature_stds

        # Initialize weights
        self.weights = np.random.normal(0, 0.01, X_norm.shape[1])
        self.bias = 0.0

        # Gradient descent
        m = X_norm.shape[0]
        for i in range(self.max_iterations):
            z = X_norm @ self.weights + self.bias
            predictions = self._sigmoid(z)

            # Calculate gradients
            dw = (1 / m) * X_norm.T @ (predictions - y)
            db = (1 / m) * np.sum(predictions - y)

            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_norm = (X - self.feature_means) / self.feature_stds
        z = X_norm @ self.weights + self.bias
        probabilities = self._sigmoid(z)
        return (probabilities > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_norm = (X - self.feature_means) / self.feature_stds
        z = X_norm @ self.weights + self.bias
        return self._sigmoid(z)


class MLModelManager:
    """
    Manages machine learning models for workflow optimization.

    Provides lightweight models without requiring heavy ML frameworks.

    Created by: Vatsal216
    Date: 2025-07-23 06:16:00 UTC
    """

    def __init__(self, model_dir: str = "models"):
        """
        Initialize ML model manager.

        Args:
            model_dir: Directory to store trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.models: Dict[ModelType, Any] = {}
        self.model_metrics: Dict[ModelType, ModelMetrics] = {}
        self._lock = threading.RLock()

        logger.info(f"MLModelManager initialized")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"User: Vatsal216, Time: 2025-07-23 06:16:00")

    def create_model(self, model_type: ModelType, **kwargs) -> Any:
        """Create a new model instance"""
        with self._lock:
            if model_type in [ModelType.PERFORMANCE_PREDICTOR, ModelType.RESOURCE_PREDICTOR]:
                model = SimpleLinearRegressor()
            elif model_type in [ModelType.BOTTLENECK_DETECTOR, ModelType.OPTIMIZATION_RECOMMENDER]:
                model = SimpleClassifier(**kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            self.models[model_type] = model
            logger.info(f"Created model: {model_type.value}")
            return model

    def train_model(
        self, model_type: ModelType, training_data: TrainingData, **kwargs
    ) -> ModelMetrics:
        """
        Train a model with provided data.

        Args:
            model_type: Type of model to train
            training_data: Training data structure
            **kwargs: Additional training parameters

        Returns:
            Model performance metrics
        """
        with self._lock:
            start_time = time.time()

            # Create model if not exists
            if model_type not in self.models:
                self.create_model(model_type, **kwargs)

            model = self.models[model_type]

            # Train the model
            logger.info(f"Training {model_type.value} model...")
            model.fit(training_data.features, training_data.targets)

            training_time = time.time() - start_time

            # Calculate metrics
            metrics = self._calculate_metrics(model, training_data, training_time)
            self.model_metrics[model_type] = metrics

            logger.info(f"Model {model_type.value} trained successfully")
            logger.info(f"Training time: {training_time:.2f}s")

            return metrics

    def predict(
        self, model_type: ModelType, features: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using trained model.

        Args:
            model_type: Type of model to use
            features: Input features

        Returns:
            Predictions (and probabilities for classifiers)
        """
        with self._lock:
            if model_type not in self.models:
                raise ValueError(f"Model {model_type.value} not found")

            model = self.models[model_type]

            start_time = time.time()
            predictions = model.predict(features)
            prediction_time = time.time() - start_time

            # Return probabilities for classifiers
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)
                return predictions, probabilities

            return predictions

    def save_model(self, model_type: ModelType, filename: Optional[str] = None) -> str:
        """Save trained model to disk"""
        with self._lock:
            if model_type not in self.models:
                raise ValueError(f"Model {model_type.value} not found")

            if filename is None:
                filename = f"{model_type.value}_model.pkl"

            filepath = self.model_dir / filename

            with open(filepath, "wb") as f:
                pickle.dump(self.models[model_type], f)

            logger.info(f"Model {model_type.value} saved to {filepath}")
            return str(filepath)

    def load_model(self, model_type: ModelType, filename: str) -> Any:
        """Load trained model from disk"""
        with self._lock:
            filepath = self.model_dir / filename

            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")

            with open(filepath, "rb") as f:
                model = pickle.load(f)

            self.models[model_type] = model
            logger.info(f"Model {model_type.value} loaded from {filepath}")
            return model

    def get_model_info(self, model_type: ModelType) -> Dict[str, Any]:
        """Get information about a model"""
        with self._lock:
            if model_type not in self.models:
                return {"exists": False}

            model = self.models[model_type]
            metrics = self.model_metrics.get(model_type)

            info = {
                "exists": True,
                "trained": getattr(model, "is_trained", False),
                "type": model_type.value,
                "model_class": model.__class__.__name__,
                "metrics": metrics.__dict__ if metrics else None,
                "created_by": "Vatsal216",
                "timestamp": "2025-07-23 06:16:00",
            }

            return info

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models"""
        with self._lock:
            return {model_type.value: self.get_model_info(model_type) for model_type in self.models}

    def _calculate_metrics(
        self, model: Any, training_data: TrainingData, training_time: float
    ) -> ModelMetrics:
        """Calculate model performance metrics"""

        # Make predictions on training data
        predictions = model.predict(training_data.features)

        if hasattr(model, "predict_proba"):
            # Classification metrics
            accuracy = np.mean(predictions == training_data.targets)

            # Calculate precision, recall, F1 (binary classification)
            tp = np.sum((predictions == 1) & (training_data.targets == 1))
            fp = np.sum((predictions == 1) & (training_data.targets == 0))
            fn = np.sum((predictions == 0) & (training_data.targets == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            )

            mae = np.mean(np.abs(predictions - training_data.targets))
            rmse = np.sqrt(np.mean((predictions - training_data.targets) ** 2))
        else:
            # Regression metrics
            accuracy = model.score(training_data.features, training_data.targets)
            precision = recall = f1_score = 0.0  # Not applicable for regression

            mae = np.mean(np.abs(predictions - training_data.targets))
            rmse = np.sqrt(np.mean((predictions - training_data.targets) ** 2))

        # Estimate model size (simplified)
        model_size_mb = len(pickle.dumps(model)) / (1024 * 1024)

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mae=mae,
            rmse=rmse,
            training_time=training_time,
            prediction_time=0.0,  # Will be updated during predictions
            model_size_mb=model_size_mb,
        )

    def generate_synthetic_training_data(
        self, model_type: ModelType, num_samples: int = 1000
    ) -> TrainingData:
        """
        Generate synthetic training data for testing.

        This method creates realistic synthetic data for development and testing.
        """
        np.random.seed(42)  # For reproducible results

        if model_type == ModelType.PERFORMANCE_PREDICTOR:
            # Features: workflow_size, agent_count, task_complexity, data_size
            features = np.random.rand(num_samples, 4)
            features[:, 0] *= 100  # workflow_size (0-100)
            features[:, 1] *= 20  # agent_count (0-20)
            features[:, 2] *= 10  # task_complexity (0-10)
            features[:, 3] *= 1000  # data_size (0-1000)

            # Target: execution_time (simulated relationship)
            targets = (
                features[:, 0] * 0.1
                + features[:, 1] * 2.0
                + features[:, 2] * 5.0
                + features[:, 3] * 0.01
                + np.random.normal(0, 5, num_samples)
            )

            feature_names = ["workflow_size", "agent_count", "task_complexity", "data_size"]
            target_names = ["execution_time"]

        elif model_type == ModelType.RESOURCE_PREDICTOR:
            # Features: task_count, concurrent_agents, memory_usage, cpu_usage
            features = np.random.rand(num_samples, 4)
            features[:, 0] *= 50  # task_count (0-50)
            features[:, 1] *= 10  # concurrent_agents (0-10)
            features[:, 2] *= 8192  # memory_usage (0-8192 MB)
            features[:, 3] *= 100  # cpu_usage (0-100%)

            # Target: resource_score
            targets = (
                features[:, 0] * 0.5
                + features[:, 1] * 10.0
                + features[:, 2] * 0.01
                + features[:, 3] * 1.0
                + np.random.normal(0, 10, num_samples)
            )

            feature_names = ["task_count", "concurrent_agents", "memory_usage", "cpu_usage"]
            target_names = ["resource_score"]

        elif model_type == ModelType.BOTTLENECK_DETECTOR:
            # Features: queue_length, response_time, error_rate, throughput
            features = np.random.rand(num_samples, 4)
            features[:, 0] *= 100  # queue_length (0-100)
            features[:, 1] *= 1000  # response_time (0-1000ms)
            features[:, 2] *= 50  # error_rate (0-50%)
            features[:, 3] *= 1000  # throughput (0-1000 req/s)

            # Target: is_bottleneck (binary classification)
            bottleneck_score = (
                features[:, 0] * 0.02
                + features[:, 1] * 0.001
                + features[:, 2] * 0.05
                - features[:, 3] * 0.001
            )
            targets = (bottleneck_score > 2.0).astype(int)

            feature_names = ["queue_length", "response_time", "error_rate", "throughput"]
            target_names = ["is_bottleneck"]

        else:
            raise ValueError(f"Synthetic data generation not implemented for {model_type}")

        return TrainingData(
            features=features,
            targets=targets,
            feature_names=feature_names,
            target_names=target_names,
            metadata={
                "num_samples": num_samples,
                "synthetic": True,
                "model_type": model_type.value,
                "created_by": "Vatsal216",
                "created_at": "2025-07-23 06:16:00",
            },
        )
