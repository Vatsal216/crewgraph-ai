"""
ML Training Infrastructure for CrewGraph AI

Provides comprehensive training pipelines for all ML models including:
- Model training orchestration
- Feature processing and engineering  
- Model versioning and management
- Automated retraining pipelines
- Training data management

Author: Vatsal216
Created: 2025-01-27
"""

import json
import pickle
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    ML_AVAILABLE = True
except ImportError:
    np = pd = cross_val_score = train_test_split = None
    mean_squared_error = accuracy_score = f1_score = None
    StandardScaler = LabelEncoder = None
    ML_AVAILABLE = False

from ..models import WorkflowPatternLearner, ResourceScalingPredictor, AnomalyDetector
from ...types import WorkflowId
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for ML model training."""
    
    model_type: str
    training_data_path: str
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    max_training_time: int = 3600  # seconds
    
    # Feature engineering
    feature_selection: bool = True
    feature_scaling: bool = True
    handle_missing_values: bool = True
    
    # Model parameters
    hyperparameter_tuning: bool = True
    grid_search_params: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Training monitoring
    early_stopping: bool = True
    patience: int = 10
    min_improvement: float = 0.001
    
    # Output configuration
    model_output_dir: str = "models"
    save_training_artifacts: bool = True


@dataclass
class TrainingResult:
    """Result of a model training process."""
    
    success: bool
    model_type: str
    training_time: float
    model_path: Optional[str] = None
    
    # Performance metrics
    training_score: float = 0.0
    validation_score: float = 0.0
    cross_val_score: float = 0.0
    
    # Training details
    total_samples: int = 0
    feature_count: int = 0
    selected_features: List[str] = field(default_factory=list)
    
    # Model metadata
    model_version: str = "1.0.0"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Errors and warnings
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ModelVersion:
    """Model version information."""
    
    version: str
    timestamp: float
    model_path: str
    config_path: str
    
    # Performance metrics
    validation_score: float
    training_samples: int
    
    # Metadata
    git_commit: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "validation_score": self.validation_score,
            "training_samples": self.training_samples,
            "git_commit": self.git_commit,
            "tags": self.tags,
            "notes": self.notes
        }


class FeatureProcessor:
    """
    Handles feature engineering and preprocessing for ML models.
    """
    
    def __init__(self):
        """Initialize feature processor."""
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
    def process_features(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        config: Optional[TrainingConfig] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process features for training.
        
        Args:
            data: Raw training data
            target_column: Target variable column name
            config: Training configuration
            
        Returns:
            Processed features and target variable
        """
        if not ML_AVAILABLE:
            raise RuntimeError("ML libraries not available")
        
        processed_data = data.copy()
        
        # Handle missing values
        if config and config.handle_missing_values:
            processed_data = self._handle_missing_values(processed_data)
        
        # Separate features and target
        if target_column:
            target = processed_data[target_column]
            features = processed_data.drop(columns=[target_column])
        else:
            target = pd.Series()
            features = processed_data
        
        # Encode categorical variables
        features = self._encode_categorical_features(features)
        
        # Feature scaling
        if config and config.feature_scaling:
            features = self._scale_features(features)
        
        # Feature selection
        if config and config.feature_selection and not target.empty:
            features = self._select_features(features, target)
        
        # Calculate feature statistics
        self._calculate_feature_stats(features)
        
        logger.info(f"Processed features: {features.shape[1]} features, {features.shape[0]} samples")
        
        return features, target
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # For numeric columns, fill with median
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().any():
                data[col].fillna(data[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if data[col].isnull().any():
                mode_value = data[col].mode()
                if not mode_value.empty:
                    data[col].fillna(mode_value[0], inplace=True)
                else:
                    data[col].fillna('unknown', inplace=True)
        
        return data
    
    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        encoded_data = data.copy()
        
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                encoded_data[col] = self.encoders[col].fit_transform(data[col])
            else:
                # Handle new categories not seen during training
                known_categories = set(self.encoders[col].classes_)
                new_categories = set(data[col].unique()) - known_categories
                
                if new_categories:
                    # Add new categories to encoder
                    all_categories = list(known_categories) + list(new_categories)
                    self.encoders[col].classes_ = np.array(all_categories)
                
                encoded_data[col] = self.encoders[col].transform(data[col])
        
        return encoded_data
    
    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        scaled_data = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if not numeric_columns.empty:
            if 'scaler' not in self.scalers:
                self.scalers['scaler'] = StandardScaler()
                scaled_data[numeric_columns] = self.scalers['scaler'].fit_transform(data[numeric_columns])
            else:
                scaled_data[numeric_columns] = self.scalers['scaler'].transform(data[numeric_columns])
        
        return scaled_data
    
    def _select_features(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select most important features."""
        try:
            from sklearn.feature_selection import SelectKBest, f_regression, f_classif
            
            # Determine if classification or regression
            if target.dtype == 'object' or len(target.unique()) < 10:
                score_func = f_classif
            else:
                score_func = f_regression
            
            # Select top 50% of features
            k = max(1, int(features.shape[1] * 0.5))
            selector = SelectKBest(score_func=score_func, k=k)
            
            selected_features = selector.fit_transform(features, target)
            selected_columns = features.columns[selector.get_support()]
            
            return pd.DataFrame(selected_features, columns=selected_columns, index=features.index)
            
        except ImportError:
            logger.warning("Feature selection not available, using all features")
            return features
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}, using all features")
            return features
    
    def _calculate_feature_stats(self, features: pd.DataFrame):
        """Calculate and store feature statistics."""
        for col in features.columns:
            if features[col].dtype in [np.number]:
                self.feature_stats[col] = {
                    'mean': float(features[col].mean()),
                    'std': float(features[col].std()),
                    'min': float(features[col].min()),
                    'max': float(features[col].max()),
                    'missing_rate': float(features[col].isnull().mean())
                }
    
    def transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted processors."""
        if not ML_AVAILABLE:
            raise RuntimeError("ML libraries not available")
        
        transformed_data = data.copy()
        
        # Apply same transformations as training
        transformed_data = self._encode_categorical_features(transformed_data)
        
        if 'scaler' in self.scalers:
            numeric_columns = transformed_data.select_dtypes(include=[np.number]).columns
            if not numeric_columns.empty:
                transformed_data[numeric_columns] = self.scalers['scaler'].transform(
                    transformed_data[numeric_columns]
                )
        
        return transformed_data
    
    def save_processors(self, output_dir: str):
        """Save fitted processors to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save scalers
        if self.scalers:
            with open(output_path / "scalers.pkl", "wb") as f:
                pickle.dump(self.scalers, f)
        
        # Save encoders
        if self.encoders:
            with open(output_path / "encoders.pkl", "wb") as f:
                pickle.dump(self.encoders, f)
        
        # Save feature stats
        if self.feature_stats:
            with open(output_path / "feature_stats.json", "w") as f:
                json.dump(self.feature_stats, f, indent=2)
    
    def load_processors(self, input_dir: str):
        """Load fitted processors from disk."""
        input_path = Path(input_dir)
        
        # Load scalers
        scalers_path = input_path / "scalers.pkl"
        if scalers_path.exists():
            with open(scalers_path, "rb") as f:
                self.scalers = pickle.load(f)
        
        # Load encoders
        encoders_path = input_path / "encoders.pkl"
        if encoders_path.exists():
            with open(encoders_path, "rb") as f:
                self.encoders = pickle.load(f)
        
        # Load feature stats
        stats_path = input_path / "feature_stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                self.feature_stats = json.load(f)


class ModelVersionManager:
    """
    Manages model versions and provides model lifecycle management.
    """
    
    def __init__(self, models_dir: str = "models"):
        """Initialize model version manager."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.versions_file = self.models_dir / "versions.json"
        self.versions: Dict[str, List[ModelVersion]] = {}
        
        self._load_versions()
    
    def register_model(
        self,
        model_type: str,
        model_path: str,
        config_path: str,
        validation_score: float,
        training_samples: int,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: str = ""
    ) -> ModelVersion:
        """Register a new model version."""
        if version is None:
            version = self._generate_version(model_type)
        
        model_version = ModelVersion(
            version=version,
            timestamp=time.time(),
            model_path=model_path,
            config_path=config_path,
            validation_score=validation_score,
            training_samples=training_samples,
            tags=tags or [],
            notes=notes
        )
        
        if model_type not in self.versions:
            self.versions[model_type] = []
        
        self.versions[model_type].append(model_version)
        self._save_versions()
        
        logger.info(f"Registered model {model_type} version {version}")
        return model_version
    
    def get_latest_version(self, model_type: str) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        if model_type not in self.versions or not self.versions[model_type]:
            return None
        
        return max(self.versions[model_type], key=lambda v: v.timestamp)
    
    def get_best_version(self, model_type: str) -> Optional[ModelVersion]:
        """Get the best performing version of a model."""
        if model_type not in self.versions or not self.versions[model_type]:
            return None
        
        return max(self.versions[model_type], key=lambda v: v.validation_score)
    
    def list_versions(self, model_type: str) -> List[ModelVersion]:
        """List all versions of a model."""
        return self.versions.get(model_type, [])
    
    def promote_version(self, model_type: str, version: str, tag: str = "production"):
        """Promote a model version to production."""
        if model_type not in self.versions:
            raise ValueError(f"Model type {model_type} not found")
        
        for model_version in self.versions[model_type]:
            if model_version.version == version:
                if tag not in model_version.tags:
                    model_version.tags.append(tag)
                self._save_versions()
                logger.info(f"Promoted {model_type} version {version} to {tag}")
                return
        
        raise ValueError(f"Version {version} not found for model {model_type}")
    
    def cleanup_old_versions(self, model_type: str, keep_count: int = 5):
        """Clean up old model versions, keeping only the most recent."""
        if model_type not in self.versions:
            return
        
        versions = sorted(self.versions[model_type], key=lambda v: v.timestamp, reverse=True)
        
        if len(versions) <= keep_count:
            return
        
        versions_to_remove = versions[keep_count:]
        
        for version in versions_to_remove:
            # Don't remove tagged versions
            if not version.tags:
                try:
                    # Remove model files
                    if Path(version.model_path).exists():
                        Path(version.model_path).unlink()
                    if Path(version.config_path).exists():
                        Path(version.config_path).unlink()
                    
                    # Remove from registry
                    self.versions[model_type].remove(version)
                    
                    logger.info(f"Cleaned up {model_type} version {version.version}")
                    
                except Exception as e:
                    logger.error(f"Failed to cleanup version {version.version}: {e}")
        
        self._save_versions()
    
    def _generate_version(self, model_type: str) -> str:
        """Generate a new version number."""
        if model_type not in self.versions or not self.versions[model_type]:
            return "1.0.0"
        
        latest = self.get_latest_version(model_type)
        if latest:
            # Simple version increment
            parts = latest.version.split('.')
            if len(parts) == 3:
                major, minor, patch = map(int, parts)
                return f"{major}.{minor}.{patch + 1}"
        
        return "1.0.0"
    
    def _load_versions(self):
        """Load version registry from disk."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                
                self.versions = {}
                for model_type, versions_data in data.items():
                    self.versions[model_type] = [
                        ModelVersion(**version_data) for version_data in versions_data
                    ]
                    
            except Exception as e:
                logger.error(f"Failed to load version registry: {e}")
                self.versions = {}
    
    def _save_versions(self):
        """Save version registry to disk."""
        try:
            data = {}
            for model_type, versions in self.versions.items():
                data[model_type] = [version.to_dict() for version in versions]
            
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save version registry: {e}")


class MLTrainingPipeline:
    """
    Orchestrates the complete ML training pipeline including data processing,
    model training, evaluation, and deployment.
    """
    
    def __init__(self, models_dir: str = "models"):
        """Initialize training pipeline."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_processor = FeatureProcessor()
        self.version_manager = ModelVersionManager(str(self.models_dir))
        
        # Training statistics
        self.training_history: List[TrainingResult] = []
    
    def train_model(
        self,
        model_type: str,
        training_data: Union[str, pd.DataFrame],
        config: TrainingConfig
    ) -> TrainingResult:
        """
        Train a machine learning model with the specified configuration.
        
        Args:
            model_type: Type of model to train
            training_data: Training data (file path or DataFrame)
            config: Training configuration
            
        Returns:
            Training result with metrics and model information
        """
        start_time = time.time()
        
        try:
            # Load training data
            if isinstance(training_data, str):
                data = pd.read_csv(training_data)
            else:
                data = training_data.copy()
            
            # Determine target column
            target_column = self._determine_target_column(model_type, data)
            
            # Process features
            features, target = self.feature_processor.process_features(
                data, target_column, config
            )
            
            # Split data
            if not target.empty:
                X_train, X_val, y_train, y_val = train_test_split(
                    features, target, 
                    test_size=config.validation_split,
                    random_state=42
                )
            else:
                X_train, X_val = features, pd.DataFrame()
                y_train, y_val = pd.Series(), pd.Series()
            
            # Create and train model
            model = self._create_model(model_type, config)
            
            if model_type in ["pattern_learning", "anomaly_detection"]:
                # Unsupervised learning
                model.fit(X_train)
                training_score = self._evaluate_unsupervised_model(model, X_train)
                validation_score = self._evaluate_unsupervised_model(model, X_val) if not X_val.empty else training_score
            else:
                # Supervised learning
                model.fit(X_train, y_train)
                training_score = self._evaluate_supervised_model(model, X_train, y_train)
                validation_score = self._evaluate_supervised_model(model, X_val, y_val) if not X_val.empty else training_score
            
            # Cross-validation
            cv_score = 0.0
            if config.cross_validation_folds > 1 and not target.empty:
                cv_scores = cross_val_score(
                    model, features, target, 
                    cv=config.cross_validation_folds
                )
                cv_score = float(np.mean(cv_scores))
            
            # Save model and artifacts
            model_path, config_path = self._save_model_artifacts(
                model, model_type, config, start_time
            )
            
            # Register model version
            model_version = self.version_manager.register_model(
                model_type=model_type,
                model_path=model_path,
                config_path=config_path,
                validation_score=validation_score,
                training_samples=len(features),
                notes=f"Trained with {len(features)} samples"
            )
            
            training_time = time.time() - start_time
            
            result = TrainingResult(
                success=True,
                model_type=model_type,
                training_time=training_time,
                model_path=model_path,
                training_score=training_score,
                validation_score=validation_score,
                cross_val_score=cv_score,
                total_samples=len(features),
                feature_count=features.shape[1],
                selected_features=list(features.columns),
                model_version=model_version.version,
                hyperparameters=self._get_model_hyperparameters(model)
            )
            
            self.training_history.append(result)
            logger.info(f"Successfully trained {model_type} model in {training_time:.2f}s")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            error_msg = str(e)
            
            result = TrainingResult(
                success=False,
                model_type=model_type,
                training_time=training_time,
                error_message=error_msg
            )
            
            self.training_history.append(result)
            logger.error(f"Training failed for {model_type}: {error_msg}")
            
            return result
    
    def _determine_target_column(self, model_type: str, data: pd.DataFrame) -> Optional[str]:
        """Determine target column based on model type and data."""
        if model_type in ["pattern_learning", "anomaly_detection"]:
            return None  # Unsupervised learning
        
        # Common target column names
        target_candidates = [
            "target", "label", "y", "output", "prediction",
            "duration", "performance", "cost", "success"
        ]
        
        for candidate in target_candidates:
            if candidate in data.columns:
                return candidate
        
        # If no obvious target, use last column
        if len(data.columns) > 1:
            return data.columns[-1]
        
        return None
    
    def _create_model(self, model_type: str, config: TrainingConfig):
        """Create appropriate model based on type."""
        if not ML_AVAILABLE:
            raise RuntimeError("ML libraries not available")
        
        if model_type == "pattern_learning":
            return WorkflowPatternLearner()
        elif model_type == "resource_scaling":
            return ResourceScalingPredictor()
        elif model_type == "anomaly_detection":
            return AnomalyDetector()
        else:
            # Generic ML models
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            if config.hyperparameter_tuning and config.grid_search_params:
                # Use GridSearchCV for hyperparameter tuning
                from sklearn.model_selection import GridSearchCV
                base_model = RandomForestRegressor(random_state=42)
                model = GridSearchCV(
                    base_model, 
                    config.grid_search_params,
                    cv=3,
                    scoring='neg_mean_squared_error'
                )
            else:
                model = RandomForestRegressor(random_state=42)
            
            return model
    
    def _evaluate_supervised_model(self, model, X, y) -> float:
        """Evaluate supervised learning model."""
        if X.empty or y.empty:
            return 0.0
        
        try:
            predictions = model.predict(X)
            
            # Determine if classification or regression
            if y.dtype == 'object' or len(y.unique()) < 10:
                # Classification
                return float(accuracy_score(y, predictions))
            else:
                # Regression - return negative MSE for consistency
                mse = mean_squared_error(y, predictions)
                return float(-mse)
                
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return 0.0
    
    def _evaluate_unsupervised_model(self, model, X) -> float:
        """Evaluate unsupervised learning model."""
        if X.empty:
            return 0.0
        
        try:
            # For unsupervised models, use silhouette score or similar
            from sklearn.metrics import silhouette_score
            
            if hasattr(model, 'predict'):
                labels = model.predict(X)
                if len(set(labels)) > 1:
                    return float(silhouette_score(X, labels))
            
            return 0.5  # Default neutral score
            
        except Exception:
            return 0.5  # Default neutral score
    
    def _save_model_artifacts(
        self,
        model,
        model_type: str,
        config: TrainingConfig,
        timestamp: float
    ) -> Tuple[str, str]:
        """Save model and configuration artifacts."""
        timestamp_str = str(int(timestamp))
        model_dir = self.models_dir / model_type / timestamp_str
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = str(model_dir / "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save configuration
        config_path = str(model_dir / "config.json")
        config_dict = {
            "model_type": model_type,
            "training_data_path": config.training_data_path,
            "validation_split": config.validation_split,
            "feature_selection": config.feature_selection,
            "feature_scaling": config.feature_scaling,
            "hyperparameter_tuning": config.hyperparameter_tuning,
            "timestamp": timestamp
        }
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save feature processors
        self.feature_processor.save_processors(str(model_dir))
        
        return model_path, config_path
    
    def _get_model_hyperparameters(self, model) -> Dict[str, Any]:
        """Extract hyperparameters from trained model."""
        if hasattr(model, 'get_params'):
            return model.get_params()
        elif hasattr(model, '__dict__'):
            # Extract basic attributes
            params = {}
            for key, value in model.__dict__.items():
                if not key.startswith('_') and isinstance(value, (int, float, str, bool)):
                    params[key] = value
            return params
        else:
            return {}
    
    def retrain_model(
        self,
        model_type: str,
        new_data: Union[str, pd.DataFrame],
        config: Optional[TrainingConfig] = None
    ) -> TrainingResult:
        """Retrain an existing model with new data."""
        if config is None:
            # Use default configuration
            config = TrainingConfig(
                model_type=model_type,
                training_data_path="" if isinstance(new_data, pd.DataFrame) else new_data
            )
        
        # Get existing model for warm start if possible
        latest_version = self.version_manager.get_latest_version(model_type)
        
        if latest_version:
            logger.info(f"Retraining {model_type} from version {latest_version.version}")
        else:
            logger.info(f"Training new {model_type} model")
        
        return self.train_model(model_type, new_data, config)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training pipeline statistics."""
        if not self.training_history:
            return {"total_trainings": 0}
        
        successful_trainings = [r for r in self.training_history if r.success]
        failed_trainings = [r for r in self.training_history if not r.success]
        
        stats = {
            "total_trainings": len(self.training_history),
            "successful_trainings": len(successful_trainings),
            "failed_trainings": len(failed_trainings),
            "success_rate": len(successful_trainings) / len(self.training_history),
            "average_training_time": sum(r.training_time for r in self.training_history) / len(self.training_history),
            "model_types_trained": list(set(r.model_type for r in self.training_history)),
            "best_models": {}
        }
        
        # Best model by type
        for model_type in stats["model_types_trained"]:
            type_results = [r for r in successful_trainings if r.model_type == model_type]
            if type_results:
                best_result = max(type_results, key=lambda r: r.validation_score)
                stats["best_models"][model_type] = {
                    "version": best_result.model_version,
                    "validation_score": best_result.validation_score,
                    "training_time": best_result.training_time,
                    "total_samples": best_result.total_samples
                }
        
        return stats


# Export all training classes
__all__ = [
    "TrainingConfig",
    "TrainingResult", 
    "ModelVersion",
    "FeatureProcessor",
    "ModelVersionManager",
    "MLTrainingPipeline"
]