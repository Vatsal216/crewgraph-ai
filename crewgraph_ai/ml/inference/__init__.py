"""
ML Inference Engine for CrewGraph AI

Provides real-time inference capabilities for all ML models including:
- Real-time prediction serving
- Batch processing capabilities
- Model loading and caching
- Performance monitoring
- A/B testing support

Author: Vatsal216
Created: 2025-01-27
"""

import asyncio
import json
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
    import pandas as pd
    ML_AVAILABLE = True
except ImportError:
    np = pd = None
    ML_AVAILABLE = False

from ..models import WorkflowPatternLearner, ResourceScalingPredictor, AnomalyDetector
from ..training import ModelVersionManager, FeatureProcessor
from ...types import WorkflowId
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionRequest:
    """Request for ML inference."""
    
    request_id: str
    model_type: str
    features: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    
    # Optional parameters
    model_version: Optional[str] = None
    return_confidence: bool = True
    return_explanation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "model_type": self.model_type,
            "features": self.features,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "return_confidence": self.return_confidence,
            "return_explanation": self.return_explanation
        }


@dataclass
class PredictionResponse:
    """Response from ML inference."""
    
    request_id: str
    prediction: Any
    confidence_score: float
    model_version: str
    processing_time: float
    
    # Optional fields
    explanation: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "prediction": self.prediction,
            "confidence_score": self.confidence_score,
            "model_version": self.model_version,
            "processing_time": self.processing_time,
            "explanation": self.explanation,
            "feature_importance": self.feature_importance,
            "error_message": self.error_message
        }


@dataclass
class BatchPredictionRequest:
    """Request for batch ML inference."""
    
    batch_id: str
    model_type: str
    features_batch: List[Dict[str, Any]]
    model_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_id": self.batch_id,
            "model_type": self.model_type,
            "features_batch": self.features_batch,
            "model_version": self.model_version
        }


@dataclass
class ModelCacheEntry:
    """Cached model entry."""
    
    model: Any
    model_version: str
    feature_processor: FeatureProcessor
    last_used: float
    load_time: float
    prediction_count: int = 0
    
    def update_usage(self):
        """Update usage statistics."""
        self.last_used = time.time()
        self.prediction_count += 1


class RealTimePredictor:
    """
    Real-time ML prediction service with model caching and performance monitoring.
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        cache_size: int = 10,
        cache_ttl: int = 3600  # 1 hour
    ):
        """
        Initialize real-time predictor.
        
        Args:
            models_dir: Directory containing trained models
            cache_size: Maximum number of models to cache
            cache_ttl: Cache time-to-live in seconds
        """
        self.models_dir = Path(models_dir)
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        
        # Model cache
        self.model_cache: Dict[str, ModelCacheEntry] = {}
        self.cache_lock = threading.RLock()
        
        # Version manager
        self.version_manager = ModelVersionManager(str(self.models_dir))
        
        # Performance tracking
        self.prediction_stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_response_time": 0.0,
            "error_count": 0
        }
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Real-time predictor initialized")
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make a real-time prediction.
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction response with results and metadata
        """
        start_time = time.time()
        
        try:
            # Load model (from cache or disk)
            model, feature_processor, model_version = self._load_model(
                request.model_type, request.model_version
            )
            
            # Prepare features
            features_df = self._prepare_features(request.features, feature_processor)
            
            # Make prediction
            prediction, confidence = self._make_prediction(
                model, features_df, request.model_type
            )
            
            # Get feature importance if requested
            feature_importance = None
            if request.return_explanation and hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(
                    features_df.columns,
                    model.feature_importances_
                ))
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(processing_time, success=True)
            
            response = PredictionResponse(
                request_id=request.request_id,
                prediction=prediction,
                confidence_score=confidence,
                model_version=model_version,
                processing_time=processing_time,
                feature_importance=feature_importance
            )
            
            logger.debug(f"Prediction completed in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            # Update error statistics
            self._update_stats(processing_time, success=False)
            
            logger.error(f"Prediction failed: {error_msg}")
            
            return PredictionResponse(
                request_id=request.request_id,
                prediction=None,
                confidence_score=0.0,
                model_version="unknown",
                processing_time=processing_time,
                error_message=error_msg
            )
    
    async def predict_async(self, request: PredictionRequest) -> PredictionResponse:
        """Make an asynchronous prediction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.predict, request)
    
    def predict_batch(self, request: BatchPredictionRequest) -> List[PredictionResponse]:
        """
        Make batch predictions.
        
        Args:
            request: Batch prediction request
            
        Returns:
            List of prediction responses
        """
        start_time = time.time()
        responses = []
        
        try:
            # Load model once for the entire batch
            model, feature_processor, model_version = self._load_model(
                request.model_type, request.model_version
            )
            
            # Process each item in the batch
            for i, features in enumerate(request.features_batch):
                item_start = time.time()
                
                try:
                    # Prepare features
                    features_df = self._prepare_features(features, feature_processor)
                    
                    # Make prediction
                    prediction, confidence = self._make_prediction(
                        model, features_df, request.model_type
                    )
                    
                    processing_time = time.time() - item_start
                    
                    response = PredictionResponse(
                        request_id=f"{request.batch_id}_{i}",
                        prediction=prediction,
                        confidence_score=confidence,
                        model_version=model_version,
                        processing_time=processing_time
                    )
                    responses.append(response)
                    
                except Exception as e:
                    processing_time = time.time() - item_start
                    error_response = PredictionResponse(
                        request_id=f"{request.batch_id}_{i}",
                        prediction=None,
                        confidence_score=0.0,
                        model_version=model_version,
                        processing_time=processing_time,
                        error_message=str(e)
                    )
                    responses.append(error_response)
            
            total_time = time.time() - start_time
            success_rate = len([r for r in responses if r.error_message is None]) / len(responses)
            
            logger.info(f"Batch prediction completed: {len(responses)} items in {total_time:.2f}s (success rate: {success_rate:.2f})")
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # Return error responses for all items
            for i in range(len(request.features_batch)):
                error_response = PredictionResponse(
                    request_id=f"{request.batch_id}_{i}",
                    prediction=None,
                    confidence_score=0.0,
                    model_version="unknown",
                    processing_time=0.0,
                    error_message=str(e)
                )
                responses.append(error_response)
        
        return responses
    
    def _load_model(
        self, 
        model_type: str, 
        version: Optional[str] = None
    ) -> tuple[Any, FeatureProcessor, str]:
        """Load model from cache or disk."""
        cache_key = f"{model_type}_{version or 'latest'}"
        
        with self.cache_lock:
            # Check cache first
            if cache_key in self.model_cache:
                entry = self.model_cache[cache_key]
                
                # Check if cache entry is still valid
                if time.time() - entry.last_used < self.cache_ttl:
                    entry.update_usage()
                    self.prediction_stats["cache_hits"] += 1
                    return entry.model, entry.feature_processor, entry.model_version
                else:
                    # Remove expired entry
                    del self.model_cache[cache_key]
            
            # Cache miss - load from disk
            self.prediction_stats["cache_misses"] += 1
            
            # Get model version info
            if version:
                model_versions = self.version_manager.list_versions(model_type)
                model_version_obj = next(
                    (v for v in model_versions if v.version == version), None
                )
            else:
                model_version_obj = self.version_manager.get_best_version(model_type)
            
            if not model_version_obj:
                raise ValueError(f"No model found for type {model_type}, version {version}")
            
            # Load model
            load_start = time.time()
            with open(model_version_obj.model_path, "rb") as f:
                model = pickle.load(f)
            
            # Load feature processor
            feature_processor = FeatureProcessor()
            processor_dir = Path(model_version_obj.model_path).parent
            feature_processor.load_processors(str(processor_dir))
            
            load_time = time.time() - load_start
            
            # Create cache entry
            entry = ModelCacheEntry(
                model=model,
                model_version=model_version_obj.version,
                feature_processor=feature_processor,
                last_used=time.time(),
                load_time=load_time
            )
            
            # Add to cache (remove oldest if cache is full)
            if len(self.model_cache) >= self.cache_size:
                self._evict_oldest_cache_entry()
            
            self.model_cache[cache_key] = entry
            
            logger.debug(f"Loaded model {model_type} v{model_version_obj.version} in {load_time:.3f}s")
            
            return model, feature_processor, model_version_obj.version
    
    def _prepare_features(
        self, 
        features: Dict[str, Any], 
        feature_processor: FeatureProcessor
    ) -> pd.DataFrame:
        """Prepare features for prediction."""
        if not ML_AVAILABLE:
            raise RuntimeError("ML libraries not available")
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Apply feature processing
        processed_features = feature_processor.transform_features(features_df)
        
        return processed_features
    
    def _make_prediction(
        self, 
        model: Any, 
        features_df: pd.DataFrame, 
        model_type: str
    ) -> tuple[Any, float]:
        """Make prediction using the model."""
        try:
            # Handle different model types
            if model_type == "pattern_learning":
                # Get similar patterns
                features_dict = features_df.iloc[0].to_dict()
                patterns = model.get_similar_patterns(features_dict)
                prediction = [p.to_dict() for p in patterns[:3]]  # Top 3 patterns
                confidence = 0.8 if patterns else 0.3
                
            elif model_type == "resource_scaling":
                # Predict resource needs
                features_dict = features_df.iloc[0].to_dict()
                resource_pred = model.predict_resource_needs(features_dict)
                prediction = {
                    "cpu": resource_pred.predicted_cpu,
                    "memory": resource_pred.predicted_memory,
                    "instances": resource_pred.predicted_instances,
                    "recommendation": resource_pred.recommendation
                }
                confidence = resource_pred.confidence_score
                
            elif model_type == "anomaly_detection":
                # Detect anomalies
                features_dict = features_df.iloc[0].to_dict()
                anomalies = model.detect_anomalies(features_dict)
                prediction = [a.__dict__ for a in anomalies]
                confidence = 1.0 if anomalies else 0.9
                
            else:
                # Standard ML model prediction
                if hasattr(model, 'predict_proba'):
                    # Classification with probabilities
                    probabilities = model.predict_proba(features_df)[0]
                    prediction = int(np.argmax(probabilities))
                    confidence = float(np.max(probabilities))
                elif hasattr(model, 'predict'):
                    # Regression or classification
                    prediction = model.predict(features_df)[0]
                    if hasattr(model, 'score'):
                        # Try to get confidence from model
                        confidence = 0.8  # Default confidence
                    else:
                        confidence = 0.7
                else:
                    raise ValueError(f"Model {type(model)} doesn't support prediction")
                
                # Convert numpy types to Python types for JSON serialization
                if hasattr(prediction, 'item'):
                    prediction = prediction.item()
                elif isinstance(prediction, np.ndarray):
                    prediction = prediction.tolist()
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed for model type {model_type}: {e}")
            raise
    
    def _evict_oldest_cache_entry(self):
        """Remove the oldest cache entry."""
        if not self.model_cache:
            return
        
        oldest_key = min(
            self.model_cache.keys(),
            key=lambda k: self.model_cache[k].last_used
        )
        del self.model_cache[oldest_key]
        logger.debug(f"Evicted cache entry: {oldest_key}")
    
    def _update_stats(self, processing_time: float, success: bool):
        """Update prediction statistics."""
        self.prediction_stats["total_predictions"] += 1
        
        if not success:
            self.prediction_stats["error_count"] += 1
        
        # Update average response time
        total = self.prediction_stats["total_predictions"]
        current_avg = self.prediction_stats["average_response_time"]
        self.prediction_stats["average_response_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            cache_info = {}
            for key, entry in self.model_cache.items():
                cache_info[key] = {
                    "model_version": entry.model_version,
                    "last_used": entry.last_used,
                    "prediction_count": entry.prediction_count,
                    "load_time": entry.load_time
                }
            
            return {
                "cache_size": len(self.model_cache),
                "max_cache_size": self.cache_size,
                "cache_entries": cache_info,
                **self.prediction_stats
            }
    
    def clear_cache(self):
        """Clear the model cache."""
        with self.cache_lock:
            self.model_cache.clear()
            logger.info("Model cache cleared")
    
    def preload_models(self, model_types: List[str]):
        """Preload models into cache."""
        for model_type in model_types:
            try:
                self._load_model(model_type)
                logger.info(f"Preloaded model: {model_type}")
            except Exception as e:
                logger.warning(f"Failed to preload model {model_type}: {e}")


class BatchProcessor:
    """
    Batch processing engine for large-scale ML inference.
    """
    
    def __init__(
        self,
        predictor: RealTimePredictor,
        batch_size: int = 100,
        max_workers: int = 4
    ):
        """
        Initialize batch processor.
        
        Args:
            predictor: Real-time predictor instance
            batch_size: Size of processing batches
            max_workers: Number of worker threads
        """
        self.predictor = predictor
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"Batch processor initialized with batch_size={batch_size}, workers={max_workers}")
    
    def process_file(
        self,
        input_file: str,
        output_file: str,
        model_type: str,
        feature_columns: List[str],
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a file with batch predictions.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            model_type: Type of model to use
            feature_columns: List of feature column names
            model_version: Specific model version to use
            
        Returns:
            Processing statistics
        """
        if not ML_AVAILABLE:
            raise RuntimeError("ML libraries not available")
        
        start_time = time.time()
        
        try:
            # Read input file
            data = pd.read_csv(input_file)
            total_rows = len(data)
            
            logger.info(f"Processing {total_rows} rows from {input_file}")
            
            # Prepare results list
            all_results = []
            
            # Process in batches
            for batch_start in range(0, total_rows, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_rows)
                batch_data = data.iloc[batch_start:batch_end]
                
                # Prepare batch request
                features_batch = []
                for _, row in batch_data.iterrows():
                    features = {col: row[col] for col in feature_columns if col in row}
                    features_batch.append(features)
                
                batch_request = BatchPredictionRequest(
                    batch_id=f"file_batch_{batch_start}",
                    model_type=model_type,
                    features_batch=features_batch,
                    model_version=model_version
                )
                
                # Process batch
                batch_results = self.predictor.predict_batch(batch_request)
                all_results.extend(batch_results)
                
                logger.debug(f"Processed batch {batch_start}-{batch_end}")
            
            # Create output DataFrame
            output_data = data.copy()
            output_data['prediction'] = [r.prediction for r in all_results]
            output_data['confidence'] = [r.confidence_score for r in all_results]
            output_data['model_version'] = [r.model_version for r in all_results]
            output_data['processing_time'] = [r.processing_time for r in all_results]
            output_data['error_message'] = [r.error_message for r in all_results]
            
            # Save output file
            output_data.to_csv(output_file, index=False)
            
            processing_time = time.time() - start_time
            successful_predictions = len([r for r in all_results if r.error_message is None])
            
            stats = {
                "total_rows": total_rows,
                "successful_predictions": successful_predictions,
                "failed_predictions": total_rows - successful_predictions,
                "success_rate": successful_predictions / total_rows,
                "processing_time": processing_time,
                "rows_per_second": total_rows / processing_time,
                "average_prediction_time": sum(r.processing_time for r in all_results) / len(all_results)
            }
            
            logger.info(f"Batch processing completed: {stats}")
            return stats
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Batch processing failed: {error_msg}")
            raise RuntimeError(f"Batch processing failed: {error_msg}")
    
    async def process_stream(
        self,
        input_stream: asyncio.Queue,
        output_stream: asyncio.Queue,
        model_type: str,
        model_version: Optional[str] = None
    ):
        """
        Process a stream of prediction requests.
        
        Args:
            input_stream: Queue of prediction requests
            output_stream: Queue for prediction responses
            model_type: Type of model to use
            model_version: Specific model version to use
        """
        logger.info(f"Starting stream processing for model {model_type}")
        
        batch_buffer = []
        
        while True:
            try:
                # Get requests from input stream (with timeout)
                request = await asyncio.wait_for(input_stream.get(), timeout=1.0)
                batch_buffer.append(request)
                
                # Process batch when full or timeout
                if len(batch_buffer) >= self.batch_size:
                    await self._process_stream_batch(
                        batch_buffer, output_stream, model_type, model_version
                    )
                    batch_buffer = []
                    
            except asyncio.TimeoutError:
                # Process remaining requests in buffer
                if batch_buffer:
                    await self._process_stream_batch(
                        batch_buffer, output_stream, model_type, model_version
                    )
                    batch_buffer = []
                    
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                break
        
        logger.info("Stream processing stopped")
    
    async def _process_stream_batch(
        self,
        requests: List[PredictionRequest],
        output_stream: asyncio.Queue,
        model_type: str,
        model_version: Optional[str]
    ):
        """Process a batch of stream requests."""
        try:
            # Create batch request
            features_batch = [req.features for req in requests]
            batch_request = BatchPredictionRequest(
                batch_id=f"stream_batch_{int(time.time())}",
                model_type=model_type,
                features_batch=features_batch,
                model_version=model_version
            )
            
            # Process batch
            results = self.predictor.predict_batch(batch_request)
            
            # Send results to output stream
            for i, result in enumerate(results):
                # Update request ID to match original request
                result.request_id = requests[i].request_id
                await output_stream.put(result)
                
        except Exception as e:
            logger.error(f"Stream batch processing failed: {e}")
            
            # Send error responses
            for request in requests:
                error_response = PredictionResponse(
                    request_id=request.request_id,
                    prediction=None,
                    confidence_score=0.0,
                    model_version="unknown",
                    processing_time=0.0,
                    error_message=str(e)
                )
                await output_stream.put(error_response)


class MLInferenceEngine:
    """
    Complete ML inference engine with real-time and batch capabilities.
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        cache_size: int = 10,
        batch_size: int = 100,
        max_workers: int = 4
    ):
        """Initialize ML inference engine."""
        self.predictor = RealTimePredictor(
            models_dir=models_dir,
            cache_size=cache_size
        )
        self.batch_processor = BatchProcessor(
            predictor=self.predictor,
            batch_size=batch_size,
            max_workers=max_workers
        )
        
        logger.info("ML inference engine initialized")
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make a real-time prediction."""
        return self.predictor.predict(request)
    
    async def predict_async(self, request: PredictionRequest) -> PredictionResponse:
        """Make an asynchronous prediction."""
        return await self.predictor.predict_async(request)
    
    def predict_batch(self, request: BatchPredictionRequest) -> List[PredictionResponse]:
        """Make batch predictions."""
        return self.predictor.predict_batch(request)
    
    def process_file(
        self,
        input_file: str,
        output_file: str,
        model_type: str,
        feature_columns: List[str],
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a file with batch predictions."""
        return self.batch_processor.process_file(
            input_file, output_file, model_type, feature_columns, model_version
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        return {
            "cache_stats": self.predictor.get_cache_stats(),
            "batch_processor_config": {
                "batch_size": self.batch_processor.batch_size,
                "max_workers": self.batch_processor.max_workers
            }
        }
    
    def preload_models(self, model_types: List[str]):
        """Preload models into cache."""
        self.predictor.preload_models(model_types)
    
    def clear_cache(self):
        """Clear the model cache."""
        self.predictor.clear_cache()


# Export all inference classes
__all__ = [
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "ModelCacheEntry",
    "RealTimePredictor",
    "BatchProcessor",
    "MLInferenceEngine"
]