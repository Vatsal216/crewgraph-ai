"""
Machine Learning Components for CrewGraph AI

This module provides comprehensive ML capabilities including:
- Workflow pattern learning
- Predictive resource scaling  
- Intelligent task scheduling
- Performance anomaly detection
- Cost prediction models
- Auto-tuning parameters

Author: Vatsal216
Created: 2025-01-27
"""

from .models import *
from .training import *
from .inference import *
from .optimization import *

__all__ = [
    # Core ML components
    "WorkflowPatternLearner",
    "ResourceScalingPredictor", 
    "TaskSchedulingOptimizer",
    "AnomalyDetector",
    "CostPredictor",
    "ParameterTuner",
    
    # Training infrastructure
    "MLTrainingPipeline",
    "ModelVersionManager",
    "FeatureProcessor",
    
    # Inference components  
    "MLInferenceEngine",
    "RealTimePredictor",
    "BatchProcessor",
    
    # Optimization
    "HyperparameterOptimizer",
    "ModelSelectionOptimizer",
    "ResourceOptimizer"
]