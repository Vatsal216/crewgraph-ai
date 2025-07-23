"""
AI Intelligence Layer for CrewGraph AI

This module provides AI-driven workflow optimization, performance prediction,
and resource management capabilities.

Author: Vatsal216
Created: 2025-07-23 06:16:00 UTC
"""

from .optimizer import WorkflowOptimizer, OptimizationStrategy
from .predictor import PerformancePredictor, ResourcePredictor
from .analyzer import BottleneckAnalyzer, ResourceAnalyzer
from .ml_models import MLModelManager, ModelType

__all__ = [
    "WorkflowOptimizer",
    "OptimizationStrategy", 
    "PerformancePredictor",
    "ResourcePredictor",
    "BottleneckAnalyzer",
    "ResourceAnalyzer",
    "MLModelManager",
    "ModelType"
]

__version__ = "1.0.0"
__author__ = "Vatsal216"
__created__ = "2025-07-23 06:16:00"

print(f"🧠 CrewGraph AI Intelligence Layer v{__version__} loaded")
print(f"🤖 AI-driven workflow optimization enabled")
print(f"👤 Created by: {__author__}")
print(f"⏰ Timestamp: {__created__}")