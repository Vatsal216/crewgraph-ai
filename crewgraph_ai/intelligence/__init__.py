"""
AI Intelligence Layer for CrewGraph AI

This module provides AI-driven workflow optimization, performance prediction,
and resource management capabilities.

Author: Vatsal216
Created: 2025-07-23 06:16:00 UTC
"""

from .analyzer import BottleneckAnalyzer, ResourceAnalyzer
from .ml_models import MLModelManager, ModelType
from .optimizer import OptimizationStrategy, WorkflowOptimizer
from .predictor import PerformancePredictor, ResourcePredictor

__all__ = [
    "WorkflowOptimizer",
    "OptimizationStrategy",
    "PerformancePredictor",
    "ResourcePredictor",
    "BottleneckAnalyzer",
    "ResourceAnalyzer",
    "MLModelManager",
    "ModelType",
]

__version__ = "1.0.0"
__author__ = "Vatsal216"
__created__ = "2025-07-23 06:16:00"

print(f"🧠 CrewGraph AI Intelligence Layer v{__version__} loaded")
print(f"🤖 AI-driven workflow optimization enabled")
print(f"👤 Created by: {__author__}")
print(f"⏰ Timestamp: {__created__}")

"""
AI-Driven Intelligence Layer for CrewGraph AI

This module provides intelligent optimization and prediction capabilities
for workflow orchestration, including performance prediction, resource
optimization, and adaptive planning.

Key Components:
    - Performance Predictor: ML-based workflow performance prediction
    - Resource Optimizer: Intelligent resource allocation and scheduling  
    - Adaptive Planner: Dynamic workflow optimization based on execution feedback
    - Pattern Analyzer: Workflow pattern recognition and recommendations

Features:
    - Lightweight ML models for real-time predictions
    - Performance benchmarking and optimization suggestions
    - Adaptive workflow tuning based on historical data
    - Resource usage optimization and bottleneck detection

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

from .analyzer import PatternAnalyzer, WorkflowPattern
from .optimizer import OptimizationResult, ResourceOptimizer
from .planner import AdaptivePlanner, PlanningRecommendation
from .predictor import PerformancePredictor, WorkflowMetrics

__all__ = [
    "PerformancePredictor",
    "WorkflowMetrics",
    "ResourceOptimizer",
    "OptimizationResult",
    "AdaptivePlanner",
    "PlanningRecommendation",
    "PatternAnalyzer",
    "WorkflowPattern",
]

__version__ = "1.0.0"
__author__ = "Vatsal216"
