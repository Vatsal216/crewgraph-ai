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

from .predictor import PerformancePredictor, WorkflowMetrics
from .optimizer import ResourceOptimizer, OptimizationResult
from .planner import AdaptivePlanner, PlanningRecommendation
from .analyzer import PatternAnalyzer, WorkflowPattern

__all__ = [
    "PerformancePredictor",
    "WorkflowMetrics", 
    "ResourceOptimizer",
    "OptimizationResult",
    "AdaptivePlanner",
    "PlanningRecommendation",
    "PatternAnalyzer",
    "WorkflowPattern"
]

__version__ = "1.0.0"
__author__ = "Vatsal216"