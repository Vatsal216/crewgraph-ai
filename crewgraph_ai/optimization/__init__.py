"""
Optimization Module for CrewGraph AI

This module provides advanced optimization capabilities for workflow
performance, resource utilization, and execution efficiency with ML-driven
intelligence and cost optimization.

Key Components:
    - Workflow Optimizer: End-to-end workflow optimization
    - ML Optimizer: Machine learning-based performance optimization  
    - Reinforcement Learning: Q-learning scheduler for intelligent task scheduling
    - Performance Predictor: ML-based performance prediction with historical analysis
    - Resource Manager: Intelligent resource allocation and cost optimization
    - Auto Scaler: Predictive auto-scaling with cost vs performance trade-offs

Features:
    - ML-based performance optimization and prediction
    - Reinforcement learning for task scheduling
    - Intelligent resource allocation and auto-scaling
    - Cost optimization with performance trade-off analysis
    - Predictive scaling based on usage patterns
    - Multi-objective optimization (speed, cost, quality, efficiency)

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
Updated: 2025-07-23 17:40:00 UTC
"""

from .workflow_optimizer import OptimizationResult, OptimizationType, WorkflowOptimizer

# ML-based optimization
from .ml_optimizer import (
    MLOptimizer,
    MLPrediction, 
    OptimizationRecommendation,
    PerformanceFeatures
)

# Reinforcement learning for scheduling
from .reinforcement_learning import (
    QLearningScheduler,
    MultiObjectiveRLOptimizer,
    RLAction,
    RLReward,
    RLState,
    SchedulingDecision,
    create_standard_actions
)

# Performance prediction
from .performance_predictor import (
    PerformancePredictor,
    PerformancePrediction,
    HistoricalExecution,
    PredictionFeatures,
    BottleneckPrediction
)

# Resource management
from .resource_manager import (
    ResourceManager,
    ResourceMetrics,
    ResourceAllocation,
    ResourceForecast,
    ResourceType,
    ScalingRecommendation,
    ScalingDirection
)

# Auto-scaling
from .auto_scaler import (
    AutoScaler,
    ScalingPolicy,
    ScalingRule,
    ScalingEvent,
    CostOptimizationSettings,
    ScalingMetrics,
    ScalingTrigger
)

__all__ = [
    # Core optimization
    "WorkflowOptimizer", 
    "OptimizationResult",
    "OptimizationType",
    
    # ML optimization
    "MLOptimizer",
    "MLPrediction",
    "OptimizationRecommendation", 
    "PerformanceFeatures",
    
    # Reinforcement learning
    "QLearningScheduler",
    "MultiObjectiveRLOptimizer",
    "RLAction",
    "RLReward", 
    "RLState",
    "SchedulingDecision",
    "create_standard_actions",
    
    # Performance prediction
    "PerformancePredictor",
    "PerformancePrediction",
    "HistoricalExecution",
    "PredictionFeatures", 
    "BottleneckPrediction",
    
    # Resource management
    "ResourceManager",
    "ResourceMetrics",
    "ResourceAllocation",
    "ResourceForecast",
    "ResourceType",
    "ScalingRecommendation",
    "ScalingDirection",
    
    # Auto-scaling
    "AutoScaler",
    "ScalingPolicy",
    "ScalingRule",
    "ScalingEvent", 
    "CostOptimizationSettings",
    "ScalingMetrics",
    "ScalingTrigger"
]

__version__ = "2.0.0"
__author__ = "Vatsal216"
