"""
Advanced Analytics & Visualization Module for CrewGraph AI

This module provides comprehensive analytics and real-time visualization
capabilities for workflow performance monitoring and optimization insights.

Key Components:
    - Performance Dashboard: Real-time performance monitoring
    - Workflow Analyzer: Deep workflow analysis and insights
    - Cost Optimizer: Cost analysis and optimization recommendations
    - Trend Analyzer: Historical trend analysis and forecasting

Features:
    - Real-time performance dashboards
    - Interactive workflow visualization
    - Bottleneck analysis and recommendations
    - Cost optimization insights
    - Historical trend analysis and forecasting

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

from .dashboard import PerformanceDashboard, DashboardConfig
from .analyzer import WorkflowAnalyzer, AnalysisReport

__all__ = [
    "PerformanceDashboard",
    "DashboardConfig",
    "WorkflowAnalyzer",
    "AnalysisReport"
]

__version__ = "1.0.0"
__author__ = "Vatsal216"