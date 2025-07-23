"""
Workflow Analyzer for CrewGraph AI Analytics Module

Provides deep workflow analysis capabilities for performance insights
and optimization recommendations.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisReport:
    """Comprehensive analysis report for workflow performance."""

    workflow_id: str
    analysis_type: str
    performance_score: float
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]
    trends: Dict[str, Any]
    generated_at: datetime


class WorkflowAnalyzer:
    """
    Deep workflow analyzer that provides comprehensive performance
    insights and optimization recommendations.
    """

    def __init__(self):
        """Initialize the workflow analyzer."""
        logger.info("WorkflowAnalyzer initialized")

    def analyze_workflow_performance(
        self,
        workflow_data: Dict[str, Any],
        execution_history: Optional[List[Dict[str, Any]]] = None,
    ) -> AnalysisReport:
        """
        Analyze workflow performance comprehensively.

        Args:
            workflow_data: Workflow definition and metadata
            execution_history: Historical execution data

        Returns:
            Comprehensive analysis report
        """
        workflow_id = workflow_data.get("id", "unknown")

        # Calculate performance score
        performance_score = self._calculate_performance_score(workflow_data, execution_history)

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(workflow_data, execution_history)

        # Generate recommendations
        recommendations = self._generate_recommendations(workflow_data, execution_history)

        # Analyze trends
        trends = self._analyze_trends(execution_history) if execution_history else {}

        report = AnalysisReport(
            workflow_id=workflow_id,
            analysis_type="comprehensive",
            performance_score=performance_score,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            trends=trends,
            generated_at=datetime.now(),
        )

        logger.info(f"Generated analysis report for workflow {workflow_id}")
        return report

    def _calculate_performance_score(
        self, workflow_data: Dict[str, Any], execution_history: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Calculate overall performance score (0-100)."""
        if not execution_history:
            return 50.0  # Neutral score without data

        # Calculate metrics
        success_rates = [h.get("success_rate", 1.0) for h in execution_history]
        execution_times = [h.get("execution_time", 0) for h in execution_history]

        if not success_rates:
            return 50.0

        avg_success_rate = sum(success_rates) / len(success_rates)

        # Performance score based on success rate and consistency
        score = avg_success_rate * 100

        # Adjust for execution time consistency
        if execution_times and len(execution_times) > 1:
            import statistics

            time_variance = statistics.stdev(execution_times) / max(
                statistics.mean(execution_times), 1
            )
            consistency_penalty = min(20, time_variance * 100)
            score -= consistency_penalty

        return max(0, min(100, score))

    def _identify_bottlenecks(
        self, workflow_data: Dict[str, Any], execution_history: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        # Structural bottlenecks
        tasks = workflow_data.get("tasks", [])
        if len(tasks) > 10:
            bottlenecks.append(
                {
                    "type": "complexity",
                    "severity": "medium",
                    "description": f"High task count ({len(tasks)}) may impact performance",
                    "recommendation": "Consider breaking into smaller workflows",
                }
            )

        # Execution history bottlenecks
        if execution_history:
            long_executions = [h for h in execution_history if h.get("execution_time", 0) > 60]
            if len(long_executions) > len(execution_history) * 0.3:
                bottlenecks.append(
                    {
                        "type": "execution_time",
                        "severity": "high",
                        "description": f"{len(long_executions)} executions took over 60 seconds",
                        "recommendation": "Optimize long-running tasks or add parallelization",
                    }
                )

        return bottlenecks

    def _generate_recommendations(
        self, workflow_data: Dict[str, Any], execution_history: Optional[List[Dict[str, Any]]]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Basic recommendations
        tasks = workflow_data.get("tasks", [])
        if len(tasks) > 5:
            recommendations.append("Consider parallelizing independent tasks")

        if execution_history:
            error_rates = [1.0 - h.get("success_rate", 1.0) for h in execution_history]
            if error_rates and sum(error_rates) / len(error_rates) > 0.1:
                recommendations.append("Implement better error handling and retry mechanisms")

        return recommendations

    def _analyze_trends(self, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(execution_history) < 3:
            return {"trend": "insufficient_data"}

        # Analyze recent vs historical performance
        recent = execution_history[-3:]
        historical = execution_history[:-3] if len(execution_history) > 3 else execution_history

        recent_avg = sum(h.get("success_rate", 1.0) for h in recent) / len(recent)
        historical_avg = sum(h.get("success_rate", 1.0) for h in historical) / len(historical)

        if recent_avg > historical_avg * 1.1:
            trend = "improving"
        elif recent_avg < historical_avg * 0.9:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "performance_trend": trend,
            "recent_performance": recent_avg,
            "historical_performance": historical_avg,
        }
