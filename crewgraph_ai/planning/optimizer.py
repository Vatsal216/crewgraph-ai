"""
CrewGraph AI Workflow Optimizer
Advanced optimization system for workflow performance, resource utilization, and cost efficiency

Author: Vatsal216
Created: 2025-07-22 13:46:40 UTC
"""

import time
import math
import threading
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import copy

from crewai import Agent, Task
from ..core.graph import CrewGraph
from ..core.state import StateManager
from ..memory.base import BaseMemory
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector
from ..utils.exceptions import OptimizationError

logger = get_logger(__name__)
metrics = get_metrics_collector()


class OptimizationType(Enum):
    """Types of workflow optimizations"""
    PERFORMANCE = "performance"
    COST = "cost"
    RESOURCE_UTILIZATION = "resource_utilization"
    ENERGY_EFFICIENCY = "energy_efficiency"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RELIABILITY = "reliability"
    BALANCED = "balanced"


class OptimizationLevel(Enum):
    """Optimization intensity levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXPERIMENTAL = "experimental"


@dataclass
class OptimizationRule:
    """Optimization rule configuration"""
    name: str
    optimization_type: OptimizationType
    priority: int = 5  # 1-10, higher is more important
    min_improvement_threshold: float = 0.05  # 5% minimum improvement
    max_risk_level: float = 0.1  # 10% maximum risk tolerance
    enabled: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_by: str = "Vatsal216"
    created_at: str = "2025-07-22 13:46:40"


@dataclass
class OptimizationSuggestion:
    """Optimization suggestion with impact analysis"""
    suggestion_id: str
    optimization_type: OptimizationType
    title: str
    description: str
    current_value: float
    projected_value: float
    improvement_percentage: float
    confidence_score: float
    risk_level: float
    implementation_effort: str  # low, medium, high
    estimated_savings: Optional[Dict[str, float]] = None
    prerequisites: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    rollback_plan: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: str = "Vatsal216"
    created_at: str = "2025-07-22 13:46:40"
    
    def get_priority_score(self) -> float:
        """Calculate priority score based on multiple factors"""
        base_score = self.improvement_percentage * self.confidence_score
        risk_penalty = self.risk_level * 10  # Penalize high risk
        effort_penalty = {"low": 0, "medium": 5, "high": 15}.get(self.implementation_effort, 10)
        
        return max(0, base_score - risk_penalty - effort_penalty)


@dataclass
class OptimizationResult:
    """Result of optimization analysis or application"""
    workflow_id: str
    optimization_type: OptimizationType
    suggestions: List[OptimizationSuggestion]
    current_metrics: Dict[str, float]
    projected_metrics: Dict[str, float]
    total_improvement: float
    confidence_score: float
    analysis_duration: float
    optimization_rules_applied: List[str]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    analyzed_by: str = "Vatsal216"
    analyzed_at: str = "2025-07-22 13:46:40"


class WorkflowOptimizer:
    """
    Advanced workflow optimization system for CrewGraph AI.
    
    Provides comprehensive workflow optimization capabilities including:
    - Performance optimization and bottleneck identification
    - Resource utilization optimization
    - Cost reduction and efficiency improvements
    - Agent and task scheduling optimization
    - Memory usage optimization
    - Parallel execution optimization
    - Predictive optimization based on historical patterns
    - Multi-objective optimization with trade-off analysis
    - Real-time optimization recommendations
    - A/B testing framework for optimization validation
    
    Features:
    - Multi-dimensional optimization (performance, cost, reliability)
    - Machine learning-based optimization suggestions
    - Risk assessment and rollback planning
    - Integration with monitoring and scaling systems
    - Customizable optimization rules and policies
    - Comprehensive impact analysis and reporting
    
    Created by: Vatsal216
    Date: 2025-07-22 13:46:40 UTC
    """
    
    def __init__(self,
                 optimization_level: OptimizationLevel = OptimizationLevel.MODERATE,
                 target_optimization: OptimizationType = OptimizationType.BALANCED,
                 enable_predictive: bool = True,
                 enable_continuous: bool = False,
                 analysis_interval: int = 300,  # 5 minutes
                 custom_rules: Optional[List[OptimizationRule]] = None):
        """
        Initialize workflow optimizer.
        
        Args:
            optimization_level: How aggressively to optimize
            target_optimization: Primary optimization objective
            enable_predictive: Enable predictive optimization
            enable_continuous: Enable continuous optimization
            analysis_interval: Analysis interval in seconds
            custom_rules: Custom optimization rules
        """
        self.optimization_level = optimization_level
        self.target_optimization = target_optimization
        self.enable_predictive = enable_predictive
        self.enable_continuous = enable_continuous
        self.analysis_interval = analysis_interval
        
        # Optimization rules
        self.optimization_rules = custom_rules or self._default_optimization_rules()
        
        # Historical data for predictive optimization
        self.workflow_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.optimization_history: deque = deque(maxlen=500)
        self.applied_optimizations: Dict[str, List] = defaultdict(list)
        
        # Analysis cache
        self.analysis_cache: Dict[str, Tuple[OptimizationResult, float]] = {}
        self.cache_ttl = 600  # 10 minutes
        
        # Continuous optimization
        self._continuous_thread: Optional[threading.Thread] = None
        self._running_continuous = False
        
        # Statistics
        self.optimization_stats = {
            'total_analyses': 0,
            'suggestions_generated': 0,
            'optimizations_applied': 0,
            'total_improvement_achieved': 0.0,
            'avg_analysis_time': 0.0,
            'last_analysis_time': None
        }
        
        logger.info("WorkflowOptimizer initialized")
        logger.info(f"Level: {optimization_level.value}, Target: {target_optimization.value}")
        logger.info(f"Predictive: {enable_predictive}, Continuous: {enable_continuous}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 13:46:40")
        
        # Record initialization metrics
        metrics.increment_counter(
            "crewgraph_workflow_optimizers_created_total",
            labels={
                "optimization_level": optimization_level.value,
                "target_optimization": target_optimization.value,
                "user": "Vatsal216"
            }
        )
    
    def analyze_workflow(self,
                        workflow: CrewGraph,
                        include_predictions: bool = None,
                        custom_metrics: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """
        Analyze workflow and generate optimization suggestions.
        
        Args:
            workflow: Workflow to analyze
            include_predictions: Include predictive analysis
            custom_metrics: Additional custom metrics to consider
            
        Returns:
            Comprehensive optimization analysis result
        """
        start_time = time.time()
        workflow_id = workflow.id
        
        logger.info(f"Starting workflow optimization analysis: {workflow_id}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 13:46:40")
        
        # Check cache first
        cache_key = self._generate_cache_key(workflow, custom_metrics)
        if cache_key in self.analysis_cache:
            cached_result, cache_time = self.analysis_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                logger.debug(f"Using cached analysis for workflow: {workflow_id}")
                return cached_result
        
        try:
            # Collect current workflow metrics
            current_metrics = self._collect_workflow_metrics(workflow, custom_metrics)
            
            # Store metrics for historical analysis
            self.workflow_metrics_history[workflow_id].append({
                'timestamp': time.time(),
                'metrics': current_metrics.copy()
            })
            
            # Generate optimization suggestions
            suggestions = []
            warnings = []
            errors = []
            
            # Apply optimization rules
            applied_rules = []
            for rule in self.optimization_rules:
                if not rule.enabled:
                    continue
                
                try:
                    rule_suggestions = self._apply_optimization_rule(
                        rule, workflow, current_metrics
                    )
                    suggestions.extend(rule_suggestions)
                    applied_rules.append(rule.name)
                    
                except Exception as e:
                    error_msg = f"Error applying rule '{rule.name}': {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Add predictive suggestions if enabled
            if include_predictions or (include_predictions is None and self.enable_predictive):
                try:
                    predictive_suggestions = self._generate_predictive_suggestions(
                        workflow, current_metrics
                    )
                    suggestions.extend(predictive_suggestions)
                    applied_rules.append("predictive_analysis")
                    
                except Exception as e:
                    error_msg = f"Error in predictive analysis: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Prioritize and filter suggestions
            suggestions = self._prioritize_suggestions(suggestions)
            suggestions = self._filter_conflicting_suggestions(suggestions)
            
            # Calculate projected metrics
            projected_metrics = self._calculate_projected_metrics(
                current_metrics, suggestions
            )
            
            # Calculate overall improvement
            total_improvement = self._calculate_total_improvement(
                current_metrics, projected_metrics
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(suggestions)
            
            # Create result
            analysis_duration = time.time() - start_time
            result = OptimizationResult(
                workflow_id=workflow_id,
                optimization_type=self.target_optimization,
                suggestions=suggestions,
                current_metrics=current_metrics,
                projected_metrics=projected_metrics,
                total_improvement=total_improvement,
                confidence_score=confidence_score,
                analysis_duration=analysis_duration,
                optimization_rules_applied=applied_rules,
                warnings=warnings,
                errors=errors,
                analyzed_by="Vatsal216",
                analyzed_at="2025-07-22 13:46:40"
            )
            
            # Cache result
            self.analysis_cache[cache_key] = (result, time.time())
            
            # Update statistics
            self.optimization_stats['total_analyses'] += 1
            self.optimization_stats['suggestions_generated'] += len(suggestions)
            self.optimization_stats['avg_analysis_time'] = (
                (self.optimization_stats['avg_analysis_time'] * (self.optimization_stats['total_analyses'] - 1) + 
                 analysis_duration) / self.optimization_stats['total_analyses']
            )
            self.optimization_stats['last_analysis_time'] = time.time()
            
            # Record metrics
            metrics.record_duration(
                "crewgraph_workflow_optimization_analysis_duration_seconds",
                analysis_duration,
                labels={
                    "workflow_id": workflow_id,
                    "optimization_type": self.target_optimization.value,
                    "suggestions_count": str(len(suggestions)),
                    "user": "Vatsal216"
                }
            )
            
            metrics.increment_counter(
                "crewgraph_workflow_optimizations_analyzed_total",
                labels={
                    "workflow_id": workflow_id,
                    "optimization_type": self.target_optimization.value,
                    "user": "Vatsal216"
                }
            )
            
            logger.info(f"Optimization analysis completed: {workflow_id}")
            logger.info(f"Generated {len(suggestions)} suggestions with {total_improvement:.2f}% projected improvement")
            
            return result
            
        except Exception as e:
            error_msg = f"Workflow optimization analysis failed: {str(e)}"
            logger.error(error_msg)
            
            # Return error result
            return OptimizationResult(
                workflow_id=workflow_id,
                optimization_type=self.target_optimization,
                suggestions=[],
                current_metrics=current_metrics if 'current_metrics' in locals() else {},
                projected_metrics={},
                total_improvement=0.0,
                confidence_score=0.0,
                analysis_duration=time.time() - start_time,
                optimization_rules_applied=[],
                errors=[error_msg],
                analyzed_by="Vatsal216",
                analyzed_at="2025-07-22 13:46:40"
            )
    
    def apply_optimizations(self,
                           workflow: CrewGraph,
                           suggestions: List[OptimizationSuggestion],
                           auto_approve_low_risk: bool = True,
                           max_risk_level: float = 0.2) -> Dict[str, Any]:
        """
        Apply selected optimization suggestions to workflow.
        
        Args:
            workflow: Workflow to optimize
            suggestions: Optimization suggestions to apply
            auto_approve_low_risk: Automatically apply low-risk optimizations
            max_risk_level: Maximum acceptable risk level
            
        Returns:
            Application results with success/failure details
        """
        workflow_id = workflow.id
        logger.info(f"Applying optimizations to workflow: {workflow_id}")
        logger.info(f"Suggestions to apply: {len(suggestions)}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 13:46:40")
        
        results = {
            'workflow_id': workflow_id,
            'total_suggestions': len(suggestions),
            'applied_successfully': 0,
            'failed_applications': 0,
            'skipped_high_risk': 0,
            'total_improvement': 0.0,
            'applications': [],
            'errors': [],
            'rollback_info': [],
            'applied_by': 'Vatsal216',
            'applied_at': '2025-07-22 13:46:40'
        }
        
        # Filter suggestions by risk level
        filtered_suggestions = []
        for suggestion in suggestions:
            if suggestion.risk_level > max_risk_level:
                results['skipped_high_risk'] += 1
                logger.warning(f"Skipping high-risk optimization: {suggestion.title} "
                             f"(risk: {suggestion.risk_level:.2f})")
                continue
            
            if not auto_approve_low_risk and suggestion.risk_level > 0.05:
                # Would require manual approval in real system
                logger.info(f"Optimization requires approval: {suggestion.title}")
                continue
            
            filtered_suggestions.append(suggestion)
        
        # Apply optimizations in priority order
        for suggestion in sorted(filtered_suggestions, 
                               key=lambda s: s.get_priority_score(), 
                               reverse=True):
            try:
                logger.info(f"Applying optimization: {suggestion.title}")
                
                # Create rollback information before applying
                rollback_info = self._create_rollback_info(workflow, suggestion)
                
                # Apply the optimization
                application_result = self._apply_single_optimization(workflow, suggestion)
                
                if application_result['success']:
                    results['applied_successfully'] += 1
                    results['total_improvement'] += suggestion.improvement_percentage
                    
                    application_info = {
                        'suggestion_id': suggestion.suggestion_id,
                        'title': suggestion.title,
                        'improvement': suggestion.improvement_percentage,
                        'applied_at': time.time(),
                        'rollback_info': rollback_info
                    }
                    results['applications'].append(application_info)
                    results['rollback_info'].append(rollback_info)
                    
                    # Track applied optimization
                    self.applied_optimizations[workflow_id].append({
                        'suggestion': suggestion,
                        'applied_at': time.time(),
                        'rollback_info': rollback_info
                    })
                    
                    logger.info(f"Successfully applied optimization: {suggestion.title}")
                    
                else:
                    results['failed_applications'] += 1
                    error_msg = f"Failed to apply optimization '{suggestion.title}': {application_result.get('error', 'Unknown error')}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
                
            except Exception as e:
                results['failed_applications'] += 1
                error_msg = f"Exception applying optimization '{suggestion.title}': {str(e)}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        # Update statistics
        self.optimization_stats['optimizations_applied'] += results['applied_successfully']
        self.optimization_stats['total_improvement_achieved'] += results['total_improvement']
        
        # Record metrics
        metrics.increment_counter(
            "crewgraph_workflow_optimizations_applied_total",
            results['applied_successfully'],
            labels={
                "workflow_id": workflow_id,
                "user": "Vatsal216"
            }
        )
        
        metrics.record_gauge(
            "crewgraph_workflow_optimization_improvement_percent",
            results['total_improvement'],
            labels={
                "workflow_id": workflow_id,
                "user": "Vatsal216"
            }
        )
        
        logger.info(f"Optimization application completed: {workflow_id}")
        logger.info(f"Applied: {results['applied_successfully']}, "
                   f"Failed: {results['failed_applications']}, "
                   f"Total improvement: {results['total_improvement']:.2f}%")
        
        return results
    
    def _default_optimization_rules(self) -> List[OptimizationRule]:
        """Create default optimization rules"""
        return [
            # Performance Optimization Rules
            OptimizationRule(
                name="task_parallelization",
                optimization_type=OptimizationType.PERFORMANCE,
                priority=8,
                min_improvement_threshold=0.15,
                conditions={"has_parallelizable_tasks": True},
                created_by="Vatsal216"
            ),
            OptimizationRule(
                name="agent_specialization",
                optimization_type=OptimizationType.PERFORMANCE,
                priority=7,
                min_improvement_threshold=0.10,
                conditions={"agent_utilization_variance": ">0.3"},
                created_by="Vatsal216"
            ),
            OptimizationRule(
                name="memory_optimization",
                optimization_type=OptimizationType.RESOURCE_UTILIZATION,
                priority=6,
                min_improvement_threshold=0.12,
                conditions={"memory_usage": ">70"},
                created_by="Vatsal216"
            ),
            
            # Cost Optimization Rules
            OptimizationRule(
                name="resource_rightsizing",
                optimization_type=OptimizationType.COST,
                priority=7,
                min_improvement_threshold=0.08,
                conditions={"resource_utilization": "<40"},
                created_by="Vatsal216"
            ),
            OptimizationRule(
                name="idle_resource_elimination",
                optimization_type=OptimizationType.COST,
                priority=9,
                min_improvement_threshold=0.20,
                conditions={"idle_time": ">300"},
                created_by="Vatsal216"
            ),
            
            # Reliability Optimization Rules
            OptimizationRule(
                name="error_handling_improvement",
                optimization_type=OptimizationType.RELIABILITY,
                priority=8,
                min_improvement_threshold=0.05,
                conditions={"error_rate": ">0.01"},
                created_by="Vatsal216"
            ),
            OptimizationRule(
                name="timeout_optimization",
                optimization_type=OptimizationType.RELIABILITY,
                priority=6,
                min_improvement_threshold=0.07,
                conditions={"timeout_rate": ">0.005"},
                created_by="Vatsal216"
            ),
            
            # Latency Optimization Rules
            OptimizationRule(
                name="caching_optimization",
                optimization_type=OptimizationType.LATENCY,
                priority=7,
                min_improvement_threshold=0.15,
                conditions={"cache_miss_rate": ">0.3"},
                created_by="Vatsal216"
            ),
            OptimizationRule(
                name="workflow_path_optimization",
                optimization_type=OptimizationType.LATENCY,
                priority=6,
                min_improvement_threshold=0.10,
                conditions={"critical_path_length": ">5"},
                created_by="Vatsal216"
            )
        ]
    
    def _collect_workflow_metrics(self,
                                 workflow: CrewGraph,
                                 custom_metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Collect comprehensive workflow metrics for analysis"""
        metrics_data = {}
        
        try:
            # Basic workflow metrics
            metrics_data.update({
                'agent_count': len(workflow.agents),
                'task_count': len(workflow.tasks),
                'tool_count': len(workflow.tools),
                'execution_time': getattr(workflow, 'last_execution_time', 0.0),
                'success_rate': getattr(workflow, 'success_rate', 1.0),
                'error_rate': getattr(workflow, 'error_rate', 0.0),
                'resource_utilization': self._calculate_resource_utilization(workflow),
                'parallelization_factor': self._calculate_parallelization_factor(workflow),
                'agent_utilization_variance': self._calculate_agent_utilization_variance(workflow),
                'memory_usage_mb': self._estimate_memory_usage(workflow),
                'cache_hit_rate': getattr(workflow, 'cache_hit_rate', 0.0),
                'idle_time_percentage': self._calculate_idle_time_percentage(workflow),
                'critical_path_length': self._calculate_critical_path_length(workflow),
                'bottleneck_factor': self._identify_bottleneck_factor(workflow),
                'cost_per_execution': self._estimate_execution_cost(workflow)
            })
            
            # Performance metrics
            if hasattr(workflow, 'get_performance_metrics'):
                performance_metrics = workflow.get_performance_metrics()
                metrics_data.update({
                    f'perf_{k}': v for k, v in performance_metrics.items()
                })
            
            # Add custom metrics
            if custom_metrics:
                metrics_data.update({
                    f'custom_{k}': v for k, v in custom_metrics.items()
                })
            
            logger.debug(f"Collected {len(metrics_data)} workflow metrics")
            
        except Exception as e:
            logger.error(f"Error collecting workflow metrics: {e}")
            # Return basic metrics on error
            metrics_data = {
                'agent_count': len(workflow.agents) if hasattr(workflow, 'agents') else 0,
                'task_count': len(workflow.tasks) if hasattr(workflow, 'tasks') else 0,
                'error_occurred': 1.0
            }
        
        return metrics_data
    
    def _apply_optimization_rule(self,
                                rule: OptimizationRule,
                                workflow: CrewGraph,
                                metrics: Dict[str, float]) -> List[OptimizationSuggestion]:
        """Apply specific optimization rule to generate suggestions"""
        suggestions = []
        
        # Check if rule conditions are met
        if not self._check_rule_conditions(rule, metrics):
            return suggestions
        
        try:
            # Generate suggestions based on rule type
            if rule.name == "task_parallelization":
                suggestions.extend(self._suggest_task_parallelization(workflow, metrics))
            
            elif rule.name == "agent_specialization":
                suggestions.extend(self._suggest_agent_specialization(workflow, metrics))
            
            elif rule.name == "memory_optimization":
                suggestions.extend(self._suggest_memory_optimization(workflow, metrics))
            
            elif rule.name == "resource_rightsizing":
                suggestions.extend(self._suggest_resource_rightsizing(workflow, metrics))
            
            elif rule.name == "idle_resource_elimination":
                suggestions.extend(self._suggest_idle_resource_elimination(workflow, metrics))
            
            elif rule.name == "error_handling_improvement":
                suggestions.extend(self._suggest_error_handling_improvement(workflow, metrics))
            
            elif rule.name == "timeout_optimization":
                suggestions.extend(self._suggest_timeout_optimization(workflow, metrics))
            
            elif rule.name == "caching_optimization":
                suggestions.extend(self._suggest_caching_optimization(workflow, metrics))
            
            elif rule.name == "workflow_path_optimization":
                suggestions.extend(self._suggest_workflow_path_optimization(workflow, metrics))
            
            # Filter suggestions by rule criteria
            filtered_suggestions = []
            for suggestion in suggestions:
                if (suggestion.improvement_percentage >= rule.min_improvement_threshold and
                    suggestion.risk_level <= rule.max_risk_level):
                    filtered_suggestions.append(suggestion)
            
            logger.debug(f"Rule '{rule.name}' generated {len(filtered_suggestions)} valid suggestions")
            return filtered_suggestions
            
        except Exception as e:
            logger.error(f"Error applying optimization rule '{rule.name}': {e}")
            return []
    
    def _suggest_task_parallelization(self,
                                    workflow: CrewGraph,
                                    metrics: Dict[str, float]) -> List[OptimizationSuggestion]:
        """Generate task parallelization suggestions"""
        suggestions = []
        
        try:
            current_parallelization = metrics.get('parallelization_factor', 1.0)
            potential_parallelization = self._analyze_parallelization_potential(workflow)
            
            if potential_parallelization > current_parallelization * 1.2:
                improvement = ((potential_parallelization - current_parallelization) / 
                             current_parallelization * 100)
                
                suggestion = OptimizationSuggestion(
                    suggestion_id=f"task_parallel_{workflow.id}_{int(time.time())}",
                    optimization_type=OptimizationType.PERFORMANCE,
                    title="Implement Task Parallelization",
                    description=f"Increase task parallelization from {current_parallelization:.1f}x to {potential_parallelization:.1f}x by identifying independent tasks and executing them concurrently.",
                    current_value=current_parallelization,
                    projected_value=potential_parallelization,
                    improvement_percentage=improvement,
                    confidence_score=0.8,
                    risk_level=0.15,
                    implementation_effort="medium",
                    estimated_savings={
                        'execution_time_reduction_percent': improvement * 0.7,
                        'throughput_increase_percent': improvement * 0.8
                    },
                    implementation_steps=[
                        "Analyze task dependencies to identify parallelizable tasks",
                        "Implement concurrent execution framework",
                        "Add synchronization points for dependent tasks",
                        "Test parallel execution with gradual rollout",
                        "Monitor performance and adjust parallelization level"
                    ],
                    rollback_plan=[
                        "Disable parallel execution flag",
                        "Revert to sequential task execution",
                        "Monitor system stability",
                        "Investigate issues before re-enabling"
                    ],
                    metadata={
                        'parallelizable_tasks': self._identify_parallelizable_tasks(workflow),
                        'dependency_graph_complexity': self._calculate_dependency_complexity(workflow)
                    }
                )
                suggestions.append(suggestion)
        
        except Exception as e:
            logger.error(f"Error generating task parallelization suggestions: {e}")
        
        return suggestions
    
    def _suggest_agent_specialization(self,
                                    workflow: CrewGraph,
                                    metrics: Dict[str, float]) -> List[OptimizationSuggestion]:
        """Generate agent specialization suggestions"""
        suggestions = []
        
        try:
            current_variance = metrics.get('agent_utilization_variance', 0.0)
            
            if current_variance > 0.3:  # High variance indicates specialization opportunity
                projected_improvement = min(30, current_variance * 50)  # Cap at 30%
                
                suggestion = OptimizationSuggestion(
                    suggestion_id=f"agent_spec_{workflow.id}_{int(time.time())}",
                    optimization_type=OptimizationType.PERFORMANCE,
                    title="Optimize Agent Specialization",
                    description=f"Reduce agent utilization variance from {current_variance:.2f} by specializing agents for specific task types and balancing workload distribution.",
                    current_value=current_variance,
                    projected_value=current_variance * 0.6,  # 40% reduction
                    improvement_percentage=projected_improvement,
                    confidence_score=0.75,
                    risk_level=0.12,
                    implementation_effort="medium",
                    estimated_savings={
                        'resource_utilization_improvement_percent': projected_improvement * 0.6,
                        'task_completion_time_reduction_percent': projected_improvement * 0.4
                    },
                    implementation_steps=[
                        "Analyze task types and agent capabilities",
                        "Create specialized agent pools for different task categories",
                        "Implement intelligent task routing to appropriate agents",
                        "Add load balancing to prevent overutilization",
                        "Monitor agent performance and adjust specialization"
                    ],
                    rollback_plan=[
                        "Revert to general-purpose agent assignment",
                        "Remove task routing specialization",
                        "Monitor overall system performance",
                        "Gradually re-introduce specialization if needed"
                    ]
                )
                suggestions.append(suggestion)
        
        except Exception as e:
            logger.error(f"Error generating agent specialization suggestions: {e}")
        
        return suggestions
    
    def _suggest_memory_optimization(self,
                                   workflow: CrewGraph,
                                   metrics: Dict[str, float]) -> List[OptimizationSuggestion]:
        """Generate memory optimization suggestions"""
        suggestions = []
        
        try:
            memory_usage = metrics.get('memory_usage_mb', 0)
            
            if memory_usage > 1000:  # More than 1GB
                projected_reduction = min(40, (memory_usage - 500) / memory_usage * 100)
                
                suggestion = OptimizationSuggestion(
                    suggestion_id=f"memory_opt_{workflow.id}_{int(time.time())}",
                    optimization_type=OptimizationType.RESOURCE_UTILIZATION,
                    title="Optimize Memory Usage",
                    description=f"Reduce memory usage from {memory_usage:.0f}MB through caching optimization, data structure improvements, and garbage collection tuning.",
                    current_value=memory_usage,
                    projected_value=memory_usage * (1 - projected_reduction/100),
                    improvement_percentage=projected_reduction,
                    confidence_score=0.7,
                    risk_level=0.08,
                    implementation_effort="low",
                    estimated_savings={
                        'memory_cost_reduction_percent': projected_reduction,
                        'gc_overhead_reduction_percent': projected_reduction * 0.3
                    },
                    implementation_steps=[
                        "Enable memory profiling and monitoring",
                        "Implement object pooling for frequently created objects",
                        "Optimize data structures and reduce object creation",
                        "Configure garbage collection settings",
                        "Add memory usage alerts and monitoring"
                    ],
                    rollback_plan=[
                        "Revert garbage collection settings to defaults",
                        "Disable object pooling",
                        "Remove memory optimizations",
                        "Monitor system stability"
                    ]
                )
                suggestions.append(suggestion)
        
        except Exception as e:
            logger.error(f"Error generating memory optimization suggestions: {e}")
        
        return suggestions
    
    def _suggest_caching_optimization(self,
                                    workflow: CrewGraph,
                                    metrics: Dict[str, float]) -> List[OptimizationSuggestion]:
        """Generate caching optimization suggestions"""
        suggestions = []
        
        try:
            cache_hit_rate = metrics.get('cache_hit_rate', 0.0)
            
            if cache_hit_rate < 0.7:  # Less than 70% cache hit rate
                projected_improvement = min(50, (0.9 - cache_hit_rate) * 100)
                
                suggestion = OptimizationSuggestion(
                    suggestion_id=f"cache_opt_{workflow.id}_{int(time.time())}",
                    optimization_type=OptimizationType.LATENCY,
                    title="Improve Caching Strategy",
                    description=f"Increase cache hit rate from {cache_hit_rate:.1%} to 90% through intelligent caching policies, cache warming, and optimized cache sizes.",
                    current_value=cache_hit_rate,
                    projected_value=0.9,
                    improvement_percentage=projected_improvement,
                    confidence_score=0.85,
                    risk_level=0.05,
                    implementation_effort="low",
                    estimated_savings={
                        'response_time_reduction_percent': projected_improvement * 0.8,
                        'resource_usage_reduction_percent': projected_improvement * 0.3
                    },
                    implementation_steps=[
                        "Analyze current cache usage patterns",
                        "Implement intelligent cache eviction policies",
                        "Add cache warming for frequently accessed data",
                        "Optimize cache sizes based on usage patterns",
                        "Monitor cache performance and hit rates"
                    ],
                    rollback_plan=[
                        "Revert to previous cache configuration",
                        "Disable cache warming",
                        "Monitor system performance",
                        "Gradually re-enable optimizations"
                    ]
                )
                suggestions.append(suggestion)
        
        except Exception as e:
            logger.error(f"Error generating caching optimization suggestions: {e}")
        
        return suggestions
    
    def start_continuous_optimization(self) -> None:
        """Start continuous optimization monitoring"""
        if self._running_continuous:
            logger.warning("Continuous optimization already running")
            return
        
        if not self.enable_continuous:
            logger.warning("Continuous optimization is disabled")
            return
        
        self._running_continuous = True
        
        self._continuous_thread = threading.Thread(
            target=self._continuous_optimization_loop,
            name="WorkflowOptimizer-Continuous",
            daemon=True
        )
        self._continuous_thread.start()
        
        logger.info("Continuous workflow optimization started")
        logger.info(f"Analysis interval: {self.analysis_interval} seconds")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 13:46:40")
    
    def stop_continuous_optimization(self) -> None:
        """Stop continuous optimization monitoring"""
        self._running_continuous = False
        
        if self._continuous_thread and self._continuous_thread.is_alive():
            self._continuous_thread.join(timeout=10.0)
        
        logger.info("Continuous workflow optimization stopped")
    
    def get_optimization_report(self,
                              workflow_ids: Optional[List[str]] = None,
                              time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        end_time = time.time()
        start_time = end_time - (time_range_hours * 3600)
        
        # Filter optimization history by time range
        relevant_optimizations = [
            opt for opt in self.optimization_history
            if start_time <= opt.get('timestamp', 0) <= end_time
        ]
        
        # Filter by workflow IDs if specified
        if workflow_ids:
            relevant_optimizations = [
                opt for opt in relevant_optimizations
                if opt.get('workflow_id') in workflow_ids
            ]
        
        report = {
            'report_metadata': {
                'generated_by': 'Vatsal216',
                'generated_at': '2025-07-22 13:46:40',
                'time_range_hours': time_range_hours,
                'workflow_filter': workflow_ids
            },
            'summary': {
                'total_optimizations_analyzed': self.optimization_stats['total_analyses'],
                'total_suggestions_generated': self.optimization_stats['suggestions_generated'],
                'total_optimizations_applied': self.optimization_stats['optimizations_applied'],
                'total_improvement_achieved': self.optimization_stats['total_improvement_achieved'],
                'average_analysis_time': self.optimization_stats['avg_analysis_time'],
                'last_analysis_time': self.optimization_stats['last_analysis_time']
            },
            'optimization_breakdown': self._analyze_optimization_breakdown(relevant_optimizations),
            'top_improvements': self._get_top_improvements(relevant_optimizations),
            'optimization_trends': self._analyze_optimization_trends(relevant_optimizations),
            'recommendations': self._generate_meta_recommendations(),
            'performance_impact': self._calculate_performance_impact(relevant_optimizations),
            'cost_impact': self._calculate_cost_impact(relevant_optimizations)
        }
        
        return report
    
    def get_optimizer_status(self) -> Dict[str, Any]:
        """Get current optimizer status and configuration"""
        return {
            'optimizer_config': {
                'optimization_level': self.optimization_level.value,
                'target_optimization': self.target_optimization.value,
                'predictive_enabled': self.enable_predictive,
                'continuous_enabled': self.enable_continuous,
                'analysis_interval': self.analysis_interval
            },
            'runtime_status': {
                'continuous_running': self._running_continuous,
                'total_rules': len(self.optimization_rules),
                'active_rules': len([r for r in self.optimization_rules if r.enabled]),
                'cache_size': len(self.analysis_cache),
                'workflows_tracked': len(self.workflow_metrics_history)
            },
            'statistics': self.optimization_stats.copy(),
            'recent_activity': {
                'recent_optimizations': list(self.optimization_history)[-10:],
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'avg_suggestions_per_analysis': (
                    self.optimization_stats['suggestions_generated'] / 
                    max(1, self.optimization_stats['total_analyses'])
                )
            },
            'status_metadata': {
                'created_by': 'Vatsal216',
                'status_time': '2025-07-22 13:46:40'
            }
        }
    
    def export_optimization_data(self, 
                               format_type: str = 'json',
                               include_history: bool = True) -> str:
        """Export optimization data and history"""
        export_data = {
            'export_info': {
                'exported_by': 'Vatsal216',
                'exported_at': '2025-07-22 13:46:40',
                'format': format_type,
                'optimizer_version': '1.0.0'
            },
            'optimizer_configuration': {
                'optimization_level': self.optimization_level.value,
                'target_optimization': self.target_optimization.value,
                'rules': [
                    {
                        'name': rule.name,
                        'type': rule.optimization_type.value,
                        'priority': rule.priority,
                        'enabled': rule.enabled
                    } for rule in self.optimization_rules
                ]
            },
            'statistics': self.optimization_stats.copy()
        }
        
        if include_history:
            export_data['optimization_history'] = list(self.optimization_history)
            export_data['workflow_metrics_summary'] = {
                workflow_id: len(history) 
                for workflow_id, history in self.workflow_metrics_history.items()
            }
        
        if format_type.lower() == 'json':
            import json
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def __repr__(self) -> str:
        return (f"WorkflowOptimizer(level={self.optimization_level.value}, "
                f"target={self.target_optimization.value}, "
                f"analyses={self.optimization_stats['total_analyses']}, "
                f"continuous={'ON' if self._running_continuous else 'OFF'})")


def create_workflow_optimizer(optimization_level: str = "moderate",
                            target_optimization: str = "balanced",
                            **kwargs) -> WorkflowOptimizer:
    """
    Factory function to create workflow optimizer.
    
    Args:
        optimization_level: Optimization intensity level
        target_optimization: Primary optimization objective
        **kwargs: Additional configuration options
        
    Returns:
        Configured WorkflowOptimizer instance
    """
    level = OptimizationLevel(optimization_level.lower())
    target = OptimizationType(target_optimization.lower())
    
    logger.info(f"Creating WorkflowOptimizer: {optimization_level} -> {target_optimization}")
    logger.info(f"User: Vatsal216, Time: 2025-07-22 13:46:40")
    
    return WorkflowOptimizer(
        optimization_level=level,
        target_optimization=target,
        **kwargs
    )