"""
Pattern Analyzer for CrewGraph AI Intelligence Layer

This module provides workflow pattern recognition and analysis capabilities
to identify optimization opportunities and best practices.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime

from ..utils.logging import get_logger
from ..types import WorkflowId

logger = get_logger(__name__)


@dataclass
class WorkflowPattern:
    """Represents a recognized workflow pattern."""
    pattern_id: str
    pattern_type: str
    frequency: int
    average_performance: float
    optimization_potential: float
    characteristics: Dict[str, Any]
    examples: List[WorkflowId]
    recommendations: List[str]


class PatternAnalyzer:
    """
    Analyzes workflow execution patterns to identify optimization opportunities,
    best practices, and performance trends.
    """
    
    def __init__(self, min_pattern_frequency: int = 3):
        """
        Initialize the pattern analyzer.
        
        Args:
            min_pattern_frequency: Minimum frequency for a pattern to be significant
        """
        self.min_pattern_frequency = min_pattern_frequency
        self.discovered_patterns: Dict[str, WorkflowPattern] = {}
        self.workflow_fingerprints: Dict[WorkflowId, str] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info(f"PatternAnalyzer initialized with min_frequency={min_pattern_frequency}")
    
    def analyze_workflow_structure(self, workflow_definition: Dict[str, Any]) -> str:
        """
        Analyze workflow structure and generate a pattern fingerprint.
        
        Args:
            workflow_definition: Workflow structure definition
            
        Returns:
            Pattern fingerprint string
        """
        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])
        
        # Extract structural features
        features = {
            "task_count": len(tasks),
            "dependency_count": len(dependencies),
            "has_parallel_tasks": self._has_parallel_structure(tasks, dependencies),
            "has_loops": self._has_loops(dependencies),
            "task_types": self._extract_task_types(tasks),
            "complexity_score": self._calculate_complexity(tasks, dependencies)
        }
        
        # Create fingerprint from features
        fingerprint_data = f"{features['task_count']}-{features['dependency_count']}-"
        fingerprint_data += f"{'P' if features['has_parallel_tasks'] else 'S'}-"
        fingerprint_data += f"{'L' if features['has_loops'] else 'N'}-"
        fingerprint_data += f"{features['complexity_score']:.1f}-"
        fingerprint_data += "-".join(sorted(features['task_types']))
        
        # Generate hash for compact representation
        fingerprint = hashlib.md5(fingerprint_data.encode()).hexdigest()[:12]
        
        logger.debug(f"Generated workflow fingerprint: {fingerprint}")
        return fingerprint
    
    def record_execution(self, 
                        workflow_id: WorkflowId,
                        workflow_definition: Dict[str, Any],
                        execution_metrics: Dict[str, Any]) -> None:
        """
        Record workflow execution for pattern analysis.
        
        Args:
            workflow_id: Unique workflow identifier
            workflow_definition: Workflow structure
            execution_metrics: Execution performance metrics
        """
        fingerprint = self.analyze_workflow_structure(workflow_definition)
        self.workflow_fingerprints[workflow_id] = fingerprint
        
        execution_record = {
            "workflow_id": workflow_id,
            "fingerprint": fingerprint,
            "execution_time": execution_metrics.get("execution_time", 0),
            "success_rate": execution_metrics.get("success_rate", 1.0),
            "resource_usage": execution_metrics.get("resource_usage", {}),
            "error_count": execution_metrics.get("error_count", 0),
            "timestamp": datetime.now(),
            "workflow_definition": workflow_definition
        }
        
        self.execution_history.append(execution_record)
        
        # Trigger pattern discovery if we have enough data
        if len(self.execution_history) % 10 == 0:
            self.discover_patterns()
        
        logger.debug(f"Recorded execution for workflow {workflow_id} with fingerprint {fingerprint}")
    
    def discover_patterns(self) -> Dict[str, WorkflowPattern]:
        """
        Discover workflow patterns from execution history.
        
        Returns:
            Dictionary of discovered patterns
        """
        if len(self.execution_history) < self.min_pattern_frequency:
            return self.discovered_patterns
        
        # Group executions by fingerprint
        fingerprint_groups = defaultdict(list)
        for execution in self.execution_history:
            fingerprint = execution["fingerprint"]
            fingerprint_groups[fingerprint].append(execution)
        
        # Analyze each group for patterns
        new_patterns = {}
        for fingerprint, executions in fingerprint_groups.items():
            if len(executions) >= self.min_pattern_frequency:
                pattern = self._analyze_pattern_group(fingerprint, executions)
                if pattern:
                    new_patterns[pattern.pattern_id] = pattern
        
        self.discovered_patterns.update(new_patterns)
        
        logger.info(f"Discovered {len(new_patterns)} new patterns, total: {len(self.discovered_patterns)}")
        return self.discovered_patterns
    
    def get_optimization_recommendations(self, 
                                       workflow_definition: Dict[str, Any]) -> List[str]:
        """
        Get optimization recommendations based on recognized patterns.
        
        Args:
            workflow_definition: Workflow to analyze
            
        Returns:
            List of optimization recommendations
        """
        fingerprint = self.analyze_workflow_structure(workflow_definition)
        
        recommendations = []
        
        # Check if this workflow matches any known patterns
        matching_pattern = None
        for pattern in self.discovered_patterns.values():
            if self._patterns_match(fingerprint, pattern):
                matching_pattern = pattern
                break
        
        if matching_pattern:
            recommendations.extend(matching_pattern.recommendations)
            
            # Add pattern-specific recommendations
            if matching_pattern.optimization_potential > 30:
                recommendations.append(f"High optimization potential detected ({matching_pattern.optimization_potential:.1f}%)")
            
            if matching_pattern.average_performance < 0.7:
                recommendations.append("Similar workflows have shown performance issues - consider optimization")
        
        else:
            # General recommendations for unknown patterns
            recommendations.extend(self._get_general_recommendations(workflow_definition))
        
        # Add structural recommendations
        recommendations.extend(self._get_structural_recommendations(workflow_definition))
        
        return list(set(recommendations))  # Remove duplicates
    
    def identify_performance_bottlenecks(self, 
                                       workflow_definition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify potential performance bottlenecks in workflow structure.
        
        Args:
            workflow_definition: Workflow to analyze
            
        Returns:
            List of identified bottlenecks with details
        """
        bottlenecks = []
        
        tasks = workflow_definition.get("tasks", [])
        dependencies = workflow_definition.get("dependencies", [])
        
        # Check for sequential bottlenecks
        sequential_chains = self._find_sequential_chains(tasks, dependencies)
        for chain in sequential_chains:
            if len(chain) > 5:
                bottlenecks.append({
                    "type": "sequential_chain",
                    "severity": "high" if len(chain) > 10 else "medium",
                    "description": f"Long sequential chain of {len(chain)} tasks",
                    "affected_tasks": chain,
                    "recommendation": "Consider parallelizing independent tasks in the chain"
                })
        
        # Check for resource-intensive task clusters
        resource_intensive_tasks = [
            task for task in tasks 
            if task.get("resource_requirements", {}).get("memory", 0) > 1000
        ]
        
        if len(resource_intensive_tasks) > 2:
            bottlenecks.append({
                "type": "resource_bottleneck",
                "severity": "medium",
                "description": f"{len(resource_intensive_tasks)} resource-intensive tasks detected",
                "affected_tasks": [task.get("id", "unknown") for task in resource_intensive_tasks],
                "recommendation": "Consider scheduling resource-intensive tasks separately"
            })
        
        # Check for dependency bottlenecks
        dependency_counts = Counter()
        for dep in dependencies:
            dependency_counts[dep.get("target")] += 1
        
        high_dependency_tasks = [task_id for task_id, count in dependency_counts.items() if count > 3]
        if high_dependency_tasks:
            bottlenecks.append({
                "type": "dependency_bottleneck",
                "severity": "high",
                "description": f"Tasks with high dependency count: {high_dependency_tasks}",
                "affected_tasks": high_dependency_tasks,
                "recommendation": "Review dependencies to reduce coupling"
            })
        
        return bottlenecks
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of discovered patterns and analysis."""
        if not self.discovered_patterns:
            return {
                "total_patterns": 0,
                "total_executions_analyzed": len(self.execution_history),
                "message": "Insufficient data for pattern analysis"
            }
        
        # Calculate summary statistics
        total_optimizations = sum(1 for p in self.discovered_patterns.values() 
                                if p.optimization_potential > 20)
        
        avg_performance = sum(p.average_performance for p in self.discovered_patterns.values()) / len(self.discovered_patterns)
        
        most_common_pattern = max(self.discovered_patterns.values(), key=lambda p: p.frequency)
        
        return {
            "total_patterns": len(self.discovered_patterns),
            "total_executions_analyzed": len(self.execution_history),
            "patterns_with_optimization_potential": total_optimizations,
            "average_pattern_performance": avg_performance,
            "most_common_pattern": {
                "type": most_common_pattern.pattern_type,
                "frequency": most_common_pattern.frequency,
                "performance": most_common_pattern.average_performance
            },
            "analysis_coverage": len(set(self.workflow_fingerprints.values())),
            "last_analysis": datetime.now().isoformat()
        }
    
    def _has_parallel_structure(self, tasks: List[Dict], dependencies: List[Dict]) -> bool:
        """Check if workflow has parallel task execution opportunities."""
        if len(tasks) < 2:
            return False
        
        # Build dependency graph
        dependent_tasks = set()
        for dep in dependencies:
            dependent_tasks.add(dep.get("target"))
        
        # If there are tasks without dependencies, there's potential for parallelism
        independent_tasks = [task for task in tasks if task.get("id") not in dependent_tasks]
        return len(independent_tasks) > 1
    
    def _has_loops(self, dependencies: List[Dict]) -> bool:
        """Check if workflow has loops (cycles) in dependencies."""
        # Simple cycle detection - build adjacency list and check for back edges
        graph = defaultdict(list)
        for dep in dependencies:
            source = dep.get("source")
            target = dep.get("target")
            if source and target:
                graph[source].append(target)
        
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False
    
    def _extract_task_types(self, tasks: List[Dict]) -> List[str]:
        """Extract and categorize task types."""
        types = []
        for task in tasks:
            task_type = task.get("type", "unknown")
            # Normalize task type
            if "data" in task_type.lower():
                types.append("data")
            elif "compute" in task_type.lower() or "process" in task_type.lower():
                types.append("compute")
            elif "io" in task_type.lower() or "file" in task_type.lower():
                types.append("io")
            elif "ml" in task_type.lower() or "model" in task_type.lower():
                types.append("ml")
            else:
                types.append("general")
        
        return list(set(types))  # Return unique types
    
    def _calculate_complexity(self, tasks: List[Dict], dependencies: List[Dict]) -> float:
        """Calculate workflow complexity score."""
        task_count = len(tasks)
        dependency_count = len(dependencies)
        
        if task_count == 0:
            return 0.0
        
        # Base complexity from task and dependency counts
        base_complexity = task_count + (dependency_count * 0.5)
        
        # Normalize to 0-10 scale
        normalized = min(10.0, base_complexity / 2.0)
        
        return normalized
    
    def _analyze_pattern_group(self, fingerprint: str, executions: List[Dict]) -> Optional[WorkflowPattern]:
        """Analyze a group of similar executions to extract patterns."""
        if not executions:
            return None
        
        # Calculate performance metrics
        execution_times = [e["execution_time"] for e in executions]
        success_rates = [e["success_rate"] for e in executions]
        
        avg_performance = sum(success_rates) / len(success_rates)
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        # Calculate optimization potential
        time_variance = max(execution_times) - min(execution_times) if len(execution_times) > 1 else 0
        optimization_potential = min(50.0, (time_variance / avg_execution_time) * 100) if avg_execution_time > 0 else 0
        
        # Extract characteristics
        sample_workflow = executions[0]["workflow_definition"]
        characteristics = {
            "task_count": len(sample_workflow.get("tasks", [])),
            "dependency_count": len(sample_workflow.get("dependencies", [])),
            "average_execution_time": avg_execution_time,
            "execution_count": len(executions)
        }
        
        # Generate recommendations
        recommendations = []
        if optimization_potential > 25:
            recommendations.append("High execution time variance suggests optimization opportunities")
        if avg_performance < 0.8:
            recommendations.append("Success rate could be improved with error handling")
        if avg_execution_time > 60:
            recommendations.append("Long execution time - consider task parallelization")
        
        # Determine pattern type
        task_count = characteristics["task_count"]
        if task_count <= 3:
            pattern_type = "simple"
        elif task_count <= 10:
            pattern_type = "moderate"
        else:
            pattern_type = "complex"
        
        pattern = WorkflowPattern(
            pattern_id=fingerprint,
            pattern_type=pattern_type,
            frequency=len(executions),
            average_performance=avg_performance,
            optimization_potential=optimization_potential,
            characteristics=characteristics,
            examples=[e["workflow_id"] for e in executions[:5]],  # Store up to 5 examples
            recommendations=recommendations
        )
        
        return pattern
    
    def _patterns_match(self, fingerprint: str, pattern: WorkflowPattern) -> bool:
        """Check if a fingerprint matches a discovered pattern."""
        return fingerprint == pattern.pattern_id
    
    def _get_general_recommendations(self, workflow_definition: Dict[str, Any]) -> List[str]:
        """Get general optimization recommendations."""
        recommendations = []
        
        task_count = len(workflow_definition.get("tasks", []))
        dependency_count = len(workflow_definition.get("dependencies", []))
        
        if task_count > 5 and dependency_count < task_count * 0.3:
            recommendations.append("Consider parallel execution for independent tasks")
        
        if dependency_count > task_count:
            recommendations.append("High dependency ratio - review workflow structure")
        
        return recommendations
    
    def _get_structural_recommendations(self, workflow_definition: Dict[str, Any]) -> List[str]:
        """Get recommendations based on workflow structure."""
        recommendations = []
        
        tasks = workflow_definition.get("tasks", [])
        
        # Check for resource imbalance
        memory_requirements = [task.get("resource_requirements", {}).get("memory", 0) for task in tasks]
        if memory_requirements and max(memory_requirements) > sum(memory_requirements) / len(memory_requirements) * 3:
            recommendations.append("Memory requirements are imbalanced - consider resource leveling")
        
        return recommendations
    
    def _find_sequential_chains(self, tasks: List[Dict], dependencies: List[Dict]) -> List[List[str]]:
        """Find sequential chains in the workflow."""
        # Build dependency graph
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        
        for dep in dependencies:
            source = dep.get("source")
            target = dep.get("target")
            if source and target:
                graph[source].append(target)
                reverse_graph[target].append(source)
        
        # Find chains (paths with no branching)
        chains = []
        visited = set()
        
        for task in tasks:
            task_id = task.get("id")
            if task_id and task_id not in visited:
                # Start a new chain if this task has at most one predecessor
                if len(reverse_graph[task_id]) <= 1:
                    chain = self._build_chain(task_id, graph, visited)
                    if len(chain) > 1:
                        chains.append(chain)
        
        return chains
    
    def _build_chain(self, start_task: str, graph: Dict, visited: Set[str]) -> List[str]:
        """Build a sequential chain starting from a task."""
        chain = []
        current = start_task
        
        while current and current not in visited and len(graph[current]) <= 1:
            chain.append(current)
            visited.add(current)
            
            # Move to next task if it exists and has only one dependency
            next_tasks = graph[current]
            if next_tasks:
                current = next_tasks[0]
            else:
                break
        
        return chain