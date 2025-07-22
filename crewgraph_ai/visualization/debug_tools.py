"""
Debug Tools - General debugging utilities for CrewGraph AI

Provides workflow validation, configuration inspection, agent/task dependency
analysis, and performance bottleneck identification.
"""

import time
import inspect
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback
from collections import defaultdict, deque

try:
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from ..core.agents import AgentWrapper
from ..core.tasks import TaskWrapper
from ..core.state import SharedState
from ..tools.registry import ToolRegistry
from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError, ValidationError

logger = get_logger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue found during workflow analysis."""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'dependency', 'configuration', 'performance', etc.
    component: str  # Component where issue was found
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class PerformanceBottleneck:
    """Represents a performance bottleneck identified in the workflow."""
    component_type: str  # 'task', 'agent', 'tool', 'memory'
    component_id: str
    bottleneck_type: str  # 'slow_execution', 'high_memory', 'frequent_calls'
    severity: float  # 0.0 to 1.0
    impact_description: str
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class DebugTools:
    """
    Comprehensive debugging utilities for CrewGraph AI workflows.
    
    Provides workflow validation, configuration analysis, dependency checking,
    and performance bottleneck identification.
    """
    
    def __init__(self, output_dir: str = "debug_analysis"):
        """
        Initialize the DebugTools.
        
        Args:
            output_dir: Directory to save debug analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Analysis data
        self.validation_issues: List[ValidationIssue] = []
        self.performance_bottlenecks: List[PerformanceBottleneck] = []
        self.dependency_graph: Optional[nx.DiGraph] = None
        
        # Performance tracking
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.memory_usage: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("DebugTools initialized")
    
    def validate_workflow(self, 
                         agents: Dict[str, AgentWrapper],
                         tasks: Dict[str, TaskWrapper],
                         state: SharedState,
                         tool_registry: ToolRegistry) -> Dict[str, Any]:
        """
        Perform comprehensive workflow validation.
        
        Args:
            agents: Dictionary of workflow agents
            tasks: Dictionary of workflow tasks
            state: Shared state object
            tool_registry: Tool registry
            
        Returns:
            Validation report with issues and recommendations
        """
        self.validation_issues.clear()
        
        logger.info("Starting comprehensive workflow validation")
        
        # Validate agents
        self._validate_agents(agents)
        
        # Validate tasks
        self._validate_tasks(tasks, agents, tool_registry)
        
        # Validate dependencies
        self._validate_dependencies(tasks)
        
        # Validate state configuration
        self._validate_state(state)
        
        # Validate tool registry
        self._validate_tools(tool_registry)
        
        # Generate validation report
        report = self._generate_validation_report()
        
        logger.info(f"Workflow validation completed. Found {len(self.validation_issues)} issues")
        return report
    
    def analyze_dependencies(self, tasks: Dict[str, TaskWrapper]) -> Dict[str, Any]:
        """
        Analyze task dependencies and identify potential issues.
        
        Args:
            tasks: Dictionary of workflow tasks
            
        Returns:
            Dependency analysis report
        """
        logger.info("Analyzing task dependencies")
        
        # Build dependency graph
        self.dependency_graph = self._build_dependency_graph(tasks)
        
        analysis = {
            "total_tasks": len(tasks),
            "total_dependencies": self.dependency_graph.number_of_edges(),
            "dependency_analysis": {},
            "issues": [],
            "recommendations": []
        }
        
        # Check for circular dependencies
        circular_deps = self._detect_circular_dependencies()
        if circular_deps:
            analysis["issues"].append({
                "type": "circular_dependencies",
                "cycles": circular_deps,
                "severity": "error"
            })
        
        # Identify critical path
        critical_path = self._find_critical_path()
        analysis["dependency_analysis"]["critical_path"] = critical_path
        
        # Identify isolated tasks
        isolated_tasks = self._find_isolated_tasks()
        analysis["dependency_analysis"]["isolated_tasks"] = isolated_tasks
        
        # Calculate dependency depth
        dependency_depths = self._calculate_dependency_depths()
        analysis["dependency_analysis"]["dependency_depths"] = dependency_depths
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_dependency_recommendations(analysis)
        
        return analysis
    
    def inspect_configuration(self, 
                            workflow_config: Dict[str, Any],
                            agent_configs: Optional[Dict[str, Any]] = None,
                            task_configs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Inspect and validate workflow configuration.
        
        Args:
            workflow_config: Main workflow configuration
            agent_configs: Agent-specific configurations
            task_configs: Task-specific configurations
            
        Returns:
            Configuration inspection report
        """
        logger.info("Inspecting workflow configuration")
        
        inspection = {
            "timestamp": datetime.now().isoformat(),
            "workflow_config": self._inspect_workflow_config(workflow_config),
            "agent_configs": self._inspect_agent_configs(agent_configs or {}),
            "task_configs": self._inspect_task_configs(task_configs or {}),
            "configuration_issues": [],
            "recommendations": []
        }
        
        # Validate configuration values
        config_issues = self._validate_configuration_values(workflow_config, agent_configs, task_configs)
        inspection["configuration_issues"].extend(config_issues)
        
        # Generate configuration recommendations
        inspection["recommendations"] = self._generate_config_recommendations(inspection)
        
        return inspection
    
    def identify_performance_bottlenecks(self, 
                                       execution_data: Dict[str, Any],
                                       threshold_percentile: float = 0.95) -> List[PerformanceBottleneck]:
        """
        Identify performance bottlenecks in workflow execution.
        
        Args:
            execution_data: Execution performance data
            threshold_percentile: Percentile threshold for identifying bottlenecks
            
        Returns:
            List of identified performance bottlenecks
        """
        logger.info("Identifying performance bottlenecks")
        
        self.performance_bottlenecks.clear()
        
        # Analyze execution times
        self._analyze_execution_times(execution_data, threshold_percentile)
        
        # Analyze memory usage
        self._analyze_memory_bottlenecks(execution_data)
        
        # Analyze call frequency
        self._analyze_call_frequency_bottlenecks(execution_data)
        
        # Analyze resource utilization
        self._analyze_resource_bottlenecks(execution_data)
        
        return self.performance_bottlenecks
    
    def generate_debug_report(self, 
                            workflow_data: Dict[str, Any],
                            include_visualizations: bool = True) -> str:
        """
        Generate comprehensive debug report.
        
        Args:
            workflow_data: Complete workflow data for analysis
            include_visualizations: Whether to include visual diagrams
            
        Returns:
            Path to generated debug report file
        """
        logger.info("Generating comprehensive debug report")
        
        report = {
            "report_metadata": {
                "timestamp": datetime.now().isoformat(),
                "crewgraph_version": "1.0.0",
                "python_version": sys.version,
                "analysis_scope": list(workflow_data.keys())
            },
            "validation_summary": {
                "total_issues": len(self.validation_issues),
                "error_count": len([i for i in self.validation_issues if i.severity == "error"]),
                "warning_count": len([i for i in self.validation_issues if i.severity == "warning"]),
                "info_count": len([i for i in self.validation_issues if i.severity == "info"])
            },
            "validation_issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "component": issue.component,
                    "message": issue.message,
                    "details": issue.details,
                    "suggestions": issue.suggestions
                }
                for issue in self.validation_issues
            ],
            "performance_analysis": {
                "bottlenecks_count": len(self.performance_bottlenecks),
                "bottlenecks": [
                    {
                        "component_type": b.component_type,
                        "component_id": b.component_id,
                        "bottleneck_type": b.bottleneck_type,
                        "severity": b.severity,
                        "impact_description": b.impact_description,
                        "recommendations": b.recommendations,
                        "metrics": b.metrics
                    }
                    for b in self.performance_bottlenecks
                ]
            },
            "system_information": self._gather_system_information(),
            "recommendations": self._generate_overall_recommendations()
        }
        
        # Add dependency analysis if available
        if self.dependency_graph:
            report["dependency_analysis"] = self._analyze_dependency_graph()
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_filename = f"debug_report_{timestamp}.json"
        json_filepath = self.output_dir / json_filename
        
        with open(json_filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # HTML report
        html_filename = f"debug_report_{timestamp}.html"
        html_filepath = self.output_dir / html_filename
        self._generate_html_debug_report(report, html_filepath)
        
        # Generate visualizations if requested
        if include_visualizations and VISUALIZATION_AVAILABLE:
            self._generate_debug_visualizations(workflow_data, timestamp)
        
        logger.info(f"Debug report generated: {html_filepath}")
        return str(html_filepath)
    
    def trace_execution_path(self, 
                           execution_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Trace and analyze execution path through the workflow.
        
        Args:
            execution_trace: List of execution events
            
        Returns:
            Execution path analysis
        """
        logger.info("Tracing execution path")
        
        if not execution_trace:
            return {"error": "No execution trace provided"}
        
        # Build execution timeline
        timeline = sorted(execution_trace, key=lambda x: x.get('timestamp', ''))
        
        # Analyze execution flow
        flow_analysis = {
            "total_events": len(timeline),
            "execution_start": timeline[0].get('timestamp') if timeline else None,
            "execution_end": timeline[-1].get('timestamp') if timeline else None,
            "unique_components": len(set(event.get('node_id', '') for event in timeline)),
            "execution_path": [event.get('node_id', '') for event in timeline],
            "branching_points": [],
            "merge_points": [],
            "error_points": []
        }
        
        # Identify branching and merge points
        component_counts = defaultdict(int)
        for event in timeline:
            node_id = event.get('node_id', '')
            component_counts[node_id] += 1
            
            # Check for errors
            if event.get('event_type') == 'error':
                flow_analysis["error_points"].append({
                    "node_id": node_id,
                    "timestamp": event.get('timestamp'),
                    "error_message": event.get('message', '')
                })
        
        # Identify components with multiple visits (potential loops or retries)
        flow_analysis["revisited_components"] = {
            node_id: count for node_id, count in component_counts.items() if count > 1
        }
        
        return flow_analysis
    
    def _validate_agents(self, agents: Dict[str, AgentWrapper]) -> None:
        """Validate agent configurations."""
        for agent_id, agent in agents.items():
            try:
                # Check if agent has required attributes
                if not hasattr(agent, 'crew_agent') or agent.crew_agent is None:
                    self._add_validation_issue(
                        "error", "configuration", f"agent_{agent_id}",
                        "Agent missing crew_agent configuration",
                        suggestions=["Ensure agent is properly initialized with CrewAI agent"]
                    )
                
                # Check agent name consistency
                if agent.name != agent_id:
                    self._add_validation_issue(
                        "warning", "configuration", f"agent_{agent_id}",
                        f"Agent name '{agent.name}' doesn't match agent ID '{agent_id}'",
                        suggestions=["Consider using consistent naming"]
                    )
                
                # Check if agent has state access
                if not hasattr(agent, 'state') or agent.state is None:
                    self._add_validation_issue(
                        "warning", "configuration", f"agent_{agent_id}",
                        "Agent doesn't have access to shared state",
                        suggestions=["Ensure agent is connected to workflow state"]
                    )
                
            except Exception as e:
                self._add_validation_issue(
                    "error", "configuration", f"agent_{agent_id}",
                    f"Error validating agent: {str(e)}"
                )
    
    def _validate_tasks(self, 
                       tasks: Dict[str, TaskWrapper],
                       agents: Dict[str, AgentWrapper],
                       tool_registry: ToolRegistry) -> None:
        """Validate task configurations."""
        for task_id, task in tasks.items():
            try:
                # Check task description
                if not task.description or len(task.description.strip()) < 10:
                    self._add_validation_issue(
                        "warning", "configuration", f"task_{task_id}",
                        "Task has insufficient description",
                        suggestions=["Provide detailed task description for better execution"]
                    )
                
                # Check agent assignment
                if hasattr(task, 'assigned_agent') and task.assigned_agent:
                    agent_name = task.assigned_agent.name
                    if agent_name not in agents:
                        self._add_validation_issue(
                            "error", "dependency", f"task_{task_id}",
                            f"Task assigned to non-existent agent '{agent_name}'",
                            suggestions=["Ensure agent exists before assignment"]
                        )
                
                # Check tool availability
                if hasattr(task, 'tools') and task.tools:
                    for tool in task.tools:
                        tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                        if not tool_registry.get_tool(tool_name):
                            self._add_validation_issue(
                                "warning", "dependency", f"task_{task_id}",
                                f"Task references unavailable tool '{tool_name}'",
                                suggestions=["Ensure tool is registered before use"]
                            )
                
                # Check dependencies
                if hasattr(task, 'dependencies') and task.dependencies:
                    for dep in task.dependencies:
                        if dep not in tasks:
                            self._add_validation_issue(
                                "error", "dependency", f"task_{task_id}",
                                f"Task depends on non-existent task '{dep}'",
                                suggestions=["Remove invalid dependency or create missing task"]
                            )
                
            except Exception as e:
                self._add_validation_issue(
                    "error", "configuration", f"task_{task_id}",
                    f"Error validating task: {str(e)}"
                )
    
    def _validate_dependencies(self, tasks: Dict[str, TaskWrapper]) -> None:
        """Validate task dependencies for circular references and validity."""
        # Build dependency graph for validation
        graph = nx.DiGraph()
        
        for task_id, task in tasks.items():
            graph.add_node(task_id)
            if hasattr(task, 'dependencies') and task.dependencies:
                for dep in task.dependencies:
                    if dep in tasks:  # Only add valid dependencies
                        graph.add_edge(dep, task_id)
        
        # Check for circular dependencies
        try:
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                self._add_validation_issue(
                    "error", "dependency", "workflow",
                    f"Circular dependency detected: {' -> '.join(cycle + [cycle[0]])}",
                    details={"cycle": cycle},
                    suggestions=["Break circular dependency by removing or restructuring dependencies"]
                )
        except Exception as e:
            self._add_validation_issue(
                "warning", "dependency", "workflow",
                f"Could not analyze dependencies: {str(e)}"
            )
    
    def _validate_state(self, state: SharedState) -> None:
        """Validate shared state configuration."""
        try:
            # Check if state has memory backend
            if not hasattr(state, 'memory') or state.memory is None:
                self._add_validation_issue(
                    "warning", "configuration", "state",
                    "Shared state has no memory backend configured",
                    suggestions=["Configure memory backend for state persistence"]
                )
            
            # Check state access patterns
            if hasattr(state, '_access_log'):
                if len(state._access_log) == 0:
                    self._add_validation_issue(
                        "info", "usage", "state",
                        "Shared state has not been accessed yet"
                    )
                
        except Exception as e:
            self._add_validation_issue(
                "error", "configuration", "state",
                f"Error validating state: {str(e)}"
            )
    
    def _validate_tools(self, tool_registry: ToolRegistry) -> None:
        """Validate tool registry configuration."""
        try:
            # Check if tools are registered
            if hasattr(tool_registry, '_tools'):
                tool_count = len(tool_registry._tools)
                if tool_count == 0:
                    self._add_validation_issue(
                        "warning", "configuration", "tools",
                        "No tools registered in tool registry",
                        suggestions=["Register tools if workflow requires them"]
                    )
                elif tool_count > 50:
                    self._add_validation_issue(
                        "warning", "performance", "tools",
                        f"Large number of tools registered ({tool_count})",
                        suggestions=["Consider tool organization and lazy loading"]
                    )
            
        except Exception as e:
            self._add_validation_issue(
                "error", "configuration", "tools",
                f"Error validating tools: {str(e)}"
            )
    
    def _build_dependency_graph(self, tasks: Dict[str, TaskWrapper]) -> nx.DiGraph:
        """Build NetworkX graph from task dependencies."""
        graph = nx.DiGraph()
        
        # Add all tasks as nodes
        for task_id in tasks.keys():
            graph.add_node(task_id)
        
        # Add dependency edges
        for task_id, task in tasks.items():
            if hasattr(task, 'dependencies') and task.dependencies:
                for dep in task.dependencies:
                    if dep in tasks:
                        graph.add_edge(dep, task_id)
        
        return graph
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the workflow."""
        if not self.dependency_graph:
            return []
        
        try:
            return list(nx.simple_cycles(self.dependency_graph))
        except Exception:
            return []
    
    def _find_critical_path(self) -> List[str]:
        """Find the critical path through the dependency graph."""
        if not self.dependency_graph:
            return []
        
        try:
            # Find longest path (critical path)
            if nx.is_directed_acyclic_graph(self.dependency_graph):
                return nx.dag_longest_path(self.dependency_graph)
            else:
                return []
        except Exception:
            return []
    
    def _find_isolated_tasks(self) -> List[str]:
        """Find tasks with no dependencies or dependents."""
        if not self.dependency_graph:
            return []
        
        isolated = []
        for node in self.dependency_graph.nodes():
            if self.dependency_graph.in_degree(node) == 0 and self.dependency_graph.out_degree(node) == 0:
                isolated.append(node)
        
        return isolated
    
    def _calculate_dependency_depths(self) -> Dict[str, int]:
        """Calculate dependency depth for each task."""
        if not self.dependency_graph:
            return {}
        
        depths = {}
        try:
            for node in nx.topological_sort(self.dependency_graph):
                if self.dependency_graph.in_degree(node) == 0:
                    depths[node] = 0
                else:
                    max_parent_depth = max(depths[parent] for parent in self.dependency_graph.predecessors(node))
                    depths[node] = max_parent_depth + 1
        except Exception:
            # Fallback for non-DAG
            for node in self.dependency_graph.nodes():
                depths[node] = 0
        
        return depths
    
    def _add_validation_issue(self, 
                            severity: str,
                            category: str,
                            component: str,
                            message: str,
                            details: Optional[Dict[str, Any]] = None,
                            suggestions: Optional[List[str]] = None) -> None:
        """Add a validation issue to the list."""
        issue = ValidationIssue(
            severity=severity,
            category=category,
            component=component,
            message=message,
            details=details or {},
            suggestions=suggestions or []
        )
        self.validation_issues.append(issue)
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        issues_by_severity = defaultdict(list)
        issues_by_category = defaultdict(list)
        
        for issue in self.validation_issues:
            issues_by_severity[issue.severity].append(issue)
            issues_by_category[issue.category].append(issue)
        
        return {
            "summary": {
                "total_issues": len(self.validation_issues),
                "errors": len(issues_by_severity["error"]),
                "warnings": len(issues_by_severity["warning"]),
                "info": len(issues_by_severity["info"])
            },
            "issues_by_severity": dict(issues_by_severity),
            "issues_by_category": dict(issues_by_category),
            "all_issues": self.validation_issues,
            "recommendations": self._generate_validation_recommendations()
        }
    
    def _generate_validation_recommendations(self) -> List[str]:
        """Generate recommendations based on validation issues."""
        recommendations = []
        
        error_count = len([i for i in self.validation_issues if i.severity == "error"])
        if error_count > 0:
            recommendations.append(f"Fix {error_count} critical errors before deployment")
        
        dependency_issues = len([i for i in self.validation_issues if i.category == "dependency"])
        if dependency_issues > 0:
            recommendations.append("Review task dependencies for consistency")
        
        config_issues = len([i for i in self.validation_issues if i.category == "configuration"])
        if config_issues > 0:
            recommendations.append("Review component configurations")
        
        return recommendations
    
    def _analyze_execution_times(self, execution_data: Dict[str, Any], threshold_percentile: float) -> None:
        """Analyze execution times for bottlenecks."""
        # Implementation would analyze execution time data
        # This is a simplified version
        pass
    
    def _analyze_memory_bottlenecks(self, execution_data: Dict[str, Any]) -> None:
        """Analyze memory usage for bottlenecks."""
        # Implementation would analyze memory usage patterns
        pass
    
    def _analyze_call_frequency_bottlenecks(self, execution_data: Dict[str, Any]) -> None:
        """Analyze call frequency for bottlenecks."""
        # Implementation would analyze call patterns
        pass
    
    def _analyze_resource_bottlenecks(self, execution_data: Dict[str, Any]) -> None:
        """Analyze resource utilization for bottlenecks."""
        # Implementation would analyze resource usage
        pass
    
    def _inspect_workflow_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect workflow configuration."""
        return {
            "config_keys": list(config.keys()),
            "config_size": len(config),
            "has_memory_config": "memory_backend" in config,
            "has_logging_config": any(key.startswith("log") for key in config.keys())
        }
    
    def _inspect_agent_configs(self, configs: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect agent configurations."""
        return {
            "agent_count": len(configs),
            "agents": list(configs.keys()),
            "common_config_keys": self._find_common_keys(configs.values()) if configs else []
        }
    
    def _inspect_task_configs(self, configs: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect task configurations."""
        return {
            "task_count": len(configs),
            "tasks": list(configs.keys()),
            "common_config_keys": self._find_common_keys(configs.values()) if configs else []
        }
    
    def _find_common_keys(self, config_list: List[Dict[str, Any]]) -> List[str]:
        """Find common keys across configuration dictionaries."""
        if not config_list:
            return []
        
        common_keys = set(config_list[0].keys())
        for config in config_list[1:]:
            common_keys &= set(config.keys())
        
        return list(common_keys)
    
    def _validate_configuration_values(self, 
                                     workflow_config: Dict[str, Any],
                                     agent_configs: Optional[Dict[str, Any]],
                                     task_configs: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate configuration values."""
        issues = []
        
        # Check for common configuration issues
        if "timeout" in workflow_config:
            timeout = workflow_config["timeout"]
            if isinstance(timeout, (int, float)) and timeout <= 0:
                issues.append({
                    "severity": "error",
                    "message": "Timeout value must be positive",
                    "component": "workflow_config"
                })
        
        return issues
    
    def _generate_config_recommendations(self, inspection: Dict[str, Any]) -> List[str]:
        """Generate configuration recommendations."""
        recommendations = []
        
        if not inspection["workflow_config"]["has_memory_config"]:
            recommendations.append("Consider configuring memory backend for better performance")
        
        if not inspection["workflow_config"]["has_logging_config"]:
            recommendations.append("Configure logging for better debugging")
        
        return recommendations
    
    def _generate_dependency_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate dependency-specific recommendations."""
        recommendations = []
        
        if analysis["issues"]:
            recommendations.append("Resolve dependency issues before deployment")
        
        isolated_count = len(analysis["dependency_analysis"].get("isolated_tasks", []))
        if isolated_count > 0:
            recommendations.append(f"Review {isolated_count} isolated tasks for workflow integration")
        
        return recommendations
    
    def _gather_system_information(self) -> Dict[str, Any]:
        """Gather system information for debugging."""
        import platform
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "hostname": platform.node(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations based on all analyses."""
        recommendations = []
        
        if self.validation_issues:
            error_count = len([i for i in self.validation_issues if i.severity == "error"])
            if error_count > 0:
                recommendations.append(f"Address {error_count} critical validation errors")
        
        if self.performance_bottlenecks:
            high_severity_count = len([b for b in self.performance_bottlenecks if b.severity > 0.7])
            if high_severity_count > 0:
                recommendations.append(f"Optimize {high_severity_count} high-severity performance bottlenecks")
        
        return recommendations
    
    def _analyze_dependency_graph(self) -> Dict[str, Any]:
        """Analyze the dependency graph structure."""
        if not self.dependency_graph:
            return {}
        
        return {
            "node_count": self.dependency_graph.number_of_nodes(),
            "edge_count": self.dependency_graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self.dependency_graph),
            "density": nx.density(self.dependency_graph),
            "average_clustering": nx.average_clustering(self.dependency_graph.to_undirected()),
            "weakly_connected_components": nx.number_weakly_connected_components(self.dependency_graph)
        }
    
    def _generate_html_debug_report(self, report: Dict[str, Any], filepath: Path) -> None:
        """Generate HTML debug report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CrewGraph AI Debug Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }}
                .error {{ border-left-color: #e74c3c; }}
                .warning {{ border-left-color: #f39c12; }}
                .success {{ border-left-color: #27ae60; }}
                .metric {{ background-color: #ecf0f1; padding: 10px; margin: 5px 0; }}
                .issue {{ margin: 10px 0; padding: 10px; }}
                .issue.error {{ background-color: #fadbd8; }}
                .issue.warning {{ background-color: #fdeaa7; }}
                .issue.info {{ background-color: #d6eaf8; }}
                pre {{ background-color: #f8f9fa; padding: 15px; overflow-x: auto; }}
                .recommendations {{ background-color: #e8f5e8; padding: 15px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 CrewGraph AI Debug Report</h1>
                <p>Generated: {report['report_metadata']['timestamp']}</p>
                <p>Version: {report['report_metadata']['crewgraph_version']}</p>
            </div>
            
            <div class="section {'error' if report['validation_summary']['error_count'] > 0 else 'success'}">
                <h2>📋 Validation Summary</h2>
                <div class="metric">Total Issues: {report['validation_summary']['total_issues']}</div>
                <div class="metric">Errors: {report['validation_summary']['error_count']}</div>
                <div class="metric">Warnings: {report['validation_summary']['warning_count']}</div>
                <div class="metric">Info: {report['validation_summary']['info_count']}</div>
            </div>
            
            <div class="section">
                <h2>⚡ Performance Analysis</h2>
                <div class="metric">Bottlenecks Found: {report['performance_analysis']['bottlenecks_count']}</div>
            </div>
            
            <div class="section">
                <h2>🔧 Recommendations</h2>
                <div class="recommendations">
                    {'<br>'.join(f'• {rec}' for rec in report['recommendations'])}
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Detailed Analysis</h2>
                <pre>{json.dumps(report, indent=2, default=str)[:5000]}...</pre>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
    
    def _generate_debug_visualizations(self, workflow_data: Dict[str, Any], timestamp: str) -> None:
        """Generate debug visualizations."""
        if not VISUALIZATION_AVAILABLE:
            return
        
        # This would generate various debug visualizations
        # Implementation depends on available workflow data
        pass