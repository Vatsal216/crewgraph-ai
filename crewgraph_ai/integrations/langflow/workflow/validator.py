"""
Workflow Validator for CrewGraph AI + Langflow Integration

This module provides validation functionality for workflows and components.

Created by: Vatsal216
Date: 2025-07-23
"""

import ast
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime

# Import CrewGraph components
try:
    from ....utils.logging import get_logger
    from ....utils.exceptions import ValidationError
except ImportError:
    def get_logger(name):
        import logging
        return logging.getLogger(name)
    
    class ValidationError(Exception):
        pass

logger = get_logger(__name__)


class WorkflowValidator:
    """
    Validator for CrewGraph AI + Langflow Integration
    
    This class provides comprehensive validation for:
    - Langflow workflow data
    - CrewGraph workflow definitions
    - Component code and configurations
    - Workflow execution compatibility
    """
    
    def __init__(self):
        self.supported_node_types = {
            "CrewGraphAgent", "CrewGraphTask", "CrewGraphTool", 
            "CrewGraphMemory", "ConditionalNode", "LoopNode", "ParallelNode"
        }
        self.required_python_modules = {
            "crewai", "langchain", "pydantic", "typing"
        }
        self.security_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'import\s+sys',
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__\s*\(',
            r'open\s*\(',
            r'file\s*\(',
        ]
    
    async def validate_langflow_data(self, langflow_data: Dict[str, Any]) -> List[str]:
        """
        Validate Langflow workflow data for import compatibility
        
        Args:
            langflow_data: Langflow workflow data
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        try:
            logger.info("Validating Langflow workflow data")
            
            # Basic structure validation
            structure_issues = self._validate_basic_structure(langflow_data)
            issues.extend(structure_issues)
            
            # Node validation
            if "flow" in langflow_data and "nodes" in langflow_data["flow"]:
                for node in langflow_data["flow"]["nodes"]:
                    node_issues = self._validate_langflow_node(node)
                    issues.extend(node_issues)
            
            # Edge validation
            if "flow" in langflow_data:
                edge_issues = self._validate_langflow_edges(langflow_data["flow"])
                issues.extend(edge_issues)
            
            # Workflow logic validation
            logic_issues = await self._validate_workflow_logic(langflow_data)
            issues.extend(logic_issues)
            
            # Compatibility validation
            compat_issues = self._validate_crewgraph_compatibility(langflow_data)
            issues.extend(compat_issues)
            
            logger.info(f"Langflow validation completed: {len(issues)} issues found")
            
        except Exception as e:
            logger.error(f"Langflow validation failed: {e}")
            issues.append(f"Validation error: {str(e)}")
        
        return issues
    
    def _validate_basic_structure(self, data: Dict[str, Any]) -> List[str]:
        """Validate basic workflow structure"""
        issues = []
        
        # Required top-level fields
        required_fields = ["flow"]
        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")
        
        # Flow structure
        if "flow" in data:
            flow = data["flow"]
            if not isinstance(flow, dict):
                issues.append("Flow must be a dictionary")
            else:
                if "nodes" not in flow:
                    issues.append("Flow missing nodes array")
                elif not isinstance(flow["nodes"], list):
                    issues.append("Flow nodes must be a list")
                
                if "edges" not in flow:
                    issues.append("Flow missing edges array")
                elif not isinstance(flow["edges"], list):
                    issues.append("Flow edges must be a list")
        
        return issues
    
    def _validate_langflow_node(self, node: Dict[str, Any]) -> List[str]:
        """Validate a single Langflow node"""
        issues = []
        
        # Required node fields
        required_fields = ["id", "type", "data"]
        for field in required_fields:
            if field not in node:
                issues.append(f"Node missing required field: {field}")
        
        # Node ID validation
        if "id" in node:
            node_id = node["id"]
            if not isinstance(node_id, str) or not node_id.strip():
                issues.append(f"Node ID must be a non-empty string")
            elif not re.match(r'^[a-zA-Z0-9_-]+$', node_id):
                issues.append(f"Node ID '{node_id}' contains invalid characters")
        
        # Node type validation
        if "type" in node:
            node_type = node["type"]
            if node_type not in self.supported_node_types:
                issues.append(f"Unsupported node type: {node_type}")
        
        # Node data validation
        if "data" in node:
            data_issues = self._validate_node_data(node["data"], node.get("type"))
            issues.extend(data_issues)
        
        # Position validation
        if "position" in node:
            pos = node["position"]
            if not isinstance(pos, dict) or "x" not in pos or "y" not in pos:
                issues.append(f"Node {node.get('id')} has invalid position format")
            elif not all(isinstance(v, (int, float)) for v in [pos["x"], pos["y"]]):
                issues.append(f"Node {node.get('id')} position coordinates must be numeric")
        
        return issues
    
    def _validate_node_data(self, data: Dict[str, Any], node_type: Optional[str]) -> List[str]:
        """Validate node data based on type"""
        issues = []
        
        if not isinstance(data, dict):
            issues.append("Node data must be a dictionary")
            return issues
        
        # Type-specific validation
        if node_type == "CrewGraphAgent":
            inputs = data.get("inputs", {})
            if not inputs.get("role", {}).get("value"):
                issues.append("Agent node missing role")
            if not inputs.get("goal", {}).get("value"):
                issues.append("Agent node missing goal")
            
            # Validate LLM model
            llm_model = inputs.get("llm_model", {}).get("value")
            if llm_model and not isinstance(llm_model, str):
                issues.append("Agent LLM model must be a string")
        
        elif node_type == "CrewGraphTask":
            inputs = data.get("inputs", {})
            if not inputs.get("description", {}).get("value"):
                issues.append("Task node missing description")
            
            # Validate expected output
            expected = inputs.get("expected_output", {}).get("value", "")
            if expected and len(expected) > 1000:
                issues.append("Task expected output is too long (max 1000 characters)")
        
        elif node_type == "CrewGraphTool":
            inputs = data.get("inputs", {})
            tool_name = inputs.get("tool_name", {}).get("value")
            if not tool_name:
                issues.append("Tool node missing tool name")
            elif not isinstance(tool_name, str):
                issues.append("Tool name must be a string")
            
            # Validate tool configuration
            config = inputs.get("tool_config", {}).get("value", {})
            if config and not isinstance(config, dict):
                issues.append("Tool configuration must be a dictionary")
        
        return issues
    
    def _validate_langflow_edges(self, flow: Dict[str, Any]) -> List[str]:
        """Validate workflow edges"""
        issues = []
        
        nodes = flow.get("nodes", [])
        edges = flow.get("edges", [])
        
        # Create node ID set for validation
        node_ids = {node.get("id") for node in nodes if node.get("id")}
        
        # Validate each edge
        edge_ids = set()
        for edge in edges:
            # Check required fields
            if "id" not in edge:
                issues.append("Edge missing required 'id' field")
                continue
            
            edge_id = edge["id"]
            
            # Check for duplicate edge IDs
            if edge_id in edge_ids:
                issues.append(f"Duplicate edge ID: {edge_id}")
            edge_ids.add(edge_id)
            
            # Check source and target
            if "source" not in edge:
                issues.append(f"Edge {edge_id} missing source")
            elif edge["source"] not in node_ids:
                issues.append(f"Edge {edge_id} references non-existent source: {edge['source']}")
            
            if "target" not in edge:
                issues.append(f"Edge {edge_id} missing target")
            elif edge["target"] not in node_ids:
                issues.append(f"Edge {edge_id} references non-existent target: {edge['target']}")
            
            # Check for self-loops
            if edge.get("source") == edge.get("target"):
                issues.append(f"Edge {edge_id} creates self-loop")
        
        return issues
    
    async def _validate_workflow_logic(self, langflow_data: Dict[str, Any]) -> List[str]:
        """Validate workflow execution logic"""
        issues = []
        
        try:
            flow = langflow_data.get("flow", {})
            nodes = flow.get("nodes", [])
            edges = flow.get("edges", [])
            
            # Build graph structure
            graph = self._build_graph(nodes, edges)
            
            # Check for cycles
            if self._has_cycles(graph):
                issues.append("Workflow contains cycles that may cause infinite loops")
            
            # Check for disconnected components
            disconnected = self._find_disconnected_components(graph)
            if disconnected:
                issues.append(f"Workflow has disconnected components: {disconnected}")
            
            # Check for missing entry points
            entry_points = self._find_entry_points(graph)
            if not entry_points:
                issues.append("Workflow has no entry points (nodes with no inputs)")
            
            # Check for missing exit points
            exit_points = self._find_exit_points(graph)
            if not exit_points:
                issues.append("Workflow has no exit points (nodes with no outputs)")
            
            # Validate agent-task relationships
            agent_task_issues = self._validate_agent_task_relationships(nodes, edges)
            issues.extend(agent_task_issues)
            
        except Exception as e:
            logger.error(f"Workflow logic validation failed: {e}")
            issues.append(f"Logic validation error: {str(e)}")
        
        return issues
    
    def _build_graph(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Set[str]]:
        """Build adjacency list representation of the workflow graph"""
        graph = {}
        
        # Initialize all nodes
        for node in nodes:
            node_id = node.get("id")
            if node_id:
                graph[node_id] = set()
        
        # Add edges
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target and source in graph:
                graph[source].add(target)
        
        return graph
    
    def _has_cycles(self, graph: Dict[str, Set[str]]) -> bool:
        """Check if graph has cycles using DFS"""
        white = set(graph.keys())  # Unvisited
        gray = set()               # Currently visiting
        black = set()              # Visited
        
        def dfs(node):
            if node in black:
                return False
            if node in gray:
                return True  # Back edge found, cycle detected
            
            gray.add(node)
            white.discard(node)
            
            for neighbor in graph.get(node, set()):
                if dfs(neighbor):
                    return True
            
            gray.remove(node)
            black.add(node)
            return False
        
        for node in list(white):
            if dfs(node):
                return True
        
        return False
    
    def _find_disconnected_components(self, graph: Dict[str, Set[str]]) -> List[str]:
        """Find disconnected components in the graph"""
        visited = set()
        components = []
        
        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.append(node)
            
            # Visit neighbors (both outgoing and incoming)
            for neighbor in graph.get(node, set()):
                dfs(neighbor, component)
            
            # Visit nodes that point to this node
            for other_node, neighbors in graph.items():
                if node in neighbors:
                    dfs(other_node, component)
        
        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                if len(component) > 1:
                    components.append(component)
        
        # Return isolated nodes
        isolated = [comp[0] for comp in components if len(comp) == 1]
        return isolated
    
    def _find_entry_points(self, graph: Dict[str, Set[str]]) -> List[str]:
        """Find nodes with no incoming edges"""
        has_incoming = set()
        for neighbors in graph.values():
            has_incoming.update(neighbors)
        
        return [node for node in graph if node not in has_incoming]
    
    def _find_exit_points(self, graph: Dict[str, Set[str]]) -> List[str]:
        """Find nodes with no outgoing edges"""
        return [node for node, neighbors in graph.items() if not neighbors]
    
    def _validate_agent_task_relationships(self, nodes: List[Dict], edges: List[Dict]) -> List[str]:
        """Validate relationships between agents and tasks"""
        issues = []
        
        # Find agents and tasks
        agents = {node["id"]: node for node in nodes if node.get("type") == "CrewGraphAgent"}
        tasks = {node["id"]: node for node in nodes if node.get("type") == "CrewGraphTask"}
        
        # Build edge map
        edge_map = {}
        for edge in edges:
            source, target = edge.get("source"), edge.get("target")
            if source:
                edge_map.setdefault(source, []).append(target)
        
        # Check if tasks have assigned agents
        for task_id, task in tasks.items():
            # Check if task has agent assignment in data
            agent_assignment = task.get("data", {}).get("inputs", {}).get("agent", {}).get("value")
            
            # Check if task is connected to an agent
            connected_agents = []
            for agent_id in agents:
                if task_id in edge_map.get(agent_id, []):
                    connected_agents.append(agent_id)
            
            if not agent_assignment and not connected_agents:
                issues.append(f"Task {task_id} has no assigned agent")
            elif len(connected_agents) > 1:
                issues.append(f"Task {task_id} is connected to multiple agents: {connected_agents}")
        
        return issues
    
    def _validate_crewgraph_compatibility(self, langflow_data: Dict[str, Any]) -> List[str]:
        """Validate compatibility with CrewGraph execution engine"""
        issues = []
        
        try:
            flow = langflow_data.get("flow", {})
            nodes = flow.get("nodes", [])
            
            # Check for required CrewGraph components
            has_agent = any(node.get("type") == "CrewGraphAgent" for node in nodes)
            has_task = any(node.get("type") == "CrewGraphTask" for node in nodes)
            
            if not has_agent:
                issues.append("Workflow requires at least one CrewGraphAgent")
            
            if not has_task:
                issues.append("Workflow requires at least one CrewGraphTask")
            
            # Check for unsupported Langflow components
            unsupported_types = []
            for node in nodes:
                node_type = node.get("type")
                if node_type and node_type not in self.supported_node_types:
                    unsupported_types.append(node_type)
            
            if unsupported_types:
                unique_types = list(set(unsupported_types))
                issues.append(f"Unsupported components for CrewGraph: {', '.join(unique_types)}")
            
            # Check for complex Langflow features that don't map to CrewGraph
            for node in nodes:
                data = node.get("data", {})
                
                # Check for complex input/output connections
                inputs = data.get("inputs", {})
                for input_name, input_data in inputs.items():
                    if isinstance(input_data, dict) and input_data.get("type") == "file":
                        issues.append(f"File inputs not fully supported in CrewGraph conversion")
                
                # Check for advanced configuration
                if data.get("advanced_config"):
                    issues.append(f"Advanced configurations may not be fully preserved")
            
        except Exception as e:
            logger.error(f"Compatibility validation failed: {e}")
            issues.append(f"Compatibility validation error: {str(e)}")
        
        return issues
    
    async def validate_component_code(
        self,
        component_code: str,
        component_type: str
    ) -> Dict[str, Any]:
        """
        Validate custom component code for security and compatibility
        
        Args:
            component_code: Python code for the component
            component_type: Type of component (agent, tool, etc.)
            
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "security_issues": [],
            "syntax_valid": False,
            "imports_valid": False
        }
        
        try:
            logger.info(f"Validating {component_type} component code")
            
            # Syntax validation
            try:
                ast.parse(component_code)
                result["syntax_valid"] = True
            except SyntaxError as e:
                result["valid"] = False
                result["issues"].append(f"Syntax error: {str(e)}")
            
            # Security validation
            security_issues = self._check_security_issues(component_code)
            result["security_issues"] = security_issues
            if security_issues:
                result["valid"] = False
                result["issues"].extend([f"Security issue: {issue}" for issue in security_issues])
            
            # Import validation
            import_issues = self._validate_imports(component_code)
            if not import_issues:
                result["imports_valid"] = True
            else:
                result["warnings"].extend(import_issues)
            
            # Component-specific validation
            type_issues = self._validate_component_type(component_code, component_type)
            result["issues"].extend(type_issues)
            if type_issues:
                result["valid"] = False
            
            # Complexity analysis
            complexity = self._analyze_complexity(component_code)
            if complexity["high_complexity"]:
                result["warnings"].append("Component has high complexity")
            
            result["complexity_metrics"] = complexity
            
        except Exception as e:
            logger.error(f"Component validation failed: {e}")
            result["valid"] = False
            result["issues"].append(f"Validation error: {str(e)}")
        
        return result
    
    def _check_security_issues(self, code: str) -> List[str]:
        """Check for potential security issues in code"""
        issues = []
        
        for pattern in self.security_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Potentially unsafe pattern found: {pattern}")
        
        # Check for dangerous function calls
        dangerous_funcs = ["exec", "eval", "__import__", "compile", "globals", "locals"]
        for func in dangerous_funcs:
            if f"{func}(" in code:
                issues.append(f"Dangerous function call: {func}")
        
        return issues
    
    def _validate_imports(self, code: str) -> List[str]:
        """Validate import statements"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.required_python_modules:
                            issues.append(f"Non-standard import: {alias.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self.required_python_modules:
                        if not node.module.startswith(('crewgraph_ai', 'langchain', 'pydantic')):
                            issues.append(f"Non-standard module import: {node.module}")
        
        except Exception as e:
            issues.append(f"Import analysis failed: {str(e)}")
        
        return issues
    
    def _validate_component_type(self, code: str, component_type: str) -> List[str]:
        """Validate component code matches expected type"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            # Look for required base class
            expected_bases = {
                "agent": "CrewGraphAgentComponent",
                "tool": "CrewGraphToolComponent", 
                "memory": "BaseMemory",
                "custom": "LangflowComponent"
            }
            
            expected_base = expected_bases.get(component_type, "LangflowComponent")
            
            # Find class definitions
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            if not classes:
                issues.append("Component must define at least one class")
            else:
                has_proper_base = False
                for cls in classes:
                    for base in cls.bases:
                        if isinstance(base, ast.Name) and base.id == expected_base:
                            has_proper_base = True
                            break
                
                if not has_proper_base:
                    issues.append(f"Component should inherit from {expected_base}")
            
            # Check for required methods
            required_methods = ["execute", "_get_metadata", "_get_inputs", "_get_outputs"]
            method_names = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    method_names.append(node.name)
            
            for method in required_methods:
                if method not in method_names:
                    issues.append(f"Missing required method: {method}")
        
        except Exception as e:
            issues.append(f"Type validation failed: {str(e)}")
        
        return issues
    
    def _analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity"""
        try:
            tree = ast.parse(code)
            
            # Count various constructs
            lines = len(code.split('\n'))
            functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            loops = len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))])
            conditions = len([n for n in ast.walk(tree) if isinstance(n, ast.If)])
            
            # Calculate complexity score
            complexity_score = (
                lines * 0.1 +
                functions * 2 +
                classes * 3 +
                loops * 5 +
                conditions * 2
            )
            
            return {
                "lines_of_code": lines,
                "function_count": functions,
                "class_count": classes,
                "loop_count": loops,
                "condition_count": conditions,
                "complexity_score": complexity_score,
                "high_complexity": complexity_score > 100
            }
            
        except Exception:
            return {
                "lines_of_code": 0,
                "function_count": 0,
                "class_count": 0,
                "loop_count": 0,
                "condition_count": 0,
                "complexity_score": 0,
                "high_complexity": False
            }
    
    def validate_workflow_execution_requirements(
        self,
        workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate workflow execution requirements
        
        Args:
            workflow_data: Workflow data to validate
            
        Returns:
            Validation result with execution requirements
        """
        result = {
            "can_execute": True,
            "issues": [],
            "warnings": [],
            "requirements": {
                "agents_needed": 0,
                "tools_needed": [],
                "memory_required": False,
                "estimated_execution_time": 0
            }
        }
        
        try:
            nodes = workflow_data.get("nodes", [])
            
            # Count agents
            agents = [n for n in nodes if n.get("type") == "agent"]
            result["requirements"]["agents_needed"] = len(agents)
            
            # Collect tools
            tools = set()
            for node in nodes:
                if node.get("type") == "tool":
                    tool_name = node.get("data", {}).get("tool_name", "")
                    if tool_name:
                        tools.add(tool_name)
                elif node.get("type") == "task":
                    task_tools = node.get("data", {}).get("tools", [])
                    tools.update(task_tools)
            
            result["requirements"]["tools_needed"] = list(tools)
            
            # Check memory requirements
            has_memory = any(n.get("type") == "memory" for n in nodes)
            result["requirements"]["memory_required"] = has_memory
            
            # Estimate execution time (rough)
            task_count = len([n for n in nodes if n.get("type") == "task"])
            estimated_time = task_count * 30  # 30 seconds per task estimate
            result["requirements"]["estimated_execution_time"] = estimated_time
            
            # Check for execution blockers
            if not agents:
                result["can_execute"] = False
                result["issues"].append("No agents defined - workflow cannot execute")
            
            if not any(n.get("type") == "task" for n in nodes):
                result["can_execute"] = False
                result["issues"].append("No tasks defined - workflow cannot execute")
            
        except Exception as e:
            logger.error(f"Execution requirements validation failed: {e}")
            result["can_execute"] = False
            result["issues"].append(f"Validation error: {str(e)}")
        
        return result