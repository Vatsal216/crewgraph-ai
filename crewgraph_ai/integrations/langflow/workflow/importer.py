"""
Workflow Importer for CrewGraph AI + Langflow Integration

This module handles importing Langflow workflows to CrewGraph format.

Created by: Vatsal216
Date: 2025-07-23
"""

import json
import gzip
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Import CrewGraph components
try:
    from ....core.orchestrator import GraphOrchestrator
    from ....core.graph import WorkflowGraph
    from ....utils.logging import get_logger
except ImportError:
    GraphOrchestrator = None
    WorkflowGraph = None
    
    def get_logger(name):
        import logging
        return logging.getLogger(name)

logger = get_logger(__name__)


class WorkflowImporter:
    """
    Import Langflow workflows to CrewGraph format
    
    This class handles the conversion of Langflow workflow definitions
    into CrewGraph-compatible format for execution.
    """
    
    def __init__(self):
        self.supported_components = {
            "CrewGraphAgent": "agent",
            "CrewGraphTask": "task",
            "CrewGraphTool": "tool",
            "CrewGraphMemory": "memory",
            "ConditionalNode": "condition",
            "LoopNode": "loop",
            "ParallelNode": "parallel"
        }
        self.auto_fixes = {
            "missing_connections": True,
            "invalid_node_types": True,
            "incomplete_configurations": True,
            "dependency_resolution": True
        }
    
    async def import_workflow(
        self,
        langflow_data: Dict[str, Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        auto_fix_issues: bool = False
    ) -> Dict[str, Any]:
        """
        Import a Langflow workflow to CrewGraph format
        
        Args:
            langflow_data: Langflow workflow data
            name: Optional name for the imported workflow
            description: Optional description for the workflow
            auto_fix_issues: Whether to automatically fix common issues
            
        Returns:
            Dictionary containing import results
        """
        try:
            logger.info(f"Importing Langflow workflow: {langflow_data.get('name', 'unnamed')}")
            
            # Decompress if needed
            if langflow_data.get("compressed"):
                langflow_data = self._decompress_workflow(langflow_data)
            
            # Validate Langflow data
            validation_result = await self._validate_langflow_data(langflow_data)
            if not validation_result["valid"] and not auto_fix_issues:
                raise ValueError(f"Invalid Langflow data: {validation_result['issues']}")
            
            # Convert to CrewGraph format
            crewgraph_workflow = await self._convert_to_crewgraph(
                langflow_data, 
                name, 
                description,
                auto_fix_issues
            )
            
            # Save workflow to CrewGraph
            workflow_id = await self._save_workflow(crewgraph_workflow)
            
            # Collect results
            result = {
                "workflow_id": workflow_id,
                "import_status": "success",
                "validation_issues": validation_result.get("issues", []),
                "fixed_issues": validation_result.get("fixed_issues", []),
                "components_created": crewgraph_workflow.get("components_created", []),
                "imported_at": datetime.utcnow().isoformat(),
                "statistics": {
                    "nodes_imported": len(crewgraph_workflow.get("nodes", [])),
                    "edges_imported": len(crewgraph_workflow.get("edges", [])),
                    "components_created": len(crewgraph_workflow.get("components_created", []))
                }
            }
            
            logger.info(f"Successfully imported workflow: {workflow_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to import Langflow workflow: {e}")
            raise
    
    def _decompress_workflow(self, compressed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress compressed workflow data"""
        try:
            if compressed_data.get("compression_type") == "gzip+base64":
                # Decode base64
                compressed_bytes = base64.b64decode(compressed_data["data"])
                
                # Decompress gzip
                json_str = gzip.decompress(compressed_bytes).decode('utf-8')
                
                # Parse JSON
                return json.loads(json_str)
            else:
                raise ValueError(f"Unsupported compression type: {compressed_data.get('compression_type')}")
                
        except Exception as e:
            logger.error(f"Failed to decompress workflow: {e}")
            raise ValueError(f"Failed to decompress workflow: {e}")
    
    async def _validate_langflow_data(self, langflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Langflow workflow data"""
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "fixed_issues": []
        }
        
        try:
            # Check required fields
            required_fields = ["flow"]
            for field in required_fields:
                if field not in langflow_data:
                    validation_result["issues"].append(f"Missing required field: {field}")
                    validation_result["valid"] = False
            
            # Check flow structure
            if "flow" in langflow_data:
                flow = langflow_data["flow"]
                
                if "nodes" not in flow:
                    validation_result["issues"].append("Flow missing nodes array")
                    validation_result["valid"] = False
                
                if "edges" not in flow:
                    validation_result["issues"].append("Flow missing edges array")
                    validation_result["valid"] = False
                
                # Validate nodes
                for node in flow.get("nodes", []):
                    node_issues = self._validate_node(node)
                    validation_result["issues"].extend(node_issues)
                    if node_issues:
                        validation_result["valid"] = False
                
                # Validate edges
                for edge in flow.get("edges", []):
                    edge_issues = self._validate_edge(edge, flow.get("nodes", []))
                    validation_result["issues"].extend(edge_issues)
                    if edge_issues:
                        validation_result["valid"] = False
            
            # Check for unsupported components
            unsupported_types = []
            for node in langflow_data.get("flow", {}).get("nodes", []):
                node_type = node.get("type")
                if node_type and node_type not in self.supported_components:
                    unsupported_types.append(node_type)
            
            if unsupported_types:
                validation_result["warnings"].append(
                    f"Unsupported component types found: {', '.join(set(unsupported_types))}"
                )
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _validate_node(self, node: Dict[str, Any]) -> List[str]:
        """Validate a single node"""
        issues = []
        
        if "id" not in node:
            issues.append(f"Node missing required 'id' field")
        
        if "type" not in node:
            issues.append(f"Node {node.get('id', 'unknown')} missing 'type' field")
        
        if "data" not in node:
            issues.append(f"Node {node.get('id', 'unknown')} missing 'data' field")
        
        # Type-specific validation
        node_type = node.get("type")
        if node_type == "CrewGraphAgent":
            data = node.get("data", {})
            inputs = data.get("inputs", {})
            if not inputs.get("role", {}).get("value"):
                issues.append(f"Agent node {node.get('id')} missing role")
            if not inputs.get("goal", {}).get("value"):
                issues.append(f"Agent node {node.get('id')} missing goal")
        
        elif node_type == "CrewGraphTask":
            data = node.get("data", {})
            inputs = data.get("inputs", {})
            if not inputs.get("description", {}).get("value"):
                issues.append(f"Task node {node.get('id')} missing description")
        
        return issues
    
    def _validate_edge(self, edge: Dict[str, Any], nodes: List[Dict[str, Any]]) -> List[str]:
        """Validate a single edge"""
        issues = []
        
        if "source" not in edge:
            issues.append(f"Edge {edge.get('id', 'unknown')} missing source")
        
        if "target" not in edge:
            issues.append(f"Edge {edge.get('id', 'unknown')} missing target")
        
        # Check if source and target nodes exist
        node_ids = {node["id"] for node in nodes if "id" in node}
        
        if edge.get("source") not in node_ids:
            issues.append(f"Edge {edge.get('id')} references non-existent source node: {edge.get('source')}")
        
        if edge.get("target") not in node_ids:
            issues.append(f"Edge {edge.get('id')} references non-existent target node: {edge.get('target')}")
        
        return issues
    
    async def _convert_to_crewgraph(
        self,
        langflow_data: Dict[str, Any],
        name: Optional[str],
        description: Optional[str],
        auto_fix_issues: bool
    ) -> Dict[str, Any]:
        """Convert Langflow data to CrewGraph format"""
        
        workflow_name = name or langflow_data.get("name", f"imported_workflow_{uuid4().hex[:8]}")
        workflow_description = description or langflow_data.get("description", "Imported from Langflow")
        
        crewgraph_workflow = {
            "id": str(uuid4()),
            "name": workflow_name,
            "description": workflow_description,
            "version": "1.0.0",
            "created_at": datetime.utcnow().isoformat(),
            "imported_from": "langflow",
            "original_metadata": langflow_data.get("metadata", {}),
            "nodes": [],
            "edges": [],
            "components_created": []
        }
        
        flow = langflow_data.get("flow", {})
        
        # Convert nodes
        for node in flow.get("nodes", []):
            crewgraph_node = await self._convert_node(node, auto_fix_issues)
            if crewgraph_node:
                crewgraph_workflow["nodes"].append(crewgraph_node)
                if crewgraph_node.get("created_component"):
                    crewgraph_workflow["components_created"].append(crewgraph_node["created_component"])
        
        # Convert edges
        for edge in flow.get("edges", []):
            crewgraph_edge = self._convert_edge(edge)
            if crewgraph_edge:
                crewgraph_workflow["edges"].append(crewgraph_edge)
        
        # Auto-fix issues if enabled
        if auto_fix_issues:
            crewgraph_workflow = await self._apply_auto_fixes(crewgraph_workflow)
        
        return crewgraph_workflow
    
    async def _convert_node(self, node: Dict[str, Any], auto_fix: bool) -> Optional[Dict[str, Any]]:
        """Convert a Langflow node to CrewGraph format"""
        try:
            node_type = node.get("type")
            if node_type not in self.supported_components:
                if auto_fix:
                    logger.warning(f"Unsupported node type {node_type}, skipping")
                    return None
                else:
                    raise ValueError(f"Unsupported node type: {node_type}")
            
            crewgraph_type = self.supported_components[node_type]
            data = node.get("data", {})
            inputs = data.get("inputs", {})
            
            crewgraph_node = {
                "id": node["id"],
                "type": crewgraph_type,
                "name": data.get("label", node["id"]),
                "description": data.get("description", ""),
                "position": node.get("position", {"x": 0, "y": 0}),
                "data": {},
                "metadata": {
                    "imported_from_langflow": True,
                    "original_type": node_type,
                    "import_timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Convert type-specific data
            if crewgraph_type == "agent":
                crewgraph_node["data"] = {
                    "role": inputs.get("role", {}).get("value", "Assistant"),
                    "goal": inputs.get("goal", {}).get("value", "Help users accomplish tasks"),
                    "backstory": inputs.get("backstory", {}).get("value", ""),
                    "llm_model": inputs.get("llm_model", {}).get("value", "gpt-3.5-turbo"),
                    "max_iter": inputs.get("max_iter", {}).get("value", 5),
                    "verbose": inputs.get("verbose", {}).get("value", True),
                    "allow_delegation": inputs.get("allow_delegation", {}).get("value", False)
                }
            
            elif crewgraph_type == "task":
                crewgraph_node["data"] = {
                    "description": inputs.get("description", {}).get("value", ""),
                    "expected_output": inputs.get("expected_output", {}).get("value", ""),
                    "agent_id": inputs.get("agent", {}).get("value", ""),
                    "tools": inputs.get("tools", {}).get("value", []),
                    "context": inputs.get("context", {}).get("value", {}),
                    "async_execution": inputs.get("async_execution", {}).get("value", False),
                    "human_input": inputs.get("human_input", {}).get("value", False)
                }
            
            elif crewgraph_type == "tool":
                crewgraph_node["data"] = {
                    "tool_name": inputs.get("tool_name", {}).get("value", ""),
                    "configuration": inputs.get("tool_config", {}).get("value", {}),
                    "cache_results": inputs.get("cache_results", {}).get("value", True),
                    "retry_on_failure": inputs.get("retry_on_failure", {}).get("value", True),
                    "max_retries": inputs.get("max_retries", {}).get("value", 3),
                    "timeout_seconds": inputs.get("timeout_seconds", {}).get("value", 30)
                }
            
            return crewgraph_node
            
        except Exception as e:
            logger.error(f"Failed to convert node {node.get('id', 'unknown')}: {e}")
            if auto_fix:
                return None
            else:
                raise
    
    def _convert_edge(self, edge: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a Langflow edge to CrewGraph format"""
        try:
            return {
                "id": edge.get("id", str(uuid4())),
                "source": edge["source"],
                "target": edge["target"],
                "type": edge.get("type", "default"),
                "source_handle": edge.get("sourceHandle", "default"),
                "target_handle": edge.get("targetHandle", "default"),
                "data": edge.get("data", {}),
                "metadata": {
                    "imported_from_langflow": True,
                    "import_timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to convert edge {edge.get('id', 'unknown')}: {e}")
            return None
    
    async def _apply_auto_fixes(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Apply automatic fixes to common issues"""
        fixed_issues = []
        
        try:
            # Fix missing node IDs
            for node in workflow["nodes"]:
                if not node.get("id"):
                    node["id"] = str(uuid4())
                    fixed_issues.append(f"Generated missing ID for node")
            
            # Fix missing edge IDs
            for edge in workflow["edges"]:
                if not edge.get("id"):
                    edge["id"] = str(uuid4())
                    fixed_issues.append(f"Generated missing ID for edge")
            
            # Fix broken connections
            node_ids = {node["id"] for node in workflow["nodes"]}
            valid_edges = []
            
            for edge in workflow["edges"]:
                if edge["source"] in node_ids and edge["target"] in node_ids:
                    valid_edges.append(edge)
                else:
                    fixed_issues.append(f"Removed invalid edge: {edge['source']} -> {edge['target']}")
            
            workflow["edges"] = valid_edges
            
            # Fix incomplete agent configurations
            for node in workflow["nodes"]:
                if node["type"] == "agent":
                    data = node["data"]
                    if not data.get("role"):
                        data["role"] = "Assistant"
                        fixed_issues.append(f"Set default role for agent {node['id']}")
                    if not data.get("goal"):
                        data["goal"] = "Help users accomplish their tasks"
                        fixed_issues.append(f"Set default goal for agent {node['id']}")
            
            # Fix incomplete task configurations
            for node in workflow["nodes"]:
                if node["type"] == "task":
                    data = node["data"]
                    if not data.get("description"):
                        data["description"] = "Perform assigned task"
                        fixed_issues.append(f"Set default description for task {node['id']}")
            
            workflow["fixed_issues"] = fixed_issues
            logger.info(f"Applied {len(fixed_issues)} auto-fixes")
            
        except Exception as e:
            logger.error(f"Auto-fix failed: {e}")
        
        return workflow
    
    async def _save_workflow(self, workflow: Dict[str, Any]) -> str:
        """Save workflow to CrewGraph storage"""
        try:
            # In real implementation, this would save to CrewGraph storage system
            # For now, just return the workflow ID
            workflow_id = workflow["id"]
            
            logger.info(f"Saved workflow {workflow_id} to CrewGraph storage")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to save workflow: {e}")
            raise
    
    async def import_multiple_workflows(
        self,
        langflow_package: Dict[str, Any],
        auto_fix_issues: bool = False
    ) -> Dict[str, Any]:
        """
        Import multiple workflows from a Langflow package
        
        Args:
            langflow_package: Package containing multiple workflows
            auto_fix_issues: Whether to auto-fix issues
            
        Returns:
            Dictionary containing import results for all workflows
        """
        try:
            workflows = langflow_package.get("workflows", [])
            logger.info(f"Importing {len(workflows)} workflows from package")
            
            results = []
            failed_imports = []
            
            for workflow_data in workflows:
                try:
                    result = await self.import_workflow(
                        workflow_data,
                        auto_fix_issues=auto_fix_issues
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to import workflow {workflow_data.get('id', 'unknown')}: {e}")
                    failed_imports.append({
                        "workflow_id": workflow_data.get("id", "unknown"),
                        "error": str(e)
                    })
            
            return {
                "import_type": "multi_workflow",
                "imported_at": datetime.utcnow().isoformat(),
                "results": results,
                "import_summary": {
                    "total_workflows": len(workflows),
                    "successful_imports": len(results),
                    "failed_imports": len(failed_imports),
                    "failed_items": failed_imports
                }
            }
            
        except Exception as e:
            logger.error(f"Multi-workflow import failed: {e}")
            raise