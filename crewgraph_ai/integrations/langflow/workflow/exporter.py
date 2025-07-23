"""
Workflow Exporter for CrewGraph AI + Langflow Integration

This module handles exporting CrewGraph workflows to Langflow format.

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


class WorkflowExporter:
    """
    Export CrewGraph workflows to Langflow format
    
    This class handles the conversion of CrewGraph workflow definitions
    into Langflow-compatible JSON format for visual editing.
    """
    
    def __init__(self):
        self.format_version = "1.0"
        self.supported_node_types = {
            "agent": "CrewGraphAgent",
            "task": "CrewGraphTask", 
            "tool": "CrewGraphTool",
            "memory": "CrewGraphMemory",
            "condition": "ConditionalNode",
            "loop": "LoopNode",
            "parallel": "ParallelNode"
        }
    
    async def export_workflow(
        self,
        workflow_id: str,
        include_metadata: bool = True,
        format_version: str = "1.0",
        compression: bool = False
    ) -> Dict[str, Any]:
        """
        Export a CrewGraph workflow to Langflow format
        
        Args:
            workflow_id: ID of the workflow to export
            include_metadata: Whether to include workflow metadata
            format_version: Langflow format version to use
            compression: Whether to compress the output
            
        Returns:
            Dictionary containing Langflow workflow data
        """
        try:
            logger.info(f"Exporting workflow {workflow_id} to Langflow format")
            
            # Get workflow from CrewGraph
            workflow_data = await self._get_workflow_data(workflow_id)
            if not workflow_data:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Convert to Langflow format
            langflow_data = await self._convert_to_langflow(
                workflow_data, 
                include_metadata,
                format_version
            )
            
            # Apply compression if requested
            if compression:
                langflow_data = self._compress_workflow(langflow_data)
            
            logger.info(f"Successfully exported workflow {workflow_id}")
            return langflow_data
            
        except Exception as e:
            logger.error(f"Failed to export workflow {workflow_id}: {e}")
            raise
    
    async def _get_workflow_data(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow data from CrewGraph"""
        try:
            # In real implementation, this would query the CrewGraph storage
            # For now, return mock workflow data
            return {
                "id": workflow_id,
                "name": f"Workflow {workflow_id}",
                "description": "Sample CrewGraph workflow",
                "version": "1.0.0",
                "created_at": "2025-07-23T20:00:00Z",
                "updated_at": "2025-07-23T20:00:00Z",
                "nodes": [
                    {
                        "id": "agent_1",
                        "type": "agent",
                        "name": "Research Agent",
                        "position": {"x": 100, "y": 100},
                        "data": {
                            "role": "Research Analyst",
                            "goal": "Conduct thorough research on given topics",
                            "backstory": "Expert research analyst with years of experience",
                            "llm_model": "gpt-4"
                        }
                    },
                    {
                        "id": "task_1", 
                        "type": "task",
                        "name": "Research Task",
                        "position": {"x": 300, "y": 100},
                        "data": {
                            "description": "Research the given topic thoroughly",
                            "expected_output": "Comprehensive research report",
                            "agent_id": "agent_1"
                        }
                    },
                    {
                        "id": "tool_1",
                        "type": "tool",
                        "name": "Web Search",
                        "position": {"x": 100, "y": 300},
                        "data": {
                            "tool_name": "web_scraper",
                            "configuration": {"max_results": 10}
                        }
                    }
                ],
                "edges": [
                    {
                        "id": "edge_1",
                        "source": "agent_1",
                        "target": "task_1",
                        "type": "default"
                    },
                    {
                        "id": "edge_2", 
                        "source": "tool_1",
                        "target": "agent_1",
                        "type": "tool_connection"
                    }
                ],
                "metadata": {
                    "author": "CrewGraph AI",
                    "tags": ["research", "analysis"],
                    "category": "data_processing"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow data: {e}")
            return None
    
    async def _convert_to_langflow(
        self,
        workflow_data: Dict[str, Any],
        include_metadata: bool,
        format_version: str
    ) -> Dict[str, Any]:
        """Convert CrewGraph workflow to Langflow format"""
        
        # Base Langflow structure
        langflow_workflow = {
            "format_version": format_version,
            "id": workflow_data["id"],
            "name": workflow_data["name"],
            "description": workflow_data.get("description", ""),
            "flow": {
                "nodes": [],
                "edges": [],
                "viewport": {
                    "x": 0,
                    "y": 0,
                    "zoom": 1
                }
            },
            "exported_at": datetime.utcnow().isoformat(),
            "exported_by": "crewgraph-langflow-integration",
            "source_platform": "crewgraph"
        }
        
        # Convert nodes
        for node in workflow_data.get("nodes", []):
            langflow_node = await self._convert_node(node)
            if langflow_node:
                langflow_workflow["flow"]["nodes"].append(langflow_node)
        
        # Convert edges
        for edge in workflow_data.get("edges", []):
            langflow_edge = self._convert_edge(edge)
            if langflow_edge:
                langflow_workflow["flow"]["edges"].append(langflow_edge)
        
        # Add metadata if requested
        if include_metadata:
            langflow_workflow["metadata"] = {
                "crewgraph_version": workflow_data.get("version", "1.0.0"),
                "original_metadata": workflow_data.get("metadata", {}),
                "node_count": len(langflow_workflow["flow"]["nodes"]),
                "edge_count": len(langflow_workflow["flow"]["edges"]),
                "export_settings": {
                    "include_metadata": include_metadata,
                    "format_version": format_version
                }
            }
        
        return langflow_workflow
    
    async def _convert_node(self, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a CrewGraph node to Langflow format"""
        try:
            node_type = node.get("type")
            if node_type not in self.supported_node_types:
                logger.warning(f"Unsupported node type: {node_type}")
                return None
            
            langflow_node = {
                "id": node["id"],
                "type": self.supported_node_types[node_type],
                "position": node.get("position", {"x": 0, "y": 0}),
                "data": {
                    "label": node.get("name", node["id"]),
                    "description": node.get("description", ""),
                    "inputs": {},
                    "outputs": {},
                    "configuration": node.get("data", {})
                },
                "width": 200,
                "height": 100,
                "selected": False,
                "dragging": False
            }
            
            # Add type-specific data
            if node_type == "agent":
                langflow_node["data"]["inputs"] = {
                    "role": {"value": node["data"].get("role", "")},
                    "goal": {"value": node["data"].get("goal", "")},
                    "backstory": {"value": node["data"].get("backstory", "")},
                    "llm_model": {"value": node["data"].get("llm_model", "gpt-3.5-turbo")}
                }
                langflow_node["data"]["outputs"] = {
                    "agent": {"type": "agent"}
                }
            
            elif node_type == "task":
                langflow_node["data"]["inputs"] = {
                    "description": {"value": node["data"].get("description", "")},
                    "expected_output": {"value": node["data"].get("expected_output", "")},
                    "agent": {"type": "agent"}
                }
                langflow_node["data"]["outputs"] = {
                    "result": {"type": "str"}
                }
            
            elif node_type == "tool":
                langflow_node["data"]["inputs"] = {
                    "tool_name": {"value": node["data"].get("tool_name", "")},
                    "tool_input": {"value": ""},
                    "tool_config": {"value": node["data"].get("configuration", {})}
                }
                langflow_node["data"]["outputs"] = {
                    "result": {"type": "str"}
                }
            
            return langflow_node
            
        except Exception as e:
            logger.error(f"Failed to convert node {node.get('id', 'unknown')}: {e}")
            return None
    
    def _convert_edge(self, edge: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a CrewGraph edge to Langflow format"""
        try:
            return {
                "id": edge["id"],
                "source": edge["source"],
                "target": edge["target"],
                "sourceHandle": edge.get("source_handle", "default"),
                "targetHandle": edge.get("target_handle", "default"),
                "type": edge.get("type", "default"),
                "animated": False,
                "style": {},
                "data": edge.get("data", {})
            }
            
        except Exception as e:
            logger.error(f"Failed to convert edge {edge.get('id', 'unknown')}: {e}")
            return None
    
    def _compress_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress workflow data using gzip"""
        try:
            # Convert to JSON string
            json_str = json.dumps(workflow_data, separators=(',', ':'))
            
            # Compress using gzip
            compressed = gzip.compress(json_str.encode('utf-8'))
            
            # Encode as base64
            encoded = base64.b64encode(compressed).decode('ascii')
            
            return {
                "format_version": workflow_data.get("format_version", "1.0"),
                "compressed": True,
                "compression_type": "gzip+base64",
                "original_size": len(json_str),
                "compressed_size": len(encoded),
                "data": encoded
            }
            
        except Exception as e:
            logger.error(f"Failed to compress workflow: {e}")
            return workflow_data
    
    async def export_multiple_workflows(
        self,
        workflow_ids: List[str],
        include_metadata: bool = True,
        format_version: str = "1.0"
    ) -> Dict[str, Any]:
        """
        Export multiple workflows to a single Langflow package
        
        Args:
            workflow_ids: List of workflow IDs to export
            include_metadata: Whether to include metadata
            format_version: Langflow format version
            
        Returns:
            Dictionary containing multiple workflows
        """
        try:
            logger.info(f"Exporting {len(workflow_ids)} workflows")
            
            workflows = []
            failed_exports = []
            
            for workflow_id in workflow_ids:
                try:
                    workflow = await self.export_workflow(
                        workflow_id, include_metadata, format_version
                    )
                    workflows.append(workflow)
                except Exception as e:
                    logger.error(f"Failed to export workflow {workflow_id}: {e}")
                    failed_exports.append({"id": workflow_id, "error": str(e)})
            
            return {
                "format_version": format_version,
                "export_type": "multi_workflow",
                "exported_at": datetime.utcnow().isoformat(),
                "workflows": workflows,
                "export_summary": {
                    "total_requested": len(workflow_ids),
                    "successful_exports": len(workflows),
                    "failed_exports": len(failed_exports),
                    "failed_items": failed_exports
                }
            }
            
        except Exception as e:
            logger.error(f"Multi-workflow export failed: {e}")
            raise
    
    def validate_export(self, exported_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate exported workflow data
        
        Args:
            exported_data: Exported workflow data
            
        Returns:
            Validation result
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # Check required fields
            required_fields = ["format_version", "id", "name", "flow"]
            for field in required_fields:
                if field not in exported_data:
                    validation_result["issues"].append(f"Missing required field: {field}")
                    validation_result["valid"] = False
            
            # Check flow structure
            if "flow" in exported_data:
                flow = exported_data["flow"]
                if "nodes" not in flow:
                    validation_result["issues"].append("Flow missing nodes array")
                    validation_result["valid"] = False
                
                if "edges" not in flow:
                    validation_result["issues"].append("Flow missing edges array")
                    validation_result["valid"] = False
                
                # Statistics
                validation_result["statistics"] = {
                    "node_count": len(flow.get("nodes", [])),
                    "edge_count": len(flow.get("edges", [])),
                    "node_types": list(set(node.get("type", "unknown") for node in flow.get("nodes", []))),
                    "has_metadata": "metadata" in exported_data
                }
            
            # Check for potential issues
            if exported_data.get("compressed"):
                validation_result["warnings"].append("Workflow is compressed - may not be directly editable")
            
            logger.info(f"Export validation: {'valid' if validation_result['valid'] else 'invalid'}")
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Validation error: {str(e)}")
        
        return validation_result