"""
CrewGraph AI + Langflow Integration Module

This module provides seamless integration between CrewGraph AI and Langflow visual workflow builder.

Key Features:
- Visual drag-and-drop workflow design interface
- Bidirectional workflow synchronization between Langflow and CrewGraph
- Enterprise-grade API bridge for seamless communication
- Production-ready deployment configuration

Created by: Vatsal216
Date: 2025-07-23
"""

# Import with fallback for development
try:
    from .api.main import create_langflow_api
    from .components.base import LangflowComponent
    from .components.agents import CrewGraphAgentComponent
    from .components.tools import CrewGraphToolComponent
    from .workflow.exporter import WorkflowExporter
    from .workflow.importer import WorkflowImporter
    from .workflow.validator import WorkflowValidator
    
    _INTEGRATION_AVAILABLE = True
except ImportError as e:
    # Create placeholder classes for development
    def create_langflow_api():
        raise ImportError(f"Langflow integration not available: {e}")
    
    class LangflowComponent:
        pass
    
    class CrewGraphAgentComponent:
        pass
    
    class CrewGraphToolComponent:
        pass
    
    class WorkflowExporter:
        pass
    
    class WorkflowImporter:
        pass
    
    class WorkflowValidator:
        pass
    
    _INTEGRATION_AVAILABLE = False

__all__ = [
    # API
    "create_langflow_api",
    
    # Components
    "LangflowComponent",
    "CrewGraphAgentComponent", 
    "CrewGraphToolComponent",
    
    # Workflow Management
    "WorkflowExporter",
    "WorkflowImporter",
    "WorkflowValidator",
]

__version__ = "1.0.0"
__author__ = "Vatsal216"