"""
API Package for CrewGraph AI + Langflow Integration

This package provides the REST API bridge between CrewGraph AI and Langflow.

Created by: Vatsal216
Date: 2025-07-23
"""

from .main import create_langflow_api, app
from .auth import AuthManager, get_current_user, require_permissions
from .models import (
    WorkflowExportRequest,
    WorkflowImportRequest,
    WorkflowExecuteRequest,
    ComponentRegisterRequest,
    ApiResponse,
    WorkflowStatus,
    ExecutionResult
)

__all__ = [
    # Main API
    "create_langflow_api",
    "app",
    
    # Authentication
    "AuthManager",
    "get_current_user", 
    "require_permissions",
    
    # Models
    "WorkflowExportRequest",
    "WorkflowImportRequest", 
    "WorkflowExecuteRequest",
    "ComponentRegisterRequest",
    "ApiResponse",
    "WorkflowStatus",
    "ExecutionResult",
]