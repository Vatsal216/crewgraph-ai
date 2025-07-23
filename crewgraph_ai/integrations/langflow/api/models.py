"""
Pydantic Models for CrewGraph AI + Langflow Integration API

This module defines the request/response models for the Langflow integration API.

Created by: Vatsal216
Date: 2025-07-23
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ComponentType(str, Enum):
    """Component types for Langflow"""
    AGENT = "agent"
    TOOL = "tool"
    MEMORY = "memory"
    CHAIN = "chain"
    CUSTOM = "custom"


class ExecutionMode(str, Enum):
    """Workflow execution modes"""
    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"


# Base Response Model
class ApiResponse(BaseModel):
    """Base API response model"""
    success: bool = True
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Workflow Models
class WorkflowExportRequest(BaseModel):
    """Request model for exporting CrewGraph workflow to Langflow"""
    workflow_id: str = Field(..., description="CrewGraph workflow ID to export")
    include_metadata: bool = Field(True, description="Include workflow metadata")
    format_version: str = Field("1.0", description="Langflow format version")
    compression: bool = Field(False, description="Compress the exported workflow")
    
    @validator('workflow_id')
    def validate_workflow_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Workflow ID cannot be empty")
        return v.strip()


class WorkflowImportRequest(BaseModel):
    """Request model for importing Langflow workflow to CrewGraph"""
    langflow_data: Dict[str, Any] = Field(..., description="Langflow workflow data")
    name: Optional[str] = Field(None, description="Name for the imported workflow")
    description: Optional[str] = Field(None, description="Description for the workflow")
    validate_components: bool = Field(True, description="Validate component compatibility")
    auto_fix_issues: bool = Field(False, description="Automatically fix common issues")
    
    @validator('langflow_data')
    def validate_langflow_data(cls, v):
        if not v:
            raise ValueError("Langflow data cannot be empty")
        if 'nodes' not in v:
            raise ValueError("Langflow data must contain 'nodes'")
        return v


class WorkflowExecuteRequest(BaseModel):
    """Request model for executing workflow"""
    workflow_id: str = Field(..., description="Workflow ID to execute")
    input_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Input data for workflow")
    mode: ExecutionMode = Field(ExecutionMode.ASYNC, description="Execution mode")
    timeout_seconds: Optional[int] = Field(3600, description="Execution timeout in seconds")
    enable_streaming: bool = Field(False, description="Enable real-time streaming of results")
    callback_url: Optional[str] = Field(None, description="Callback URL for completion notification")
    
    @validator('timeout_seconds')
    def validate_timeout(cls, v):
        if v is not None and v < 1:
            raise ValueError("Timeout must be at least 1 second")
        return v


class ComponentRegisterRequest(BaseModel):
    """Request model for registering custom Langflow components"""
    component_name: str = Field(..., description="Name of the component")
    component_type: ComponentType = Field(..., description="Type of component")
    component_code: str = Field(..., description="Python code for the component")
    display_name: Optional[str] = Field(None, description="Display name for UI")
    description: Optional[str] = Field(None, description="Component description")
    input_fields: List[Dict[str, Any]] = Field(default_factory=list, description="Input field definitions")
    output_fields: List[Dict[str, Any]] = Field(default_factory=list, description="Output field definitions")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('component_name')
    def validate_component_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Component name cannot be empty")
        # Check for valid Python identifier
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Component name must be alphanumeric with underscores/hyphens")
        return v.strip()


# Response Models
class WorkflowExportResponse(ApiResponse):
    """Response model for workflow export"""
    export_data: Optional[Dict[str, Any]] = Field(None, description="Exported Langflow data")
    export_url: Optional[str] = Field(None, description="URL to download exported workflow")
    format_version: str = Field("1.0", description="Export format version")
    file_size_bytes: Optional[int] = Field(None, description="Size of exported file")


class WorkflowImportResponse(ApiResponse):
    """Response model for workflow import"""
    workflow_id: Optional[str] = Field(None, description="ID of imported workflow")
    validation_issues: List[str] = Field(default_factory=list, description="Validation issues found")
    fixed_issues: List[str] = Field(default_factory=list, description="Issues automatically fixed")
    components_created: List[str] = Field(default_factory=list, description="Components created during import")


class WorkflowExecuteResponse(ApiResponse):
    """Response model for workflow execution"""
    execution_id: Optional[str] = Field(None, description="Execution ID for tracking")
    status: WorkflowStatus = Field(WorkflowStatus.PENDING, description="Current execution status")
    execution_url: Optional[str] = Field(None, description="URL to track execution progress")
    estimated_duration_seconds: Optional[int] = Field(None, description="Estimated execution time")


class ExecutionResult(BaseModel):
    """Model for workflow execution results"""
    execution_id: str = Field(..., description="Execution ID")
    workflow_id: str = Field(..., description="Workflow ID")
    status: WorkflowStatus = Field(..., description="Execution status")
    start_time: datetime = Field(..., description="Execution start time")
    end_time: Optional[datetime] = Field(None, description="Execution end time")
    duration_seconds: Optional[float] = Field(None, description="Execution duration")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data used")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Execution output")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    logs: List[str] = Field(default_factory=list, description="Execution logs")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Execution metrics")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ComponentListResponse(ApiResponse):
    """Response model for component listing"""
    components: List[Dict[str, Any]] = Field(default_factory=list, description="Available components")
    total_count: int = Field(0, description="Total number of components")
    crewgraph_components: List[str] = Field(default_factory=list, description="CrewGraph native components")
    custom_components: List[str] = Field(default_factory=list, description="Custom registered components")


class ComponentRegisterResponse(ApiResponse):
    """Response model for component registration"""
    component_id: Optional[str] = Field(None, description="Registered component ID")
    validation_result: Dict[str, Any] = Field(default_factory=dict, description="Component validation result")
    registration_url: Optional[str] = Field(None, description="Component access URL")


class WorkflowStatusResponse(ApiResponse):
    """Response model for workflow status"""
    execution_result: Optional[ExecutionResult] = Field(None, description="Current execution state")
    real_time_logs: List[str] = Field(default_factory=list, description="Recent log entries")
    progress_percentage: Optional[float] = Field(None, description="Execution progress (0-100)")
    current_step: Optional[str] = Field(None, description="Current execution step")
    next_steps: List[str] = Field(default_factory=list, description="Upcoming execution steps")


# Health Check Models
class HealthCheckResponse(ApiResponse):
    """Health check response model"""
    version: str = Field("1.0.0", description="API version")
    uptime_seconds: float = Field(..., description="Service uptime")
    langflow_connection: bool = Field(..., description="Langflow connectivity status")
    crewgraph_connection: bool = Field(..., description="CrewGraph connectivity status")
    active_executions: int = Field(0, description="Number of active executions")
    system_load: Optional[Dict[str, float]] = Field(None, description="System resource usage")


# Error Models
class ValidationError(BaseModel):
    """Model for validation errors"""
    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(ApiResponse):
    """Error response model"""
    success: bool = False
    error_code: str = Field(..., description="Error code")
    error_type: str = Field(..., description="Error type")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    validation_errors: List[ValidationError] = Field(default_factory=list, description="Validation errors")
    suggestions: List[str] = Field(default_factory=list, description="Suggested fixes")


# Pagination Models
class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1, description="Page number (1-based)")
    size: int = Field(20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: str = Field("asc", pattern="^(asc|desc)$", description="Sort order")


class PaginatedResponse(ApiResponse):
    """Base paginated response"""
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Items per page")
    total: int = Field(..., description="Total number of items")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")