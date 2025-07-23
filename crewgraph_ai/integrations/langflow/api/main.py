"""
Main FastAPI Application for CrewGraph AI + Langflow Integration

This module creates and configures the FastAPI application for the Langflow integration API.

Created by: Vatsal216
Date: 2025-07-23
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .auth import auth_manager, get_current_user, TokenData
from .models import (
    HealthCheckResponse,
    ApiResponse,
    ErrorResponse,
    WorkflowExportRequest,
    WorkflowExportResponse,
    WorkflowImportRequest,
    WorkflowImportResponse,
    WorkflowExecuteRequest,
    WorkflowExecuteResponse,
    WorkflowStatusResponse,
    ComponentRegisterRequest,
    ComponentRegisterResponse,
    ComponentListResponse,
)
from ..config import get_config
from ..workflow.exporter import WorkflowExporter
from ..workflow.importer import WorkflowImporter
from ..workflow.validator import WorkflowValidator

# Import existing CrewGraph components
try:
    from ....utils.logging import get_logger
    from ....utils.metrics import get_metrics_collector
except ImportError:
    def get_logger(name):
        import logging
        return logging.getLogger(name)
    
    def get_metrics_collector():
        return None

logger = get_logger(__name__)
basic_auth = HTTPBasic()

# Global state
app_start_time = time.time()
active_executions = {}
workflow_exporter = None
workflow_importer = None
workflow_validator = None
metrics_collector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    global workflow_exporter, workflow_importer, workflow_validator, metrics_collector
    
    logger.info("🚀 Starting CrewGraph AI + Langflow Integration API")
    
    # Initialize components
    try:
        workflow_exporter = WorkflowExporter()
        workflow_importer = WorkflowImporter()
        workflow_validator = WorkflowValidator()
        metrics_collector = get_metrics_collector()
        
        if metrics_collector:
            metrics_collector.record_metric("langflow_api_startup", 1.0)
        
        logger.info("✅ All components initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down CrewGraph AI + Langflow Integration API")
    
    # Cancel any active executions
    for execution_id in list(active_executions.keys()):
        try:
            # In a real implementation, we would cancel the execution
            del active_executions[execution_id]
        except Exception as e:
            logger.error(f"Error cancelling execution {execution_id}: {e}")
    
    if metrics_collector:
        metrics_collector.record_metric("langflow_api_shutdown", 1.0)


def create_langflow_api() -> FastAPI:
    """
    Create and configure the FastAPI application for Langflow integration
    
    Returns:
        Configured FastAPI application
    """
    config = get_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="CrewGraph AI + Langflow Integration API",
        description="Enterprise-grade API bridge for seamless communication between CrewGraph AI and Langflow",
        version="1.0.0",
        openapi_url=f"{config.api_prefix}/openapi.json",
        docs_url=f"{config.api_prefix}/docs",
        redoc_url=f"{config.api_prefix}/redoc",
        lifespan=lifespan
    )
    
    # Add middleware
    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add trusted host middleware for security
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["localhost", "127.0.0.1", config.api_host]
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="INTERNAL_ERROR",
                error_type="InternalServerError",
                message="An internal server error occurred",
                details={"error": str(exc)} if config.debug_mode else None
            ).dict()
        )
    
    # Authentication endpoint
    @app.post(f"{config.api_prefix}/auth/login", response_model=ApiResponse)
    async def login(credentials: HTTPBasicCredentials = Depends(basic_auth)):
        """Authenticate user and return access token"""
        token = auth_manager.authenticate_user(credentials.username, credentials.password)
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        return ApiResponse(
            success=True,
            message="Authentication successful",
            timestamp=time.time(),
            access_token=token,
            token_type="bearer"
        )
    
    # Health check endpoint
    @app.get(f"{config.api_prefix}/health", response_model=HealthCheckResponse)
    async def health_check():
        """Health check endpoint"""
        uptime = time.time() - app_start_time
        
        # Check Langflow connection
        langflow_connection = True  # In real implementation, ping Langflow API
        
        # Check CrewGraph connection
        crewgraph_connection = True  # In real implementation, check CrewGraph components
        
        # Get system load if metrics available
        system_load = None
        if metrics_collector:
            system_load = {
                "cpu_percent": 0.0,  # Get from metrics
                "memory_percent": 0.0,
                "active_connections": len(active_executions)
            }
        
        return HealthCheckResponse(
            success=True,
            message="Service is healthy",
            uptime_seconds=uptime,
            langflow_connection=langflow_connection,
            crewgraph_connection=crewgraph_connection,
            active_executions=len(active_executions),
            system_load=system_load
        )
    
    # Workflow endpoints
    @app.post(f"{config.api_prefix}/workflows/export/{{workflow_id}}", response_model=WorkflowExportResponse)
    async def export_workflow(
        workflow_id: str,
        request: WorkflowExportRequest,
        current_user: TokenData = Depends(get_current_user)
    ):
        """Export CrewGraph workflow to Langflow format"""
        try:
            if metrics_collector:
                metrics_collector.record_metric("langflow_export_requests", 1.0)
            
            # Validate workflow exists
            # In real implementation, check if workflow exists in CrewGraph
            
            # Export workflow
            export_data = await workflow_exporter.export_workflow(
                workflow_id=workflow_id,
                include_metadata=request.include_metadata,
                format_version=request.format_version,
                compression=request.compression
            )
            
            return WorkflowExportResponse(
                success=True,
                message=f"Workflow {workflow_id} exported successfully",
                export_data=export_data,
                format_version=request.format_version,
                file_size_bytes=len(str(export_data).encode()) if export_data else 0
            )
            
        except Exception as e:
            logger.error(f"Export failed for workflow {workflow_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Export failed: {str(e)}"
            )
    
    @app.post(f"{config.api_prefix}/workflows/import", response_model=WorkflowImportResponse)
    async def import_workflow(
        request: WorkflowImportRequest,
        current_user: TokenData = Depends(get_current_user)
    ):
        """Import Langflow workflow to CrewGraph"""
        try:
            if metrics_collector:
                metrics_collector.record_metric("langflow_import_requests", 1.0)
            
            # Validate workflow data if requested
            validation_issues = []
            if request.validate_components:
                validation_issues = await workflow_validator.validate_langflow_data(
                    request.langflow_data
                )
            
            # Import workflow
            result = await workflow_importer.import_workflow(
                langflow_data=request.langflow_data,
                name=request.name,
                description=request.description,
                auto_fix_issues=request.auto_fix_issues
            )
            
            return WorkflowImportResponse(
                success=True,
                message="Workflow imported successfully",
                workflow_id=result.get("workflow_id"),
                validation_issues=validation_issues,
                fixed_issues=result.get("fixed_issues", []),
                components_created=result.get("components_created", [])
            )
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Import failed: {str(e)}"
            )
    
    @app.post(f"{config.api_prefix}/workflows/execute", response_model=WorkflowExecuteResponse)
    async def execute_workflow(
        request: WorkflowExecuteRequest,
        current_user: TokenData = Depends(get_current_user)
    ):
        """Execute workflow with real-time sync"""
        try:
            if metrics_collector:
                metrics_collector.record_metric("langflow_execute_requests", 1.0)
            
            # Generate execution ID
            import uuid
            execution_id = str(uuid.uuid4())
            
            # Store execution metadata
            active_executions[execution_id] = {
                "workflow_id": request.workflow_id,
                "user_id": current_user.user_id,
                "start_time": time.time(),
                "status": "pending",
                "input_data": request.input_data
            }
            
            # In a real implementation, this would start the actual workflow execution
            # For now, we'll simulate it
            if request.mode.value == "sync":
                # Execute synchronously (mock)
                active_executions[execution_id]["status"] = "completed"
                active_executions[execution_id]["end_time"] = time.time()
            else:
                # Execute asynchronously (mock)
                active_executions[execution_id]["status"] = "running"
            
            return WorkflowExecuteResponse(
                success=True,
                message="Workflow execution started",
                execution_id=execution_id,
                status=active_executions[execution_id]["status"],
                execution_url=f"{config.get_api_url()}/workflows/{request.workflow_id}/status?execution_id={execution_id}",
                estimated_duration_seconds=300  # Mock estimate
            )
            
        except Exception as e:
            logger.error(f"Execution failed for workflow {request.workflow_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Execution failed: {str(e)}"
            )
    
    @app.get(f"{config.api_prefix}/workflows/{{workflow_id}}/status", response_model=WorkflowStatusResponse)
    async def get_workflow_status(
        workflow_id: str,
        execution_id: str = None,
        current_user: TokenData = Depends(get_current_user)
    ):
        """Get execution status"""
        try:
            # Find execution
            execution = None
            if execution_id:
                execution = active_executions.get(execution_id)
            else:
                # Find latest execution for this workflow
                for exec_id, exec_data in active_executions.items():
                    if exec_data["workflow_id"] == workflow_id:
                        execution = exec_data
                        execution["execution_id"] = exec_id
                        break
            
            if not execution:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Execution not found"
                )
            
            # Create execution result
            from .models import ExecutionResult, WorkflowStatus
            from datetime import datetime
            
            execution_result = ExecutionResult(
                execution_id=execution.get("execution_id", execution_id),
                workflow_id=workflow_id,
                status=WorkflowStatus(execution["status"]),
                start_time=datetime.fromtimestamp(execution["start_time"]),
                end_time=datetime.fromtimestamp(execution["end_time"]) if "end_time" in execution else None,
                duration_seconds=execution.get("end_time", time.time()) - execution["start_time"],
                input_data=execution["input_data"],
                output_data=execution.get("output_data"),
                error_message=execution.get("error_message"),
                logs=execution.get("logs", []),
                metrics=execution.get("metrics")
            )
            
            # Calculate progress
            progress = 100.0 if execution["status"] in ["completed", "failed"] else 50.0
            
            return WorkflowStatusResponse(
                success=True,
                message="Status retrieved successfully",
                execution_result=execution_result,
                progress_percentage=progress,
                current_step=execution.get("current_step", "Processing"),
                next_steps=execution.get("next_steps", [])
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Status check failed: {str(e)}"
            )
    
    # Component endpoints
    @app.post(f"{config.api_prefix}/components/register", response_model=ComponentRegisterResponse)
    async def register_component(
        request: ComponentRegisterRequest,
        current_user: TokenData = Depends(get_current_user)
    ):
        """Register custom Langflow component"""
        try:
            if metrics_collector:
                metrics_collector.record_metric("langflow_component_registrations", 1.0)
            
            # Validate component code
            validation_result = await workflow_validator.validate_component_code(
                request.component_code,
                request.component_type
            )
            
            if not validation_result["valid"]:
                return ComponentRegisterResponse(
                    success=False,
                    message="Component validation failed",
                    validation_result=validation_result
                )
            
            # Register component (mock implementation)
            import uuid
            component_id = str(uuid.uuid4())
            
            # In real implementation, this would register with Langflow
            
            return ComponentRegisterResponse(
                success=True,
                message="Component registered successfully",
                component_id=component_id,
                validation_result=validation_result,
                registration_url=f"{config.langflow_url}/components/{component_id}"
            )
            
        except Exception as e:
            logger.error(f"Component registration failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Registration failed: {str(e)}"
            )
    
    @app.get(f"{config.api_prefix}/components/list", response_model=ComponentListResponse)
    async def list_components(current_user: TokenData = Depends(get_current_user)):
        """List available components"""
        try:
            # In real implementation, this would query available components
            crewgraph_components = [
                "CrewGraphAgent",
                "CrewGraphTask", 
                "CrewGraphTool",
                "CrewGraphMemory",
                "CrewGraphChain"
            ]
            
            custom_components = [
                "CustomAgent",
                "CustomTool"
            ]
            
            components = []
            for comp in crewgraph_components:
                components.append({
                    "name": comp,
                    "type": "crewgraph",
                    "description": f"CrewGraph {comp} component",
                    "version": "1.0.0"
                })
            
            for comp in custom_components:
                components.append({
                    "name": comp,
                    "type": "custom",
                    "description": f"Custom {comp} component",
                    "version": "1.0.0"
                })
            
            return ComponentListResponse(
                success=True,
                message="Components listed successfully",
                components=components,
                total_count=len(components),
                crewgraph_components=crewgraph_components,
                custom_components=custom_components
            )
            
        except Exception as e:
            logger.error(f"Component listing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Listing failed: {str(e)}"
            )
    
    return app


# Create the app instance
app = create_langflow_api()


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.hot_reload,
        log_level=config.log_level.lower()
    )