"""
Configuration for CrewGraph AI + Langflow Integration

This module provides configuration management for the Langflow integration,
including API settings, authentication, and runtime configuration.

Created by: Vatsal216
Date: 2025-07-23
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class LangflowIntegrationConfig:
    """
    Configuration for CrewGraph AI + Langflow Integration
    
    This configuration manages the integration between CrewGraph AI and Langflow,
    including API settings, authentication, component registration, and deployment options.
    """
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Authentication & Security
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    enable_rbac: bool = True
    
    # Langflow Configuration
    langflow_url: str = "http://localhost:7860"
    langflow_api_key: Optional[str] = None
    auto_sync_workflows: bool = True
    sync_interval_seconds: int = 60
    
    # Component Registration
    auto_register_components: bool = True
    component_registry_url: Optional[str] = None
    custom_components_path: Optional[str] = None
    
    # Workflow Management
    workflow_storage_path: str = "./workflows"
    enable_workflow_validation: bool = True
    max_workflow_size_mb: int = 50
    workflow_backup_enabled: bool = True
    
    # Performance & Scaling
    max_concurrent_workflows: int = 10
    enable_async_execution: bool = True
    execution_timeout_seconds: int = 3600
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    
    # Monitoring & Logging
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    enable_audit_logging: bool = True
    
    # Development & Debug
    debug_mode: bool = False
    enable_dev_tools: bool = False
    hot_reload: bool = False
    
    @classmethod
    def from_env(cls) -> "LangflowIntegrationConfig":
        """Create configuration from environment variables"""
        return cls(
            # API Configuration
            api_host=os.getenv("LANGFLOW_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("LANGFLOW_API_PORT", "8000")),
            api_prefix=os.getenv("LANGFLOW_API_PREFIX", "/api/v1"),
            enable_cors=os.getenv("LANGFLOW_ENABLE_CORS", "true").lower() == "true",
            cors_origins=os.getenv("LANGFLOW_CORS_ORIGINS", "*").split(","),
            
            # Authentication & Security
            jwt_secret_key=os.getenv("LANGFLOW_JWT_SECRET"),
            jwt_algorithm=os.getenv("LANGFLOW_JWT_ALGORITHM", "HS256"),
            jwt_expire_hours=int(os.getenv("LANGFLOW_JWT_EXPIRE_HOURS", "24")),
            enable_rbac=os.getenv("LANGFLOW_ENABLE_RBAC", "true").lower() == "true",
            
            # Langflow Configuration
            langflow_url=os.getenv("LANGFLOW_URL", "http://localhost:7860"),
            langflow_api_key=os.getenv("LANGFLOW_API_KEY"),
            auto_sync_workflows=os.getenv("LANGFLOW_AUTO_SYNC", "true").lower() == "true",
            sync_interval_seconds=int(os.getenv("LANGFLOW_SYNC_INTERVAL", "60")),
            
            # Component Registration
            auto_register_components=os.getenv("LANGFLOW_AUTO_REGISTER", "true").lower() == "true",
            component_registry_url=os.getenv("LANGFLOW_COMPONENT_REGISTRY_URL"),
            custom_components_path=os.getenv("LANGFLOW_CUSTOM_COMPONENTS_PATH"),
            
            # Workflow Management
            workflow_storage_path=os.getenv("LANGFLOW_WORKFLOW_STORAGE", "./workflows"),
            enable_workflow_validation=os.getenv("LANGFLOW_ENABLE_VALIDATION", "true").lower() == "true",
            max_workflow_size_mb=int(os.getenv("LANGFLOW_MAX_WORKFLOW_SIZE_MB", "50")),
            workflow_backup_enabled=os.getenv("LANGFLOW_BACKUP_ENABLED", "true").lower() == "true",
            
            # Performance & Scaling
            max_concurrent_workflows=int(os.getenv("LANGFLOW_MAX_CONCURRENT", "10")),
            enable_async_execution=os.getenv("LANGFLOW_ASYNC_EXECUTION", "true").lower() == "true",
            execution_timeout_seconds=int(os.getenv("LANGFLOW_EXECUTION_TIMEOUT", "3600")),
            enable_caching=os.getenv("LANGFLOW_ENABLE_CACHING", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("LANGFLOW_CACHE_TTL", "300")),
            
            # Monitoring & Logging
            enable_metrics=os.getenv("LANGFLOW_ENABLE_METRICS", "true").lower() == "true",
            metrics_port=int(os.getenv("LANGFLOW_METRICS_PORT", "9090")),
            log_level=os.getenv("LANGFLOW_LOG_LEVEL", "INFO"),
            enable_audit_logging=os.getenv("LANGFLOW_AUDIT_LOGGING", "true").lower() == "true",
            
            # Development & Debug
            debug_mode=os.getenv("LANGFLOW_DEBUG", "false").lower() == "true",
            enable_dev_tools=os.getenv("LANGFLOW_DEV_TOOLS", "false").lower() == "true",
            hot_reload=os.getenv("LANGFLOW_HOT_RELOAD", "false").lower() == "true",
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate required settings
        if not self.jwt_secret_key:
            issues.append("JWT secret key is required. Set LANGFLOW_JWT_SECRET environment variable.")
        
        # Validate ports
        if not 1 <= self.api_port <= 65535:
            issues.append(f"API port must be between 1-65535, got {self.api_port}")
        
        if self.enable_metrics and not 1 <= self.metrics_port <= 65535:
            issues.append(f"Metrics port must be between 1-65535, got {self.metrics_port}")
        
        # Validate paths
        if self.custom_components_path and not Path(self.custom_components_path).exists():
            issues.append(f"Custom components path does not exist: {self.custom_components_path}")
        
        # Validate workflow storage
        try:
            Path(self.workflow_storage_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create workflow storage directory: {e}")
        
        # Validate numeric ranges
        if self.sync_interval_seconds < 10:
            issues.append("Sync interval must be at least 10 seconds")
        
        if self.max_concurrent_workflows < 1:
            issues.append("Max concurrent workflows must be at least 1")
        
        if self.execution_timeout_seconds < 60:
            issues.append("Execution timeout must be at least 60 seconds")
        
        if self.cache_ttl_seconds < 10:
            issues.append("Cache TTL must be at least 10 seconds")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            issues.append(f"Log level must be one of: {', '.join(valid_log_levels)}")
        
        return issues
    
    def get_api_url(self) -> str:
        """Get the full API URL"""
        return f"http://{self.api_host}:{self.api_port}{self.api_prefix}"
    
    def get_metrics_url(self) -> str:
        """Get the metrics URL"""
        if self.enable_metrics:
            return f"http://{self.api_host}:{self.metrics_port}/metrics"
        return None


# Global configuration instance
_global_config: Optional[LangflowIntegrationConfig] = None


def get_config() -> LangflowIntegrationConfig:
    """Get the global Langflow integration configuration"""
    global _global_config
    if _global_config is None:
        _global_config = LangflowIntegrationConfig.from_env()
    return _global_config


def set_config(config: LangflowIntegrationConfig) -> None:
    """Set the global Langflow integration configuration"""
    global _global_config
    _global_config = config


def validate_config() -> bool:
    """
    Validate current configuration and print detailed report
    
    Returns:
        True if configuration is valid
    """
    config = get_config()
    issues = config.validate()
    
    if not issues:
        print("✅ Langflow integration configuration is valid")
        print(f"🌐 API URL: {config.get_api_url()}")
        if config.enable_metrics:
            print(f"📊 Metrics URL: {config.get_metrics_url()}")
        print(f"🔒 RBAC Enabled: {config.enable_rbac}")
        print(f"🔄 Auto Sync: {config.auto_sync_workflows}")
        return True
    
    print("❌ Langflow integration configuration issues:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    return False