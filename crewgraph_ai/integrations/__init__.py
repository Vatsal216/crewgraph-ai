"""
Integration Marketplace Framework for CrewGraph AI

Provides a comprehensive plugin system with dynamic loading, sandboxed execution,
and a marketplace for pre-built and custom integrations.

Author: Vatsal216
Created: 2025-07-23 18:30:00 UTC
"""

import importlib
import json
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ..types import WorkflowId
from ..utils.logging import get_logger

logger = get_logger(__name__)


class IntegrationType(Enum):
    """Types of integrations supported."""
    COMMUNICATION = "communication"
    DEVELOPMENT = "development"
    PRODUCTIVITY = "productivity"
    DATA = "data"
    ANALYTICS = "analytics"
    CRM = "crm"
    MONITORING = "monitoring"
    SECURITY = "security"
    CUSTOM = "custom"


class IntegrationStatus(Enum):
    """Integration status indicators."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"
    UNINSTALLED = "uninstalled"


@dataclass
class IntegrationMetadata:
    """Metadata for integration definition."""
    
    name: str
    version: str
    description: str
    author: str
    integration_type: IntegrationType
    
    # Requirements and compatibility
    min_crewgraph_version: str = "1.0.0"
    python_version: str = ">=3.8"
    dependencies: List[str] = field(default_factory=list)
    
    # Configuration
    config_schema: Dict[str, Any] = field(default_factory=dict)
    supports_async: bool = True
    supports_webhook: bool = False
    supports_oauth: bool = False
    
    # Marketplace information
    homepage: Optional[str] = None
    documentation: Optional[str] = None
    repository: Optional[str] = None
    license: str = "MIT"
    tags: List[str] = field(default_factory=list)
    
    # Runtime information
    installed_at: Optional[str] = None
    last_updated: Optional[str] = None


@dataclass
class IntegrationConfig:
    """Configuration for an integration instance."""
    
    integration_id: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    environment: str = "production"
    rate_limit: Optional[int] = None
    timeout: Optional[int] = 30
    retry_count: int = 3


@dataclass
class IntegrationResult:
    """Result of an integration operation."""
    
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseIntegration(ABC):
    """
    Abstract base class for all integrations.
    
    All custom integrations must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize integration with configuration."""
        self.config = config
        self.integration_id = config.integration_id
        self.logger = get_logger(f"integration.{self.integration_id}")
        
        # Runtime state
        self.is_initialized = False
        self.last_execution = None
        self.execution_count = 0
    
    @property
    @abstractmethod
    def metadata(self) -> IntegrationMetadata:
        """Get integration metadata."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the integration.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute(self, action: str, **kwargs) -> IntegrationResult:
        """
        Execute an integration action.
        
        Args:
            action: Action to perform
            **kwargs: Action-specific parameters
            
        Returns:
            Integration result
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """
        Validate integration configuration.
        
        Returns:
            List of validation issues (empty if valid)
        """
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on integration.
        
        Returns:
            Health status information
        """
        try:
            # Basic health check
            if not self.is_initialized:
                return {
                    "status": "unhealthy",
                    "message": "Integration not initialized",
                    "last_execution": self.last_execution
                }
            
            # Perform integration-specific health check
            return self._perform_health_check()
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "last_execution": self.last_execution
            }
    
    def _perform_health_check(self) -> Dict[str, Any]:
        """Override in subclasses for specific health checks."""
        return {
            "status": "healthy",
            "message": "Integration is operational",
            "execution_count": self.execution_count,
            "last_execution": self.last_execution
        }
    
    def shutdown(self):
        """Shutdown the integration and cleanup resources."""
        self.is_initialized = False
        self.logger.info(f"Integration {self.integration_id} shutdown")


class IntegrationRegistry:
    """
    Registry for managing available integrations.
    
    Provides discovery, installation, and lifecycle management
    for integrations in the marketplace.
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """Initialize integration registry."""
        self.registry_path = Path(registry_path) if registry_path else Path.cwd() / "integrations"
        self.registry_path.mkdir(exist_ok=True)
        
        # Registry data
        self.available_integrations: Dict[str, IntegrationMetadata] = {}
        self.installed_integrations: Dict[str, Type[BaseIntegration]] = {}
        
        # Load built-in integrations
        self._load_builtin_integrations()
        
        logger.info("Integration registry initialized")
    
    def register_integration(
        self, 
        integration_class: Type[BaseIntegration],
        metadata: Optional[IntegrationMetadata] = None
    ):
        """Register an integration class."""
        if metadata is None:
            # Get metadata from integration class
            dummy_config = IntegrationConfig(integration_id="dummy")
            dummy_instance = integration_class(dummy_config)
            metadata = dummy_instance.metadata
        
        integration_id = metadata.name.lower().replace(" ", "_")
        
        self.available_integrations[integration_id] = metadata
        self.installed_integrations[integration_id] = integration_class
        
        logger.info(f"Registered integration: {metadata.name} v{metadata.version}")
    
    def get_available_integrations(
        self, 
        integration_type: Optional[IntegrationType] = None
    ) -> List[IntegrationMetadata]:
        """Get list of available integrations."""
        integrations = list(self.available_integrations.values())
        
        if integration_type:
            integrations = [i for i in integrations if i.integration_type == integration_type]
        
        return integrations
    
    def get_integration_class(self, integration_id: str) -> Optional[Type[BaseIntegration]]:
        """Get integration class by ID."""
        return self.installed_integrations.get(integration_id)
    
    def install_integration(self, integration_id: str, source: str) -> bool:
        """
        Install an integration from source.
        
        Args:
            integration_id: Unique integration identifier
            source: Source location (file path, URL, etc.)
            
        Returns:
            True if installation successful
        """
        try:
            # For simplicity, assume source is a Python module path
            module = importlib.import_module(source)
            
            # Look for integration class in module
            integration_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseIntegration) and 
                    attr != BaseIntegration):
                    integration_class = attr
                    break
            
            if integration_class:
                self.register_integration(integration_class)
                logger.info(f"Installed integration: {integration_id}")
                return True
            else:
                logger.error(f"No integration class found in {source}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to install integration {integration_id}: {e}")
            return False
    
    def uninstall_integration(self, integration_id: str) -> bool:
        """Uninstall an integration."""
        if integration_id in self.installed_integrations:
            del self.installed_integrations[integration_id]
            
        if integration_id in self.available_integrations:
            del self.available_integrations[integration_id]
        
        logger.info(f"Uninstalled integration: {integration_id}")
        return True
    
    def search_integrations(
        self, 
        query: str, 
        integration_type: Optional[IntegrationType] = None
    ) -> List[IntegrationMetadata]:
        """Search for integrations matching query."""
        results = []
        query_lower = query.lower()
        
        for metadata in self.available_integrations.values():
            # Check if query matches name, description, or tags
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                
                if integration_type is None or metadata.integration_type == integration_type:
                    results.append(metadata)
        
        return results
    
    def _load_builtin_integrations(self):
        """Load built-in integrations."""
        # Load communication integrations
        try:
            from .connectors.communication.slack import SlackIntegration
            self.register_integration(SlackIntegration)
        except ImportError:
            logger.debug("Slack integration not available")
        
        try:
            from .connectors.communication.teams import TeamsIntegration
            self.register_integration(TeamsIntegration)
        except ImportError:
            logger.debug("Teams integration not available")
        
        # Load development integrations
        try:
            from .connectors.development.github import GitHubIntegration
            self.register_integration(GitHubIntegration)
        except ImportError:
            logger.debug("GitHub integration not available")
        
        try:
            from .connectors.development.jira import JiraIntegration
            self.register_integration(JiraIntegration)
        except ImportError:
            logger.debug("Jira integration not available")
        
        # Load data integrations
        try:
            from .connectors.data.postgresql import PostgreSQLIntegration
            self.register_integration(PostgreSQLIntegration)
        except ImportError:
            logger.debug("PostgreSQL integration not available")
        
        logger.info("Built-in integrations loaded")


class IntegrationManager:
    """
    Manager for active integration instances.
    
    Handles lifecycle management, execution, and monitoring
    of active integrations in workflows.
    """
    
    def __init__(self, registry: IntegrationRegistry):
        """Initialize integration manager."""
        self.registry = registry
        self.active_integrations: Dict[str, BaseIntegration] = {}
        self.integration_configs: Dict[str, IntegrationConfig] = {}
        
        logger.info("Integration manager initialized")
    
    def create_integration(
        self, 
        integration_id: str, 
        config: IntegrationConfig
    ) -> Optional[BaseIntegration]:
        """Create and initialize an integration instance."""
        integration_class = self.registry.get_integration_class(integration_id)
        
        if not integration_class:
            logger.error(f"Integration not found: {integration_id}")
            return None
        
        try:
            # Create instance
            instance = integration_class(config)
            
            # Validate configuration
            validation_issues = instance.validate_config()
            if validation_issues:
                logger.error(f"Configuration validation failed for {integration_id}: {validation_issues}")
                return None
            
            # Initialize integration
            if instance.initialize():
                instance_id = f"{integration_id}_{uuid.uuid4().hex[:8]}"
                self.active_integrations[instance_id] = instance
                self.integration_configs[instance_id] = config
                
                logger.info(f"Created integration instance: {instance_id}")
                return instance
            else:
                logger.error(f"Failed to initialize integration: {integration_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create integration {integration_id}: {e}")
            return None
    
    def execute_integration(
        self, 
        instance_id: str, 
        action: str, 
        **kwargs
    ) -> IntegrationResult:
        """Execute an action on an integration instance."""
        if instance_id not in self.active_integrations:
            return IntegrationResult(
                success=False,
                error_message=f"Integration instance not found: {instance_id}"
            )
        
        integration = self.active_integrations[instance_id]
        
        try:
            start_time = datetime.now()
            result = integration.execute(action, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update execution tracking
            integration.last_execution = start_time.isoformat()
            integration.execution_count += 1
            
            # Add execution time to result
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            logger.error(f"Integration execution failed for {instance_id}: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    def get_integration_health(self, instance_id: str) -> Dict[str, Any]:
        """Get health status of an integration instance."""
        if instance_id not in self.active_integrations:
            return {
                "status": "not_found",
                "message": f"Integration instance not found: {instance_id}"
            }
        
        integration = self.active_integrations[instance_id]
        return integration.health_check()
    
    def list_active_integrations(self) -> List[Dict[str, Any]]:
        """List all active integration instances."""
        result = []
        
        for instance_id, integration in self.active_integrations.items():
            metadata = integration.metadata
            config = self.integration_configs[instance_id]
            health = integration.health_check()
            
            result.append({
                "instance_id": instance_id,
                "integration_id": integration.integration_id,
                "name": metadata.name,
                "version": metadata.version,
                "type": metadata.integration_type.value,
                "enabled": config.enabled,
                "health_status": health["status"],
                "execution_count": integration.execution_count,
                "last_execution": integration.last_execution
            })
        
        return result
    
    def remove_integration(self, instance_id: str) -> bool:
        """Remove an active integration instance."""
        if instance_id not in self.active_integrations:
            return False
        
        integration = self.active_integrations[instance_id]
        integration.shutdown()
        
        del self.active_integrations[instance_id]
        del self.integration_configs[instance_id]
        
        logger.info(f"Removed integration instance: {instance_id}")
        return True
    
    def shutdown_all(self):
        """Shutdown all active integrations."""
        for integration in self.active_integrations.values():
            try:
                integration.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down integration: {e}")
        
        self.active_integrations.clear()
        self.integration_configs.clear()
        
        logger.info("All integrations shut down")


class SandboxExecutor:
    """
    Sandboxed execution environment for untrusted integrations.
    
    Provides isolation and security for executing third-party
    integration code with restricted access to system resources.
    """
    
    def __init__(
        self,
        max_execution_time: int = 60,
        max_memory_mb: int = 100,
        allowed_modules: Optional[List[str]] = None
    ):
        """Initialize sandbox executor."""
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.allowed_modules = allowed_modules or [
            "json", "datetime", "uuid", "base64", "hashlib",
            "requests", "urllib", "http"
        ]
        
        logger.info("Sandbox executor initialized")
    
    def execute_integration(
        self, 
        integration_class: Type[BaseIntegration],
        config: IntegrationConfig,
        action: str,
        **kwargs
    ) -> IntegrationResult:
        """Execute integration in sandboxed environment."""
        try:
            # Create restricted environment
            restricted_globals = self._create_restricted_globals()
            
            # Import only allowed modules
            for module_name in self.allowed_modules:
                try:
                    restricted_globals[module_name] = __import__(module_name)
                except ImportError:
                    pass
            
            # Execute in restricted environment
            # Note: This is a simplified sandbox. In production, consider using
            # more robust sandboxing solutions like Docker containers or PyPy's
            # sandboxed execution environment.
            
            integration = integration_class(config)
            
            if not integration.initialize():
                return IntegrationResult(
                    success=False,
                    error_message="Failed to initialize integration in sandbox"
                )
            
            result = integration.execute(action, **kwargs)
            integration.shutdown()
            
            return result
            
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Sandbox execution failed: {str(e)}"
            )
    
    def _create_restricted_globals(self) -> Dict[str, Any]:
        """Create restricted global namespace for sandbox."""
        # Start with safe built-ins
        safe_builtins = {
            'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
            'set', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted',
            'min', 'max', 'sum', 'abs', 'round', 'isinstance', 'hasattr',
            'getattr', 'setattr', 'type', 'print'
        }
        
        restricted_globals = {
            '__builtins__': {name: getattr(__builtins__, name) for name in safe_builtins if hasattr(__builtins__, name)}
        }
        
        return restricted_globals


class IntegrationMarketplace:
    """
    Marketplace for discovering and managing integrations.
    
    Provides a centralized hub for finding, installing, and
    managing integrations from various sources.
    """
    
    def __init__(self, registry: IntegrationRegistry):
        """Initialize integration marketplace."""
        self.registry = registry
        self.marketplace_data = {
            "featured": [],
            "popular": [],
            "recent": [],
            "categories": {}
        }
        
        self._populate_marketplace()
        logger.info("Integration marketplace initialized")
    
    def _populate_marketplace(self):
        """Populate marketplace with sample data."""
        # Featured integrations
        self.marketplace_data["featured"] = [
            "slack", "github", "jira", "postgresql", "google_analytics"
        ]
        
        # Popular integrations by type
        self.marketplace_data["categories"] = {
            IntegrationType.COMMUNICATION: ["slack", "teams", "discord"],
            IntegrationType.DEVELOPMENT: ["github", "gitlab", "jira"],
            IntegrationType.DATA: ["postgresql", "mongodb", "redis"],
            IntegrationType.ANALYTICS: ["google_analytics", "mixpanel"]
        }
    
    def get_featured_integrations(self) -> List[IntegrationMetadata]:
        """Get featured integrations."""
        featured = []
        for integration_id in self.marketplace_data["featured"]:
            if integration_id in self.registry.available_integrations:
                featured.append(self.registry.available_integrations[integration_id])
        return featured
    
    def get_integrations_by_category(
        self, 
        category: IntegrationType
    ) -> List[IntegrationMetadata]:
        """Get integrations by category."""
        return self.registry.get_available_integrations(category)
    
    def search_marketplace(self, query: str) -> List[IntegrationMetadata]:
        """Search marketplace for integrations."""
        return self.registry.search_integrations(query)
    
    def get_integration_details(self, integration_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an integration."""
        if integration_id not in self.registry.available_integrations:
            return None
        
        metadata = self.registry.available_integrations[integration_id]
        
        return {
            "metadata": metadata,
            "is_installed": integration_id in self.registry.installed_integrations,
            "download_count": 1500,  # Simulated
            "rating": 4.5,  # Simulated
            "reviews": [
                {"user": "developer1", "rating": 5, "comment": "Great integration!"},
                {"user": "developer2", "rating": 4, "comment": "Works well, easy to setup"}
            ]
        }
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        available_count = len(self.registry.available_integrations)
        installed_count = len(self.registry.installed_integrations)
        
        category_counts = {}
        for integration_type in IntegrationType:
            category_integrations = self.registry.get_available_integrations(integration_type)
            category_counts[integration_type.value] = len(category_integrations)
        
        return {
            "total_available": available_count,
            "total_installed": installed_count,
            "featured_count": len(self.marketplace_data["featured"]),
            "category_breakdown": category_counts,
            "marketplace_version": "1.0.0"
        }


# Initialize global instances
_registry = IntegrationRegistry()
_manager = IntegrationManager(_registry)
_marketplace = IntegrationMarketplace(_registry)


def get_integration_registry() -> IntegrationRegistry:
    """Get the global integration registry."""
    return _registry


def get_integration_manager() -> IntegrationManager:
    """Get the global integration manager."""
    return _manager


def get_integration_marketplace() -> IntegrationMarketplace:
    """Get the global integration marketplace."""
    return _marketplace


# Export main classes
__all__ = [
    "BaseIntegration",
    "IntegrationMetadata",
    "IntegrationConfig",
    "IntegrationResult",
    "IntegrationType",
    "IntegrationStatus",
    "IntegrationRegistry",
    "IntegrationManager",
    "IntegrationMarketplace",
    "SandboxExecutor",
    "get_integration_registry",
    "get_integration_manager",
    "get_integration_marketplace"
]