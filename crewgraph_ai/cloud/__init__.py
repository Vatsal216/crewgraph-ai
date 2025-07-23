"""
Cloud Provider Abstraction Framework for CrewGraph AI

Provides unified interface for multi-cloud deployment across AWS, GCP, and Azure
with support for containers, serverless, and infrastructure as code.

Author: Vatsal216
Created: 2025-07-23 17:50:00 UTC
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..types import WorkflowId
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"


class DeploymentType(Enum):
    """Types of deployment strategies."""
    CONTAINER = "container"
    SERVERLESS = "serverless"
    VIRTUAL_MACHINE = "virtual_machine"
    KUBERNETES = "kubernetes"
    MANAGED_SERVICE = "managed_service"


class ResourceStatus(Enum):
    """Cloud resource status."""
    CREATING = "creating"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DELETED = "deleted"


@dataclass
class CloudCredentials:
    """Cloud provider credentials configuration."""
    
    provider: CloudProvider
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    region: Optional[str] = None
    project_id: Optional[str] = None
    subscription_id: Optional[str] = None
    tenant_id: Optional[str] = None
    service_account_path: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate credentials for specific provider."""
        issues = []
        
        if self.provider == CloudProvider.AWS:
            if not self.access_key:
                issues.append("AWS access_key is required")
            if not self.secret_key:
                issues.append("AWS secret_key is required")
            if not self.region:
                issues.append("AWS region is required")
                
        elif self.provider == CloudProvider.GCP:
            if not self.project_id:
                issues.append("GCP project_id is required")
            if not (self.service_account_path or (self.access_key and self.secret_key)):
                issues.append("GCP service account or access keys required")
                
        elif self.provider == CloudProvider.AZURE:
            if not self.subscription_id:
                issues.append("Azure subscription_id is required")
            if not self.tenant_id:
                issues.append("Azure tenant_id is required")
        
        return issues


@dataclass
class DeploymentConfig:
    """Deployment configuration for cloud resources."""
    
    name: str
    provider: CloudProvider
    deployment_type: DeploymentType
    region: str
    
    # Resource specifications
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    storage_gb: float = 10.0
    
    # Scaling configuration
    min_instances: int = 1
    max_instances: int = 5
    auto_scaling_enabled: bool = True
    
    # Network configuration
    vpc_id: Optional[str] = None
    subnet_ids: List[str] = field(default_factory=list)
    security_group_ids: List[str] = field(default_factory=list)
    
    # Container configuration
    image_uri: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Serverless configuration
    function_timeout: int = 300
    function_memory: int = 512
    
    # Cost optimization
    use_spot_instances: bool = False
    use_reserved_instances: bool = False


@dataclass
class CloudResource:
    """Represents a deployed cloud resource."""
    
    resource_id: str
    name: str
    provider: CloudProvider
    deployment_type: DeploymentType
    status: ResourceStatus
    region: str
    
    # Resource details
    endpoint_url: Optional[str] = None
    internal_ip: Optional[str] = None
    public_ip: Optional[str] = None
    
    # Cost information
    hourly_cost: float = 0.0
    total_cost: float = 0.0
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: Optional[str] = None
    last_updated: Optional[str] = None


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    
    success: bool
    resource: Optional[CloudResource] = None
    error_message: Optional[str] = None
    deployment_logs: List[str] = field(default_factory=list)
    infrastructure_code: Optional[str] = None


class BaseCloudProvider(ABC):
    """
    Abstract base class for cloud provider implementations.
    
    Defines the common interface that all cloud providers must implement
    for unified multi-cloud deployment capabilities.
    """
    
    def __init__(self, credentials: CloudCredentials):
        """Initialize cloud provider with credentials."""
        self.credentials = credentials
        self.provider = credentials.provider
        
        # Validate credentials
        issues = credentials.validate()
        if issues:
            raise ValueError(f"Invalid credentials: {', '.join(issues)}")
        
        logger.info(f"Initialized {self.provider.value} cloud provider")
    
    @abstractmethod
    def deploy_workflow(
        self, 
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> DeploymentResult:
        """
        Deploy a workflow to the cloud provider.
        
        Args:
            workflow_id: Unique workflow identifier
            config: Deployment configuration
            workflow_definition: Workflow structure and tasks
            
        Returns:
            Deployment result with resource information
        """
        pass
    
    @abstractmethod
    def get_resource_status(self, resource_id: str) -> CloudResource:
        """Get current status of a deployed resource."""
        pass
    
    @abstractmethod
    def scale_resource(
        self, 
        resource_id: str, 
        target_instances: int
    ) -> bool:
        """Scale a deployed resource to target number of instances."""
        pass
    
    @abstractmethod
    def stop_resource(self, resource_id: str) -> bool:
        """Stop a running cloud resource."""
        pass
    
    @abstractmethod
    def delete_resource(self, resource_id: str) -> bool:
        """Delete a cloud resource permanently."""
        pass
    
    @abstractmethod
    def list_resources(self, workflow_id: Optional[WorkflowId] = None) -> List[CloudResource]:
        """List all deployed resources, optionally filtered by workflow."""
        pass
    
    @abstractmethod
    def get_resource_logs(self, resource_id: str, lines: int = 100) -> List[str]:
        """Get recent logs from a deployed resource."""
        pass
    
    @abstractmethod
    def get_resource_metrics(
        self, 
        resource_id: str, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance metrics for a deployed resource."""
        pass
    
    @abstractmethod
    def estimate_cost(self, config: DeploymentConfig) -> Dict[str, float]:
        """Estimate deployment cost for given configuration."""
        pass
    
    @abstractmethod
    def generate_infrastructure_code(
        self, 
        config: DeploymentConfig,
        template_type: str = "terraform"
    ) -> str:
        """Generate Infrastructure as Code templates."""
        pass
    
    # Common utility methods
    def validate_config(self, config: DeploymentConfig) -> List[str]:
        """Validate deployment configuration for this provider."""
        issues = []
        
        if config.provider != self.provider:
            issues.append(f"Config provider {config.provider} doesn't match {self.provider}")
        
        if config.cpu_cores <= 0:
            issues.append("CPU cores must be positive")
        
        if config.memory_gb <= 0:
            issues.append("Memory must be positive")
        
        if config.min_instances > config.max_instances:
            issues.append("Min instances cannot exceed max instances")
        
        return issues
    
    def generate_resource_name(self, workflow_id: WorkflowId, config: DeploymentConfig) -> str:
        """Generate a standardized resource name."""
        safe_workflow_id = workflow_id.replace("_", "-").replace(" ", "-").lower()
        safe_name = config.name.replace("_", "-").replace(" ", "-").lower()
        return f"crewgraph-{safe_workflow_id}-{safe_name}"
    
    def get_common_tags(self, workflow_id: WorkflowId, config: DeploymentConfig) -> Dict[str, str]:
        """Get common tags for cloud resources."""
        return {
            "CreatedBy": "CrewGraphAI",
            "WorkflowId": workflow_id,
            "DeploymentName": config.name,
            "DeploymentType": config.deployment_type.value,
            "Provider": self.provider.value,
            "AutoScaling": str(config.auto_scaling_enabled),
            "Environment": "production"
        }
    
    def calculate_resource_cost(
        self, 
        config: DeploymentConfig, 
        hours: float = 1.0
    ) -> float:
        """Calculate estimated resource cost for given hours."""
        # Base cost calculation (override in provider implementations)
        base_cost_per_hour = (
            config.cpu_cores * 0.05 +  # $0.05 per CPU hour
            config.memory_gb * 0.01 +   # $0.01 per GB memory hour
            config.storage_gb * 0.0001  # $0.0001 per GB storage hour
        )
        
        # Apply spot instance discount
        if config.use_spot_instances:
            base_cost_per_hour *= 0.3  # 70% discount for spot instances
        
        # Apply reserved instance discount
        if config.use_reserved_instances:
            base_cost_per_hour *= 0.6  # 40% discount for reserved instances
        
        return base_cost_per_hour * hours
    
    def create_deployment_metadata(
        self, 
        workflow_id: WorkflowId,
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Create metadata for deployment tracking."""
        return {
            "workflow_id": workflow_id,
            "deployment_name": config.name,
            "provider": self.provider.value,
            "deployment_type": config.deployment_type.value,
            "region": config.region,
            "cpu_cores": config.cpu_cores,
            "memory_gb": config.memory_gb,
            "auto_scaling": config.auto_scaling_enabled,
            "min_instances": config.min_instances,
            "max_instances": config.max_instances,
            "use_spot_instances": config.use_spot_instances,
            "estimated_hourly_cost": self.calculate_resource_cost(config),
            "created_at": datetime.now().isoformat(),
            "tags": self.get_common_tags(workflow_id, config)
        }


class CloudProviderFactory:
    """Factory for creating cloud provider instances."""
    
    _providers = {}
    
    @classmethod
    def register_provider(cls, provider_type: CloudProvider, provider_class):
        """Register a cloud provider implementation."""
        cls._providers[provider_type] = provider_class
        logger.info(f"Registered cloud provider: {provider_type.value}")
    
    @classmethod
    def create_provider(cls, credentials: CloudCredentials) -> BaseCloudProvider:
        """Create a cloud provider instance."""
        provider_type = credentials.provider
        
        if provider_type not in cls._providers:
            raise ValueError(f"Unsupported cloud provider: {provider_type}")
        
        provider_class = cls._providers[provider_type]
        return provider_class(credentials)
    
    @classmethod
    def get_supported_providers(cls) -> List[CloudProvider]:
        """Get list of supported cloud providers."""
        return list(cls._providers.keys())


class MultiCloudManager:
    """
    Manager for multi-cloud deployments across different providers.
    
    Provides unified interface for deploying workflows across multiple
    cloud providers with cost optimization and failover capabilities.
    """
    
    def __init__(self):
        """Initialize multi-cloud manager."""
        self.providers: Dict[CloudProvider, BaseCloudProvider] = {}
        self.deployments: Dict[str, CloudResource] = {}
        
        logger.info("Multi-cloud manager initialized")
    
    def add_provider(self, credentials: CloudCredentials):
        """Add a cloud provider to the manager."""
        provider = CloudProviderFactory.create_provider(credentials)
        self.providers[credentials.provider] = provider
        logger.info(f"Added {credentials.provider.value} provider")
    
    def deploy_to_optimal_provider(
        self,
        workflow_id: WorkflowId,
        configs: List[DeploymentConfig],
        workflow_definition: Dict[str, Any],
        cost_priority: float = 0.7,
        performance_priority: float = 0.3
    ) -> DeploymentResult:
        """
        Deploy to the optimal cloud provider based on cost and performance.
        
        Args:
            workflow_id: Workflow to deploy
            configs: List of deployment configs for different providers
            workflow_definition: Workflow structure
            cost_priority: Weight for cost optimization (0.0-1.0)
            performance_priority: Weight for performance optimization (0.0-1.0)
            
        Returns:
            Best deployment result
        """
        if not configs:
            return DeploymentResult(
                success=False,
                error_message="No deployment configurations provided"
            )
        
        # Evaluate each configuration
        best_config = None
        best_score = -1
        cost_estimates = {}
        
        for config in configs:
            if config.provider not in self.providers:
                logger.warning(f"Provider {config.provider} not available, skipping")
                continue
            
            provider = self.providers[config.provider]
            
            # Get cost estimate
            cost_estimate = provider.estimate_cost(config)
            cost_estimates[config.provider] = cost_estimate
            
            # Calculate optimization score
            normalized_cost = cost_estimate.get("total_hourly_cost", 1.0)
            normalized_performance = 1.0 / max(config.cpu_cores * config.memory_gb, 1.0)
            
            score = (
                cost_priority * (1.0 / normalized_cost) +
                performance_priority * normalized_performance
            )
            
            if score > best_score:
                best_score = score
                best_config = config
        
        if not best_config:
            return DeploymentResult(
                success=False,
                error_message="No suitable provider found for deployment"
            )
        
        # Deploy to best provider
        provider = self.providers[best_config.provider]
        result = provider.deploy_workflow(workflow_id, best_config, workflow_definition)
        
        if result.success and result.resource:
            self.deployments[result.resource.resource_id] = result.resource
            logger.info(f"Deployed {workflow_id} to {best_config.provider.value}")
        
        return result
    
    def deploy_with_failover(
        self,
        workflow_id: WorkflowId,
        configs: List[DeploymentConfig],
        workflow_definition: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy with automatic failover to backup providers."""
        # Sort configs by priority (could be cost, reliability, etc.)
        sorted_configs = sorted(configs, key=lambda c: c.cpu_cores * c.memory_gb, reverse=True)
        
        last_error = "No configurations provided"
        
        for config in sorted_configs:
            if config.provider not in self.providers:
                continue
            
            provider = self.providers[config.provider]
            result = provider.deploy_workflow(workflow_id, config, workflow_definition)
            
            if result.success:
                if result.resource:
                    self.deployments[result.resource.resource_id] = result.resource
                logger.info(f"Deployed {workflow_id} to {config.provider.value} (failover)")
                return result
            else:
                last_error = result.error_message or "Deployment failed"
                logger.warning(f"Deployment to {config.provider.value} failed: {last_error}")
        
        return DeploymentResult(
            success=False,
            error_message=f"All deployment attempts failed. Last error: {last_error}"
        )
    
    def get_deployment_status(self, resource_id: str) -> Optional[CloudResource]:
        """Get status of a specific deployment."""
        if resource_id in self.deployments:
            resource = self.deployments[resource_id]
            provider = self.providers.get(resource.provider)
            
            if provider:
                # Refresh status from cloud provider
                updated_resource = provider.get_resource_status(resource_id)
                self.deployments[resource_id] = updated_resource
                return updated_resource
        
        return None
    
    def list_all_deployments(self) -> List[CloudResource]:
        """List all deployments across all providers."""
        all_resources = []
        
        for provider in self.providers.values():
            try:
                resources = provider.list_resources()
                all_resources.extend(resources)
            except Exception as e:
                logger.error(f"Failed to list resources for {provider.provider}: {e}")
        
        return all_resources
    
    def optimize_costs(self) -> Dict[str, Any]:
        """Analyze and optimize costs across all deployments."""
        total_cost = 0.0
        provider_costs = {}
        optimization_opportunities = []
        
        for resource in self.deployments.values():
            total_cost += resource.hourly_cost
            
            provider_name = resource.provider.value
            if provider_name not in provider_costs:
                provider_costs[provider_name] = 0.0
            provider_costs[provider_name] += resource.hourly_cost
            
            # Identify optimization opportunities
            if resource.hourly_cost > 1.0:  # High cost resources
                optimization_opportunities.append({
                    "resource_id": resource.resource_id,
                    "provider": provider_name,
                    "current_cost": resource.hourly_cost,
                    "optimization": "Consider rightsizing or using spot instances"
                })
        
        return {
            "total_hourly_cost": total_cost,
            "provider_breakdown": provider_costs,
            "optimization_opportunities": optimization_opportunities,
            "potential_savings": sum(
                opp["current_cost"] * 0.3 for opp in optimization_opportunities
            )
        }
    
    def cleanup_unused_resources(self, max_age_hours: int = 24) -> Dict[str, int]:
        """Clean up unused resources older than specified age."""
        cleanup_results = {"deleted": 0, "errors": 0}
        
        for provider in self.providers.values():
            try:
                resources = provider.list_resources()
                
                for resource in resources:
                    # Simple cleanup logic (should be more sophisticated in production)
                    if resource.status == ResourceStatus.STOPPED:
                        if provider.delete_resource(resource.resource_id):
                            cleanup_results["deleted"] += 1
                            if resource.resource_id in self.deployments:
                                del self.deployments[resource.resource_id]
                        else:
                            cleanup_results["errors"] += 1
                            
            except Exception as e:
                logger.error(f"Cleanup failed for {provider.provider}: {e}")
                cleanup_results["errors"] += 1
        
        return cleanup_results


# Auto-register cloud providers
def _register_providers():
    """Auto-register available cloud providers."""
    try:
        from .aws import AWSProvider
        CloudProviderFactory.register_provider(CloudProvider.AWS, AWSProvider)
    except ImportError:
        logger.debug("AWS provider not available")
    
    try:
        from .gcp import GCPProvider
        CloudProviderFactory.register_provider(CloudProvider.GCP, GCPProvider)
    except ImportError:
        logger.debug("GCP provider not available")
    
    try:
        from .azure import AzureProvider
        CloudProviderFactory.register_provider(CloudProvider.AZURE, AzureProvider)
    except ImportError:
        logger.debug("Azure provider not available")


# Register providers on import
_register_providers()

# Export commonly used classes
__all__ = [
    "CloudProvider",
    "DeploymentType",
    "ResourceStatus",
    "CloudCredentials",
    "DeploymentConfig",
    "CloudResource",
    "DeploymentResult",
    "BaseCloudProvider",
    "CloudProviderFactory",
    "MultiCloudManager"
]