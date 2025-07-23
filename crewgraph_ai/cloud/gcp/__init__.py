"""
Google Cloud Platform Provider Implementation for CrewGraph AI

Provides GCP-specific deployment capabilities including Cloud Run, Compute Engine,
Cloud Functions, and Deployment Manager template generation.

Author: Vatsal216
Created: 2025-07-23 18:10:00 UTC
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .. import (
    BaseCloudProvider,
    CloudCredentials, 
    CloudProvider,
    CloudResource,
    DeploymentConfig,
    DeploymentResult,
    DeploymentType,
    ResourceStatus
)
from ...types import WorkflowId
from ...utils.logging import get_logger

logger = get_logger(__name__)


class GCPProvider(BaseCloudProvider):
    """
    Google Cloud Platform provider implementation.
    
    Supports deployment to Cloud Run, Compute Engine, Cloud Functions,
    and GKE with Deployment Manager template generation.
    """
    
    def __init__(self, credentials: CloudCredentials):
        """Initialize GCP provider."""
        super().__init__(credentials)
        
        # GCP service endpoints by region
        self.service_endpoints = {
            "us-central1": "https://us-central1-run.googleapis.com",
            "us-east1": "https://us-east1-run.googleapis.com",
            "europe-west1": "https://europe-west1-run.googleapis.com"
        }
        
        # GCP pricing (simplified - actual pricing would come from GCP APIs)
        self.pricing = {
            "compute_engine": {
                "e2-micro": {"cpu": 0.25, "memory": 1, "cost_per_hour": 0.006},
                "e2-small": {"cpu": 0.5, "memory": 2, "cost_per_hour": 0.012},
                "e2-medium": {"cpu": 1, "memory": 4, "cost_per_hour": 0.024},
                "n1-standard-1": {"cpu": 1, "memory": 3.75, "cost_per_hour": 0.0475},
                "n1-standard-2": {"cpu": 2, "memory": 7.5, "cost_per_hour": 0.095}
            },
            "cloud_run": {
                "cpu_per_hour": 0.000024,
                "memory_per_gb_hour": 0.0000025,
                "requests_per_million": 0.40
            },
            "cloud_functions": {
                "invocations_per_million": 0.40,
                "gb_second": 0.0000025,
                "cpu_ghz_second": 0.0000100
            },
            "gke": {
                "cluster_management_fee": 0.10,  # per hour
                "node_cost_multiplier": 1.0
            }
        }
        
        # Simulated GCP resources
        self._simulated_resources: Dict[str, CloudResource] = {}
        
        logger.info(f"GCP provider initialized for project {credentials.project_id}")
    
    def deploy_workflow(
        self, 
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy workflow to GCP using specified deployment type."""
        # Validate configuration
        validation_issues = self.validate_config(config)
        if validation_issues:
            return DeploymentResult(
                success=False,
                error_message=f"Configuration validation failed: {', '.join(validation_issues)}"
            )
        
        # Select deployment method based on type
        if config.deployment_type == DeploymentType.CONTAINER:
            return self._deploy_to_cloud_run(workflow_id, config, workflow_definition)
        elif config.deployment_type == DeploymentType.SERVERLESS:
            return self._deploy_to_cloud_functions(workflow_id, config, workflow_definition)
        elif config.deployment_type == DeploymentType.VIRTUAL_MACHINE:
            return self._deploy_to_compute_engine(workflow_id, config, workflow_definition)
        elif config.deployment_type == DeploymentType.KUBERNETES:
            return self._deploy_to_gke(workflow_id, config, workflow_definition)
        else:
            return DeploymentResult(
                success=False,
                error_message=f"Unsupported deployment type: {config.deployment_type}"
            )
    
    def _deploy_to_cloud_run(
        self,
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy workflow to Google Cloud Run."""
        try:
            resource_id = f"run-{uuid.uuid4().hex[:8]}"
            resource_name = self.generate_resource_name(workflow_id, config)
            
            # Create Cloud Run service configuration
            service_config = self._create_cloud_run_service(config, workflow_definition)
            
            # Calculate cost (Cloud Run charges per request and compute time)
            estimated_requests_per_hour = 100
            cpu_time_hours = estimated_requests_per_hour * 0.1 / 3600  # 0.1 seconds per request
            memory_gb_hours = config.memory_gb * cpu_time_hours
            
            hourly_cost = (
                config.cpu_cores * cpu_time_hours * self.pricing["cloud_run"]["cpu_per_hour"] +
                memory_gb_hours * self.pricing["cloud_run"]["memory_per_gb_hour"] +
                (estimated_requests_per_hour / 1000000) * self.pricing["cloud_run"]["requests_per_million"]
            )
            
            resource = CloudResource(
                resource_id=resource_id,
                name=resource_name,
                provider=CloudProvider.GCP,
                deployment_type=DeploymentType.CONTAINER,
                status=ResourceStatus.CREATING,
                region=config.region,
                endpoint_url=f"https://{resource_name}-{self.credentials.project_id}.{config.region}.run.app",
                hourly_cost=hourly_cost,
                tags=self.get_common_tags(workflow_id, config),
                created_at=datetime.now().isoformat()
            )
            
            self._simulated_resources[resource_id] = resource
            
            # Generate Deployment Manager template
            dm_template = self._generate_cloud_run_deployment_manager(config, service_config)
            
            deployment_logs = [
                f"Created Cloud Run service: {service_config['metadata']['name']}",
                f"Container image: {service_config['spec']['template']['spec']['containers'][0]['image']}",
                f"Service endpoint: {resource.endpoint_url}",
                "Automatic HTTPS enabled"
            ]
            
            logger.info(f"Deployed {workflow_id} to Cloud Run: {resource_id}")
            
            # Simulate deployment completion
            resource.status = ResourceStatus.RUNNING
            
            return DeploymentResult(
                success=True,
                resource=resource,
                deployment_logs=deployment_logs,
                infrastructure_code=dm_template
            )
            
        except Exception as e:
            logger.error(f"Cloud Run deployment failed: {e}")
            return DeploymentResult(
                success=False,
                error_message=f"Cloud Run deployment failed: {str(e)}"
            )
    
    def _deploy_to_cloud_functions(
        self,
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy workflow to Google Cloud Functions."""
        try:
            resource_id = f"function-{uuid.uuid4().hex[:8]}"
            resource_name = self.generate_resource_name(workflow_id, config)
            
            # Create Cloud Function configuration
            function_config = self._create_cloud_function_config(config, workflow_definition)
            
            # Estimate cost
            estimated_invocations_per_hour = 100
            memory_gb = config.function_memory / 1024
            duration_seconds = config.function_timeout
            gb_seconds = memory_gb * duration_seconds * estimated_invocations_per_hour
            
            hourly_cost = (
                (estimated_invocations_per_hour / 1000000) * self.pricing["cloud_functions"]["invocations_per_million"] +
                gb_seconds * self.pricing["cloud_functions"]["gb_second"]
            )
            
            resource = CloudResource(
                resource_id=resource_id,
                name=resource_name,
                provider=CloudProvider.GCP,
                deployment_type=DeploymentType.SERVERLESS,
                status=ResourceStatus.RUNNING,
                region=config.region,
                endpoint_url=f"https://{config.region}-{self.credentials.project_id}.cloudfunctions.net/{resource_name}",
                hourly_cost=hourly_cost,
                tags=self.get_common_tags(workflow_id, config),
                created_at=datetime.now().isoformat()
            )
            
            self._simulated_resources[resource_id] = resource
            
            dm_template = self._generate_cloud_function_deployment_manager(config, function_config)
            
            deployment_logs = [
                f"Created Cloud Function: {function_config['name']}",
                f"Runtime: {function_config['runtime']}",
                f"Memory: {function_config['availableMemoryMb']}MB",
                f"Timeout: {function_config['timeout']}s",
                f"Trigger: {function_config['httpsTrigger']}"
            ]
            
            logger.info(f"Deployed {workflow_id} to Cloud Functions: {resource_id}")
            
            return DeploymentResult(
                success=True,
                resource=resource,
                deployment_logs=deployment_logs,
                infrastructure_code=dm_template
            )
            
        except Exception as e:
            logger.error(f"Cloud Functions deployment failed: {e}")
            return DeploymentResult(
                success=False,
                error_message=f"Cloud Functions deployment failed: {str(e)}"
            )
    
    def _deploy_to_compute_engine(
        self,
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy workflow to Google Compute Engine."""
        try:
            resource_id = f"gce-{uuid.uuid4().hex[:8]}"
            resource_name = self.generate_resource_name(workflow_id, config)
            
            # Select appropriate machine type
            machine_type = self._select_gce_machine_type(config)
            machine_pricing = self.pricing["compute_engine"][machine_type]
            
            # Apply sustained use discount (simplified)
            hourly_cost = machine_pricing["cost_per_hour"] * config.min_instances
            if config.use_spot_instances:
                hourly_cost *= 0.2  # 80% discount for preemptible instances
            
            resource = CloudResource(
                resource_id=resource_id,
                name=resource_name,
                provider=CloudProvider.GCP,
                deployment_type=DeploymentType.VIRTUAL_MACHINE,
                status=ResourceStatus.CREATING,
                region=config.region,
                public_ip="35.123.45.67",  # Simulated IP
                internal_ip="10.128.0.2",  # Simulated internal IP
                hourly_cost=hourly_cost,
                tags=self.get_common_tags(workflow_id, config),
                created_at=datetime.now().isoformat()
            )
            
            self._simulated_resources[resource_id] = resource
            
            # Generate startup script for workflow execution
            startup_script = self._generate_gce_startup_script(workflow_definition)
            
            dm_template = self._generate_gce_deployment_manager(config, machine_type, startup_script)
            
            deployment_logs = [
                f"Created Compute Engine instance: {resource_id}",
                f"Machine type: {machine_type}",
                f"Public IP: {resource.public_ip}",
                f"Preemptible: {config.use_spot_instances}",
                "Startup script configured for workflow execution"
            ]
            
            logger.info(f"Deployed {workflow_id} to Compute Engine: {resource_id}")
            
            # Simulate startup time
            resource.status = ResourceStatus.RUNNING
            
            return DeploymentResult(
                success=True,
                resource=resource,
                deployment_logs=deployment_logs,
                infrastructure_code=dm_template
            )
            
        except Exception as e:
            logger.error(f"Compute Engine deployment failed: {e}")
            return DeploymentResult(
                success=False,
                error_message=f"Compute Engine deployment failed: {str(e)}"
            )
    
    def _deploy_to_gke(
        self,
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy workflow to Google Kubernetes Engine."""
        try:
            resource_id = f"gke-{uuid.uuid4().hex[:8]}"
            resource_name = self.generate_resource_name(workflow_id, config)
            
            # Generate Kubernetes manifests
            k8s_manifests = self._generate_kubernetes_manifests(config, workflow_definition)
            
            # Estimate GKE cost
            node_cost = self.pricing["compute_engine"]["n1-standard-1"]["cost_per_hour"] * config.min_instances
            hourly_cost = (
                self.pricing["gke"]["cluster_management_fee"] +
                node_cost * self.pricing["gke"]["node_cost_multiplier"]
            )
            
            resource = CloudResource(
                resource_id=resource_id,
                name=resource_name,
                provider=CloudProvider.GCP,
                deployment_type=DeploymentType.KUBERNETES,
                status=ResourceStatus.CREATING,
                region=config.region,
                endpoint_url=f"https://{resource_name}.{config.region}.container.cluster",
                hourly_cost=hourly_cost,
                tags=self.get_common_tags(workflow_id, config),
                created_at=datetime.now().isoformat()
            )
            
            self._simulated_resources[resource_id] = resource
            
            dm_template = self._generate_gke_deployment_manager(config)
            
            deployment_logs = [
                f"Created GKE cluster: {resource_name}",
                f"Node pool: {config.min_instances} nodes",
                f"Deployed Kubernetes manifests",
                f"Service endpoint: {resource.endpoint_url}",
                "Autopilot mode enabled for simplified management"
            ]
            
            logger.info(f"Deployed {workflow_id} to GKE: {resource_id}")
            
            resource.status = ResourceStatus.RUNNING
            
            return DeploymentResult(
                success=True,
                resource=resource,
                deployment_logs=deployment_logs,
                infrastructure_code=dm_template
            )
            
        except Exception as e:
            logger.error(f"GKE deployment failed: {e}")
            return DeploymentResult(
                success=False,
                error_message=f"GKE deployment failed: {str(e)}"
            )
    
    def get_resource_status(self, resource_id: str) -> CloudResource:
        """Get current status of GCP resource."""
        if resource_id in self._simulated_resources:
            resource = self._simulated_resources[resource_id]
            
            # Simulate status updates
            if resource.status == ResourceStatus.CREATING:
                resource.status = ResourceStatus.RUNNING
                resource.last_updated = datetime.now().isoformat()
            
            return resource
        else:
            return CloudResource(
                resource_id=resource_id,
                name="unknown",
                provider=CloudProvider.GCP,
                deployment_type=DeploymentType.CONTAINER,
                status=ResourceStatus.ERROR,
                region="unknown"
            )
    
    def scale_resource(self, resource_id: str, target_instances: int) -> bool:
        """Scale GCP resource to target number of instances."""
        if resource_id not in self._simulated_resources:
            return False
        
        resource = self._simulated_resources[resource_id]
        
        logger.info(f"Scaling {resource_id} to {target_instances} instances")
        
        # Update cost based on scaling
        if resource.deployment_type == DeploymentType.CONTAINER:
            # Cloud Run scales automatically, no cost change
            pass
        elif resource.deployment_type == DeploymentType.VIRTUAL_MACHINE:
            # Compute Engine scaling
            base_cost = resource.hourly_cost
            current_instances = int(resource.tags.get("InstanceCount", "1"))
            resource.hourly_cost = base_cost / current_instances * target_instances
        
        resource.tags["InstanceCount"] = str(target_instances)
        resource.last_updated = datetime.now().isoformat()
        
        return True
    
    def stop_resource(self, resource_id: str) -> bool:
        """Stop GCP resource."""
        if resource_id not in self._simulated_resources:
            return False
        
        resource = self._simulated_resources[resource_id]
        resource.status = ResourceStatus.STOPPED
        resource.last_updated = datetime.now().isoformat()
        
        logger.info(f"Stopped GCP resource: {resource_id}")
        return True
    
    def delete_resource(self, resource_id: str) -> bool:
        """Delete GCP resource."""
        if resource_id not in self._simulated_resources:
            return False
        
        resource = self._simulated_resources[resource_id]
        resource.status = ResourceStatus.DELETED
        resource.last_updated = datetime.now().isoformat()
        
        del self._simulated_resources[resource_id]
        
        logger.info(f"Deleted GCP resource: {resource_id}")
        return True
    
    def list_resources(self, workflow_id: Optional[WorkflowId] = None) -> List[CloudResource]:
        """List GCP resources."""
        resources = list(self._simulated_resources.values())
        
        if workflow_id:
            resources = [r for r in resources if r.tags.get("WorkflowId") == workflow_id]
        
        return resources
    
    def get_resource_logs(self, resource_id: str, lines: int = 100) -> List[str]:
        """Get logs from GCP resource."""
        # Simulate Cloud Logging
        sample_logs = [
            f"[{datetime.now().isoformat()}] INFO Starting CrewGraph workflow execution",
            f"[{datetime.now().isoformat()}] INFO Workflow task 1 initiated",
            f"[{datetime.now().isoformat()}] INFO Task 1 completed successfully",
            f"[{datetime.now().isoformat()}] INFO Workflow execution completed"
        ]
        
        return sample_logs[-lines:]
    
    def get_resource_metrics(
        self, 
        resource_id: str, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get Cloud Monitoring metrics for GCP resource."""
        # Simulate Cloud Monitoring metrics
        return {
            "cpu_utilization": {"mean": 62.4, "maximum": 88.2, "minimum": 15.7},
            "memory_utilization": {"mean": 54.8, "maximum": 72.1, "minimum": 28.3},
            "network_sent_bytes": {"sum": 2048.9},  # KB
            "network_received_bytes": {"sum": 1536.4},  # KB
            "request_count": {"sum": 1267},
            "request_latency": {"mean": 187.3, "p95": 456.7},  # ms
            "error_rate": {"mean": 0.8}  # percentage
        }
    
    def estimate_cost(self, config: DeploymentConfig) -> Dict[str, float]:
        """Estimate GCP deployment cost."""
        if config.deployment_type == DeploymentType.CONTAINER:
            # Cloud Run pricing
            estimated_requests = 100
            cpu_time = estimated_requests * 0.1 / 3600  # 0.1 seconds per request
            memory_gb_hours = config.memory_gb * cpu_time
            
            hourly_cost = (
                config.cpu_cores * cpu_time * self.pricing["cloud_run"]["cpu_per_hour"] +
                memory_gb_hours * self.pricing["cloud_run"]["memory_per_gb_hour"] +
                (estimated_requests / 1000000) * self.pricing["cloud_run"]["requests_per_million"]
            )
            
        elif config.deployment_type == DeploymentType.SERVERLESS:
            # Cloud Functions pricing
            estimated_invocations = 100
            memory_gb = config.function_memory / 1024
            gb_seconds = memory_gb * config.function_timeout * estimated_invocations
            
            hourly_cost = (
                (estimated_invocations / 1000000) * self.pricing["cloud_functions"]["invocations_per_million"] +
                gb_seconds * self.pricing["cloud_functions"]["gb_second"]
            )
            
        elif config.deployment_type == DeploymentType.VIRTUAL_MACHINE:
            # Compute Engine pricing
            machine_type = self._select_gce_machine_type(config)
            hourly_cost = self.pricing["compute_engine"][machine_type]["cost_per_hour"] * config.min_instances
            
            if config.use_spot_instances:
                hourly_cost *= 0.2  # Preemptible discount
                
        elif config.deployment_type == DeploymentType.KUBERNETES:
            # GKE pricing
            node_cost = self.pricing["compute_engine"]["n1-standard-1"]["cost_per_hour"] * config.min_instances
            hourly_cost = self.pricing["gke"]["cluster_management_fee"] + node_cost
            
        else:
            hourly_cost = 0.5  # Default
        
        return {
            "hourly_cost": hourly_cost,
            "daily_cost": hourly_cost * 24,
            "monthly_cost": hourly_cost * 24 * 30,
            "total_hourly_cost": hourly_cost
        }
    
    def generate_infrastructure_code(
        self, 
        config: DeploymentConfig,
        template_type: str = "terraform"
    ) -> str:
        """Generate Infrastructure as Code templates."""
        if template_type == "terraform":
            return self._generate_terraform_template(config)
        elif template_type == "deployment_manager":
            return self._generate_deployment_manager_template(config)
        else:
            raise ValueError(f"Unsupported template type: {template_type}")
    
    # Helper methods for GCP-specific operations
    
    def _create_cloud_run_service(
        self, 
        config: DeploymentConfig, 
        workflow_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create Cloud Run service configuration."""
        return {
            "apiVersion": "serving.knative.dev/v1",
            "kind": "Service",
            "metadata": {
                "name": f"crewgraph-{config.name}",
                "namespace": self.credentials.project_id,
                "annotations": {
                    "run.googleapis.com/client-name": "crewgraph-ai"
                }
            },
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "autoscaling.knative.dev/minScale": str(config.min_instances),
                            "autoscaling.knative.dev/maxScale": str(config.max_instances),
                            "run.googleapis.com/cpu-throttling": "false"
                        }
                    },
                    "spec": {
                        "containerConcurrency": 80,
                        "timeoutSeconds": config.function_timeout,
                        "containers": [
                            {
                                "image": config.image_uri or "gcr.io/crewgraph/workflow-runner:latest",
                                "resources": {
                                    "limits": {
                                        "cpu": str(config.cpu_cores),
                                        "memory": f"{int(config.memory_gb)}Gi"
                                    }
                                },
                                "env": [
                                    {"name": k, "value": v} for k, v in config.environment_variables.items()
                                ],
                                "ports": [{"containerPort": 8080}]
                            }
                        ]
                    }
                },
                "traffic": [{"percent": 100, "latestRevision": True}]
            }
        }
    
    def _create_cloud_function_config(
        self, 
        config: DeploymentConfig, 
        workflow_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create Cloud Function configuration."""
        return {
            "name": f"projects/{self.credentials.project_id}/locations/{config.region}/functions/crewgraph-{config.name}",
            "sourceArchiveUrl": "gs://crewgraph-functions/workflow.zip",
            "entryPoint": "execute_workflow",
            "runtime": "python39",
            "availableMemoryMb": config.function_memory,
            "timeout": f"{config.function_timeout}s",
            "environmentVariables": config.environment_variables,
            "httpsTrigger": {},
            "labels": {
                "created-by": "crewgraph-ai",
                "workflow-id": workflow_definition.get("id", "unknown")
            }
        }
    
    def _select_gce_machine_type(self, config: DeploymentConfig) -> str:
        """Select appropriate Compute Engine machine type."""
        for machine_type, specs in self.pricing["compute_engine"].items():
            if (specs["cpu"] >= config.cpu_cores and 
                specs["memory"] >= config.memory_gb):
                return machine_type
        
        return "n1-standard-2"  # Default to larger machine if requirements not met
    
    def _generate_gce_startup_script(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate Compute Engine startup script."""
        return """#!/bin/bash
# CrewGraph AI workflow execution startup script
apt-get update
apt-get install -y python3 python3-pip

# Install CrewGraph AI
pip3 install crewgraph-ai

# Create workflow execution script
cat > /opt/crewgraph_workflow.py << 'EOF'
import json
from crewgraph_ai import CrewGraph

# Initialize workflow
workflow = CrewGraph("gce_workflow")

# Execute workflow
try:
    result = workflow.execute()
    print(f"Workflow completed successfully: {result}")
except Exception as e:
    print(f"Workflow execution failed: {e}")
    exit(1)
EOF

# Execute workflow
python3 /opt/crewgraph_workflow.py

# Log completion
echo "CrewGraph workflow execution completed" >> /var/log/crewgraph.log
"""
    
    def _generate_kubernetes_manifests(
        self, 
        config: DeploymentConfig, 
        workflow_definition: Dict[str, Any]
    ) -> str:
        """Generate Kubernetes deployment manifests for GKE."""
        return f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crewgraph-{config.name}
  labels:
    app: crewgraph-{config.name}
spec:
  replicas: {config.min_instances}
  selector:
    matchLabels:
      app: crewgraph-{config.name}
  template:
    metadata:
      labels:
        app: crewgraph-{config.name}
    spec:
      containers:
      - name: crewgraph-container
        image: {config.image_uri or 'gcr.io/crewgraph/workflow-runner:latest'}
        resources:
          requests:
            memory: "{config.memory_gb}Gi"
            cpu: "{config.cpu_cores}"
          limits:
            memory: "{config.memory_gb}Gi" 
            cpu: "{config.cpu_cores}"
        env:
{chr(10).join(f'        - name: {k}{chr(10)}          value: "{v}"' for k, v in config.environment_variables.items())}
        ports:
        - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: crewgraph-{config.name}-service
spec:
  selector:
    app: crewgraph-{config.name}
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: crewgraph-{config.name}-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: crewgraph-{config.name}
  minReplicas: {config.min_instances}
  maxReplicas: {config.max_instances}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""
    
    def _generate_cloud_run_deployment_manager(
        self, 
        config: DeploymentConfig, 
        service_config: Dict[str, Any]
    ) -> str:
        """Generate Deployment Manager template for Cloud Run."""
        return json.dumps({
            "resources": [
                {
                    "name": f"crewgraph-{config.name}",
                    "type": "run.googleapis.com/v1:namespaces.services",
                    "metadata": {
                        "annotations": {
                            "cnrm.cloud.google.com/deletion-policy": "abandon"
                        }
                    },
                    "properties": service_config
                }
            ]
        }, indent=2)
    
    def _generate_cloud_function_deployment_manager(
        self, 
        config: DeploymentConfig, 
        function_config: Dict[str, Any]
    ) -> str:
        """Generate Deployment Manager template for Cloud Functions."""
        return json.dumps({
            "resources": [
                {
                    "name": f"crewgraph-{config.name}",
                    "type": "cloudfunctions.v1.function",
                    "properties": function_config
                }
            ]
        }, indent=2)
    
    def _generate_gce_deployment_manager(
        self, 
        config: DeploymentConfig, 
        machine_type: str,
        startup_script: str
    ) -> str:
        """Generate Deployment Manager template for Compute Engine."""
        return json.dumps({
            "resources": [
                {
                    "name": f"crewgraph-{config.name}",
                    "type": "compute.v1.instance",
                    "properties": {
                        "zone": f"{config.region}-a",
                        "machineType": f"zones/{config.region}-a/machineTypes/{machine_type}",
                        "disks": [
                            {
                                "boot": True,
                                "autoDelete": True,
                                "initializeParams": {
                                    "sourceImage": "projects/debian-cloud/global/images/family/debian-11"
                                }
                            }
                        ],
                        "networkInterfaces": [
                            {
                                "network": "projects/{0}/global/networks/default".format(self.credentials.project_id),
                                "accessConfigs": [
                                    {
                                        "type": "ONE_TO_ONE_NAT",
                                        "name": "External NAT"
                                    }
                                ]
                            }
                        ],
                        "metadata": {
                            "items": [
                                {
                                    "key": "startup-script",
                                    "value": startup_script
                                }
                            ]
                        },
                        "tags": {
                            "items": ["crewgraph", "workflow"]
                        },
                        "labels": {
                            "created-by": "crewgraph-ai",
                            "environment": "production"
                        }
                    }
                }
            ]
        }, indent=2)
    
    def _generate_gke_deployment_manager(self, config: DeploymentConfig) -> str:
        """Generate Deployment Manager template for GKE."""
        return json.dumps({
            "resources": [
                {
                    "name": f"crewgraph-{config.name}-cluster",
                    "type": "container.v1.cluster",
                    "properties": {
                        "zone": f"{config.region}-a",
                        "cluster": {
                            "name": f"crewgraph-{config.name}",
                            "initialNodeCount": config.min_instances,
                            "nodeConfig": {
                                "machineType": "n1-standard-1",
                                "diskSizeGb": 20,
                                "oauthScopes": [
                                    "https://www.googleapis.com/auth/devstorage.read_only",
                                    "https://www.googleapis.com/auth/logging.write",
                                    "https://www.googleapis.com/auth/monitoring"
                                ]
                            },
                            "masterAuth": {
                                "username": "",
                                "password": ""
                            },
                            "loggingService": "logging.googleapis.com",
                            "monitoringService": "monitoring.googleapis.com",
                            "network": "default",
                            "addonsConfig": {
                                "httpLoadBalancing": {"disabled": False},
                                "horizontalPodAutoscaling": {"disabled": False}
                            }
                        }
                    }
                }
            ]
        }, indent=2)
    
    def _generate_terraform_template(self, config: DeploymentConfig) -> str:
        """Generate Terraform template for GCP deployment."""
        if config.deployment_type == DeploymentType.CONTAINER:
            return self._generate_terraform_cloud_run(config)
        elif config.deployment_type == DeploymentType.SERVERLESS:
            return self._generate_terraform_cloud_functions(config)
        elif config.deployment_type == DeploymentType.VIRTUAL_MACHINE:
            return self._generate_terraform_compute_engine(config)
        elif config.deployment_type == DeploymentType.KUBERNETES:
            return self._generate_terraform_gke(config)
        else:
            return "# Terraform template not implemented for this deployment type"
    
    def _generate_terraform_cloud_run(self, config: DeploymentConfig) -> str:
        """Generate Terraform template for Cloud Run."""
        return f"""
# CrewGraph AI Cloud Run Deployment
terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 4.0"
    }}
  }}
}}

provider "google" {{
  project = "{self.credentials.project_id}"
  region  = "{config.region}"
}}

resource "google_cloud_run_service" "crewgraph" {{
  name     = "crewgraph-{config.name}"
  location = "{config.region}"

  template {{
    metadata {{
      annotations = {{
        "autoscaling.knative.dev/minScale" = "{config.min_instances}"
        "autoscaling.knative.dev/maxScale" = "{config.max_instances}"
        "run.googleapis.com/cpu-throttling" = "false"
      }}
    }}

    spec {{
      container_concurrency = 80
      timeout_seconds       = {config.function_timeout}

      containers {{
        image = "{config.image_uri or 'gcr.io/crewgraph/workflow-runner:latest'}"

        resources {{
          limits = {{
            cpu    = "{config.cpu_cores}"
            memory = "{int(config.memory_gb)}Gi"
          }}
        }}

{chr(10).join(f'        env {{{chr(10)}          name  = "{k}"{chr(10)}          value = "{v}"{chr(10)}        }}' for k, v in config.environment_variables.items())}
      }}
    }}
  }}

  traffic {{
    percent         = 100
    latest_revision = true
  }}
}}

resource "google_cloud_run_service_iam_member" "public" {{
  service  = google_cloud_run_service.crewgraph.name
  location = google_cloud_run_service.crewgraph.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}}

output "url" {{
  value = google_cloud_run_service.crewgraph.status[0].url
}}
"""
    
    def _generate_terraform_cloud_functions(self, config: DeploymentConfig) -> str:
        """Generate Terraform template for Cloud Functions."""
        return f"""
# CrewGraph AI Cloud Functions Deployment
terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 4.0"
    }}
  }}
}}

provider "google" {{
  project = "{self.credentials.project_id}"
  region  = "{config.region}"
}}

resource "google_storage_bucket" "function_bucket" {{
  name     = "{self.credentials.project_id}-crewgraph-functions"
  location = "US"
}}

resource "google_storage_bucket_object" "function_source" {{
  name   = "crewgraph-{config.name}.zip"
  bucket = google_storage_bucket.function_bucket.name
  source = "crewgraph_function.zip"
}}

resource "google_cloudfunctions_function" "crewgraph" {{
  name        = "crewgraph-{config.name}"
  description = "CrewGraph AI workflow function"
  runtime     = "python39"

  available_memory_mb   = {config.function_memory}
  timeout               = {config.function_timeout}
  entry_point          = "execute_workflow"

  source_archive_bucket = google_storage_bucket.function_bucket.name
  source_archive_object = google_storage_bucket_object.function_source.name

  trigger {{
    https_trigger {{}}
  }}

  environment_variables = {{
{chr(10).join(f'    {k} = "{v}"' for k, v in config.environment_variables.items())}
  }}
}}

resource "google_cloudfunctions_function_iam_member" "invoker" {{
  project        = google_cloudfunctions_function.crewgraph.project
  region         = google_cloudfunctions_function.crewgraph.region
  cloud_function = google_cloudfunctions_function.crewgraph.name

  role   = "roles/cloudfunctions.invoker"
  member = "allUsers"
}}

output "function_url" {{
  value = google_cloudfunctions_function.crewgraph.https_trigger_url
}}
"""
    
    def _generate_terraform_compute_engine(self, config: DeploymentConfig) -> str:
        """Generate Terraform template for Compute Engine."""
        machine_type = self._select_gce_machine_type(config)
        
        return f"""
# CrewGraph AI Compute Engine Deployment
terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 4.0"
    }}
  }}
}}

provider "google" {{
  project = "{self.credentials.project_id}"
  region  = "{config.region}"
}}

resource "google_compute_instance" "crewgraph" {{
  name         = "crewgraph-{config.name}"
  machine_type = "{machine_type}"
  zone         = "{config.region}-a"

  boot_disk {{
    initialize_params {{
      image = "debian-cloud/debian-11"
    }}
  }}

  network_interface {{
    network = "default"
    access_config {{
      // Ephemeral public IP
    }}
  }}

  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y python3 python3-pip
    pip3 install crewgraph-ai
    
    # Execute workflow
    python3 -c "
    from crewgraph_ai import CrewGraph
    workflow = CrewGraph('gce_workflow')
    result = workflow.execute()
    print(f'Workflow completed: {{result}}')
    "
  EOF

  tags = ["crewgraph", "workflow"]

  labels = {{
    environment = "production"
    created-by  = "crewgraph-ai"
  }}
}}

output "external_ip" {{
  value = google_compute_instance.crewgraph.network_interface[0].access_config[0].nat_ip
}}
"""
    
    def _generate_terraform_gke(self, config: DeploymentConfig) -> str:
        """Generate Terraform template for GKE."""
        return f"""
# CrewGraph AI GKE Deployment
terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 4.0"
    }}
  }}
}}

provider "google" {{
  project = "{self.credentials.project_id}"
  region  = "{config.region}"
}}

resource "google_container_cluster" "crewgraph" {{
  name     = "crewgraph-{config.name}"
  location = "{config.region}"

  remove_default_node_pool = true
  initial_node_count       = 1

  workload_identity_config {{
    workload_pool = "{self.credentials.project_id}.svc.id.goog"
  }}

  addons_config {{
    horizontal_pod_autoscaling {{
      disabled = false
    }}
    http_load_balancing {{
      disabled = false
    }}
  }}
}}

resource "google_container_node_pool" "crewgraph_nodes" {{
  name       = "crewgraph-{config.name}-nodes"
  location   = "{config.region}"
  cluster    = google_container_cluster.crewgraph.name
  node_count = {config.min_instances}

  node_config {{
    machine_type = "n1-standard-1"
    disk_size_gb = 20

    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only"
    ]

    labels = {{
      environment = "production"
      app         = "crewgraph"
    }}

    tags = ["crewgraph", "gke-node"]
  }}

  autoscaling {{
    min_node_count = {config.min_instances}
    max_node_count = {config.max_instances}
  }}
}}

output "cluster_endpoint" {{
  value = google_container_cluster.crewgraph.endpoint
}}

output "cluster_ca_certificate" {{
  value = base64decode(google_container_cluster.crewgraph.master_auth.0.cluster_ca_certificate)
}}
"""
    
    def _generate_deployment_manager_template(self, config: DeploymentConfig) -> str:
        """Generate Deployment Manager template based on deployment type."""
        if config.deployment_type == DeploymentType.CONTAINER:
            return self._generate_cloud_run_deployment_manager(config, {})
        elif config.deployment_type == DeploymentType.SERVERLESS:
            return self._generate_cloud_function_deployment_manager(config, {})
        elif config.deployment_type == DeploymentType.VIRTUAL_MACHINE:
            return self._generate_gce_deployment_manager(config, "n1-standard-1", "")
        elif config.deployment_type == DeploymentType.KUBERNETES:
            return self._generate_gke_deployment_manager(config)
        else:
            return json.dumps({"error": "Unsupported deployment type"}, indent=2)