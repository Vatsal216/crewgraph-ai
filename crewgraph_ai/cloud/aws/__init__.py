"""
AWS Cloud Provider Implementation for CrewGraph AI

Provides AWS-specific deployment capabilities including ECS, Lambda, EC2,
and CloudFormation template generation.

Author: Vatsal216
Created: 2025-07-23 18:00:00 UTC
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


class AWSProvider(BaseCloudProvider):
    """
    AWS cloud provider implementation.
    
    Supports deployment to ECS, Lambda, EC2, and provides CloudFormation
    template generation for infrastructure as code.
    """
    
    def __init__(self, credentials: CloudCredentials):
        """Initialize AWS provider."""
        super().__init__(credentials)
        
        # AWS service endpoints by region
        self.service_endpoints = {
            "us-east-1": "https://us-east-1.amazonaws.com",
            "us-west-2": "https://us-west-2.amazonaws.com", 
            "eu-west-1": "https://eu-west-1.amazonaws.com"
        }
        
        # AWS pricing (simplified - actual pricing would come from AWS API)
        self.pricing = {
            "ec2": {
                "t3.micro": {"cpu": 2, "memory": 1, "cost_per_hour": 0.0104},
                "t3.small": {"cpu": 2, "memory": 2, "cost_per_hour": 0.0208},
                "t3.medium": {"cpu": 2, "memory": 4, "cost_per_hour": 0.0416},
                "m5.large": {"cpu": 2, "memory": 8, "cost_per_hour": 0.096},
                "m5.xlarge": {"cpu": 4, "memory": 16, "cost_per_hour": 0.192}
            },
            "lambda": {
                "requests_per_million": 0.20,
                "gb_second": 0.0000166667
            },
            "ecs": {
                "fargate_cpu_per_hour": 0.04048,
                "fargate_memory_per_gb_hour": 0.004445
            }
        }
        
        # Simulated AWS resources (in production, this would use boto3)
        self._simulated_resources: Dict[str, CloudResource] = {}
        
        logger.info(f"AWS provider initialized for region {credentials.region}")
    
    def deploy_workflow(
        self, 
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy workflow to AWS using specified deployment type."""
        # Validate configuration
        validation_issues = self.validate_config(config)
        if validation_issues:
            return DeploymentResult(
                success=False,
                error_message=f"Configuration validation failed: {', '.join(validation_issues)}"
            )
        
        # Select deployment method based on type
        if config.deployment_type == DeploymentType.CONTAINER:
            return self._deploy_to_ecs(workflow_id, config, workflow_definition)
        elif config.deployment_type == DeploymentType.SERVERLESS:
            return self._deploy_to_lambda(workflow_id, config, workflow_definition)
        elif config.deployment_type == DeploymentType.VIRTUAL_MACHINE:
            return self._deploy_to_ec2(workflow_id, config, workflow_definition)
        elif config.deployment_type == DeploymentType.KUBERNETES:
            return self._deploy_to_eks(workflow_id, config, workflow_definition)
        else:
            return DeploymentResult(
                success=False,
                error_message=f"Unsupported deployment type: {config.deployment_type}"
            )
    
    def _deploy_to_ecs(
        self,
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy workflow to AWS ECS Fargate."""
        try:
            # Generate resource ID
            resource_id = f"ecs-{uuid.uuid4().hex[:8]}"
            resource_name = self.generate_resource_name(workflow_id, config)
            
            # Create ECS task definition
            task_definition = self._create_ecs_task_definition(config, workflow_definition)
            
            # Create ECS service
            service_config = self._create_ecs_service_config(config, task_definition)
            
            # Calculate cost
            hourly_cost = (
                config.cpu_cores * self.pricing["ecs"]["fargate_cpu_per_hour"] +
                config.memory_gb * self.pricing["ecs"]["fargate_memory_per_gb_hour"]
            ) * config.min_instances
            
            # Create cloud resource
            resource = CloudResource(
                resource_id=resource_id,
                name=resource_name,
                provider=CloudProvider.AWS,
                deployment_type=DeploymentType.CONTAINER,
                status=ResourceStatus.CREATING,
                region=config.region,
                endpoint_url=f"https://{resource_name}.{config.region}.elb.amazonaws.com",
                hourly_cost=hourly_cost,
                tags=self.get_common_tags(workflow_id, config),
                created_at=datetime.now().isoformat()
            )
            
            # Store resource (simulate deployment)
            self._simulated_resources[resource_id] = resource
            
            # Generate CloudFormation template
            cf_template = self._generate_ecs_cloudformation(config, task_definition, service_config)
            
            deployment_logs = [
                f"Created ECS task definition: {task_definition['family']}",
                f"Created ECS service: {service_config['serviceName']}",
                f"Deployed to cluster: {service_config['cluster']}",
                f"Service endpoint: {resource.endpoint_url}"
            ]
            
            logger.info(f"Deployed {workflow_id} to ECS: {resource_id}")
            
            # Simulate deployment completion
            resource.status = ResourceStatus.RUNNING
            
            return DeploymentResult(
                success=True,
                resource=resource,
                deployment_logs=deployment_logs,
                infrastructure_code=cf_template
            )
            
        except Exception as e:
            logger.error(f"ECS deployment failed: {e}")
            return DeploymentResult(
                success=False,
                error_message=f"ECS deployment failed: {str(e)}"
            )
    
    def _deploy_to_lambda(
        self,
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy workflow to AWS Lambda."""
        try:
            resource_id = f"lambda-{uuid.uuid4().hex[:8]}"
            resource_name = self.generate_resource_name(workflow_id, config)
            
            # Create Lambda function configuration
            lambda_config = self._create_lambda_function_config(config, workflow_definition)
            
            # Estimate cost (simplified)
            estimated_invocations_per_hour = 100  # Example
            estimated_duration_ms = config.function_timeout * 1000
            estimated_memory_gb_seconds = (config.function_memory / 1024) * estimated_duration_ms / 1000
            
            hourly_cost = (
                (estimated_invocations_per_hour / 1000000) * self.pricing["lambda"]["requests_per_million"] +
                estimated_memory_gb_seconds * self.pricing["lambda"]["gb_second"]
            )
            
            resource = CloudResource(
                resource_id=resource_id,
                name=resource_name,
                provider=CloudProvider.AWS,
                deployment_type=DeploymentType.SERVERLESS,
                status=ResourceStatus.RUNNING,
                region=config.region,
                endpoint_url=f"https://{resource_id}.lambda-url.{config.region}.on.aws/",
                hourly_cost=hourly_cost,
                tags=self.get_common_tags(workflow_id, config),
                created_at=datetime.now().isoformat()
            )
            
            self._simulated_resources[resource_id] = resource
            
            cf_template = self._generate_lambda_cloudformation(config, lambda_config)
            
            deployment_logs = [
                f"Created Lambda function: {lambda_config['FunctionName']}",
                f"Runtime: {lambda_config['Runtime']}",
                f"Memory: {lambda_config['MemorySize']}MB",
                f"Timeout: {lambda_config['Timeout']}s"
            ]
            
            logger.info(f"Deployed {workflow_id} to Lambda: {resource_id}")
            
            return DeploymentResult(
                success=True,
                resource=resource,
                deployment_logs=deployment_logs,
                infrastructure_code=cf_template
            )
            
        except Exception as e:
            logger.error(f"Lambda deployment failed: {e}")
            return DeploymentResult(
                success=False,
                error_message=f"Lambda deployment failed: {str(e)}"
            )
    
    def _deploy_to_ec2(
        self,
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy workflow to AWS EC2."""
        try:
            resource_id = f"ec2-{uuid.uuid4().hex[:8]}"
            resource_name = self.generate_resource_name(workflow_id, config)
            
            # Select appropriate instance type
            instance_type = self._select_ec2_instance_type(config)
            instance_pricing = self.pricing["ec2"][instance_type]
            
            # Apply spot instance discount if enabled
            hourly_cost = instance_pricing["cost_per_hour"] * config.min_instances
            if config.use_spot_instances:
                hourly_cost *= 0.3  # 70% discount for spot instances
            
            resource = CloudResource(
                resource_id=resource_id,
                name=resource_name,
                provider=CloudProvider.AWS,
                deployment_type=DeploymentType.VIRTUAL_MACHINE,
                status=ResourceStatus.CREATING,
                region=config.region,
                public_ip="52.123.45.67",  # Simulated IP
                internal_ip="10.0.1.100",  # Simulated internal IP
                hourly_cost=hourly_cost,
                tags=self.get_common_tags(workflow_id, config),
                created_at=datetime.now().isoformat()
            )
            
            self._simulated_resources[resource_id] = resource
            
            # Generate user data script for workflow execution
            user_data = self._generate_ec2_user_data(workflow_definition)
            
            cf_template = self._generate_ec2_cloudformation(config, instance_type, user_data)
            
            deployment_logs = [
                f"Created EC2 instance: {resource_id}",
                f"Instance type: {instance_type}",
                f"Public IP: {resource.public_ip}",
                f"Using spot instances: {config.use_spot_instances}"
            ]
            
            logger.info(f"Deployed {workflow_id} to EC2: {resource_id}")
            
            # Simulate startup time
            resource.status = ResourceStatus.RUNNING
            
            return DeploymentResult(
                success=True,
                resource=resource,
                deployment_logs=deployment_logs,
                infrastructure_code=cf_template
            )
            
        except Exception as e:
            logger.error(f"EC2 deployment failed: {e}")
            return DeploymentResult(
                success=False,
                error_message=f"EC2 deployment failed: {str(e)}"
            )
    
    def _deploy_to_eks(
        self,
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy workflow to AWS EKS (Kubernetes)."""
        try:
            resource_id = f"eks-{uuid.uuid4().hex[:8]}"
            resource_name = self.generate_resource_name(workflow_id, config)
            
            # Generate Kubernetes manifests
            k8s_manifests = self._generate_kubernetes_manifests(config, workflow_definition)
            
            # Estimate EKS cost
            hourly_cost = (
                0.10 +  # EKS cluster cost per hour
                config.cpu_cores * 0.04048 * config.min_instances +  # Fargate CPU
                config.memory_gb * 0.004445 * config.min_instances   # Fargate memory
            )
            
            resource = CloudResource(
                resource_id=resource_id,
                name=resource_name,
                provider=CloudProvider.AWS,
                deployment_type=DeploymentType.KUBERNETES,
                status=ResourceStatus.CREATING,
                region=config.region,
                endpoint_url=f"https://{resource_name}.{config.region}.eks.amazonaws.com",
                hourly_cost=hourly_cost,
                tags=self.get_common_tags(workflow_id, config),
                created_at=datetime.now().isoformat()
            )
            
            self._simulated_resources[resource_id] = resource
            
            cf_template = self._generate_eks_cloudformation(config)
            
            deployment_logs = [
                f"Created EKS cluster: {resource_name}",
                f"Deployed Kubernetes manifests",
                f"Service endpoint: {resource.endpoint_url}",
                "Fargate profile configured for serverless pods"
            ]
            
            logger.info(f"Deployed {workflow_id} to EKS: {resource_id}")
            
            resource.status = ResourceStatus.RUNNING
            
            return DeploymentResult(
                success=True,
                resource=resource,
                deployment_logs=deployment_logs,
                infrastructure_code=cf_template
            )
            
        except Exception as e:
            logger.error(f"EKS deployment failed: {e}")
            return DeploymentResult(
                success=False,
                error_message=f"EKS deployment failed: {str(e)}"
            )
    
    def get_resource_status(self, resource_id: str) -> CloudResource:
        """Get current status of AWS resource."""
        if resource_id in self._simulated_resources:
            resource = self._simulated_resources[resource_id]
            
            # Simulate status updates
            if resource.status == ResourceStatus.CREATING:
                # Simulate creation completion
                resource.status = ResourceStatus.RUNNING
                resource.last_updated = datetime.now().isoformat()
            
            return resource
        else:
            # Return a default "not found" resource
            return CloudResource(
                resource_id=resource_id,
                name="unknown",
                provider=CloudProvider.AWS,
                deployment_type=DeploymentType.CONTAINER,
                status=ResourceStatus.ERROR,
                region="unknown"
            )
    
    def scale_resource(self, resource_id: str, target_instances: int) -> bool:
        """Scale AWS resource to target number of instances."""
        if resource_id not in self._simulated_resources:
            return False
        
        resource = self._simulated_resources[resource_id]
        
        # Simulate scaling operation
        logger.info(f"Scaling {resource_id} to {target_instances} instances")
        
        # Update cost based on new instance count
        base_cost = resource.hourly_cost
        if resource.deployment_type == DeploymentType.CONTAINER:
            # For ECS, scale cost proportionally
            resource.hourly_cost = base_cost / max(1, len(resource.tags.get("InstanceCount", "1"))) * target_instances
        
        resource.tags["InstanceCount"] = str(target_instances)
        resource.last_updated = datetime.now().isoformat()
        
        return True
    
    def stop_resource(self, resource_id: str) -> bool:
        """Stop AWS resource."""
        if resource_id not in self._simulated_resources:
            return False
        
        resource = self._simulated_resources[resource_id]
        resource.status = ResourceStatus.STOPPED
        resource.last_updated = datetime.now().isoformat()
        
        logger.info(f"Stopped AWS resource: {resource_id}")
        return True
    
    def delete_resource(self, resource_id: str) -> bool:
        """Delete AWS resource."""
        if resource_id not in self._simulated_resources:
            return False
        
        resource = self._simulated_resources[resource_id]
        resource.status = ResourceStatus.DELETED
        resource.last_updated = datetime.now().isoformat()
        
        # Remove from simulated resources
        del self._simulated_resources[resource_id]
        
        logger.info(f"Deleted AWS resource: {resource_id}")
        return True
    
    def list_resources(self, workflow_id: Optional[WorkflowId] = None) -> List[CloudResource]:
        """List AWS resources."""
        resources = list(self._simulated_resources.values())
        
        if workflow_id:
            resources = [r for r in resources if r.tags.get("WorkflowId") == workflow_id]
        
        return resources
    
    def get_resource_logs(self, resource_id: str, lines: int = 100) -> List[str]:
        """Get logs from AWS resource."""
        # Simulate CloudWatch logs
        sample_logs = [
            f"[{datetime.now().isoformat()}] INFO Starting workflow execution",
            f"[{datetime.now().isoformat()}] INFO Task 1 completed successfully",
            f"[{datetime.now().isoformat()}] INFO Task 2 processing...",
            f"[{datetime.now().isoformat()}] INFO Workflow execution completed"
        ]
        
        return sample_logs[-lines:]
    
    def get_resource_metrics(
        self, 
        resource_id: str, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get CloudWatch metrics for AWS resource."""
        # Simulate CloudWatch metrics
        return {
            "cpu_utilization": {"average": 65.2, "maximum": 89.1, "minimum": 12.3},
            "memory_utilization": {"average": 58.7, "maximum": 75.4, "minimum": 22.1},
            "network_in": {"sum": 1024.5},  # KB
            "network_out": {"sum": 2048.7},  # KB
            "request_count": {"sum": 1450},
            "error_count": {"sum": 12},
            "duration": {"average": 245.6}  # ms
        }
    
    def estimate_cost(self, config: DeploymentConfig) -> Dict[str, float]:
        """Estimate AWS deployment cost."""
        if config.deployment_type == DeploymentType.CONTAINER:
            # ECS Fargate pricing
            cpu_cost = config.cpu_cores * self.pricing["ecs"]["fargate_cpu_per_hour"]
            memory_cost = config.memory_gb * self.pricing["ecs"]["fargate_memory_per_gb_hour"]
            hourly_cost = (cpu_cost + memory_cost) * config.min_instances
            
        elif config.deployment_type == DeploymentType.SERVERLESS:
            # Lambda pricing (estimated)
            estimated_invocations = 100  # per hour
            memory_gb_seconds = (config.function_memory / 1024) * (config.function_timeout / 3600)
            
            request_cost = (estimated_invocations / 1000000) * self.pricing["lambda"]["requests_per_million"]
            compute_cost = memory_gb_seconds * self.pricing["lambda"]["gb_second"]
            hourly_cost = request_cost + compute_cost
            
        elif config.deployment_type == DeploymentType.VIRTUAL_MACHINE:
            # EC2 pricing
            instance_type = self._select_ec2_instance_type(config)
            hourly_cost = self.pricing["ec2"][instance_type]["cost_per_hour"] * config.min_instances
            
            if config.use_spot_instances:
                hourly_cost *= 0.3
                
        else:
            hourly_cost = 1.0  # Default
        
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
        elif template_type == "cloudformation":
            return self._generate_cloudformation_template(config)
        else:
            raise ValueError(f"Unsupported template type: {template_type}")
    
    # Helper methods for AWS-specific operations
    
    def _create_ecs_task_definition(
        self, 
        config: DeploymentConfig, 
        workflow_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create ECS task definition."""
        return {
            "family": f"crewgraph-{config.name}",
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": str(int(config.cpu_cores * 1024)),  # CPU units
            "memory": str(int(config.memory_gb * 1024)),  # MB
            "containerDefinitions": [
                {
                    "name": "crewgraph-container",
                    "image": config.image_uri or "crewgraph/workflow-runner:latest",
                    "memory": int(config.memory_gb * 1024),
                    "cpu": int(config.cpu_cores * 1024),
                    "environment": [
                        {"name": k, "value": v} for k, v in config.environment_variables.items()
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": f"/ecs/crewgraph-{config.name}",
                            "awslogs-region": config.region,
                            "awslogs-stream-prefix": "ecs"
                        }
                    }
                }
            ]
        }
    
    def _create_ecs_service_config(
        self, 
        config: DeploymentConfig, 
        task_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create ECS service configuration."""
        return {
            "serviceName": f"crewgraph-{config.name}-service",
            "cluster": "crewgraph-cluster",
            "taskDefinition": task_definition["family"],
            "desiredCount": config.min_instances,
            "launchType": "FARGATE",
            "networkConfiguration": {
                "awsvpcConfiguration": {
                    "subnets": config.subnet_ids,
                    "securityGroups": config.security_group_ids,
                    "assignPublicIp": "ENABLED"
                }
            }
        }
    
    def _create_lambda_function_config(
        self, 
        config: DeploymentConfig, 
        workflow_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create Lambda function configuration."""
        return {
            "FunctionName": f"crewgraph-{config.name}",
            "Runtime": "python3.9",
            "Code": {
                "ZipFile": self._generate_lambda_code(workflow_definition)
            },
            "Handler": "lambda_function.lambda_handler",
            "Role": f"arn:aws:iam::123456789012:role/crewgraph-lambda-role",
            "MemorySize": config.function_memory,
            "Timeout": config.function_timeout,
            "Environment": {
                "Variables": config.environment_variables
            }
        }
    
    def _select_ec2_instance_type(self, config: DeploymentConfig) -> str:
        """Select appropriate EC2 instance type based on requirements."""
        for instance_type, specs in self.pricing["ec2"].items():
            if (specs["cpu"] >= config.cpu_cores and 
                specs["memory"] >= config.memory_gb):
                return instance_type
        
        return "m5.xlarge"  # Default to larger instance if requirements not met
    
    def _generate_lambda_code(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate Lambda function code for workflow execution."""
        return """
import json
import os

def lambda_handler(event, context):
    # CrewGraph AI workflow execution logic
    print("Executing CrewGraph workflow...")
    
    # Process workflow tasks
    tasks = event.get('tasks', [])
    results = []
    
    for task in tasks:
        print(f"Processing task: {task.get('id')}")
        # Simulate task execution
        result = {
            'task_id': task.get('id'),
            'status': 'completed',
            'result': 'Task executed successfully'
        }
        results.append(result)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Workflow completed successfully',
            'results': results
        })
    }
"""
    
    def _generate_ec2_user_data(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate EC2 user data script for workflow execution."""
        return """#!/bin/bash
yum update -y
yum install -y python3 python3-pip docker
systemctl start docker
systemctl enable docker

# Install CrewGraph AI
pip3 install crewgraph-ai

# Create workflow execution script
cat > /opt/crewgraph_workflow.py << 'EOF'
import json
from crewgraph_ai import CrewGraph

# Load workflow definition
workflow = CrewGraph("ec2_workflow")

# Execute workflow
try:
    result = workflow.execute()
    print(f"Workflow completed: {result}")
except Exception as e:
    print(f"Workflow failed: {e}")
EOF

# Run workflow
python3 /opt/crewgraph_workflow.py
"""
    
    def _generate_kubernetes_manifests(
        self, 
        config: DeploymentConfig, 
        workflow_definition: Dict[str, Any]
    ) -> str:
        """Generate Kubernetes deployment manifests."""
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
        image: {config.image_uri or 'crewgraph/workflow-runner:latest'}
        resources:
          requests:
            memory: "{config.memory_gb}Gi"
            cpu: "{config.cpu_cores}"
          limits:
            memory: "{config.memory_gb}Gi"
            cpu: "{config.cpu_cores}"
        env:
{chr(10).join(f'        - name: {k}{chr(10)}          value: "{v}"' for k, v in config.environment_variables.items())}
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
"""
    
    def _generate_ecs_cloudformation(
        self, 
        config: DeploymentConfig, 
        task_definition: Dict[str, Any],
        service_config: Dict[str, Any]
    ) -> str:
        """Generate CloudFormation template for ECS deployment."""
        return json.dumps({
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"CrewGraph AI ECS deployment for {config.name}",
            "Resources": {
                "TaskDefinition": {
                    "Type": "AWS::ECS::TaskDefinition",
                    "Properties": task_definition
                },
                "Service": {
                    "Type": "AWS::ECS::Service",
                    "Properties": service_config
                },
                "LogGroup": {
                    "Type": "AWS::Logs::LogGroup",
                    "Properties": {
                        "LogGroupName": f"/ecs/crewgraph-{config.name}",
                        "RetentionInDays": 7
                    }
                }
            }
        }, indent=2)
    
    def _generate_lambda_cloudformation(
        self, 
        config: DeploymentConfig, 
        lambda_config: Dict[str, Any]
    ) -> str:
        """Generate CloudFormation template for Lambda deployment."""
        return json.dumps({
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"CrewGraph AI Lambda deployment for {config.name}",
            "Resources": {
                "LambdaFunction": {
                    "Type": "AWS::Lambda::Function",
                    "Properties": lambda_config
                },
                "LambdaRole": {
                    "Type": "AWS::IAM::Role",
                    "Properties": {
                        "AssumeRolePolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Principal": {"Service": "lambda.amazonaws.com"},
                                    "Action": "sts:AssumeRole"
                                }
                            ]
                        },
                        "ManagedPolicyArns": [
                            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
                        ]
                    }
                }
            }
        }, indent=2)
    
    def _generate_ec2_cloudformation(
        self, 
        config: DeploymentConfig, 
        instance_type: str,
        user_data: str
    ) -> str:
        """Generate CloudFormation template for EC2 deployment."""
        return json.dumps({
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"CrewGraph AI EC2 deployment for {config.name}",
            "Resources": {
                "Instance": {
                    "Type": "AWS::EC2::Instance",
                    "Properties": {
                        "InstanceType": instance_type,
                        "ImageId": "ami-0abcdef1234567890",  # Amazon Linux 2
                        "UserData": {"Fn::Base64": user_data},
                        "IamInstanceProfile": {"Ref": "InstanceProfile"},
                        "SecurityGroupIds": config.security_group_ids,
                        "SubnetId": config.subnet_ids[0] if config.subnet_ids else None,
                        "Tags": [
                            {"Key": k, "Value": v} for k, v in self.get_common_tags("workflow", config).items()
                        ]
                    }
                },
                "InstanceProfile": {
                    "Type": "AWS::IAM::InstanceProfile",
                    "Properties": {
                        "Roles": [{"Ref": "InstanceRole"}]
                    }
                },
                "InstanceRole": {
                    "Type": "AWS::IAM::Role",
                    "Properties": {
                        "AssumeRolePolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Principal": {"Service": "ec2.amazonaws.com"},
                                    "Action": "sts:AssumeRole"
                                }
                            ]
                        }
                    }
                }
            }
        }, indent=2)
    
    def _generate_eks_cloudformation(self, config: DeploymentConfig) -> str:
        """Generate CloudFormation template for EKS deployment."""
        return json.dumps({
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"CrewGraph AI EKS deployment for {config.name}",
            "Resources": {
                "EKSCluster": {
                    "Type": "AWS::EKS::Cluster",
                    "Properties": {
                        "Name": f"crewgraph-{config.name}",
                        "Version": "1.21",
                        "RoleArn": {"Fn::GetAtt": ["EKSServiceRole", "Arn"]},
                        "ResourcesVpcConfig": {
                            "SubnetIds": config.subnet_ids
                        }
                    }
                },
                "EKSServiceRole": {
                    "Type": "AWS::IAM::Role",
                    "Properties": {
                        "AssumeRolePolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Principal": {"Service": "eks.amazonaws.com"},
                                    "Action": "sts:AssumeRole"
                                }
                            ]
                        },
                        "ManagedPolicyArns": [
                            "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
                        ]
                    }
                }
            }
        }, indent=2)
    
    def _generate_terraform_template(self, config: DeploymentConfig) -> str:
        """Generate Terraform template for AWS deployment."""
        if config.deployment_type == DeploymentType.CONTAINER:
            return self._generate_terraform_ecs(config)
        elif config.deployment_type == DeploymentType.SERVERLESS:
            return self._generate_terraform_lambda(config)
        elif config.deployment_type == DeploymentType.VIRTUAL_MACHINE:
            return self._generate_terraform_ec2(config)
        else:
            return "# Terraform template not implemented for this deployment type"
    
    def _generate_terraform_ecs(self, config: DeploymentConfig) -> str:
        """Generate Terraform template for ECS."""
        return f"""
# CrewGraph AI ECS Deployment
terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }}
  }}
}}

provider "aws" {{
  region = "{config.region}"
}}

resource "aws_ecs_cluster" "crewgraph" {{
  name = "crewgraph-{config.name}"
}}

resource "aws_ecs_task_definition" "crewgraph" {{
  family                   = "crewgraph-{config.name}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "{int(config.cpu_cores * 1024)}"
  memory                   = "{int(config.memory_gb * 1024)}"
  
  container_definitions = jsonencode([
    {{
      name  = "crewgraph-container"
      image = "{config.image_uri or 'crewgraph/workflow-runner:latest'}"
      memory = {int(config.memory_gb * 1024)}
      cpu    = {int(config.cpu_cores * 1024)}
      
      environment = [
{chr(10).join(f'        {{ name = "{k}", value = "{v}" }},' for k, v in config.environment_variables.items())}
      ]
      
      logConfiguration = {{
        logDriver = "awslogs"
        options = {{
          awslogs-group         = "/ecs/crewgraph-{config.name}"
          awslogs-region        = "{config.region}"
          awslogs-stream-prefix = "ecs"
        }}
      }}
    }}
  ])
}}

resource "aws_ecs_service" "crewgraph" {{
  name            = "crewgraph-{config.name}"
  cluster         = aws_ecs_cluster.crewgraph.id
  task_definition = aws_ecs_task_definition.crewgraph.arn
  desired_count   = {config.min_instances}
  launch_type     = "FARGATE"
  
  network_configuration {{
    subnets          = {config.subnet_ids}
    security_groups  = {config.security_group_ids}
    assign_public_ip = true
  }}
}}
"""
    
    def _generate_terraform_lambda(self, config: DeploymentConfig) -> str:
        """Generate Terraform template for Lambda."""
        return f"""
# CrewGraph AI Lambda Deployment
terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }}
  }}
}}

provider "aws" {{
  region = "{config.region}"
}}

resource "aws_lambda_function" "crewgraph" {{
  filename      = "crewgraph_workflow.zip"
  function_name = "crewgraph-{config.name}"
  role          = aws_iam_role.lambda_role.arn
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.9"
  memory_size   = {config.function_memory}
  timeout       = {config.function_timeout}
  
  environment {{
    variables = {{
{chr(10).join(f'      {k} = "{v}"' for k, v in config.environment_variables.items())}
    }}
  }}
}}

resource "aws_iam_role" "lambda_role" {{
  name = "crewgraph-{config.name}-lambda-role"
  
  assume_role_policy = jsonencode({{
    Version = "2012-10-17"
    Statement = [
      {{
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {{
          Service = "lambda.amazonaws.com"
        }}
      }}
    ]
  }})
}}

resource "aws_iam_role_policy_attachment" "lambda_basic" {{
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  role       = aws_iam_role.lambda_role.name
}}
"""
    
    def _generate_terraform_ec2(self, config: DeploymentConfig) -> str:
        """Generate Terraform template for EC2."""
        instance_type = self._select_ec2_instance_type(config)
        
        return f"""
# CrewGraph AI EC2 Deployment
terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }}
  }}
}}

provider "aws" {{
  region = "{config.region}"
}}

resource "aws_instance" "crewgraph" {{
  ami           = "ami-0abcdef1234567890"  # Amazon Linux 2
  instance_type = "{instance_type}"
  
  vpc_security_group_ids = {config.security_group_ids}
  subnet_id              = "{config.subnet_ids[0] if config.subnet_ids else ''}"
  
  user_data = base64encode(<<-EOF
    #!/bin/bash
    yum update -y
    yum install -y python3 python3-pip
    pip3 install crewgraph-ai
    # Add workflow execution logic here
  EOF
  )
  
  tags = {{
    Name = "crewgraph-{config.name}"
    Environment = "production"
    CreatedBy = "CrewGraphAI"
  }}
}}
"""
    
    def _generate_cloudformation_template(self, config: DeploymentConfig) -> str:
        """Generate CloudFormation template based on deployment type."""
        if config.deployment_type == DeploymentType.CONTAINER:
            return self._generate_ecs_cloudformation(config, {}, {})
        elif config.deployment_type == DeploymentType.SERVERLESS:
            return self._generate_lambda_cloudformation(config, {})
        elif config.deployment_type == DeploymentType.VIRTUAL_MACHINE:
            return self._generate_ec2_cloudformation(config, "t3.medium", "")
        else:
            return json.dumps({"error": "Unsupported deployment type"}, indent=2)