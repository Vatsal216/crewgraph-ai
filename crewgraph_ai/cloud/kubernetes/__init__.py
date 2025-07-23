"""
Kubernetes Deployment Support for CrewGraph AI

Provides Kubernetes deployment configurations, Helm charts, and manifests
for deploying CrewGraph workflows to any Kubernetes cluster.

Author: Vatsal216
Created: 2025-07-23 18:20:00 UTC
"""

import json
import yaml
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .. import DeploymentConfig
from ...types import WorkflowId
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KubernetesConfig:
    """Kubernetes-specific deployment configuration."""
    
    namespace: str = "crewgraph"
    service_type: str = "LoadBalancer"  # ClusterIP, NodePort, LoadBalancer
    enable_ingress: bool = True
    ingress_class: str = "nginx"
    enable_hpa: bool = True  # Horizontal Pod Autoscaler
    enable_pdb: bool = True  # Pod Disruption Budget
    storage_class: str = "standard"
    persistent_volume_size: str = "10Gi"
    
    # Security
    enable_network_policies: bool = True
    enable_pod_security_policy: bool = True
    service_account_name: str = "crewgraph-sa"
    
    # Monitoring
    enable_prometheus_monitoring: bool = True
    enable_grafana_dashboard: bool = True


class KubernetesDeployer:
    """
    Kubernetes deployment generator for CrewGraph AI workflows.
    
    Generates complete Kubernetes manifests, Helm charts, and deployment
    configurations for multi-cloud Kubernetes environments.
    """
    
    def __init__(self, k8s_config: Optional[KubernetesConfig] = None):
        """Initialize Kubernetes deployer."""
        self.k8s_config = k8s_config or KubernetesConfig()
        logger.info("Kubernetes deployer initialized")
    
    def generate_manifests(
        self,
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate complete Kubernetes manifests for workflow deployment.
        
        Returns:
            Dictionary of manifest files (filename -> content)
        """
        manifests = {}
        
        # Generate namespace
        manifests["01-namespace.yaml"] = self._generate_namespace()
        
        # Generate service account and RBAC
        manifests["02-serviceaccount.yaml"] = self._generate_service_account()
        manifests["03-rbac.yaml"] = self._generate_rbac()
        
        # Generate configmap for workflow definition
        manifests["04-configmap.yaml"] = self._generate_configmap(workflow_id, workflow_definition)
        
        # Generate secret for sensitive data
        manifests["05-secret.yaml"] = self._generate_secret(config)
        
        # Generate deployment
        manifests["06-deployment.yaml"] = self._generate_deployment(workflow_id, config)
        
        # Generate service
        manifests["07-service.yaml"] = self._generate_service(workflow_id, config)
        
        # Generate ingress if enabled
        if self.k8s_config.enable_ingress:
            manifests["08-ingress.yaml"] = self._generate_ingress(workflow_id, config)
        
        # Generate HPA if enabled
        if self.k8s_config.enable_hpa:
            manifests["09-hpa.yaml"] = self._generate_hpa(workflow_id, config)
        
        # Generate PDB if enabled
        if self.k8s_config.enable_pdb:
            manifests["10-pdb.yaml"] = self._generate_pdb(workflow_id, config)
        
        # Generate network policy if enabled
        if self.k8s_config.enable_network_policies:
            manifests["11-networkpolicy.yaml"] = self._generate_network_policy(workflow_id)
        
        # Generate persistent volume claim
        manifests["12-pvc.yaml"] = self._generate_pvc(workflow_id)
        
        # Generate monitoring resources
        if self.k8s_config.enable_prometheus_monitoring:
            manifests["13-servicemonitor.yaml"] = self._generate_service_monitor(workflow_id)
        
        logger.info(f"Generated {len(manifests)} Kubernetes manifests for {workflow_id}")
        return manifests
    
    def generate_helm_chart(
        self,
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        workflow_definition: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate Helm chart for workflow deployment.
        
        Returns:
            Dictionary of chart files (filepath -> content)
        """
        chart_files = {}
        
        # Chart.yaml
        chart_files["Chart.yaml"] = self._generate_chart_yaml(workflow_id)
        
        # values.yaml
        chart_files["values.yaml"] = self._generate_values_yaml(config, workflow_definition)
        
        # Templates
        chart_files["templates/deployment.yaml"] = self._generate_helm_deployment()
        chart_files["templates/service.yaml"] = self._generate_helm_service()
        chart_files["templates/configmap.yaml"] = self._generate_helm_configmap()
        chart_files["templates/secret.yaml"] = self._generate_helm_secret()
        chart_files["templates/serviceaccount.yaml"] = self._generate_helm_service_account()
        chart_files["templates/ingress.yaml"] = self._generate_helm_ingress()
        chart_files["templates/hpa.yaml"] = self._generate_helm_hpa()
        chart_files["templates/pdb.yaml"] = self._generate_helm_pdb()
        
        # Helper templates
        chart_files["templates/_helpers.tpl"] = self._generate_helm_helpers()
        
        logger.info(f"Generated Helm chart for {workflow_id}")
        return chart_files
    
    def generate_kustomization(
        self,
        workflow_id: WorkflowId,
        config: DeploymentConfig,
        environments: List[str] = None
    ) -> Dict[str, str]:
        """
        Generate Kustomize configuration for multi-environment deployments.
        
        Returns:
            Dictionary of kustomization files
        """
        if environments is None:
            environments = ["dev", "staging", "prod"]
        
        kustomize_files = {}
        
        # Base kustomization
        kustomize_files["base/kustomization.yaml"] = self._generate_base_kustomization()
        
        # Environment-specific overlays
        for env in environments:
            kustomize_files[f"overlays/{env}/kustomization.yaml"] = self._generate_env_kustomization(env, config)
            kustomize_files[f"overlays/{env}/config-patch.yaml"] = self._generate_env_config_patch(env, config)
        
        logger.info(f"Generated Kustomize configuration for {workflow_id}")
        return kustomize_files
    
    # Manifest generation methods
    
    def _generate_namespace(self) -> str:
        """Generate namespace manifest."""
        return yaml.dump({
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.k8s_config.namespace,
                "labels": {
                    "name": self.k8s_config.namespace,
                    "app.kubernetes.io/name": "crewgraph",
                    "app.kubernetes.io/managed-by": "crewgraph-ai"
                }
            }
        })
    
    def _generate_service_account(self) -> str:
        """Generate service account manifest."""
        return yaml.dump({
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": self.k8s_config.service_account_name,
                "namespace": self.k8s_config.namespace,
                "labels": {
                    "app.kubernetes.io/name": "crewgraph",
                    "app.kubernetes.io/component": "serviceaccount"
                }
            }
        })
    
    def _generate_rbac(self) -> str:
        """Generate RBAC manifests."""
        manifests = [
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "ClusterRole",
                "metadata": {
                    "name": "crewgraph-role"
                },
                "rules": [
                    {
                        "apiGroups": [""],
                        "resources": ["pods", "services", "configmaps", "secrets"],
                        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"]
                    },
                    {
                        "apiGroups": ["apps"],
                        "resources": ["deployments", "replicasets"],
                        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"]
                    }
                ]
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "ClusterRoleBinding",
                "metadata": {
                    "name": "crewgraph-rolebinding"
                },
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "ClusterRole",
                    "name": "crewgraph-role"
                },
                "subjects": [
                    {
                        "kind": "ServiceAccount",
                        "name": self.k8s_config.service_account_name,
                        "namespace": self.k8s_config.namespace
                    }
                ]
            }
        ]
        
        return yaml.dump_all(manifests)
    
    def _generate_configmap(self, workflow_id: WorkflowId, workflow_definition: Dict[str, Any]) -> str:
        """Generate configmap for workflow configuration."""
        return yaml.dump({
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"crewgraph-{workflow_id}-config",
                "namespace": self.k8s_config.namespace,
                "labels": {
                    "app.kubernetes.io/name": "crewgraph",
                    "app.kubernetes.io/instance": workflow_id
                }
            },
            "data": {
                "workflow.json": json.dumps(workflow_definition, indent=2),
                "config.yaml": yaml.dump({
                    "workflow_id": workflow_id,
                    "log_level": "INFO",
                    "enable_metrics": True,
                    "metrics_port": 9090
                })
            }
        })
    
    def _generate_secret(self, config: DeploymentConfig) -> str:
        """Generate secret for sensitive configuration."""
        return yaml.dump({
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "crewgraph-secrets",
                "namespace": self.k8s_config.namespace
            },
            "type": "Opaque",
            "data": {
                # Base64 encoded secret values
                "api-key": "Y3Jld2dyYXBoLWFwaS1rZXk=",  # crewgraph-api-key
                "database-password": "cGFzc3dvcmQ="  # password
            }
        })
    
    def _generate_deployment(self, workflow_id: WorkflowId, config: DeploymentConfig) -> str:
        """Generate deployment manifest."""
        return yaml.dump({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"crewgraph-{workflow_id}",
                "namespace": self.k8s_config.namespace,
                "labels": {
                    "app.kubernetes.io/name": "crewgraph",
                    "app.kubernetes.io/instance": workflow_id,
                    "app.kubernetes.io/version": "1.0.0"
                }
            },
            "spec": {
                "replicas": config.min_instances,
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": "crewgraph",
                        "app.kubernetes.io/instance": workflow_id
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app.kubernetes.io/name": "crewgraph",
                            "app.kubernetes.io/instance": workflow_id
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "9090",
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "serviceAccountName": self.k8s_config.service_account_name,
                        "containers": [
                            {
                                "name": "crewgraph",
                                "image": config.image_uri or "crewgraph/workflow-runner:latest",
                                "imagePullPolicy": "Always",
                                "ports": [
                                    {"containerPort": 8080, "name": "http"},
                                    {"containerPort": 9090, "name": "metrics"}
                                ],
                                "resources": {
                                    "requests": {
                                        "memory": f"{config.memory_gb}Gi",
                                        "cpu": f"{config.cpu_cores}"
                                    },
                                    "limits": {
                                        "memory": f"{config.memory_gb * 1.5}Gi",
                                        "cpu": f"{config.cpu_cores * 1.5}"
                                    }
                                },
                                "env": [
                                    {"name": "WORKFLOW_ID", "value": workflow_id},
                                    {"name": "LOG_LEVEL", "value": "INFO"},
                                    {
                                        "name": "API_KEY",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": "crewgraph-secrets",
                                                "key": "api-key"
                                            }
                                        }
                                    }
                                ] + [
                                    {"name": k, "value": v} for k, v in config.environment_variables.items()
                                ],
                                "volumeMounts": [
                                    {
                                        "name": "config-volume",
                                        "mountPath": "/app/config"
                                    },
                                    {
                                        "name": "data-volume",
                                        "mountPath": "/app/data"
                                    }
                                ],
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/ready",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                }
                            }
                        ],
                        "volumes": [
                            {
                                "name": "config-volume",
                                "configMap": {
                                    "name": f"crewgraph-{workflow_id}-config"
                                }
                            },
                            {
                                "name": "data-volume",
                                "persistentVolumeClaim": {
                                    "claimName": f"crewgraph-{workflow_id}-pvc"
                                }
                            }
                        ]
                    }
                }
            }
        })
    
    def _generate_service(self, workflow_id: WorkflowId, config: DeploymentConfig) -> str:
        """Generate service manifest."""
        return yaml.dump({
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"crewgraph-{workflow_id}-service",
                "namespace": self.k8s_config.namespace,
                "labels": {
                    "app.kubernetes.io/name": "crewgraph",
                    "app.kubernetes.io/instance": workflow_id
                }
            },
            "spec": {
                "type": self.k8s_config.service_type,
                "ports": [
                    {
                        "port": 80,
                        "targetPort": 8080,
                        "protocol": "TCP",
                        "name": "http"
                    },
                    {
                        "port": 9090,
                        "targetPort": 9090,
                        "protocol": "TCP",
                        "name": "metrics"
                    }
                ],
                "selector": {
                    "app.kubernetes.io/name": "crewgraph",
                    "app.kubernetes.io/instance": workflow_id
                }
            }
        })
    
    def _generate_ingress(self, workflow_id: WorkflowId, config: DeploymentConfig) -> str:
        """Generate ingress manifest."""
        return yaml.dump({
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"crewgraph-{workflow_id}-ingress",
                "namespace": self.k8s_config.namespace,
                "annotations": {
                    "kubernetes.io/ingress.class": self.k8s_config.ingress_class,
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod"
                }
            },
            "spec": {
                "tls": [
                    {
                        "hosts": [f"crewgraph-{workflow_id}.example.com"],
                        "secretName": f"crewgraph-{workflow_id}-tls"
                    }
                ],
                "rules": [
                    {
                        "host": f"crewgraph-{workflow_id}.example.com",
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": f"crewgraph-{workflow_id}-service",
                                            "port": {"number": 80}
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        })
    
    def _generate_hpa(self, workflow_id: WorkflowId, config: DeploymentConfig) -> str:
        """Generate HPA manifest."""
        return yaml.dump({
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"crewgraph-{workflow_id}-hpa",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"crewgraph-{workflow_id}"
                },
                "minReplicas": config.min_instances,
                "maxReplicas": config.max_instances,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            }
        })
    
    def _generate_pdb(self, workflow_id: WorkflowId, config: DeploymentConfig) -> str:
        """Generate Pod Disruption Budget manifest."""
        return yaml.dump({
            "apiVersion": "policy/v1",
            "kind": "PodDisruptionBudget",
            "metadata": {
                "name": f"crewgraph-{workflow_id}-pdb",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "minAvailable": 1,
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": "crewgraph",
                        "app.kubernetes.io/instance": workflow_id
                    }
                }
            }
        })
    
    def _generate_network_policy(self, workflow_id: WorkflowId) -> str:
        """Generate network policy manifest."""
        return yaml.dump({
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"crewgraph-{workflow_id}-netpol",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": "crewgraph",
                        "app.kubernetes.io/instance": workflow_id
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {"namespaceSelector": {"matchLabels": {"name": "ingress-nginx"}}},
                            {"namespaceSelector": {"matchLabels": {"name": "monitoring"}}}
                        ],
                        "ports": [
                            {"protocol": "TCP", "port": 8080},
                            {"protocol": "TCP", "port": 9090}
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [{}],
                        "ports": [
                            {"protocol": "TCP", "port": 53},
                            {"protocol": "UDP", "port": 53},
                            {"protocol": "TCP", "port": 443},
                            {"protocol": "TCP", "port": 80}
                        ]
                    }
                ]
            }
        })
    
    def _generate_pvc(self, workflow_id: WorkflowId) -> str:
        """Generate PVC manifest."""
        return yaml.dump({
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"crewgraph-{workflow_id}-pvc",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": self.k8s_config.storage_class,
                "resources": {
                    "requests": {
                        "storage": self.k8s_config.persistent_volume_size
                    }
                }
            }
        })
    
    def _generate_service_monitor(self, workflow_id: WorkflowId) -> str:
        """Generate ServiceMonitor for Prometheus."""
        return yaml.dump({
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {
                "name": f"crewgraph-{workflow_id}-monitor",
                "namespace": self.k8s_config.namespace,
                "labels": {
                    "app.kubernetes.io/name": "crewgraph",
                    "app.kubernetes.io/instance": workflow_id
                }
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": "crewgraph",
                        "app.kubernetes.io/instance": workflow_id
                    }
                },
                "endpoints": [
                    {
                        "port": "metrics",
                        "interval": "30s",
                        "path": "/metrics"
                    }
                ]
            }
        })
    
    # Helm chart generation methods
    
    def _generate_chart_yaml(self, workflow_id: WorkflowId) -> str:
        """Generate Chart.yaml for Helm chart."""
        return yaml.dump({
            "apiVersion": "v2",
            "name": "crewgraph-workflow",
            "description": f"CrewGraph AI workflow deployment for {workflow_id}",
            "type": "application",
            "version": "1.0.0",
            "appVersion": "1.0.0",
            "keywords": ["crewgraph", "ai", "workflow", "automation"],
            "maintainers": [
                {
                    "name": "CrewGraph AI",
                    "email": "support@crewgraph.ai"
                }
            ]
        })
    
    def _generate_values_yaml(self, config: DeploymentConfig, workflow_definition: Dict[str, Any]) -> str:
        """Generate values.yaml for Helm chart."""
        values = {
            "replicaCount": config.min_instances,
            "image": {
                "repository": config.image_uri or "crewgraph/workflow-runner",
                "tag": "latest",
                "pullPolicy": "Always"
            },
            "service": {
                "type": self.k8s_config.service_type,
                "port": 80,
                "targetPort": 8080
            },
            "ingress": {
                "enabled": self.k8s_config.enable_ingress,
                "className": self.k8s_config.ingress_class,
                "annotations": {},
                "hosts": [
                    {
                        "host": "crewgraph.local",
                        "paths": [
                            {
                                "path": "/",
                                "pathType": "Prefix"
                            }
                        ]
                    }
                ],
                "tls": []
            },
            "resources": {
                "requests": {
                    "memory": f"{config.memory_gb}Gi",
                    "cpu": f"{config.cpu_cores}"
                },
                "limits": {
                    "memory": f"{config.memory_gb * 1.5}Gi",
                    "cpu": f"{config.cpu_cores * 1.5}"
                }
            },
            "autoscaling": {
                "enabled": self.k8s_config.enable_hpa,
                "minReplicas": config.min_instances,
                "maxReplicas": config.max_instances,
                "targetCPUUtilizationPercentage": 70,
                "targetMemoryUtilizationPercentage": 80
            },
            "podDisruptionBudget": {
                "enabled": self.k8s_config.enable_pdb,
                "minAvailable": 1
            },
            "serviceAccount": {
                "create": True,
                "name": self.k8s_config.service_account_name
            },
            "monitoring": {
                "enabled": self.k8s_config.enable_prometheus_monitoring,
                "port": 9090,
                "path": "/metrics"
            },
            "config": {
                "workflow": workflow_definition,
                "environment": config.environment_variables
            }
        }
        
        return yaml.dump(values)
    
    def _generate_helm_deployment(self) -> str:
        """Generate Helm deployment template."""
        return """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "crewgraph.fullname" . }}
  labels:
    {{- include "crewgraph.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "crewgraph.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      {{- with .Values.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      labels:
        {{- include "crewgraph.selectorLabels" . | nindent 8 }}
    spec:
      serviceAccountName: {{ include "crewgraph.serviceAccountName" . }}
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
            - name: metrics
              containerPort: 9090
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
          readinessProbe:
            httpGet:
              path: /ready
              port: http
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          env:
            {{- range $key, $value := .Values.config.environment }}
            - name: {{ $key }}
              value: {{ $value | quote }}
            {{- end }}
"""
    
    def _generate_helm_service(self) -> str:
        """Generate Helm service template."""
        return """apiVersion: v1
kind: Service
metadata:
  name: {{ include "crewgraph.fullname" . }}
  labels:
    {{- include "crewgraph.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
    - port: 9090
      targetPort: metrics
      protocol: TCP
      name: metrics
  selector:
    {{- include "crewgraph.selectorLabels" . | nindent 4 }}
"""
    
    def _generate_helm_configmap(self) -> str:
        """Generate Helm configmap template."""
        return """apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "crewgraph.fullname" . }}-config
  labels:
    {{- include "crewgraph.labels" . | nindent 4 }}
data:
  workflow.json: |
    {{- .Values.config.workflow | toJson | nindent 4 }}
"""
    
    def _generate_helm_secret(self) -> str:
        """Generate Helm secret template."""
        return """{{- if .Values.secrets }}
apiVersion: v1
kind: Secret
metadata:
  name: {{ include "crewgraph.fullname" . }}-secrets
  labels:
    {{- include "crewgraph.labels" . | nindent 4 }}
type: Opaque
data:
  {{- range $key, $value := .Values.secrets }}
  {{ $key }}: {{ $value | b64enc }}
  {{- end }}
{{- end }}
"""
    
    def _generate_helm_service_account(self) -> str:
        """Generate Helm service account template."""
        return """{{- if .Values.serviceAccount.create -}}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ include "crewgraph.serviceAccountName" . }}
  labels:
    {{- include "crewgraph.labels" . | nindent 4 }}
{{- end }}
"""
    
    def _generate_helm_ingress(self) -> str:
        """Generate Helm ingress template."""
        return """{{- if .Values.ingress.enabled -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "crewgraph.fullname" . }}
  labels:
    {{- include "crewgraph.labels" . | nindent 4 }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{ .Values.ingress.className }}
  {{- end }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
      secretName: {{ .secretName }}
    {{- end }}
  {{- end }}
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ .host | quote }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            pathType: {{ .pathType }}
            backend:
              service:
                name: {{ include "crewgraph.fullname" $ }}
                port:
                  number: {{ $.Values.service.port }}
          {{- end }}
    {{- end }}
{{- end }}
"""
    
    def _generate_helm_hpa(self) -> str:
        """Generate Helm HPA template."""
        return """{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "crewgraph.fullname" . }}
  labels:
    {{- include "crewgraph.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "crewgraph.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
    {{- end }}
{{- end }}
"""
    
    def _generate_helm_pdb(self) -> str:
        """Generate Helm PDB template."""
        return """{{- if .Values.podDisruptionBudget.enabled }}
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {{ include "crewgraph.fullname" . }}
  labels:
    {{- include "crewgraph.labels" . | nindent 4 }}
spec:
  {{- if .Values.podDisruptionBudget.minAvailable }}
  minAvailable: {{ .Values.podDisruptionBudget.minAvailable }}
  {{- end }}
  {{- if .Values.podDisruptionBudget.maxUnavailable }}
  maxUnavailable: {{ .Values.podDisruptionBudget.maxUnavailable }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "crewgraph.selectorLabels" . | nindent 6 }}
{{- end }}
"""
    
    def _generate_helm_helpers(self) -> str:
        """Generate Helm helpers template."""
        return '''{{/*
Expand the name of the chart.
*/}}
{{- define "crewgraph.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "crewgraph.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "crewgraph.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "crewgraph.labels" -}}
helm.sh/chart: {{ include "crewgraph.chart" . }}
{{ include "crewgraph.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "crewgraph.selectorLabels" -}}
app.kubernetes.io/name: {{ include "crewgraph.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "crewgraph.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "crewgraph.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
'''
    
    # Kustomization generation methods
    
    def _generate_base_kustomization(self) -> str:
        """Generate base kustomization.yaml."""
        return yaml.dump({
            "apiVersion": "kustomize.config.k8s.io/v1beta1",
            "kind": "Kustomization",
            "resources": [
                "01-namespace.yaml",
                "02-serviceaccount.yaml",
                "03-rbac.yaml",
                "04-configmap.yaml",
                "05-secret.yaml",
                "06-deployment.yaml",
                "07-service.yaml",
                "08-ingress.yaml",
                "09-hpa.yaml",
                "10-pdb.yaml"
            ],
            "commonLabels": {
                "app.kubernetes.io/name": "crewgraph",
                "app.kubernetes.io/managed-by": "kustomize"
            }
        })
    
    def _generate_env_kustomization(self, env: str, config: DeploymentConfig) -> str:
        """Generate environment-specific kustomization."""
        return yaml.dump({
            "apiVersion": "kustomize.config.k8s.io/v1beta1",
            "kind": "Kustomization",
            "namespace": f"crewgraph-{env}",
            "resources": ["../../base"],
            "patchesStrategicMerge": ["config-patch.yaml"],
            "commonLabels": {
                "environment": env
            },
            "images": [
                {
                    "name": "crewgraph/workflow-runner",
                    "newTag": f"{env}-latest"
                }
            ]
        })
    
    def _generate_env_config_patch(self, env: str, config: DeploymentConfig) -> str:
        """Generate environment-specific configuration patch."""
        replicas = {"dev": 1, "staging": 2, "prod": config.min_instances}.get(env, 1)
        
        return yaml.dump({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "crewgraph-workflow"
            },
            "spec": {
                "replicas": replicas,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "crewgraph",
                                "env": [
                                    {"name": "ENVIRONMENT", "value": env},
                                    {"name": "LOG_LEVEL", "value": "DEBUG" if env == "dev" else "INFO"}
                                ]
                            }
                        ]
                    }
                }
            }
        })