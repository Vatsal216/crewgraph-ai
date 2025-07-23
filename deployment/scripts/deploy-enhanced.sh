#!/bin/bash
# CrewGraph AI Enhanced Deployment Script
# Deploys CrewGraph AI to Kubernetes with full monitoring and autoscaling
set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DEPLOY_DIR="$PROJECT_ROOT/deployment"

# Default values
NAMESPACE="${NAMESPACE:-crewgraph-ai}"
ENVIRONMENT="${ENVIRONMENT:-production}"
AWS_REGION="${AWS_REGION:-us-west-2}"
EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:-crewgraph-ai-production-cluster}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOMAIN="${DOMAIN:-api.crewgraph-ai.com}"
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"
ENABLE_AUTOSCALING="${ENABLE_AUTOSCALING:-true}"
ENABLE_ISTIO="${ENABLE_ISTIO:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local tools=("kubectl" "helm" "aws")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check if we can connect to the cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Update kubeconfig for EKS
setup_kubeconfig() {
    log_info "Setting up kubeconfig for EKS cluster: $EKS_CLUSTER_NAME"
    aws eks update-kubeconfig --region "$AWS_REGION" --name "$EKS_CLUSTER_NAME"
    log_success "Kubeconfig updated"
}

# Create namespaces and basic resources
setup_namespaces() {
    log_info "Setting up namespaces and RBAC..."
    kubectl apply -f "$DEPLOY_DIR/kubernetes/namespace.yaml"
    log_success "Namespaces and RBAC configured"
}

# Install monitoring stack
setup_monitoring() {
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        log_info "Installing monitoring stack..."
        
        # Install Prometheus Operator
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
        
        helm upgrade --install prometheus-operator prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --create-namespace \
            --set prometheus.prometheusSpec.retention=30d \
            --set grafana.adminPassword=admin123 \
            --wait
        
        # Apply custom monitoring configurations
        kubectl apply -f "$DEPLOY_DIR/kubernetes/monitoring/"
        
        log_success "Monitoring stack installed"
    else
        log_warning "Monitoring disabled"
    fi
}

# Install autoscaling components
setup_autoscaling() {
    if [[ "$ENABLE_AUTOSCALING" == "true" ]]; then
        log_info "Installing autoscaling components..."
        
        # Install Metrics Server
        kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
        
        log_success "Autoscaling components installed"
    else
        log_warning "Autoscaling disabled"
    fi
}

# Deploy the main application
deploy_application() {
    log_info "Deploying CrewGraph AI application..."
    
    # Create secrets
    create_secrets
    
    # Update image tag in deployment
    sed -i.bak "s|image: crewgraph-ai:.*|image: ghcr.io/vatsal216/crewgraph-ai:$IMAGE_TAG|g" "$DEPLOY_DIR/kubernetes/deployment.yaml"
    
    # Apply main deployment
    kubectl apply -f "$DEPLOY_DIR/kubernetes/deployment.yaml"
    
    # Apply autoscaling if enabled
    if [[ "$ENABLE_AUTOSCALING" == "true" ]]; then
        kubectl apply -f "$DEPLOY_DIR/kubernetes/autoscaling/"
    fi
    
    # Apply networking
    kubectl apply -f "$DEPLOY_DIR/kubernetes/networking/"
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/crewgraph-ai -n "$NAMESPACE" --timeout=600s
    
    log_success "Application deployed successfully"
}

# Create necessary secrets
create_secrets() {
    log_info "Creating secrets..."
    
    # Generate random passwords if not provided
    REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}"
    ENCRYPTION_KEY="${ENCRYPTION_KEY:-$(openssl rand -base64 32)}"
    
    # Create application secret
    kubectl create secret generic crewgraph-secret \
        --from-literal=encryption-key="$ENCRYPTION_KEY" \
        --from-literal=jwt-secret="$(openssl rand -base64 32)" \
        --namespace "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Secrets created"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=crewgraph-ai -n "$NAMESPACE" --timeout=300s
    
    log_success "Smoke tests completed successfully"
}

# Get deployment information
get_deployment_info() {
    log_info "Deployment Information:"
    echo "========================"
    
    # Get service information
    kubectl get svc -n "$NAMESPACE"
    echo ""
    
    # Get pod information
    kubectl get pods -n "$NAMESPACE"
    echo ""
    
    log_success "Application deployed successfully!"
}

# Main deployment function
main() {
    log_info "Starting CrewGraph AI deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Image Tag: $IMAGE_TAG"
    log_info "Region: $AWS_REGION"
    log_info "Cluster: $EKS_CLUSTER_NAME"
    
    check_prerequisites
    setup_kubeconfig
    setup_namespaces
    setup_monitoring
    setup_autoscaling
    deploy_application
    run_smoke_tests
    get_deployment_info
    
    log_success "🎉 CrewGraph AI deployment completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --cluster-name)
            EKS_CLUSTER_NAME="$2"
            shift 2
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --enable-monitoring)
            ENABLE_MONITORING="true"
            shift
            ;;
        --disable-monitoring)
            ENABLE_MONITORING="false"
            shift
            ;;
        --enable-autoscaling)
            ENABLE_AUTOSCALING="true"
            shift
            ;;
        --disable-autoscaling)
            ENABLE_AUTOSCALING="false"
            shift
            ;;
        --help)
            cat << EOF
Usage: $0 [OPTIONS]

Deploy CrewGraph AI to Kubernetes

OPTIONS:
    --namespace NAMESPACE       Kubernetes namespace (default: crewgraph-ai)
    --environment ENV           Environment name (default: production)
    --image-tag TAG            Docker image tag (default: latest)
    --cluster-name NAME        EKS cluster name (default: crewgraph-ai-production-cluster)
    --region REGION            AWS region (default: us-west-2)
    --enable-monitoring        Enable monitoring stack (default)
    --disable-monitoring       Disable monitoring stack
    --enable-autoscaling       Enable autoscaling (default)
    --disable-autoscaling      Disable autoscaling
    --help                     Show this help message

Examples:
    $0                                          # Deploy with defaults
    $0 --namespace dev --image-tag v1.2.3      # Deploy to dev namespace with specific tag

EOF
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"