#!/bin/bash

# CrewGraph AI Production Deployment Script
# Author: Vatsal216
# Date: 2025-07-22 10:27:20
# Description: Complete production deployment automation

set -e

echo "=========================================="
echo "CrewGraph AI Production Deployment"
echo "Author: Vatsal216"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
IMAGE_TAG=${IMAGE_TAG:-latest}
NAMESPACE=${NAMESPACE:-crewgraph-ai}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-docker.io}
PROJECT_NAME="crewgraph-ai"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check kubectl for Kubernetes deployment
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]] && ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check docker-compose for Docker deployment
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]] && ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd "$(dirname "$0")/../.."
    
    # Build the image
    docker build \
        -t "${PROJECT_NAME}:${IMAGE_TAG}" \
        -t "${PROJECT_NAME}:latest" \
        -f deployment/Dockerfile \
        .
    
    log_success "Docker image built successfully"
}

# Push image to registry
push_image() {
    if [[ "$PUSH_IMAGE" == "true" ]]; then
        log_info "Pushing image to registry..."
        
        # Tag for registry
        docker tag "${PROJECT_NAME}:${IMAGE_TAG}" "${DOCKER_REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}"
        docker tag "${PROJECT_NAME}:latest" "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest"
        
        # Push to registry
        docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}"
        docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest"
        
        log_success "Image pushed to registry"
    fi
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$(dirname "$0")/.."
    
    # Create necessary directories
    mkdir -p logs config/ssl monitoring/grafana/dashboards monitoring/grafana/datasources
    
    # Set environment variables
    export IMAGE_TAG
    export ENVIRONMENT
    export REDIS_PASSWORD=${REDIS_PASSWORD:-$(openssl rand -base64 32)}
    export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}
    export GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-$(openssl rand -base64 16)}
    export ENCRYPTION_KEY=${ENCRYPTION_KEY:-$(openssl rand -base64 32)}
    
    # Deploy services
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    if docker-compose ps | grep -q "unhealthy\|Exit"; then
        log_error "Some services are not healthy"
        docker-compose logs
        exit 1
    fi
    
    log_success "Docker deployment completed successfully"
    log_info "Services are available at:"
    log_info "  - CrewGraph API: http://localhost:8000"
    log_info "  - Metrics: http://localhost:8080"
    log_info "  - Grafana: http://localhost:3000 (admin/${GRAFANA_PASSWORD})"
    log_info "  - Prometheus: http://localhost:9090"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    cd "$(dirname "$0")/.."
    
    # Create namespace
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    kubectl create secret generic crewgraph-secret \
        --from-literal=encryption-key="${ENCRYPTION_KEY:-$(openssl rand -base64 32)}" \
        --namespace="${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret generic redis-secret \
        --from-literal=password="${REDIS_PASSWORD:-$(openssl rand -base64 32)}" \
        --namespace="${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f kubernetes/ --namespace="${NAMESPACE}"
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/crewgraph-ai --namespace="${NAMESPACE}" --timeout=300s
    
    # Get service info
    SERVICE_IP=$(kubectl get service crewgraph-ai-service --namespace="${NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [[ -z "$SERVICE_IP" ]]; then
        SERVICE_IP=$(kubectl get service crewgraph-ai-service --namespace="${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
    fi
    
    log_success "Kubernetes deployment completed successfully"
    log_info "Service is available at: http://${SERVICE_IP}:8000"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Determine health check URL based on deployment type
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        HEALTH_URL="http://localhost:8080/health"
    else
        # For Kubernetes, use port-forward or service IP
        HEALTH_URL="http://localhost:8080/health"
    fi
    
    # Wait for service to be ready
    for i in {1..30}; do
        if curl -f "$HEALTH_URL" &> /dev/null; then
            log_success "Health check passed"
            return 0
        fi
        
        log_info "Waiting for service to be ready... (attempt $i/30)"
        sleep 10
    done
    
    log_error "Health check failed after 5 minutes"
    return 1
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        cd "$(dirname "$0")/.."
        docker-compose down
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true
    fi
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting CrewGraph AI deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment Type: $DEPLOYMENT_TYPE"
    log_info "Image Tag: $IMAGE_TAG"
    
    # Check prerequisites
    check_prerequisites
    
    # Build image
    build_image
    
    # Push image if requested
    push_image
    
    # Deploy based on type
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        deploy_docker
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        deploy_kubernetes
    else
        log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
        log_info "Supported types: docker, kubernetes"
        exit 1
    fi
    
    # Run health checks
    if ! run_health_checks; then
        log_error "Deployment failed health checks"
        exit 1
    fi
    
    log_success "CrewGraph AI deployment completed successfully!"
    log_info "Deployment Summary:"
    log_info "  - Environment: $ENVIRONMENT"
    log_info "  - Type: $DEPLOYMENT_TYPE"
    log_info "  - Image: ${PROJECT_NAME}:${IMAGE_TAG}"
    log_info "  - Namespace: $NAMESPACE"
    log_info "  - Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S')"
}

# Handle script arguments
DEPLOYMENT_TYPE=${1:-docker}
PUSH_IMAGE=${2:-false}

# Handle signals for cleanup
trap cleanup SIGINT SIGTERM

# Run main function
main

# Final message
echo ""
echo "=========================================="
echo "🎉 CrewGraph AI is now running!"
echo "📝 Created by: Vatsal216"
echo "⏰ Deployed at: $(date -u '+%Y-%m-%d %H:%M:%S')"
echo "🔗 GitHub: https://github.com/Vatsal216/crewgraph-ai"
echo "=========================================="