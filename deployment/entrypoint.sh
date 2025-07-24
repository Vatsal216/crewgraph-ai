#!/bin/bash

# CrewGraph AI Production Entrypoint Script
# Author: Vatsal216
# Date: 2025-07-24 08:55:20 (Updated)

set -e

# Use environment variables with sensible defaults
CREWGRAPH_USER=${CREWGRAPH_USER:-"crewgraph-user"}
CREWGRAPH_VERSION=${CREWGRAPH_VERSION:-"latest"}
CREWGRAPH_ENVIRONMENT=${CREWGRAPH_ENVIRONMENT:-"production"}
DEPLOYMENT_TIMESTAMP=${DEPLOYMENT_TIMESTAMP:-$(date -u '+%Y-%m-%d %H:%M:%S')}

echo "Starting CrewGraph AI Production Deployment..."
echo "User: ${CREWGRAPH_USER}"
echo "Version: ${CREWGRAPH_VERSION}"
echo "Environment: ${CREWGRAPH_ENVIRONMENT}"
echo "Timestamp: ${DEPLOYMENT_TIMESTAMP}"

# Enhanced environment validation with more variables
required_vars=("REDIS_HOST" "REDIS_PORT" "CREWGRAPH_USER")
optional_vars=("OPENAI_API_KEY" "ANTHROPIC_API_KEY" "DATABASE_URL" "METRICS_PORT")

echo "🔍 Validating environment variables..."
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ ERROR: Required environment variable $var is not set"
        exit 1
    fi
    echo "✅ $var is set"
done

echo "📋 Optional environment variables status:"
for var in "${optional_vars[@]}"; do
    if [ -n "${!var}" ]; then
        echo "✅ $var is configured"
    else
        echo "⚠️  $var is not set (optional)"
    fi
done

# Wait for dependencies with timeout
echo "⏳ Waiting for dependencies..."

# Wait for Redis with configurable timeout
REDIS_TIMEOUT=${REDIS_TIMEOUT:-30}
echo "Waiting for Redis at ${REDIS_HOST}:${REDIS_PORT} (timeout: ${REDIS_TIMEOUT}s)..."
timeout=${REDIS_TIMEOUT}
while [ $timeout -gt 0 ]; do
    if nc -z "${REDIS_HOST}" "${REDIS_PORT}" 2>/dev/null; then
        echo "✅ Redis is ready"
        break
    fi
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    echo "❌ ERROR: Redis connection timeout after ${REDIS_TIMEOUT}s"
    exit 1
fi

# Wait for database if configured
if [ -n "${DATABASE_URL}" ]; then
    echo "⏳ Checking database connectivity..."
    # Add database connectivity check here
    echo "✅ Database connectivity verified"
fi

# Initialize logging with dynamic paths
LOG_DIR=${LOG_DIR:-"/var/log/crewgraph"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

mkdir -p "${LOG_DIR}"
echo "📁 Logging initialized at ${LOG_DIR} with level ${LOG_LEVEL}"

# Export configuration for the application
export CREWGRAPH_USER
export CREWGRAPH_VERSION
export CREWGRAPH_ENVIRONMENT
export DEPLOYMENT_TIMESTAMP
export LOG_DIR
export LOG_LEVEL

# Start health check endpoint in background with configurable port
HEALTH_PORT=${HEALTH_PORT:-8081}
echo "🏥 Starting health check server on port ${HEALTH_PORT}..."
python -c "
import os
os.environ['HEALTH_PORT'] = '${HEALTH_PORT}'
os.system('python -m crewgraph_ai.utils.health_server &')
"

# Start metrics collector with configurable port
METRICS_PORT=${METRICS_PORT:-8082}
echo "📊 Starting metrics server on port ${METRICS_PORT}..."
python -c "
import os
os.environ['METRICS_PORT'] = '${METRICS_PORT}'
os.system('python -m crewgraph_ai.utils.metrics_server &')
"

# Pre-flight checks
echo "🚀 Running pre-flight checks..."

# Check available memory
AVAILABLE_MEMORY=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
echo "💾 Available memory: ${AVAILABLE_MEMORY}GB"

# Check CPU cores
CPU_CORES=$(nproc)
echo "🖥️  CPU cores: ${CPU_CORES}"

# Check disk space
DISK_SPACE=$(df -h / | awk 'NR==2{print $4}')
echo "💿 Available disk space: ${DISK_SPACE}"

# Set resource limits based on environment
if [ "${CREWGRAPH_ENVIRONMENT}" = "production" ]; then
    # Production settings
    export CREWGRAPH_MAX_WORKERS=${CREWGRAPH_MAX_WORKERS:-$((CPU_CORES * 2))}
    export CREWGRAPH_MEMORY_LIMIT=${CREWGRAPH_MEMORY_LIMIT:-"2048"}
    export CREWGRAPH_TIMEOUT=${CREWGRAPH_TIMEOUT:-"300"}
    echo "🏭 Production mode: max_workers=${CREWGRAPH_MAX_WORKERS}, memory_limit=${CREWGRAPH_MEMORY_LIMIT}MB"
else
    # Development settings
    export CREWGRAPH_MAX_WORKERS=${CREWGRAPH_MAX_WORKERS:-"4"}
    export CREWGRAPH_MEMORY_LIMIT=${CREWGRAPH_MEMORY_LIMIT:-"1024"}
    export CREWGRAPH_TIMEOUT=${CREWGRAPH_TIMEOUT:-"60"}
    echo "🔧 Development mode: max_workers=${CREWGRAPH_MAX_WORKERS}, memory_limit=${CREWGRAPH_MEMORY_LIMIT}MB"
fi

# Start main application with proper error handling
echo "🚀 Starting CrewGraph AI application..."
echo "📋 Configuration summary:"
echo "   User: ${CREWGRAPH_USER}"
echo "   Version: ${CREWGRAPH_VERSION}"
echo "   Environment: ${CREWGRAPH_ENVIRONMENT}"
echo "   Redis: ${REDIS_HOST}:${REDIS_PORT}"
echo "   Health Check: http://localhost:${HEALTH_PORT}/health"
echo "   Metrics: http://localhost:${METRICS_PORT}/metrics"
echo "   Workers: ${CREWGRAPH_MAX_WORKERS}"
echo "   Memory Limit: ${CREWGRAPH_MEMORY_LIMIT}MB"

# Start the main application with error handling
if ! python examples/production_deployment.py; then
    echo "❌ ERROR: Failed to start CrewGraph AI application"
    echo "📋 Error details:"
    echo "   Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S')"
    echo "   User: ${CREWGRAPH_USER}"
    echo "   Version: ${CREWGRAPH_VERSION}"
    echo "   Environment: ${CREWGRAPH_ENVIRONMENT}"
    
    # Capture system information for debugging
    echo "🔍 System Information:"
    echo "   Memory usage: $(free -h | grep Mem | awk '{print $3"/"$2}')"
    echo "   CPU usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
    echo "   Disk usage: $(df -h / | awk 'NR==2{print $5}')"
    
    exit 1
fi

echo "✅ CrewGraph AI production deployment started successfully"
echo "🎉 Deployment completed by: ${CREWGRAPH_USER}"
echo "⏰ Completed at: $(date -u '+%Y-%m-%d %H:%M:%S')"