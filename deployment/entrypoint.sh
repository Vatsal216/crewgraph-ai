#!/bin/bash

# CrewGraph AI Production Entrypoint Script
# Author: Vatsal216
# Date: 2025-07-22 10:19:38

set -e

echo "Starting CrewGraph AI Production Deployment..."
echo "User: Vatsal216"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S')"

# Environment validation
required_vars=("REDIS_HOST" "REDIS_PORT")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

# Wait for dependencies
echo "Waiting for Redis..."
while ! curl -f "redis://${REDIS_HOST}:${REDIS_PORT}" > /dev/null 2>&1; do
    sleep 2
done
echo "Redis is ready"

# Initialize logging
mkdir -p /var/log/crewgraph
echo "Logging initialized"

# Start health check endpoint in background
python -m crewgraph_ai.utils.health_server &

# Start metrics collector
python -m crewgraph_ai.utils.metrics_server &

# Start main application
echo "Starting CrewGraph AI application..."
python examples/production_deployment.py

echo "CrewGraph AI production deployment started successfully"