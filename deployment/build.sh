#!/bin/bash

# Build with dynamic timestamp and version
BUILD_TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
CREWGRAPH_VERSION=${CREWGRAPH_VERSION:-"v1.0.0"}

echo "🔨 Building CrewGraph AI Docker image..."
echo "📅 Build timestamp: ${BUILD_TIMESTAMP}"
echo "🏷️  Version: ${CREWGRAPH_VERSION}"
echo "👤 Built by: Vatsal216"

docker build \
    --build-arg BUILD_TIMESTAMP="${BUILD_TIMESTAMP}" \
    --build-arg CREWGRAPH_VERSION="${CREWGRAPH_VERSION}" \
    --build-arg MAINTAINER_EMAIL="crewgraph@vatsal216.dev" \
    -t crewgraph-ai:${CREWGRAPH_VERSION} \
    -t crewgraph-ai:latest \
    .

echo "✅ Docker image built successfully"
echo "🏷️  Tags: crewgraph-ai:${CREWGRAPH_VERSION}, crewgraph-ai:latest"