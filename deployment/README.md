# CrewGraph AI - Production Deployment Guide

This guide provides comprehensive instructions for deploying CrewGraph AI in production environments.

## Quick Deployment

### Docker Compose (Recommended for Development/Testing)

```bash
# Clone repository
git clone https://github.com/Vatsal216/crewgraph-ai.git
cd crewgraph-ai

# Deploy with Docker Compose
./deployment/scripts/deploy.sh docker

# Services will be available at:
# - API: http://localhost:8000
# - Metrics: http://localhost:8080
# - Grafana: http://localhost:3000