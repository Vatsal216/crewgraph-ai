#!/bin/bash

# CrewGraph AI + Langflow Integration Startup Script
# This script starts the integration API server with proper configuration

echo "🚀 Starting CrewGraph AI + Langflow Integration"
echo "=" * 50

# Set default environment variables if not already set
export LANGFLOW_JWT_SECRET=${LANGFLOW_JWT_SECRET:-"crewgraph-langflow-default-secret-change-in-production"}
export LANGFLOW_API_HOST=${LANGFLOW_API_HOST:-"0.0.0.0"}
export LANGFLOW_API_PORT=${LANGFLOW_API_PORT:-"8000"}
export LANGFLOW_LOG_LEVEL=${LANGFLOW_LOG_LEVEL:-"INFO"}

# Create necessary directories
mkdir -p workflows custom_components data logs

echo "✅ Environment configured"
echo "   - API Host: $LANGFLOW_API_HOST"
echo "   - API Port: $LANGFLOW_API_PORT"
echo "   - Log Level: $LANGFLOW_LOG_LEVEL"

# Validate configuration
echo ""
echo "🔍 Validating configuration..."
python -c "
from crewgraph_ai.integrations.langflow.config import validate_config
if validate_config():
    print('✅ Configuration is valid')
else:
    print('❌ Configuration validation failed')
    exit(1)
" || exit 1

# Start the API server
echo ""
echo "🌐 Starting API server..."
echo "   - API endpoints: http://$LANGFLOW_API_HOST:$LANGFLOW_API_PORT/api/v1/"
echo "   - Documentation: http://$LANGFLOW_API_HOST:$LANGFLOW_API_PORT/api/v1/docs"
echo "   - Health check: http://$LANGFLOW_API_HOST:$LANGFLOW_API_PORT/api/v1/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start uvicorn with the integration API
python -m uvicorn crewgraph_ai.integrations.langflow.api.main:app \
    --host "$LANGFLOW_API_HOST" \
    --port "$LANGFLOW_API_PORT" \
    --log-level "$(echo $LANGFLOW_LOG_LEVEL | tr '[:upper:]' '[:lower:]')" \
    --reload