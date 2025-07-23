# CrewGraph AI Production Docker Configuration
# Author: Vatsal216
# Created: 2025-07-23 06:03:54 UTC

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY setup.py .
COPY README.md .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -e .[full]

# Copy application code
COPY crewgraph_ai/ ./crewgraph_ai/
COPY examples/ ./examples/
COPY tests/ ./tests/

# Create non-root user for security
RUN groupadd -r crewgraph && useradd -r -g crewgraph crewgraph
RUN chown -R crewgraph:crewgraph /app
USER crewgraph

# Set environment variables
ENV PYTHONPATH=/app
ENV CREWGRAPH_ENV=production
ENV CREWGRAPH_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import crewgraph_ai; print('✅ CrewGraph AI healthy')" || exit 1

# Default command
CMD ["python", "-c", "import crewgraph_ai; print('🚀 CrewGraph AI Production Ready!')"]

# Expose default port for web interfaces
EXPOSE 8080

# Labels for metadata
LABEL maintainer="Vatsal216"
LABEL version="1.0.0"
LABEL description="Production-ready AI workflow orchestration with CrewGraph AI"
LABEL created="2025-07-23T06:03:54Z"