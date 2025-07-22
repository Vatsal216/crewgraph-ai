# Multi-stage build for production
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash crewgraph

# Set work directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/crewgraph/.local

# Copy application code
COPY crewgraph_ai/ ./crewgraph_ai/
COPY examples/ ./examples/
COPY deployment/entrypoint.sh ./entrypoint.sh

# Set ownership and permissions
RUN chown -R crewgraph:crewgraph /app
RUN chmod +x entrypoint.sh

# Create log directory
RUN mkdir -p /var/log/crewgraph && chown crewgraph:crewgraph /var/log/crewgraph

# Switch to non-root user
USER crewgraph

# Add local Python packages to PATH
ENV PATH="/home/crewgraph/.local/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8000 8080

# Entry point
ENTRYPOINT ["./entrypoint.sh"]