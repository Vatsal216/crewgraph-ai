# CrewGraph AI Enterprise Deployment Guide

This guide walks through deploying CrewGraph AI with the new enterprise features.

## Prerequisites

### Required Dependencies
```bash
pip install crewai>=0.28.0
pip install langgraph>=0.0.40
pip install langchain>=0.1.0
pip install pydantic>=2.0.0
pip install structlog>=23.1.0
pip install psutil>=5.9.0
```

### Optional Dependencies (for enhanced features)
```bash
# Redis support
pip install redis>=4.0.0 hiredis>=2.0.0

# FAISS support  
pip install faiss-cpu>=1.7.0 sentence-transformers>=2.0.0

# Full feature set
pip install crewgraph-ai[full]
```

## Configuration

### 1. Environment Variables

Create a `.env` file:
```bash
# Environment
CREWGRAPH_ENVIRONMENT=production

# LLM Providers
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=your-azure-endpoint

# Memory Backend
CREWGRAPH_MEMORY_BACKEND=redis
REDIS_URL=redis://localhost:6379

# Scaling
MAX_CONCURRENT_WORKFLOWS=50
MAX_CONCURRENT_AGENTS=200
AUTO_SCALING_ENABLED=true

# Security
CREWGRAPH_ENCRYPTION=true
CREWGRAPH_ENCRYPTION_KEY=your-32-char-encryption-key
JWT_SECRET=your-jwt-secret

# Monitoring
METRICS_ENABLED=true
LOG_LEVEL=INFO
```

### 2. Configuration File (Optional)

Create `config.yaml`:
```yaml
environment: production
workflow_timeout: 300.0
task_timeout: 60.0

llm_providers:
  openai:
    provider: openai
    api_key: ${OPENAI_API_KEY}
    models: ["gpt-4", "gpt-3.5-turbo"]
    default_model: "gpt-3.5-turbo"
    rate_limit_rpm: 60
    rate_limit_tpm: 1000
    enabled: true

  anthropic:
    provider: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    models: ["claude-3-sonnet", "claude-3-haiku"]
    default_model: "claude-3-sonnet"
    rate_limit_rpm: 50
    enabled: true

security:
  encryption_enabled: true
  encryption_key: ${CREWGRAPH_ENCRYPTION_KEY}
  audit_logging: true
  rate_limiting: true

scaling:
  max_concurrent_workflows: 50
  max_concurrent_agents: 200
  auto_scaling_enabled: true
  auto_scaling_target_cpu: 70.0
  auto_scaling_min_instances: 2
  auto_scaling_max_instances: 20

monitoring:
  metrics_enabled: true
  logging_level: "INFO"
  structured_logging: true
  performance_monitoring: true
```

## Basic Usage

### 1. Initialize Enterprise Configuration

```python
from crewgraph_ai.config import get_enterprise_config, configure_enterprise

# Load from environment
config = get_enterprise_config()

# Or load from file
# config = EnterpriseConfig.from_file("config.yaml")

# Configure the system
configure_enterprise(config)
```

### 2. Interactive Flow Selection

```python
from crewgraph_ai.agents import get_flow_selector, UserRequirements, IndustryDomain

# Create requirements
requirements = UserRequirements(
    use_case="Customer support automation",
    industry=IndustryDomain.TECHNOLOGY,
    team_size=10,
    expected_volume=5000,
    budget_tier="medium",
    ai_experience_level="intermediate"
)

# Get recommendation
selector = get_flow_selector()
recommendation = await selector.analyze_requirements(requirements)

print(f"Recommended approach: {recommendation.workflow_type.value}")
print(f"Confidence: {recommendation.confidence_score:.2f}")
print(f"Setup time: {recommendation.estimated_setup_time}")
```

### 3. Auto-Scaling Setup

```python
from crewgraph_ai.scaling import get_auto_scaler, start_auto_scaling

# Start auto-scaling
await start_auto_scaling()

# Add custom metrics callback
scaler = get_auto_scaler()
scaler.add_metrics_callback(lambda: {
    "workflow_queue_size": get_queue_size(),
    "active_workflows": get_active_count()
})

# Monitor scaling
stats = scaler.get_scaling_stats()
print(f"Current instances: {stats['current_state']['current_instances']}")
```

### 4. LLM Provider Management

```python
from crewgraph_ai.providers import get_provider_manager, chat_completion

# Use provider manager
manager = get_provider_manager()

# Make requests with automatic failover
response = await chat_completion([
    {"role": "user", "content": "Hello, world!"}
], provider="openai")

print(response.response_data)
```

### 5. Distributed Memory Backend

```python
from crewgraph_ai.memory import create_distributed_memory

# Configure distributed memory
config = {
    "backends": [
        {"id": "redis1", "type": "redis", "priority": 1, "url": "redis://localhost:6379/0"},
        {"id": "redis2", "type": "redis", "priority": 2, "url": "redis://localhost:6379/1"},
        {"id": "fallback", "type": "dict", "priority": 3}
    ],
    "replication_factor": 2,
    "consistency_level": "eventual"
}

memory = create_distributed_memory(config)

# Use distributed memory
await memory.store("key", "value")
value = await memory.retrieve("key")
```

## Production Deployment

### 1. Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Install CrewGraph AI
RUN pip install -e .

# Start application
CMD ["python", "main.py"]
```

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  crewgraph:
    build: .
    environment:
      - CREWGRAPH_ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    ports:
      - "8080:8080"
    volumes:
      - ./config.yaml:/app/config.yaml
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

### 2. Kubernetes Deployment

Create `deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crewgraph-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crewgraph-ai
  template:
    metadata:
      labels:
        app: crewgraph-ai
    spec:
      containers:
      - name: crewgraph-ai
        image: crewgraph-ai:latest
        ports:
        - containerPort: 8080
        env:
        - name: CREWGRAPH_ENVIRONMENT
          value: "production"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: crewgraph-service
spec:
  selector:
    app: crewgraph-ai
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

### 3. Monitoring Setup

Create monitoring dashboard:
```python
from crewgraph_ai.scaling import get_scaling_status
from crewgraph_ai.providers import get_provider_manager

async def health_check():
    """Health check endpoint"""
    try:
        # Check auto-scaling
        scaling_status = get_scaling_status()
        
        # Check LLM providers
        provider_manager = get_provider_manager()
        provider_health = provider_manager.get_health_status()
        
        # Check memory backends
        # ... additional checks
        
        return {
            "status": "healthy",
            "scaling": scaling_status,
            "providers": provider_health,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
```

## Best Practices

### 1. Security
- Store API keys in environment variables or secret management systems
- Enable encryption for sensitive data
- Use JWT authentication for API access
- Enable audit logging for compliance
- Regularly rotate API keys and secrets

### 2. Performance
- Configure appropriate rate limits for LLM providers
- Use Redis for high-performance memory backend
- Enable auto-scaling for variable workloads
- Monitor resource utilization and adjust limits
- Implement connection pooling for database connections

### 3. Reliability
- Configure multiple LLM providers for failover
- Set up distributed memory with replication
- Implement health checks for all components
- Use circuit breaker patterns for external services
- Set appropriate timeouts and retry policies

### 4. Monitoring
- Enable comprehensive logging
- Set up metrics collection and alerting
- Monitor auto-scaling events and efficiency
- Track LLM usage and costs
- Implement distributed tracing for complex workflows

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install crewai langchain
   ```

2. **Configuration Validation Failures**
   ```python
   # Check validation issues
   config = get_enterprise_config()
   issues = config.validate()
   for issue in issues:
       print(f"Issue: {issue}")
   ```

3. **Auto-scaling Not Working**
   ```python
   # Check auto-scaling status
   scaler = get_auto_scaler()
   stats = scaler.get_scaling_stats()
   print(f"Monitoring enabled: {stats['current_state']['monitoring_enabled']}")
   ```

4. **LLM Provider Failures**
   ```python
   # Check provider health
   manager = get_provider_manager()
   health = manager.get_health_status()
   print(f"Healthy providers: {health['healthy_count']}")
   ```

### Performance Tuning

1. **Memory Backend Optimization**
   - Use Redis for high-throughput workloads
   - Configure connection pooling
   - Set appropriate TTL values
   - Monitor memory usage and eviction policies

2. **Auto-scaling Tuning**
   - Adjust scaling thresholds based on workload patterns
   - Configure appropriate cooldown periods
   - Monitor scaling efficiency and costs
   - Use custom metrics for domain-specific scaling

3. **LLM Provider Optimization**
   - Balance rate limits across providers
   - Implement intelligent load balancing
   - Cache responses when appropriate
   - Monitor costs and optimize usage patterns

## Support

For additional support:
- Check the implementation summary: `IMPLEMENTATION_SUMMARY.md`
- Review the source code for detailed documentation
- Monitor logs for debugging information
- Use health check endpoints for system status