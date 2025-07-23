# CrewGraph AI + Langflow Integration

![CrewGraph AI + Langflow](https://img.shields.io/badge/CrewGraph%20AI-Langflow%20Integration-blue)
![Version](https://img.shields.io/badge/version-1.0.0-green)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Enterprise-grade integration between CrewGraph AI and Langflow visual workflow builder, providing seamless visual design capabilities with production-ready AI orchestration.

## 🚀 Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install fastapi uvicorn pydantic pytest pytest-asyncio crewai langgraph langchain langchain-core structlog psutil asyncio-mqtt
```

### Basic Usage

1. **Start the Integration API Server**
```bash
# Navigate to the project root
cd /path/to/crewgraph-ai

# Start the Langflow integration API
python -m uvicorn crewgraph_ai.integrations.langflow.api.main:app --host 0.0.0.0 --port 8000 --reload

# API will be available at:
# - API endpoints: http://localhost:8000/api/v1/
# - Documentation: http://localhost:8000/api/v1/docs
# - Health check: http://localhost:8000/api/v1/health
```

2. **Run the Basic Example**
```bash
# Test the integration with a comprehensive example
python examples/langflow_integration/basic_example.py
```

3. **Test the Integration**
```bash
# Run the integration test suite
python test_langflow_integration.py
```

## 🏗️ Architecture

The integration provides a complete bridge between CrewGraph AI and Langflow:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Langflow UI   │◄──►│  Integration API │◄──►│  CrewGraph AI   │
│   (Visual)      │    │   (FastAPI)      │    │   (Engine)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              ┌─────▼─────┐    ┌─────────▼────┐
              │  Security │    │  Components  │
              │   & Auth  │    │  & Workflow  │
              └───────────┘    └──────────────┘
```

## 🔧 Features

### ✅ Implemented (Phase 1)

- **Enterprise Security**
  - JWT authentication with configurable expiration
  - Role-based access control (RBAC) with granular permissions
  - Comprehensive audit logging for compliance
  - Session management with validation

- **Visual Workflow Design**
  - Drag-and-drop interface through Langflow
  - Custom CrewGraph components (Agents, Tasks, Tools)
  - Real-time workflow validation
  - Visual execution monitoring

- **Bidirectional Synchronization**
  - Export CrewGraph workflows to Langflow format
  - Import Langflow workflows to CrewGraph
  - Automatic format conversion with validation
  - Workflow compression and metadata preservation

- **Production Ready**
  - Docker deployment with multi-service orchestration
  - Health checks and monitoring integration
  - Configurable CORS, timeouts, and scaling parameters
  - Redis caching and session storage

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/auth/login` | User authentication |
| `GET` | `/api/v1/health` | Health check with system status |
| `POST` | `/api/v1/workflows/export/{workflow_id}` | Export CrewGraph workflow to Langflow |
| `POST` | `/api/v1/workflows/import` | Import Langflow workflow to CrewGraph |
| `POST` | `/api/v1/workflows/execute` | Execute workflow with real-time sync |
| `GET` | `/api/v1/workflows/{workflow_id}/status` | Get execution status with progress |
| `POST` | `/api/v1/components/register` | Register custom Langflow components |
| `GET` | `/api/v1/components/list` | List available components |

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Clone the repository
git clone https://github.com/Vatsal216/crewgraph-ai.git
cd crewgraph-ai

# Set environment variables
export LANGFLOW_JWT_SECRET="your-super-secure-jwt-secret-key"
export POSTGRES_PASSWORD="your-secure-db-password"

# Start all services
docker-compose -f deployment/langflow/docker-compose.yml up -d

# Services will be available at:
# - CrewGraph Langflow API: http://localhost:8000
# - Langflow UI: http://localhost:7860
# - Redis: localhost:6379
```

### Production Deployment

```bash
# Start with production profile (includes PostgreSQL, Nginx, monitoring)
docker-compose -f deployment/langflow/docker-compose.yml --profile production up -d

# With monitoring (includes Prometheus & Grafana)
docker-compose -f deployment/langflow/docker-compose.yml --profile monitoring up -d
```

## ⚙️ Configuration

### Environment Variables

```bash
# API Configuration
export LANGFLOW_API_HOST=0.0.0.0
export LANGFLOW_API_PORT=8000
export LANGFLOW_ENABLE_CORS=true

# Authentication & Security
export LANGFLOW_JWT_SECRET=your-super-secure-jwt-secret-key
export LANGFLOW_JWT_EXPIRE_HOURS=24
export LANGFLOW_ENABLE_RBAC=true

# Langflow Configuration
export LANGFLOW_URL=http://localhost:7860
export LANGFLOW_AUTO_SYNC=true
export LANGFLOW_SYNC_INTERVAL=60

# Performance & Scaling
export LANGFLOW_MAX_CONCURRENT=10
export LANGFLOW_ASYNC_EXECUTION=true
export LANGFLOW_EXECUTION_TIMEOUT=3600

# Monitoring & Logging
export LANGFLOW_ENABLE_METRICS=true
export LANGFLOW_LOG_LEVEL=INFO
export LANGFLOW_AUDIT_LOGGING=true

# Optional: Database & Cache
export DATABASE_URL=sqlite:///app/data/workflows.db
export REDIS_URL=redis://localhost:6379
```

### Configuration Validation

```python
from crewgraph_ai.integrations.langflow.config import validate_config

# Validate your configuration
if validate_config():
    print("✅ Configuration is valid and ready!")
else:
    print("❌ Configuration issues found")
```

## 🔗 Langflow Components

### CrewGraph Agent Component

Create and configure CrewGraph agents visually:

```python
# Available in Langflow as "CrewGraphAgent"
inputs = {
    "role": "Senior Research Analyst",
    "goal": "Conduct comprehensive research on any topic",
    "backstory": "Expert with years of research experience",
    "llm_model": "gpt-4",
    "max_iter": 5,
    "verbose": True
}
```

### CrewGraph Tool Component

Access CrewGraph tools in workflows:

```python
# Available in Langflow as "CrewGraphTool"
inputs = {
    "tool_name": "web_scraper",
    "tool_input": "search query or URL",
    "tool_config": {"max_results": 10},
    "cache_results": True,
    "retry_on_failure": True
}
```

### CrewGraph Task Component

Define and execute tasks:

```python
# Available in Langflow as "CrewGraphTask"
inputs = {
    "description": "Analyze the research findings",
    "expected_output": "Detailed analysis report",
    "agent": agent_component_output,
    "tools": ["web_scraper", "text_processor"]
}
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Basic integration test
python test_langflow_integration.py

# Run with pytest (if available)
pytest test_langflow_integration.py -v

# Test specific components
python -c "
import asyncio
from crewgraph_ai.integrations.langflow.components import CrewGraphAgentComponent

async def test():
    agent = CrewGraphAgentComponent()
    result = await agent.run(
        role='Test Agent',
        goal='Test goal',
        backstory='Test backstory'
    )
    print(f'Agent created: {result[\"status\"]}')

asyncio.run(test())
"
```

## 📖 Usage Examples

### Export CrewGraph Workflow to Langflow

```python
from crewgraph_ai.integrations.langflow import WorkflowExporter

# Create exporter
exporter = WorkflowExporter()

# Export workflow
langflow_data = await exporter.export_workflow(
    workflow_id="my_workflow_123",
    include_metadata=True,
    format_version="1.0"
)

# Save or send to Langflow
print(f"Exported {len(langflow_data['flow']['nodes'])} nodes")
```

### Import Langflow Workflow to CrewGraph

```python
from crewgraph_ai.integrations.langflow import WorkflowImporter

# Create importer
importer = WorkflowImporter()

# Import workflow
result = await importer.import_workflow(
    langflow_data=langflow_workflow,
    name="Imported Research Workflow",
    description="Workflow from Langflow",
    auto_fix_issues=True
)

print(f"Imported workflow: {result['workflow_id']}")
```

### API Client Usage

```python
import httpx

async def use_api():
    async with httpx.AsyncClient() as client:
        # Authenticate
        auth = await client.post(
            "http://localhost:8000/api/v1/auth/login",
            auth=("username", "password")
        )
        token = auth.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Export workflow
        response = await client.post(
            "http://localhost:8000/api/v1/workflows/export/my_workflow",
            json={"include_metadata": True},
            headers=headers
        )
        
        print(f"Export status: {response.status_code}")
```

## 🔍 Monitoring & Debugging

### Health Check

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Expected response:
{
  "success": true,
  "message": "Service is healthy",
  "uptime_seconds": 123.45,
  "langflow_connection": true,
  "crewgraph_connection": true,
  "active_executions": 0
}
```

### Logs & Metrics

```bash
# View API logs
docker-compose logs -f crewgraph-langflow-api

# Access metrics (if enabled)
curl http://localhost:9090/metrics

# View Grafana dashboard (if monitoring profile enabled)
open http://localhost:3000
```

## 🛠️ Development

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/Vatsal216/crewgraph-ai.git
cd crewgraph-ai

# Install in development mode
pip install -e .

# Install additional dependencies
pip install fastapi uvicorn pydantic pytest pytest-asyncio

# Start development server with hot reload
uvicorn crewgraph_ai.integrations.langflow.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Adding Custom Components

1. **Create Component Class**:
```python
from crewgraph_ai.integrations.langflow.components.base import LangflowComponent

class MyCustomComponent(LangflowComponent):
    def _get_metadata(self):
        return ComponentMetadata(
            name="MyCustomComponent",
            display_name="My Custom Component",
            description="Does something awesome"
        )
    
    def _get_inputs(self):
        return [
            ComponentInput(
                name="input_text",
                display_name="Input Text",
                input_type="str",
                required=True
            )
        ]
    
    def _get_outputs(self):
        return [
            ComponentOutput(
                name="result",
                display_name="Result",
                output_type="str"
            )
        ]
    
    async def execute(self, inputs):
        # Your component logic here
        return {"result": f"Processed: {inputs['input_text']}"}
```

2. **Register Component**:
```python
# Via API
response = await client.post("/api/v1/components/register", json={
    "component_name": "MyCustomComponent",
    "component_type": "custom",
    "component_code": component_code,
    "display_name": "My Custom Component"
})
```

## 🔮 Roadmap

### Phase 2: Advanced Features (Planned)
- [ ] Real-time bidirectional sync
- [ ] Advanced workflow templates
- [ ] Custom UI components
- [ ] Workflow versioning
- [ ] A/B testing capabilities

### Phase 3: Enterprise Integration (Planned)
- [ ] Advanced monitoring and alerting
- [ ] Multi-tenant support
- [ ] Enterprise SSO integration
- [ ] Advanced security features
- [ ] Workflow marketplace

### Phase 4: Advanced Analytics (Planned)
- [ ] Workflow performance analytics
- [ ] Predictive execution modeling
- [ ] Resource optimization
- [ ] Usage analytics dashboard

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [API Docs](http://localhost:8000/api/v1/docs) (when running)
- **Issues**: [GitHub Issues](https://github.com/Vatsal216/crewgraph-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Vatsal216/crewgraph-ai/discussions)

## 🙏 Acknowledgments

- **CrewAI**: For the powerful agent framework
- **Langflow**: For the visual workflow interface
- **LangGraph**: For workflow orchestration capabilities
- **FastAPI**: For the robust API framework

---

**CrewGraph AI + Langflow Integration** - Making enterprise AI orchestration visual, intuitive, and production-ready! 🚀