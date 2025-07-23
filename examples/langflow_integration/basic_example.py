"""
CrewGraph AI + Langflow Integration Example

This example demonstrates how to use the Langflow integration to:
1. Start the API server
2. Export a CrewGraph workflow to Langflow
3. Import a Langflow workflow to CrewGraph
4. Execute workflows with the API

Created by: Vatsal216
Date: 2025-07-23
"""

import asyncio
import json
from typing import Dict, Any

# Import the integration components
from crewgraph_ai.integrations.langflow import (
    create_langflow_api,
    WorkflowExporter,
    WorkflowImporter,
    WorkflowValidator
)
from crewgraph_ai.integrations.langflow.config import get_config, validate_config


async def main():
    """Main example function"""
    print("🚀 CrewGraph AI + Langflow Integration Example")
    print("=" * 50)
    
    # 1. Validate configuration
    print("\n1. Validating configuration...")
    if validate_config():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration issues found")
        return
    
    # 2. Create workflow exporter/importer
    print("\n2. Initializing workflow components...")
    exporter = WorkflowExporter()
    importer = WorkflowImporter()
    validator = WorkflowValidator()
    print("✅ Components initialized")
    
    # 3. Create sample CrewGraph workflow for export
    print("\n3. Creating sample CrewGraph workflow...")
    sample_workflow = create_sample_crewgraph_workflow()
    print(f"✅ Sample workflow created: {sample_workflow['name']}")
    
    # 4. Export workflow to Langflow format
    print("\n4. Exporting workflow to Langflow format...")
    try:
        langflow_data = await export_workflow_example(exporter, sample_workflow["id"])
        print("✅ Workflow exported successfully")
        print(f"   - Nodes: {len(langflow_data.get('flow', {}).get('nodes', []))}")
        print(f"   - Edges: {len(langflow_data.get('flow', {}).get('edges', []))}")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return
    
    # 5. Validate Langflow data
    print("\n5. Validating Langflow data...")
    validation_issues = await validator.validate_langflow_data(langflow_data)
    if not validation_issues:
        print("✅ Langflow data is valid")
    else:
        print(f"⚠️  Found {len(validation_issues)} validation issues:")
        for issue in validation_issues[:3]:  # Show first 3 issues
            print(f"   - {issue}")
    
    # 6. Import workflow back to CrewGraph
    print("\n6. Importing workflow back to CrewGraph...")
    try:
        import_result = await import_workflow_example(importer, langflow_data)
        print("✅ Workflow imported successfully")
        print(f"   - New workflow ID: {import_result['workflow_id']}")
        print(f"   - Components created: {len(import_result['components_created'])}")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return
    
    # 7. Create and start API server (in background)
    print("\n7. Creating API server...")
    app = create_langflow_api()
    print("✅ API server created")
    print("   - To start: uvicorn main:app --host 0.0.0.0 --port 8000")
    print("   - API docs: http://localhost:8000/api/v1/docs")
    
    # 8. Show component examples
    print("\n8. Component examples...")
    await show_component_examples()
    
    print("\n🎉 Example completed successfully!")
    print("\nNext steps:")
    print("1. Start the API server: uvicorn main:app --host 0.0.0.0 --port 8000")
    print("2. Open Langflow and connect to the API")
    print("3. Import/export workflows visually")
    print("4. Execute workflows through the API")


def create_sample_crewgraph_workflow() -> Dict[str, Any]:
    """Create a sample CrewGraph workflow for demonstration"""
    return {
        "id": "sample_research_workflow",
        "name": "Research Analysis Workflow",
        "description": "A sample workflow that demonstrates research and analysis capabilities",
        "version": "1.0.0",
        "created_at": "2025-07-23T20:00:00Z",
        "nodes": [
            {
                "id": "research_agent",
                "type": "agent",
                "name": "Research Agent",
                "position": {"x": 100, "y": 100},
                "data": {
                    "role": "Senior Research Analyst",
                    "goal": "Conduct comprehensive research on any given topic",
                    "backstory": "You are an experienced research analyst with expertise in gathering, analyzing, and synthesizing information from multiple sources.",
                    "llm_model": "gpt-4",
                    "max_iter": 5,
                    "verbose": True
                }
            },
            {
                "id": "analysis_agent",
                "type": "agent", 
                "name": "Analysis Agent",
                "position": {"x": 100, "y": 300},
                "data": {
                    "role": "Data Analysis Expert",
                    "goal": "Analyze research findings and provide insights",
                    "backstory": "You are a data analysis expert who specializes in finding patterns and insights in research data.",
                    "llm_model": "gpt-4",
                    "max_iter": 3,
                    "verbose": True
                }
            },
            {
                "id": "research_task",
                "type": "task",
                "name": "Research Task",
                "position": {"x": 300, "y": 100},
                "data": {
                    "description": "Research the given topic thoroughly and gather relevant information",
                    "expected_output": "A comprehensive research report with key findings and sources",
                    "agent_id": "research_agent"
                }
            },
            {
                "id": "analysis_task",
                "type": "task",
                "name": "Analysis Task", 
                "position": {"x": 300, "y": 300},
                "data": {
                    "description": "Analyze the research findings and provide actionable insights",
                    "expected_output": "An analysis report with key insights and recommendations",
                    "agent_id": "analysis_agent"
                }
            },
            {
                "id": "web_search_tool",
                "type": "tool",
                "name": "Web Search Tool",
                "position": {"x": 500, "y": 100},
                "data": {
                    "tool_name": "web_scraper",
                    "configuration": {
                        "max_results": 10,
                        "search_depth": "comprehensive"
                    }
                }
            }
        ],
        "edges": [
            {
                "id": "agent_to_task_1",
                "source": "research_agent",
                "target": "research_task",
                "type": "agent_assignment"
            },
            {
                "id": "agent_to_task_2", 
                "source": "analysis_agent",
                "target": "analysis_task",
                "type": "agent_assignment"
            },
            {
                "id": "task_flow",
                "source": "research_task",
                "target": "analysis_task",
                "type": "data_flow"
            },
            {
                "id": "tool_connection",
                "source": "web_search_tool",
                "target": "research_agent",
                "type": "tool_assignment"
            }
        ]
    }


async def export_workflow_example(exporter: WorkflowExporter, workflow_id: str) -> Dict[str, Any]:
    """Example of exporting a workflow"""
    return await exporter.export_workflow(
        workflow_id=workflow_id,
        include_metadata=True,
        format_version="1.0",
        compression=False
    )


async def import_workflow_example(importer: WorkflowImporter, langflow_data: Dict[str, Any]) -> Dict[str, Any]:
    """Example of importing a workflow"""
    return await importer.import_workflow(
        langflow_data=langflow_data,
        name="Imported Research Workflow",
        description="Workflow imported from Langflow format",
        auto_fix_issues=True
    )


async def show_component_examples():
    """Show examples of using Langflow components"""
    from crewgraph_ai.integrations.langflow.components import (
        CrewGraphAgentComponent,
        CrewGraphToolComponent
    )
    
    print("Component examples:")
    
    # Agent component example
    agent_component = CrewGraphAgentComponent()
    print(f"   - Agent component: {agent_component.metadata.name}")
    print(f"     Inputs: {[inp.name for inp in agent_component.inputs]}")
    print(f"     Outputs: {[out.name for out in agent_component.outputs]}")
    
    # Tool component example
    tool_component = CrewGraphToolComponent()
    print(f"   - Tool component: {tool_component.metadata.name}")
    print(f"     Inputs: {[inp.name for inp in tool_component.inputs]}")
    print(f"     Outputs: {[out.name for out in tool_component.outputs]}")


async def api_client_example():
    """Example of using the API client"""
    import httpx
    
    config = get_config()
    base_url = config.get_api_url()
    
    print(f"\nAPI Client Example (using {base_url}):")
    
    try:
        async with httpx.AsyncClient() as client:
            # Health check
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API is healthy (uptime: {health_data.get('uptime_seconds', 0):.1f}s)")
            else:
                print(f"❌ API health check failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ API client example failed: {e}")
        print("   Make sure to start the API server first!")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())