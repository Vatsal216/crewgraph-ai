"""
Simple test for CrewGraph AI + Langflow Integration

This test verifies that the basic integration components work correctly.
"""

import asyncio
import json
from typing import Dict, Any


async def test_basic_integration():
    """Test basic integration functionality"""
    print("🧪 Testing CrewGraph AI + Langflow Integration")
    print("=" * 50)
    
    try:
        # Test 1: Import the integration module
        print("\n1. Testing module imports...")
        from crewgraph_ai.integrations.langflow import (
            WorkflowExporter,
            WorkflowImporter, 
            WorkflowValidator,
            LangflowComponent
        )
        print("✅ Integration modules imported successfully")
        
        # Test 2: Test configuration
        print("\n2. Testing configuration...")
        from crewgraph_ai.integrations.langflow.config import get_config, validate_config
        config = get_config()
        print(f"✅ Configuration loaded: API URL = {config.get_api_url()}")
        
        # Test 3: Test workflow exporter
        print("\n3. Testing workflow exporter...")
        exporter = WorkflowExporter()
        sample_workflow_id = "test_workflow_123"
        
        exported_data = await exporter.export_workflow(
            workflow_id=sample_workflow_id,
            include_metadata=True,
            format_version="1.0"
        )
        
        print(f"✅ Workflow exported successfully")
        print(f"   - Format version: {exported_data.get('format_version')}")
        print(f"   - Node count: {len(exported_data.get('flow', {}).get('nodes', []))}")
        print(f"   - Edge count: {len(exported_data.get('flow', {}).get('edges', []))}")
        
        # Test 4: Test workflow validator
        print("\n4. Testing workflow validator...")
        validator = WorkflowValidator()
        validation_issues = await validator.validate_langflow_data(exported_data)
        
        if not validation_issues:
            print("✅ Exported workflow data is valid")
        else:
            print(f"⚠️  Found {len(validation_issues)} validation issues:")
            for issue in validation_issues[:3]:
                print(f"   - {issue}")
        
        # Test 5: Test workflow importer
        print("\n5. Testing workflow importer...")
        importer = WorkflowImporter()
        
        import_result = await importer.import_workflow(
            langflow_data=exported_data,
            name="Test Imported Workflow",
            description="Test workflow imported from Langflow format",
            auto_fix_issues=True
        )
        
        print(f"✅ Workflow imported successfully")
        print(f"   - New workflow ID: {import_result['workflow_id']}")
        print(f"   - Import status: {import_result['import_status']}")
        print(f"   - Nodes imported: {import_result['statistics']['nodes_imported']}")
        
        # Test 6: Test component creation
        print("\n6. Testing Langflow components...")
        from crewgraph_ai.integrations.langflow.components import (
            CrewGraphAgentComponent,
            CrewGraphToolComponent
        )
        
        # Test agent component
        agent_component = CrewGraphAgentComponent()
        print(f"✅ Agent component created: {agent_component.metadata.name}")
        print(f"   - Inputs: {len(agent_component.inputs)}")
        print(f"   - Outputs: {len(agent_component.outputs)}")
        
        # Test agent execution
        agent_inputs = {
            "role": "Test Agent",
            "goal": "Test goal",
            "backstory": "Test backstory",
            "llm_model": "gpt-3.5-turbo"
        }
        
        agent_result = await agent_component.run(**agent_inputs)
        print(f"   - Execution status: {agent_result.get('status', 'unknown')}")
        
        # Test tool component
        tool_component = CrewGraphToolComponent()
        print(f"✅ Tool component created: {tool_component.metadata.name}")
        
        # Test tool execution
        tool_inputs = {
            "tool_name": "text_processor",
            "tool_input": "Sample text for processing",
            "tool_config": {}
        }
        
        tool_result = await tool_component.run(**tool_inputs)
        print(f"   - Tool execution status: {tool_result.get('status', 'unknown')}")
        
        # Test 7: Test API creation (without starting server)
        print("\n7. Testing API creation...")
        from crewgraph_ai.integrations.langflow.api.main import create_langflow_api
        
        app = create_langflow_api()
        print("✅ FastAPI application created successfully")
        print(f"   - Title: {app.title}")
        print(f"   - Version: {app.version}")
        
        # Summary
        print("\n🎉 All tests passed successfully!")
        print("\nIntegration is ready for use:")
        print("1. Start the API server with: uvicorn main:app --host 0.0.0.0 --port 8000")
        print("2. Use the workflow export/import functionality")
        print("3. Create custom Langflow components")
        print("4. Connect Langflow to the CrewGraph API")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_basic_integration())
    exit(0 if success else 1)