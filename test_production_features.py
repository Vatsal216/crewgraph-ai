#!/usr/bin/env python3
"""
Test script to verify CrewGraph AI production features implementation.

This script tests the major features implemented:
- Visualization tools
- Type annotations
- Enhanced CrewGraph interface
- Configuration options
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all modules import correctly."""
    print("🔄 Testing imports...")
    
    try:
        # Test core imports
        from crewgraph_ai import CrewGraph, CrewGraphConfig
        from crewgraph_ai.types import StateDict, NodeId, WorkflowStatus
        print("✅ Core imports successful")
        
        # Test visualization imports
        from crewgraph_ai.visualization import WorkflowVisualizer, ExecutionTracer, MemoryInspector, DebugTools
        print("✅ Visualization imports successful")
        
        # Test orchestrator with visualization methods
        from crewgraph_ai.core.orchestrator import GraphOrchestrator
        orchestrator = GraphOrchestrator("test")
        assert hasattr(orchestrator, 'visualize_workflow')
        assert hasattr(orchestrator, 'export_execution_trace')
        assert hasattr(orchestrator, 'dump_memory_state')
        assert hasattr(orchestrator, 'generate_debug_report')
        print("✅ Orchestrator visualization methods available")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_crewgraph_config():
    """Test enhanced CrewGraph configuration."""
    print("🔄 Testing CrewGraph configuration...")
    
    try:
        from crewgraph_ai import CrewGraphConfig
        
        # Test default config
        config = CrewGraphConfig()
        assert config.enable_visualization == True
        assert config.visualization_output_dir == "visualizations"
        assert config.enable_performance_tracking == True
        print("✅ Default configuration correct")
        
        # Test custom config
        custom_config = CrewGraphConfig(
            enable_visualization=True,
            visualization_output_dir="./custom_viz",
            enable_real_time_monitoring=True,
            max_concurrent_tasks=5,
            task_timeout=600.0
        )
        assert custom_config.visualization_output_dir == "./custom_viz"
        assert custom_config.enable_real_time_monitoring == True
        print("✅ Custom configuration works")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_crewgraph_interface():
    """Test enhanced CrewGraph interface."""
    print("🔄 Testing CrewGraph interface...")
    
    try:
        from crewgraph_ai import CrewGraph, CrewGraphConfig
        
        # Create workflow with visualization enabled
        config = CrewGraphConfig(enable_visualization=True)
        workflow = CrewGraph("test_workflow", config)
        
        # Test new methods exist
        assert hasattr(workflow, 'visualize_workflow')
        assert hasattr(workflow, 'start_real_time_monitoring')
        assert hasattr(workflow, 'generate_debug_report')
        assert hasattr(workflow, 'export_execution_trace')
        assert hasattr(workflow, 'analyze_performance')
        assert hasattr(workflow, 'validate_workflow')
        print("✅ CrewGraph visualization methods available")
        
        # Test validation works
        validation_result = workflow.validate_workflow()
        assert 'summary' in validation_result
        print("✅ Workflow validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ CrewGraph interface test failed: {e}")
        return False

def test_type_system():
    """Test type system implementation."""
    print("🔄 Testing type system...")
    
    try:
        from crewgraph_ai.types import (
            StateDict, NodeId, WorkflowStatus, TaskStatus,
            AgentProtocol, MemoryProtocol, ExecutionResult
        )
        
        # Test type aliases work
        state: StateDict = {"key": "value"}
        node_id: NodeId = "test_node"
        status: WorkflowStatus = "pending"
        
        # Test generic types
        result: ExecutionResult[str] = ExecutionResult(True, "success")
        assert result.is_success() == True
        assert result.get_data() == "success"
        print("✅ Type system works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Type system test failed: {e}")
        return False

def test_visualization_creation():
    """Test visualization tool creation."""
    print("🔄 Testing visualization tools...")
    
    try:
        from crewgraph_ai.visualization import WorkflowVisualizer
        
        # Create visualizer
        visualizer = WorkflowVisualizer(output_dir="/tmp/test_viz")
        
        # Test workflow data creation
        workflow_data = {
            'nodes': [
                {'id': 'task1', 'name': 'Data Collection', 'status': 'completed'},
                {'id': 'task2', 'name': 'Analysis', 'status': 'running'}
            ],
            'edges': [
                {'source': 'task1', 'target': 'task2', 'type': 'dependency'}
            ]
        }
        
        # This would normally create a file, but we just test the method exists
        assert hasattr(visualizer, 'visualize_workflow_graph')
        assert hasattr(visualizer, 'create_execution_timeline')
        assert hasattr(visualizer, 'generate_workflow_summary')
        print("✅ Visualization tools created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def run_tests():
    """Run all tests and report results."""
    print("🚀 Starting CrewGraph AI Production Features Test\n")
    
    tests = [
        test_imports,
        test_crewgraph_config,
        test_crewgraph_interface,
        test_type_system,
        test_visualization_creation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}\n")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Production features successfully implemented.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)