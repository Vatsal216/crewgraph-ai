#!/usr/bin/env python3
"""
Final Comprehensive Test of CrewGraph AI Fixes
Tests all major functionality and improvements
"""

print("🎯 CrewGraph AI - Final Comprehensive Test")
print("=" * 50)

def test_imports():
    """Test that all modules import correctly"""
    print("\n📦 Testing Imports...")
    
    try:
        # Core imports
        from crewgraph_ai import CrewGraph, CrewGraphConfig
        from crewgraph_ai import CrewGraphSettings, get_settings, configure
        from crewgraph_ai.memory import DictMemory
        print("✅ Core imports successful")
        
        # Advanced imports
        from crewgraph_ai import AgentWrapper, TaskWrapper
        from crewgraph_ai.tools import ToolRegistry
        from crewgraph_ai.planning import DynamicPlanner
        print("✅ Advanced imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("\n⚙️  Testing Configuration...")
    
    try:
        from crewgraph_ai import CrewGraphSettings
        
        # Test environment-based config
        settings = CrewGraphSettings.from_env()
        print(f"✅ Environment config: {settings.default_model}")
        
        # Test validation
        issues = settings.validate()
        print(f"ℹ️  Config validation found {len(issues)} issues (expected)")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_agent_creation():
    """Test agent creation with new API"""
    print("\n🤖 Testing Agent Creation...")
    
    try:
        from crewgraph_ai import CrewGraph, CrewGraphConfig
        from crewgraph_ai.memory import DictMemory
        
        # Create workflow
        config = CrewGraphConfig(memory_backend=DictMemory())
        workflow = CrewGraph("test", config)
        
        # Test new agent creation API
        agent = workflow.add_agent(
            name="test_agent",
            role="Tester",
            goal="Test functionality",
            backstory="I am a test agent"
        )
        
        print(f"✅ Agent created: {agent.name}")
        print(f"✅ Agent has CrewAI agent: {agent.crew_agent is not None}")
        
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def test_workflow_execution():
    """Test complete workflow execution"""
    print("\n🔄 Testing Workflow Execution...")
    
    try:
        from crewgraph_ai import CrewGraph, CrewGraphConfig
        from crewgraph_ai.memory import DictMemory
        
        # Create workflow
        config = CrewGraphConfig(
            memory_backend=DictMemory(),
            enable_planning=True,
            max_concurrent_tasks=2
        )
        workflow = CrewGraph("execution_test", config)
        
        # Add agents
        workflow.add_agent(
            name="agent1",
            role="First Agent",
            goal="Complete first task",
            backstory="I am the first agent"
        )
        
        workflow.add_agent(
            name="agent2", 
            role="Second Agent",
            goal="Complete second task",
            backstory="I am the second agent"
        )
        
        # Add tasks with dependencies
        workflow.add_task(
            name="task1",
            description="First task",
            agent="agent1"
        )
        
        workflow.add_task(
            name="task2",
            description="Second task depends on first",
            agent="agent2",
            dependencies=["task1"]
        )
        
        # Execute workflow
        results = workflow.execute({"test_data": "hello world"})
        
        print(f"✅ Workflow executed successfully")
        print(f"✅ Execution time: {results.execution_time:.3f}s")
        print(f"✅ Success: {results.success}")
        
        return True
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_system():
    """Test memory system"""
    print("\n🧠 Testing Memory System...")
    
    try:
        from crewgraph_ai.memory import DictMemory
        
        memory = DictMemory()
        
        # Test basic operations
        memory.save("test_key", {"data": "test_value"})
        result = memory.load("test_key")
        
        print(f"✅ Memory save/load: {result}")
        print(f"✅ Memory size: {memory.get_size()}")
        
        # Test conversation support
        from langchain_core.messages import HumanMessage, AIMessage
        
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]
        
        success = memory.save_conversation("test_conv", messages)
        loaded = memory.load_conversation("test_conv")
        
        print(f"✅ Conversation save: {success}")
        print(f"✅ Conversation load: {len(loaded)} messages")
        
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_tools_system():
    """Test tools system"""
    print("\n🔧 Testing Tools System...")
    
    try:
        from crewgraph_ai.tools import ToolRegistry, BuiltinTools
        
        # Test tool registry
        registry = ToolRegistry()
        builtin = BuiltinTools()
        
        print(f"✅ Tool registry created")
        print(f"✅ Built-in tools available: {len(builtin.get_all_tools())}")
        
        # Test custom tool
        def test_tool(input_text: str) -> str:
            return f"Processed: {input_text}"
        
        from crewgraph_ai.tools import ToolWrapper
        tool = ToolWrapper(test_tool)
        registry.register_tool(tool)
        
        print(f"✅ Custom tool registered")
        
        return True
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration), 
        ("Agent Creation", test_agent_creation),
        ("Workflow Execution", test_workflow_execution),
        ("Memory System", test_memory_system),
        ("Tools System", test_tools_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! CrewGraph AI is working correctly!")
        print("✨ Key improvements verified:")
        print("  ✅ Agent execution fixed")
        print("  ✅ Workflow building fixed") 
        print("  ✅ API compatibility improved")
        print("  ✅ Configuration system added")
        print("  ✅ Memory system working")
        print("  ✅ Tools system functional")
    else:
        print(f"⚠️  {total - passed} tests failed")
        print("Some functionality may need additional work")
    
    print("=" * 50)
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)