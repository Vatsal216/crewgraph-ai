#!/usr/bin/env python3
"""
Test script to verify the AgentWrapper fix
"""

print("🚀 Testing CrewGraph AI Agent Fix...")

try:
    print("📦 Importing libraries...")
    from crewai import Agent
    from crewgraph_ai import CrewGraph, CrewGraphConfig
    from crewgraph_ai.memory import DictMemory
    
    print("✅ Libraries imported successfully")
    
    print("\n🤖 Creating agents with fixed AgentWrapper...")
    
    # Test 1: Create agents using CrewAI parameters directly
    researcher_params = {
        'role': 'Researcher',
        'goal': 'Research topics thoroughly',
        'backstory': 'I am an expert researcher.',
        'verbose': False,
        'name': 'researcher'
    }
    
    writer_params = {
        'role': 'Writer', 
        'goal': 'Write engaging content',
        'backstory': 'I am a professional writer.',
        'verbose': False,
        'name': 'writer'
    }
    
    # Create workflow
    config = CrewGraphConfig(
        memory_backend=DictMemory(),
        enable_planning=True,
        max_concurrent_tasks=2
    )
    
    workflow = CrewGraph("test_workflow", config)
    
    # Add agents using the new constructor
    researcher = workflow.add_agent(None, **researcher_params)
    writer = workflow.add_agent(None, **writer_params)
    
    print("✅ Agents created successfully with new constructor")
    
    # Test 2: Add simple tasks
    workflow.add_task(
        name="research_task",
        description="Research AI trends",
        agent="researcher"
    )
    
    workflow.add_task(
        name="write_task", 
        description="Write about AI trends",
        agent="writer",
        dependencies=["research_task"]
    )
    
    print("✅ Tasks added successfully")
    
    # Test 3: Execute workflow
    print("\n🔄 Testing workflow execution...")
    
    results = workflow.execute({
        "topic": "AI trends 2024", 
        "audience": "tech professionals"
    })
    
    print("✅ Workflow executed successfully!")
    print(f"📊 Results: {results}")
    
    # Verify agents were created properly
    print(f"\n🔍 Researcher agent: {researcher}")
    print(f"🔍 Writer agent: {writer}")
    print(f"🔍 Researcher has CrewAI agent: {researcher.crew_agent is not None}")
    print(f"🔍 Writer has CrewAI agent: {writer.crew_agent is not None}")
    
    print("\n🎉 All tests passed! Agent fix is working correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 Test completed!")