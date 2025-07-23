#!/usr/bin/env python3
"""
CrewGraph AI - Updated Quick Start Example
Demonstrates the fixed API with proper configuration
"""

import os
from crewgraph_ai import CrewGraph, CrewGraphConfig, CrewGraphSettings, quick_setup
from crewgraph_ai.memory import DictMemory
from crewgraph_ai.utils import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

def main():
    """Updated quick start example with fixed API"""
    logger.info("🚀 Starting CrewGraph AI Updated Quick Start Example")
    
    print("=" * 60)
    print("🎉 CrewGraph AI - Updated Quick Start")
    print("=" * 60)
    
    # 1. Configuration Setup (NEW!)
    print("\n📋 Step 1: Configuration Setup")
    
    # Quick setup for configuration
    settings = quick_setup()
    
    # 2. Create CrewGraph workflow
    print("\n🏗️  Step 2: Creating Workflow")
    
    config = CrewGraphConfig(
        memory_backend=DictMemory(),
        enable_planning=True,
        max_concurrent_tasks=3,
        enable_visualization=True
    )
    
    workflow = CrewGraph("research_workflow", config)
    
    # 3. Add agents using the new simplified API
    print("\n🤖 Step 3: Adding Agents")
    
    # Create agents directly with CrewAI parameters
    researcher = workflow.add_agent(
        name="researcher",
        role="Research Specialist", 
        goal="Conduct thorough research on given topics",
        backstory="Expert researcher with 10 years of experience",
        verbose=True
    )
    
    writer = workflow.add_agent(
        name="writer",
        role="Content Writer",
        goal="Create engaging content based on research", 
        backstory="Professional writer specializing in technical content",
        verbose=True
    )
    
    print(f"✅ Added researcher: {researcher}")
    print(f"✅ Added writer: {writer}")
    
    # 4. Add tasks with dependencies
    print("\n📝 Step 4: Adding Tasks")
    
    research_task = workflow.add_task(
        name="research",
        description="Research AI trends in 2024",
        agent="researcher"
    )
    
    writing_task = workflow.add_task(
        name="write_article", 
        description="Write article based on research findings",
        agent="writer",
        dependencies=["research"]  # Depends on research task
    )
    
    print(f"✅ Added research task: {research_task}")
    print(f"✅ Added writing task: {writing_task}")
    
    # 5. Execute workflow
    print("\n🔄 Step 5: Executing Workflow")
    
    try:
        results = workflow.execute({
            "topic": "AI trends 2024",
            "target_audience": "technical professionals",
            "word_count": 1000
        })
        
        print("✅ Workflow completed successfully!")
        print(f"📊 Results: {results}")
        
        # 6. Show workflow state
        print("\n📈 Step 6: Workflow Analysis")
        
        # Get workflow state
        state = workflow.get_state()
        print(f"🔍 Workflow state keys: {list(state.data.keys())}")
        
        # Get agent metrics
        researcher_metrics = researcher.get_metrics()
        writer_metrics = writer.get_metrics()
        
        print(f"📊 Researcher completed {researcher_metrics.tasks_completed} tasks")
        print(f"📊 Writer completed {writer_metrics.tasks_completed} tasks")
        print(f"📊 Total execution time: {results.execution_time:.2f}s")
        
        # 7. Optional: Visualization
        print("\n🎨 Step 7: Visualization (Optional)")
        
        try:
            viz_path = workflow.visualize_workflow(format="html")
            print(f"📊 Workflow visualization saved to: {viz_path}")
        except Exception as e:
            print(f"ℹ️  Visualization not available: {e}")
        
        # 8. Show advanced features
        print("\n⭐ Step 8: Advanced Features Available")
        
        print("🔧 Available advanced features:")
        print("  - Real-time monitoring")
        print("  - Performance analytics") 
        print("  - Memory persistence")
        print("  - Error recovery")
        print("  - Debug tools")
        print("  - Security features")
        
        # Example of debug tools
        try:
            validation = workflow.validate_workflow()
            print(f"🔍 Workflow validation: {validation['summary']}")
        except Exception as e:
            print(f"ℹ️  Validation not available: {e}")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        print(f"❌ Error: {e}")
        
        # Show troubleshooting tips
        print("\n🛠️  Troubleshooting Tips:")
        print("1. Check API key configuration in .env file")
        print("2. Verify internet connection")
        print("3. Check logs for detailed error information")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Quick Start Completed Successfully!")
    print("🚀 You're ready to build advanced AI workflows!")
    print("📚 Check the documentation for more features")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)