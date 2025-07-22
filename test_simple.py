"""
Enhanced 2-Agent Test with LangChain Message Integration
Demonstrates the new message handling capabilities
"""

print("🚀 Starting Enhanced 2-Agent Test with Message Support...")

try:
    print("📦 Importing libraries...")
    from crewai import Agent
    # from crewai.tools import Tool
    from crewgraph_ai import CrewGraph, AgentWrapper, TaskWrapper
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    print("✅ CrewGraph, AgentWrapper, and Messages imported")
    
    from crewgraph_ai.memory import DictMemory
    print("✅ DictMemory imported")
    
    print("\n🤖 Creating agents...")
    
    def simple_research(topic):
        return f"Research results for {topic}: Key findings about the topic."

    def simple_write(content):
        return f"Article: {content} This is a well-written piece."

    # Create CrewAI Tool objects
    research_tool = {
        "name": "research",
        "func": simple_research,
        "description": "Research"
    }
    write_tool = {
        "name": "write",
        "func": simple_write,
        "description": "Write"
    }

    # Define agent parameters using CrewAI Tool objects
    researcher_params = {
        'role': 'Researcher',
        'goal': 'Research topics',
        'backstory': 'I research things.',
        'tools': [research_tool],
        'verbose': False,
        'name': 'researcher'
    }
    writer_params = {
        'role': 'Writer',
        'goal': 'Write content',
        'backstory': 'I write things.',
        'tools': [write_tool],
        'verbose': False,
        'name': 'writer'
    }

    # Create agents using parameters
    researcher = AgentWrapper(**researcher_params)
    writer = AgentWrapper(**writer_params)

    print("✅ Agents created successfully")

    # Create workflow with enhanced message support
    workflow = CrewGraph("test_workflow", config=CrewGraph.config_class(memory_backend=DictMemory()))
    workflow.add_agent(researcher, name=researcher_params['name'])
    workflow.add_agent(writer, name=writer_params['name'])

    print("✅ Workflow created successfully")
    
    # Create tasks with message context support
    research_task = TaskWrapper(
        name="research_task",
        description="Research AI trends"
    )
    research_task.assign_agent(researcher)
    research_task.add_tool(research_tool)

    write_task = TaskWrapper(
        name="write_task",
        description="Write about the research"
    )
    write_task.assign_agent(writer)
    write_task.add_tool(write_tool)
    write_task.dependencies = ["research_task"]

    # Add tasks to workflow
    workflow.add_task(research_task)
    workflow.add_task(write_task)

    print("✅ Tasks added successfully")
    
    # ============= ENHANCED MESSAGE TESTING =============
    print("\n💬 Testing Enhanced Message Capabilities...")
    
    # Create initial conversation messages
    initial_messages = [
        HumanMessage(content="I need a comprehensive analysis of AI trends for 2024"),
        AIMessage(content="I'll help you analyze AI trends. Let me start the research process.")
    ]
    
    # Test message-enhanced execution
    print("🔄 Executing workflow with message context...")
    
    # Use the enhanced execute_with_messages method
    enhanced_state = {
        "topic": "AI trends 2024",
        "user_requirements": "comprehensive analysis"
    }
    
    # Execute with message context (this uses our enhanced orchestrator)
    results = workflow.execute(enhanced_state)
    
    print("✅ Enhanced execution completed!")
    print(f"📊 Results: {results}")
    
    # Test message handling in tasks
    print("\n🧪 Testing Task Message Integration...")
    
    # Test execute_with_message_context on individual tasks
    test_messages = [
        HumanMessage(content="Focus on machine learning and automation trends"),
        AIMessage(content="Understanding the focus areas for research.")
    ]
    
    test_state = {"topic": "ML and Automation trends"}
    
    # This demonstrates the new message-aware task execution
    task_result = research_task.execute_with_message_context(test_messages, test_state)
    print(f"📋 Task result with message context: {task_result.success}")
    
    if task_result.success:
        # Create AI message from task result
        ai_response = research_task.create_message_from_result(task_result)
        print(f"🤖 Generated AI message: {ai_response.content[:100]}...")
    
    # Test memory with conversation support
    print("\n💾 Testing Enhanced Memory with Conversations...")
    
    memory = DictMemory()
    conversation_id = "test_conversation_001"
    
    # Create a sample conversation
    conversation_messages = [
        HumanMessage(content="What are the latest AI trends?"),
        AIMessage(content="The latest AI trends include large language models, multimodal AI, and autonomous systems."),
        HumanMessage(content="Can you elaborate on multimodal AI?"),
        AIMessage(content="Multimodal AI combines text, images, audio, and video processing in unified models.")
    ]
    
    # Save conversation to memory
    save_success = memory.save_conversation(conversation_id, conversation_messages)
    print(f"💾 Conversation saved: {save_success}")
    
    # Load conversation from memory
    loaded_messages = memory.load_conversation(conversation_id)
    print(f"📤 Loaded {len(loaded_messages)} messages from memory")
    
    # Get conversation summary
    summary = memory.get_conversation_summary(conversation_id)
    print(f"📊 Conversation summary: {summary.get('message_count', 0)} messages")
    
    # Search messages
    search_results = memory.search_messages("multimodal", limit=5)
    print(f"🔍 Found {len(search_results)} messages matching 'multimodal'")
    
    print("\n🎉 Enhanced message integration test completed successfully!")
    print("✨ New capabilities demonstrated:")
    print("   - Message-aware workflow execution")
    print("   - Task execution with conversation context")  
    print("   - Message storage and retrieval in memory")
    print("   - Conversation management and search")
    print("   - AI message generation from task results")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 Test completed!")
