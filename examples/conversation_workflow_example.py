"""
Advanced Conversation Workflow Example
Demonstrates the enhanced WorkflowBuilder with message-aware capabilities
"""

print("🚀 Starting Advanced Conversation Workflow Example...")

try:
    from crewgraph_ai.core.orchestrator import WorkflowBuilder
    from crewgraph_ai.memory import DictMemory
    from langchain_core.messages import HumanMessage, AIMessage
    import uuid
    
    print("✅ Imports successful")
    
    # Initialize memory backend
    memory = DictMemory()
    conversation_id = f"demo_conversation_{uuid.uuid4().hex[:8]}"
    
    print(f"💾 Using conversation ID: {conversation_id}")
    
    # Define conversation-aware agent functions
    def research_agent(messages, state):
        """Research agent that considers conversation context"""
        # Extract latest human message for context
        latest_human_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                latest_human_msg = msg.content
                break
        
        if latest_human_msg:
            research_result = f"Based on your request '{latest_human_msg}', I've researched the latest developments in AI and found: advanced language models, computer vision breakthroughs, and robotics integration."
        else:
            research_result = "I've conducted general AI research and found exciting developments in multiple areas."
        
        return AIMessage(
            content=research_result,
            additional_kwargs={
                'agent': 'researcher',
                'sources': ['arxiv.org', 'nature.com', 'mit.edu'],
                'confidence': 0.95
            }
        )
    
    def analysis_agent(messages, state):
        """Analysis agent that builds on previous conversation"""
        # Look for research results in conversation
        research_content = ""
        for msg in messages:
            if isinstance(msg, AIMessage) and 'researcher' in msg.additional_kwargs.get('agent', ''):
                research_content = msg.content
                break
        
        if research_content:
            analysis = f"Analyzing the research data: The trends show significant advancement in AI capabilities. Key insights: 1) Language models are becoming more efficient, 2) Multimodal AI is gaining traction, 3) Real-world applications are expanding rapidly."
        else:
            analysis = "Performing general AI trend analysis based on available data."
        
        return AIMessage(
            content=analysis,
            additional_kwargs={
                'agent': 'analyst',
                'analysis_type': 'trend_analysis',
                'key_points': 3
            }
        )
    
    def summarizer_agent(messages, state):
        """Summarizer agent that creates final summary"""
        # Collect all AI responses
        ai_responses = [msg.content for msg in messages if isinstance(msg, AIMessage)]
        
        summary = f"Summary of AI Analysis Session:\n"
        summary += f"- Conducted research on latest AI developments\n"
        summary += f"- Analyzed {len(ai_responses)} data points\n"
        summary += f"- Key finding: AI is rapidly evolving across multiple domains\n"
        summary += f"- Recommendation: Continue monitoring these trends for strategic planning"
        
        return AIMessage(
            content=summary,
            additional_kwargs={
                'agent': 'summarizer',
                'summary_type': 'executive_summary',
                'data_points': len(ai_responses)
            }
        )
    
    # Build conversation-aware workflow
    print("\n🔧 Building conversation workflow...")
    
    workflow = (WorkflowBuilder("ai_analysis_conversation")
                .with_message_flow()  # Enable message-based execution
                .with_conversation([   # Initialize with conversation context
                    HumanMessage(content="I need a comprehensive AI trend analysis for our strategic planning")
                ])
                .add_conversation_node("research", research_agent)
                .add_conversation_node("analysis", analysis_agent)  
                .add_conversation_node("summary", summarizer_agent)
                .add_memory_node("save_conversation", memory, conversation_id)
                .build())
    
    # Set up workflow connections
    workflow.add_edge("__init_messages__", "research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "summary")
    workflow.add_edge("summary", "save_conversation")
    workflow.set_finish_point("save_conversation")
    
    print("✅ Conversation workflow built successfully")
    
    # Execute the conversation workflow
    print("\n🏃 Executing conversation workflow...")
    
    initial_state = {
        "user_id": "demo_user",
        "session_id": f"session_{uuid.uuid4().hex[:8]}",
        "preferences": {"detail_level": "comprehensive"}
    }
    
    result = workflow.execute(initial_state)
    
    print("✅ Conversation workflow completed!")
    print(f"📊 Success: {result.success}")
    print(f"⏱️  Execution time: {result.execution_time:.2f}s")
    
    # Demonstrate conversation retrieval from memory
    print("\n💾 Testing conversation retrieval...")
    
    saved_conversation = memory.load_conversation(conversation_id)
    print(f"📤 Retrieved {len(saved_conversation)} messages from memory")
    
    # Display conversation flow
    print("\n💬 Conversation Flow:")
    for i, msg in enumerate(saved_conversation, 1):
        msg_type = "👤 Human" if isinstance(msg, HumanMessage) else "🤖 AI"
        agent_info = ""
        if hasattr(msg, 'additional_kwargs') and 'agent' in msg.additional_kwargs:
            agent_info = f" ({msg.additional_kwargs['agent']})"
        
        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"  {i}. {msg_type}{agent_info}: {content_preview}")
    
    # Test conversation search
    print("\n🔍 Testing conversation search...")
    
    search_results = memory.search_messages("analysis", limit=3)
    print(f"🎯 Found {len(search_results)} messages containing 'analysis'")
    
    # Get conversation summary
    summary_info = memory.get_conversation_summary(conversation_id)
    print(f"\n📊 Conversation Summary:")
    print(f"   - Total messages: {summary_info.get('message_count', 0)}")
    print(f"   - Human messages: {summary_info.get('human_messages', 0)}")
    print(f"   - AI messages: {summary_info.get('ai_messages', 0)}")
    print(f"   - Total content length: {summary_info.get('total_content_length', 0)} characters")
    
    # Demonstrate conversation continuation
    print("\n➕ Testing conversation continuation...")
    
    # Add a follow-up human message
    follow_up_message = HumanMessage(content="Can you provide more specific recommendations for implementation?")
    memory.append_message_to_conversation(conversation_id, follow_up_message)
    
    # Create a simple follow-up workflow
    def recommendation_agent(messages, state):
        """Agent that provides specific recommendations"""
        return AIMessage(
            content="Based on the analysis, here are specific implementation recommendations: 1) Invest in LLM infrastructure, 2) Develop multimodal capabilities, 3) Create AI ethics guidelines, 4) Train teams on AI tools, 5) Establish AI governance framework.",
            additional_kwargs={'agent': 'recommender', 'recommendation_count': 5}
        )
    
    follow_up_workflow = (WorkflowBuilder("follow_up_recommendations")
                         .add_conversation_node("recommendations", recommendation_agent)
                         .add_memory_node("save_follow_up", memory, conversation_id)
                         .build())
    
    follow_up_workflow.add_edge("recommendations", "save_follow_up")
    follow_up_workflow.set_entry_point("recommendations")
    follow_up_workflow.set_finish_point("save_follow_up")
    
    # Execute follow-up with updated conversation
    updated_messages = memory.load_conversation(conversation_id)
    follow_up_result = follow_up_workflow.execute_with_messages(updated_messages, {})
    
    print(f"✅ Follow-up completed: {follow_up_result.success}")
    
    # Final conversation state
    final_conversation = memory.load_conversation(conversation_id)
    print(f"📈 Final conversation length: {len(final_conversation)} messages")
    
    print("\n🎉 Advanced Conversation Workflow Example completed successfully!")
    print("✨ Demonstrated capabilities:")
    print("   - Message-aware workflow building")
    print("   - Conversation context preservation")
    print("   - Agent communication through messages")
    print("   - Memory-based conversation persistence")
    print("   - Conversation search and retrieval")
    print("   - Workflow continuation with context")
    print("   - Multi-turn conversation handling")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 Example completed!")
