"""
Simple 2-Agent Graph Example
Demonstrates a basic workflow with Research Agent -> Writer Agent
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Agent, Tool
from crewgraph_ai import CrewGraph, CrewGraphConfig
from crewgraph_ai.memory import DictMemory
from crewgraph_ai.utils import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

def research_tool(topic: str) -> str:
    """Mock research tool that simulates gathering information"""
    logger.info(f"🔍 Researching topic: {topic}")
    
    # Simulate research results
    research_data = {
        "AI trends": {
            "findings": [
                "Generative AI adoption increased 300% in 2024",
                "Multimodal AI models are becoming mainstream",
                "AI governance frameworks are being established"
            ],
            "sources": ["Tech Reports", "Industry Analysis", "Expert Interviews"]
        },
        "machine learning": {
            "findings": [
                "AutoML tools are democratizing ML development",
                "Edge AI deployment is growing rapidly",
                "Federated learning gains enterprise adoption"
            ],
            "sources": ["Research Papers", "Case Studies", "Market Data"]
        }
    }
    
    # Get data for the topic or return general findings
    data = research_data.get(topic.lower(), {
        "findings": [f"Research on {topic} shows promising developments", 
                    f"{topic} market is expanding rapidly",
                    f"Innovation in {topic} continues to accelerate"],
        "sources": ["General Research", "Market Analysis"]
    })
    
    result = f"""
Research Results for '{topic}':

Key Findings:
{chr(10).join(f'• {finding}' for finding in data['findings'])}

Sources: {', '.join(data['sources'])}
Confidence: High
Research Date: 2025-07-22
    """.strip()
    
    logger.info(f"✅ Research completed for: {topic}")
    return result

def writing_tool(research_data: str, style: str = "professional") -> str:
    """Mock writing tool that creates content based on research"""
    logger.info(f"✍️ Writing content in {style} style")
    
    # Extract key points from research data
    lines = research_data.split('\n')
    topic = ""
    findings = []
    
    for line in lines:
        if "Research Results for" in line:
            topic = line.split("'")[1] if "'" in line else "the topic"
        elif line.strip().startswith("•"):
            findings.append(line.strip()[2:])  # Remove bullet point
    
    if style == "professional":
        article = f"""
# {topic.title()} - Industry Analysis Report

## Executive Summary

Our comprehensive analysis of {topic} reveals significant developments that are reshaping the industry landscape. This report synthesizes key findings from multiple authoritative sources to provide actionable insights.

## Key Insights

{chr(10).join(f'{i+1}. **{finding}**' for i, finding in enumerate(findings))}

## Strategic Implications

The rapid evolution in {topic} presents both opportunities and challenges for organizations. Companies that adapt quickly to these trends will gain competitive advantages, while those that lag may find themselves at a disadvantage.

## Recommendations

- Monitor emerging developments closely
- Invest in relevant capabilities and infrastructure  
- Develop strategic partnerships in the ecosystem
- Ensure compliance with evolving regulations

## Conclusion

{topic.title()} continues to be a critical area for business growth and innovation. Organizations should prioritize understanding and leveraging these trends to drive future success.

---
*Report generated by CrewGraph AI Research & Writing Team*
*Date: 2025-07-22*
        """.strip()
    
    else:  # casual style
        article = f"""
# What's Happening with {topic.title()}?

Hey there! Let me break down what's going on in the world of {topic} right now.

## The Big Picture

{chr(10).join(f'• {finding}' for finding in findings)}

## Why Should You Care?

This stuff is actually pretty exciting! {topic.title()} is changing fast, and if you're in this space (or thinking about getting into it), now's a great time to pay attention.

## Bottom Line

{topic.title()} isn't just a buzzword anymore - it's real, it's happening, and it's going to impact pretty much everyone. Stay curious and keep learning!

---
*Created with CrewGraph AI - making research and writing easy!*
        """.strip()
    
    logger.info(f"✅ Article completed in {style} style")
    return article

def main():
    """Run the simple 2-agent graph example"""
    logger.info("🚀 Starting Simple 2-Agent Graph Example")
    print("="*60)
    print("🤖 CrewGraph AI - Simple 2-Agent Workflow")
    print("="*60)
    
    # Create tools
    research_crewai_tool = Tool(
        name="research_tool",
        func=research_tool,
        description="Research information about a given topic"
    )
    
    writing_crewai_tool = Tool(
        name="writing_tool", 
        func=writing_tool,
        description="Write an article based on research data"
    )
    
    # Create agents
    researcher = Agent(
        role='Research Specialist',
        goal='Conduct thorough research and gather comprehensive information',
        backstory='''You are an expert research analyst with 10+ years of experience 
                    in technology and market research. You excel at finding accurate, 
                    relevant information and presenting it clearly.''',
        tools=[research_crewai_tool],
        verbose=True,
        allow_delegation=False
    )
    
    writer = Agent(
        role='Content Writer',
        goal='Create engaging, well-structured content based on research findings',
        backstory='''You are a professional content writer and journalist with expertise 
                    in technology topics. You transform complex research into accessible, 
                    compelling articles that inform and engage readers.''',
        tools=[writing_crewai_tool],
        verbose=True,
        allow_delegation=False
    )
    
    # Create CrewGraph workflow
    config = CrewGraphConfig(
        memory_backend=DictMemory(),
        enable_planning=True,
        max_concurrent_tasks=2,
        task_timeout=120.0,
        enable_logging=True,
        log_level="INFO"
    )
    
    workflow = CrewGraph("simple_research_workflow", config)
    
    # Add agents to workflow
    workflow.add_agent(researcher, name="researcher")
    workflow.add_agent(writer, name="writer")
    
    # Add tasks with dependencies
    research_task = workflow.add_task(
        name="research_task",
        description="Research the given topic and gather comprehensive information",
        agent="researcher",
        tools=["research_tool"]
    )
    
    writing_task = workflow.add_task(
        name="writing_task", 
        description="Write a professional article based on the research findings",
        agent="writer",
        tools=["writing_tool"],
        dependencies=["research_task"]  # Writer depends on researcher
    )
    
    # Create the execution chain
    workflow.create_chain("research_task", "writing_task")
    
    print("\n📋 Workflow Setup Complete:")
    print(f"• Researcher Agent: {researcher.role}")
    print(f"• Writer Agent: {writer.role}") 
    print(f"• Task Chain: Research → Writing")
    print(f"• Memory Backend: {type(config.memory_backend).__name__}")
    
    # Execute workflow with different topics
    topics = ["AI trends", "machine learning"]
    
    for i, topic in enumerate(topics, 1):
        print(f"\n{'='*60}")
        print(f"🎯 Execution #{i}: Researching '{topic}'")
        print(f"{'='*60}")
        
        try:
            # Execute the workflow
            results = workflow.execute({
                "research_topic": topic,
                "writing_style": "professional" if i == 1 else "casual",
                "target_audience": "business professionals" if i == 1 else "general readers"
            })
            
            print(f"\n✅ Workflow completed successfully!")
            print(f"📊 Results summary:")
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"• {key}: {len(value)} characters generated")
                    else:
                        print(f"• {key}: {value}")
            else:
                print(f"• Result: {type(results).__name__}")
                
            # Display final article if available
            if isinstance(results, dict) and 'final_article' in results:
                print(f"\n📄 Generated Article Preview:")
                print("-" * 40)
                article = results['final_article']
                # Show first 300 characters
                preview = article[:300] + "..." if len(article) > 300 else article
                print(preview)
                print("-" * 40)
                
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            print(f"❌ Error: {e}")
    
    # Show workflow analytics
    print(f"\n{'='*60}")
    print("📈 Workflow Analytics")
    print(f"{'='*60}")
    
    try:
        # Get workflow state and metrics
        state = workflow.get_state()
        print(f"• Workflow Name: {workflow.name}")
        print(f"• Total Agents: {len(workflow.list_agents())}")
        print(f"• Total Tasks: {len(workflow.list_tasks())}")
        print(f"• Configuration: {workflow.config}")
        
        # List agents and their roles
        print(f"\n🤖 Agent Details:")
        for agent_name in workflow.list_agents():
            agent = workflow.get_agent(agent_name)
            print(f"  • {agent_name}: {agent.role}")
            
        # List tasks and dependencies  
        print(f"\n📋 Task Details:")
        for task_name in workflow.list_tasks():
            task = workflow.get_task(task_name)
            deps = getattr(task, 'dependencies', [])
            dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
            print(f"  • {task_name}: {task.description[:50]}...{dep_str}")
            
    except Exception as e:
        logger.warning(f"Could not retrieve analytics: {e}")
    
    print(f"\n🎉 Simple 2-Agent Graph Example Completed!")
    print(f"💡 This example demonstrates:")
    print(f"   • Agent collaboration (Research → Writing)")
    print(f"   • Task dependencies and chaining") 
    print(f"   • Memory management with DictMemory")
    print(f"   • Error handling and logging")
    print(f"   • Workflow configuration and execution")

if __name__ == "__main__":
    main()
