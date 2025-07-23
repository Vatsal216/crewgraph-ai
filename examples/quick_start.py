"""
CrewGraph AI - Quick Start Example
Demonstrates basic usage with CrewAI agents and LangGraph workflows
"""

import asyncio
from crewai import Agent, Task
from crewai.tools import BaseTool
from crewgraph_ai import CrewGraph, CrewGraphConfig
from crewgraph_ai.memory import DictMemory
from crewgraph_ai.utils import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

class SearchTool(BaseTool):
    name: str = "search"
    description: str = "Search the web for information"
    
    def _run(self, query: str) -> str:
        """Mock web search tool"""
        return f"Search results for: {query}"

class WriteTool(BaseTool):
    name: str = "write_file"
    description: str = "Write content to a file"
    
    def _run(self, content: str, filename: str = "output.txt") -> str:
        """Mock file writing tool"""
        with open(filename, 'w') as f:
            f.write(content)
        return f"Content written to {filename}"

def main():
    """Quick start example"""
    logger.info("Starting CrewGraph AI Quick Start Example")
    
    # 1. Create CrewAI agents (full compatibility)
    researcher = Agent(
        role='Research Specialist',
        goal='Conduct thorough research on given topics',
        backstory='Expert researcher with 10 years of experience',
        tools=[SearchTool()],
        verbose=True
    )
    
    writer = Agent(
        role='Content Writer', 
        goal='Create engaging content based on research',
        backstory='Professional writer specializing in technical content',
        tools=[WriteTool()],
        verbose=True
    )
    
    # 2. Create CrewGraph workflow
    config = CrewGraphConfig(
        memory_backend=DictMemory(),
        enable_planning=True,
        max_concurrent_tasks=3
    )
    
    workflow = CrewGraph("research_workflow", config)
    
    # 3. Add agents to workflow
    workflow.add_agent(researcher, name="researcher")
    workflow.add_agent(writer, name="writer")
    
    # 4. Add tasks
    research_task = workflow.add_task(
        name="research",
        description="Research AI trends in 2024",
        agent="researcher"
    )
    
    writing_task = workflow.add_task(
        name="write_article",
        description="Write article based on research",
        agent="writer",
        dependencies=["research"]  # Depends on research task
    )
    
    # 5. Create task chain
    workflow.create_chain("research", "write_article")
    
    # 6. Execute workflow
    results = workflow.execute({
        "topic": "AI trends 2024",
        "target_audience": "technical professionals"
    })
    
    logger.info("Workflow completed successfully!")
    logger.info(f"Results: {results}")
    
    # 7. Access original CrewAI objects
    original_researcher = workflow.get_agent("researcher").crew_agent
    logger.info(f"Original CrewAI agent: {original_researcher}")

if __name__ == "__main__":
    main()