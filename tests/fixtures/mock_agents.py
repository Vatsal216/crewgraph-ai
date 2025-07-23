"""
Mock agents for CrewGraph AI testing

Author: Vatsal216
Created: 2025-07-23 06:14:25 UTC
"""

from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List, Optional
import time
import uuid


class MockAgent:
    """Mock agent that simulates CrewAI Agent behavior"""
    
    def __init__(self, role: str, goal: str, backstory: str, 
                 tools: Optional[List] = None, verbose: bool = False):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.verbose = verbose
        self.execution_count = 0
        self.last_execution_time = None
        
    def execute_task(self, task_input: Any) -> str:
        """Mock task execution"""
        self.execution_count += 1
        self.last_execution_time = time.time()
        
        # Simulate some processing time
        time.sleep(0.01)
        
        return f"Mock execution result from {self.role}: {task_input}"
    
    def __str__(self):
        return f"MockAgent(role='{self.role}', goal='{self.goal}')"
    
    def __repr__(self):
        return self.__str__()


class MockTask:
    """Mock task that simulates CrewAI Task behavior"""
    
    def __init__(self, description: str, expected_output: str,
                 tools: Optional[List] = None, agent: Optional[MockAgent] = None):
        self.description = description
        self.expected_output = expected_output
        self.tools = tools or []
        self.agent = agent
        self.execution_count = 0
        self.last_execution_time = None
        
    def execute(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Mock task execution"""
        self.execution_count += 1
        self.last_execution_time = time.time()
        
        # Simulate some processing time
        time.sleep(0.01)
        
        context_str = f" with context: {context}" if context else ""
        return f"Mock task result: {self.expected_output}{context_str}"
    
    def __str__(self):
        return f"MockTask(description='{self.description}')"
    
    def __repr__(self):
        return self.__str__()


class MockTool:
    """Mock tool that simulates tool behavior"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0
        
    def run(self, input_data: Any) -> str:
        """Mock tool execution"""
        self.usage_count += 1
        time.sleep(0.005)  # Simulate tool execution time
        return f"Mock tool '{self.name}' result for: {input_data}"
    
    def __str__(self):
        return f"MockTool(name='{self.name}')"
    
    def __repr__(self):
        return self.__str__()


def create_research_agent() -> MockAgent:
    """Create a mock research agent"""
    return MockAgent(
        role="Research Analyst",
        goal="Conduct thorough research and analysis",
        backstory="Expert researcher with 10+ years of experience in data analysis",
        tools=[
            MockTool("search_tool", "Search for information"),
            MockTool("data_analyzer", "Analyze data patterns")
        ]
    )


def create_writer_agent() -> MockAgent:
    """Create a mock writer agent"""
    return MockAgent(
        role="Content Writer",
        goal="Create high-quality written content",
        backstory="Professional writer with expertise in technical writing",
        tools=[
            MockTool("text_processor", "Process and format text"),
            MockTool("grammar_checker", "Check grammar and style")
        ]
    )


def create_analyst_agent() -> MockAgent:
    """Create a mock analyst agent"""
    return MockAgent(
        role="Data Analyst",
        goal="Analyze data and extract insights",
        backstory="Data science expert with strong analytical skills",
        tools=[
            MockTool("statistics_tool", "Perform statistical analysis"),
            MockTool("visualization_tool", "Create data visualizations")
        ]
    )


def create_coordinator_agent() -> MockAgent:
    """Create a mock coordinator agent"""
    return MockAgent(
        role="Project Coordinator",
        goal="Coordinate project activities and manage workflow",
        backstory="Experienced project manager with expertise in AI workflows",
        tools=[
            MockTool("scheduling_tool", "Manage schedules and timelines"),
            MockTool("communication_tool", "Facilitate team communication")
        ]
    )


def create_quality_agent() -> MockAgent:
    """Create a mock quality assurance agent"""
    return MockAgent(
        role="Quality Assurance",
        goal="Ensure quality and accuracy of outputs",
        backstory="Quality expert with deep understanding of validation processes",
        tools=[
            MockTool("quality_checker", "Check output quality"),
            MockTool("validation_tool", "Validate results accuracy")
        ]
    )


def create_research_task(agent: Optional[MockAgent] = None) -> MockTask:
    """Create a mock research task"""
    return MockTask(
        description="Research the latest trends and developments in the field",
        expected_output="Comprehensive research report with key findings",
        agent=agent,
        tools=[MockTool("search_tool", "Search for information")]
    )


def create_analysis_task(agent: Optional[MockAgent] = None) -> MockTask:
    """Create a mock analysis task"""
    return MockTask(
        description="Analyze the collected data to identify patterns and insights",
        expected_output="Data analysis report with actionable insights",
        agent=agent,
        tools=[MockTool("analysis_tool", "Analyze data")]
    )


def create_writing_task(agent: Optional[MockAgent] = None) -> MockTask:
    """Create a mock writing task"""
    return MockTask(
        description="Write a comprehensive report based on the analysis",
        expected_output="Well-structured written report",
        agent=agent,
        tools=[MockTool("writing_tool", "Assist with writing")]
    )


def create_review_task(agent: Optional[MockAgent] = None) -> MockTask:
    """Create a mock review task"""
    return MockTask(
        description="Review and validate the quality of the output",
        expected_output="Quality assurance report with validation results",
        agent=agent,
        tools=[MockTool("review_tool", "Review content quality")]
    )


def create_planning_task(agent: Optional[MockAgent] = None) -> MockTask:
    """Create a mock planning task"""
    return MockTask(
        description="Plan the project timeline and resource allocation",
        expected_output="Detailed project plan with milestones",
        agent=agent,
        tools=[MockTool("planning_tool", "Create project plans")]
    )


def create_mock_crew(num_agents: int = 3, num_tasks: int = 3) -> Dict[str, Any]:
    """Create a mock crew with agents and tasks"""
    # Create agents
    agent_creators = [
        create_research_agent,
        create_writer_agent,
        create_analyst_agent,
        create_coordinator_agent,
        create_quality_agent
    ]
    
    agents = []
    for i in range(num_agents):
        creator = agent_creators[i % len(agent_creators)]
        agent = creator()
        agents.append(agent)
    
    # Create tasks
    task_creators = [
        create_research_task,
        create_analysis_task,
        create_writing_task,
        create_review_task,
        create_planning_task
    ]
    
    tasks = []
    for i in range(num_tasks):
        creator = task_creators[i % len(task_creators)]
        # Assign agent to task
        agent = agents[i % len(agents)] if agents else None
        task = creator(agent)
        tasks.append(task)
    
    return {
        "agents": agents,
        "tasks": tasks,
        "metadata": {
            "created_at": time.time(),
            "num_agents": len(agents),
            "num_tasks": len(tasks)
        }
    }


def create_failing_agent(error_message: str = "Mock agent failure") -> MockAgent:
    """Create a mock agent that always fails"""
    agent = MockAgent(
        role="Failing Agent",
        goal="Simulate failure scenarios",
        backstory="Agent designed to test error handling"
    )
    
    def failing_execute(task_input):
        agent.execution_count += 1
        raise Exception(error_message)
    
    agent.execute_task = failing_execute
    return agent


def create_slow_agent(delay_seconds: float = 1.0) -> MockAgent:
    """Create a mock agent that executes slowly"""
    agent = MockAgent(
        role="Slow Agent",
        goal="Simulate slow execution",
        backstory="Agent designed to test timeout scenarios"
    )
    
    def slow_execute(task_input):
        agent.execution_count += 1
        time.sleep(delay_seconds)
        return f"Slow result after {delay_seconds}s: {task_input}"
    
    agent.execute_task = slow_execute
    return agent


def create_conditional_agent(condition_func) -> MockAgent:
    """Create a mock agent that executes based on a condition"""
    agent = MockAgent(
        role="Conditional Agent",
        goal="Execute based on conditions",
        backstory="Agent designed to test conditional logic"
    )
    
    def conditional_execute(task_input):
        agent.execution_count += 1
        if condition_func(task_input):
            return f"Condition met: {task_input}"
        else:
            return f"Condition not met: {task_input}"
    
    agent.execute_task = conditional_execute
    return agent


# Pre-created mock instances for quick use
MOCK_RESEARCH_AGENT = create_research_agent()
MOCK_WRITER_AGENT = create_writer_agent()
MOCK_ANALYST_AGENT = create_analyst_agent()

MOCK_RESEARCH_TASK = create_research_task()
MOCK_ANALYSIS_TASK = create_analysis_task()
MOCK_WRITING_TASK = create_writing_task()

MOCK_BASIC_CREW = create_mock_crew(2, 2)


# Utility functions for testing
def reset_mock_metrics():
    """Reset execution counts and times for all mock objects"""
    for agent in [MOCK_RESEARCH_AGENT, MOCK_WRITER_AGENT, MOCK_ANALYST_AGENT]:
        agent.execution_count = 0
        agent.last_execution_time = None
    
    for task in [MOCK_RESEARCH_TASK, MOCK_ANALYSIS_TASK, MOCK_WRITING_TASK]:
        task.execution_count = 0
        task.last_execution_time = None


def get_mock_execution_stats() -> Dict[str, Any]:
    """Get execution statistics for mock objects"""
    return {
        "agents": {
            "research": {
                "executions": MOCK_RESEARCH_AGENT.execution_count,
                "last_execution": MOCK_RESEARCH_AGENT.last_execution_time
            },
            "writer": {
                "executions": MOCK_WRITER_AGENT.execution_count,
                "last_execution": MOCK_WRITER_AGENT.last_execution_time
            },
            "analyst": {
                "executions": MOCK_ANALYST_AGENT.execution_count,
                "last_execution": MOCK_ANALYST_AGENT.last_execution_time
            }
        },
        "tasks": {
            "research": {
                "executions": MOCK_RESEARCH_TASK.execution_count,
                "last_execution": MOCK_RESEARCH_TASK.last_execution_time
            },
            "analysis": {
                "executions": MOCK_ANALYSIS_TASK.execution_count,
                "last_execution": MOCK_ANALYSIS_TASK.last_execution_time
            },
            "writing": {
                "executions": MOCK_WRITING_TASK.execution_count,
                "last_execution": MOCK_WRITING_TASK.last_execution_time
            }
        }
    }


if __name__ == "__main__":
    # Demo the mock agents and tasks
    print("Creating mock crew...")
    crew = create_mock_crew(3, 3)
    
    print(f"Created {len(crew['agents'])} agents and {len(crew['tasks'])} tasks")
    
    # Test agent execution
    agent = crew["agents"][0]
    result = agent.execute_task("test input")
    print(f"Agent execution result: {result}")
    
    # Test task execution
    task = crew["tasks"][0]
    result = task.execute({"test": "context"})
    print(f"Task execution result: {result}")
    
    print("Mock agents and tasks are ready for testing!")