"""
Pytest configuration and fixtures for CrewGraph AI tests

Author: Vatsal216
Created: 2025-07-23 06:14:25 UTC
"""

import pytest
import tempfile
import shutil
import os
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock

# Import CrewGraph AI components
from crewgraph_ai.memory import DictMemory, create_memory
from crewgraph_ai.core import AgentWrapper, TaskWrapper, SharedState, GraphOrchestrator
from crewgraph_ai.tools import BaseTool, ToolRegistry
from crewgraph_ai.utils.logging import get_logger

# Import CrewAI components for mocking
try:
    from crewai import Agent, Task, Crew
except ImportError:
    Agent = Task = Crew = None

logger = get_logger(__name__)


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp(prefix="crewgraph_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def dict_memory():
    """Create a DictMemory instance for testing"""
    memory = DictMemory(persistent=False)
    memory.connect()
    yield memory
    try:
        memory.clear()
        memory.disconnect()
    except Exception as e:
        logger.warning(f"Memory cleanup warning: {e}")


@pytest.fixture
def shared_state(dict_memory):
    """Create a SharedState instance for testing"""
    state = SharedState(memory=dict_memory)
    yield state
    try:
        state.clear()
    except Exception as e:
        logger.warning(f"State cleanup warning: {e}")


@pytest.fixture
def mock_agent():
    """Create a mock CrewAI Agent for testing"""
    if Agent is None:
        # Create a simple mock if CrewAI not available
        agent = Mock()
        agent.role = "Test Agent"
        agent.goal = "Test goal"
        agent.backstory = "Test backstory"
        agent.tools = []
        agent.verbose = False
        return agent
    
    return Agent(
        role="Test Agent",
        goal="Perform test operations",
        backstory="A test agent for CrewGraph AI testing",
        verbose=False,
        allow_delegation=False
    )


@pytest.fixture
def mock_task():
    """Create a mock CrewAI Task for testing"""
    if Task is None:
        # Create a simple mock if CrewAI not available
        task = Mock()
        task.description = "Test task"
        task.expected_output = "Test output"
        task.tools = []
        return task
    
    return Task(
        description="Perform a test operation",
        expected_output="A successful test result",
        tools=[]
    )


@pytest.fixture
def agent_wrapper(mock_agent, shared_state):
    """Create an AgentWrapper for testing"""
    wrapper = AgentWrapper(
        agent=mock_agent,
        state=shared_state,
        agent_id="test-agent-001"
    )
    yield wrapper


@pytest.fixture
def task_wrapper(mock_task, shared_state):
    """Create a TaskWrapper for testing"""
    wrapper = TaskWrapper(
        task=mock_task,
        state=shared_state,
        task_id="test-task-001"
    )
    yield wrapper


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing"""
    tool = Mock(spec=BaseTool)
    tool.name = "test_tool"
    tool.description = "A test tool"
    tool.run = Mock(return_value="test result")
    return tool


@pytest.fixture
def tool_registry():
    """Create a ToolRegistry for testing"""
    registry = ToolRegistry()
    yield registry
    try:
        registry.clear()
    except Exception as e:
        logger.warning(f"Registry cleanup warning: {e}")


@pytest.fixture
def orchestrator(shared_state):
    """Create a GraphOrchestrator for testing"""
    orchestrator = GraphOrchestrator(state=shared_state)
    yield orchestrator


@pytest.fixture
def sample_workflow_data():
    """Provide sample workflow data for testing"""
    return {
        "agents": [
            {
                "id": "agent1",
                "role": "Researcher",
                "goal": "Research information",
                "backstory": "Expert researcher"
            },
            {
                "id": "agent2", 
                "role": "Writer",
                "goal": "Write content",
                "backstory": "Expert writer"
            }
        ],
        "tasks": [
            {
                "id": "task1",
                "description": "Research the topic",
                "expected_output": "Research report",
                "agent_id": "agent1"
            },
            {
                "id": "task2",
                "description": "Write based on research",
                "expected_output": "Written content",
                "agent_id": "agent2",
                "dependencies": ["task1"]
            }
        ]
    }


@pytest.fixture
def benchmark_data():
    """Provide data for performance benchmarking"""
    return {
        "test_sizes": [10, 50, 100],
        "value_sizes": [100, 1000, 10000],
        "operations": ["save", "load", "delete", "list_keys"],
        "iterations": 3
    }


# Test data fixtures
@pytest.fixture
def sample_agent_configs():
    """Sample agent configurations for testing"""
    return [
        {
            "role": "Data Analyst",
            "goal": "Analyze data patterns",
            "backstory": "Expert in data analysis with 5 years experience",
            "tools": ["data_processor", "chart_generator"]
        },
        {
            "role": "Report Writer", 
            "goal": "Create comprehensive reports",
            "backstory": "Professional writer specializing in technical reports",
            "tools": ["text_processor", "document_formatter"]
        }
    ]


@pytest.fixture
def sample_task_configs():
    """Sample task configurations for testing"""
    return [
        {
            "description": "Analyze the provided dataset",
            "expected_output": "Data analysis report with insights",
            "tools": ["data_processor"]
        },
        {
            "description": "Generate visualizations from analysis",
            "expected_output": "Charts and graphs showing key patterns",
            "tools": ["chart_generator"]
        }
    ]


# Performance testing fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance tests"""
    return {
        "memory_backends": ["dict"],  # Add more when available
        "test_data_sizes": [10, 50, 100],
        "concurrent_operations": [1, 5, 10],
        "timeout_seconds": 30
    }


# Mock data generators
@pytest.fixture
def generate_test_data():
    """Factory to generate test data of various sizes"""
    def _generate(size: int, value_size: int = 100) -> Dict[str, str]:
        return {
            f"test_key_{i}": "x" * value_size
            for i in range(size)
        }
    return _generate


# Cleanup helpers
@pytest.fixture(autouse=True)
def cleanup_logs():
    """Cleanup log files after each test"""
    yield
    # Clean up any log files created during tests
    log_dirs = ["logs", "test_logs"]
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            try:
                shutil.rmtree(log_dir)
            except Exception:
                pass  # Ignore cleanup errors


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings"""
    config.addinivalue_line(
        "markers", 
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "performance: mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if "performance" in str(item.fspath) or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)


# Error handling for test fixtures
@pytest.fixture
def suppress_warnings():
    """Suppress expected warnings during tests"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        yield