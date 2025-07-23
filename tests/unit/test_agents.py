"""
Unit tests for AgentWrapper and AgentPool classes

Author: Vatsal216
Created: 2025-07-23 06:14:25 UTC
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from crewgraph_ai.core.agents import AgentWrapper, AgentPool
from crewgraph_ai.core.state import SharedState
from crewgraph_ai.memory import DictMemory


class TestAgentWrapper(unittest.TestCase):
    """Test AgentWrapper functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.memory = DictMemory(persistent=False)
        self.memory.connect()
        self.state = SharedState(memory_backend=self.memory)
        
        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.role = "Test Agent"
        self.mock_agent.goal = "Test goal"
        self.mock_agent.backstory = "Test backstory"
        self.mock_agent.tools = []
        self.mock_agent.verbose = False
        
        self.agent_wrapper = AgentWrapper(
            agent=self.mock_agent,
            state=self.state,
            agent_id="test-agent-001"
        )
    
    def tearDown(self):
        """Cleanup test environment"""
        try:
            self.state.clear()
            self.memory.disconnect()
        except Exception:
            pass  # Ignore cleanup errors
    
    def test_agent_wrapper_initialization(self):
        """Test AgentWrapper initialization"""
        self.assertIsNotNone(self.agent_wrapper)
        self.assertEqual(self.agent_wrapper.agent_id, "test-agent-001")
        self.assertEqual(self.agent_wrapper.agent, self.mock_agent)
        self.assertEqual(self.agent_wrapper.state, self.state)
        self.assertIsNotNone(self.agent_wrapper.metrics)
    
    def test_agent_wrapper_metadata(self):
        """Test agent metadata handling"""
        # Test setting metadata
        metadata = {"team": "test", "priority": "high"}
        self.agent_wrapper.set_metadata(metadata)
        
        # Test retrieving metadata
        retrieved = self.agent_wrapper.get_metadata()
        self.assertEqual(retrieved["team"], "test")
        self.assertEqual(retrieved["priority"], "high")
    
    def test_agent_execution_tracking(self):
        """Test agent execution tracking"""
        # Mock execute method
        self.mock_agent.execute_task = Mock(return_value="test result")
        
        # Test task execution
        task_id = "test-task-001"
        result = self.agent_wrapper.execute_task(task_id, {"input": "test"})
        
        self.assertEqual(result, "test result")
        self.mock_agent.execute_task.assert_called_once()
    
    def test_agent_state_updates(self):
        """Test agent state updates"""
        # Test updating agent state
        state_data = {"status": "active", "last_task": "task-001"}
        self.agent_wrapper.update_state(state_data)
        
        # Verify state was updated
        agent_state = self.state.get_agent_state(self.agent_wrapper.agent_id)
        self.assertEqual(agent_state["status"], "active")
        self.assertEqual(agent_state["last_task"], "task-001")
    
    def test_agent_metrics_collection(self):
        """Test agent metrics collection"""
        # Execute some operations to generate metrics
        self.agent_wrapper.update_state({"status": "busy"})
        self.agent_wrapper.update_state({"status": "idle"})
        
        # Check metrics
        metrics = self.agent_wrapper.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("state_updates", metrics)


class TestAgentPool(unittest.TestCase):
    """Test AgentPool functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.memory = DictMemory(persistent=False)
        self.memory.connect()
        self.state = SharedState(memory_backend=self.memory)
        
        self.agent_pool = AgentPool(state=self.state)
        
        # Create mock agents
        self.mock_agents = []
        for i in range(3):
            agent = Mock()
            agent.role = f"Agent {i}"
            agent.goal = f"Goal {i}"
            agent.backstory = f"Backstory {i}"
            agent.tools = []
            agent.verbose = False
            self.mock_agents.append(agent)
    
    def tearDown(self):
        """Cleanup test environment"""
        try:
            self.state.clear()
            self.memory.disconnect()
        except Exception:
            pass
    
    def test_agent_pool_initialization(self):
        """Test AgentPool initialization"""
        self.assertIsNotNone(self.agent_pool)
        self.assertEqual(len(self.agent_pool.agents), 0)
        self.assertEqual(self.agent_pool.state, self.state)
    
    def test_add_agent_to_pool(self):
        """Test adding agents to pool"""
        # Add first agent
        agent_id = self.agent_pool.add_agent(self.mock_agents[0])
        self.assertIsNotNone(agent_id)
        self.assertEqual(len(self.agent_pool.agents), 1)
        
        # Add second agent with custom ID
        custom_id = "custom-agent-001"
        result_id = self.agent_pool.add_agent(self.mock_agents[1], agent_id=custom_id)
        self.assertEqual(result_id, custom_id)
        self.assertEqual(len(self.agent_pool.agents), 2)
    
    def test_remove_agent_from_pool(self):
        """Test removing agents from pool"""
        # Add agents
        agent_id1 = self.agent_pool.add_agent(self.mock_agents[0])
        agent_id2 = self.agent_pool.add_agent(self.mock_agents[1])
        
        # Remove one agent
        success = self.agent_pool.remove_agent(agent_id1)
        self.assertTrue(success)
        self.assertEqual(len(self.agent_pool.agents), 1)
        
        # Try to remove non-existent agent
        success = self.agent_pool.remove_agent("non-existent")
        self.assertFalse(success)
    
    def test_get_agent_from_pool(self):
        """Test retrieving agents from pool"""
        # Add agent
        agent_id = self.agent_pool.add_agent(self.mock_agents[0])
        
        # Retrieve agent
        agent_wrapper = self.agent_pool.get_agent(agent_id)
        self.assertIsNotNone(agent_wrapper)
        self.assertEqual(agent_wrapper.agent_id, agent_id)
        
        # Try to get non-existent agent
        agent_wrapper = self.agent_pool.get_agent("non-existent")
        self.assertIsNone(agent_wrapper)
    
    def test_list_agents_in_pool(self):
        """Test listing agents in pool"""
        # Add multiple agents
        agent_ids = []
        for agent in self.mock_agents:
            agent_id = self.agent_pool.add_agent(agent)
            agent_ids.append(agent_id)
        
        # List all agents
        agent_list = self.agent_pool.list_agents()
        self.assertEqual(len(agent_list), 3)
        
        # Check agent info
        for agent_info in agent_list:
            self.assertIn("agent_id", agent_info)
            self.assertIn("role", agent_info)
            self.assertIn("status", agent_info)
    
    def test_agent_pool_metrics(self):
        """Test agent pool metrics"""
        # Add agents
        for agent in self.mock_agents:
            self.agent_pool.add_agent(agent)
        
        # Get pool metrics
        metrics = self.agent_pool.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_agents", metrics)
        self.assertIn("active_agents", metrics)
        self.assertEqual(metrics["total_agents"], 3)
    
    def test_agent_pool_scaling(self):
        """Test agent pool scaling operations"""
        # Test auto-scaling capabilities
        initial_count = len(self.agent_pool.agents)
        
        # Simulate high load
        self.agent_pool.adjust_pool_size(target_size=5)
        
        # Pool should remain at current size if no template agents
        self.assertEqual(len(self.agent_pool.agents), initial_count)


# Pytest-style tests using fixtures
class TestAgentWrapperWithFixtures:
    """Test AgentWrapper using pytest fixtures"""
    
    def test_agent_wrapper_with_mock(self, mock_agent, shared_state):
        """Test AgentWrapper with pytest fixtures"""
        wrapper = AgentWrapper(
            name="fixture-test-agent",
            role=mock_agent.role,
            crew_agent=mock_agent,
            state=shared_state
        )
        
        assert wrapper is not None
        assert wrapper.name == "fixture-test-agent"
        assert wrapper.crew_agent == mock_agent
        assert wrapper.state == shared_state
    
    def test_agent_execution_with_fixtures(self, agent_wrapper):
        """Test agent execution using fixtures"""
        # Mock the agent's execute method
        agent_wrapper.agent.execute_task = Mock(return_value="fixture result")
        
        result = agent_wrapper.execute_task("test-task", {"data": "test"})
        assert result == "fixture result"
        agent_wrapper.agent.execute_task.assert_called_once()
    
    def test_agent_state_persistence(self, agent_wrapper):
        """Test agent state persistence"""
        state_data = {"status": "testing", "fixture": True}
        agent_wrapper.update_state(state_data)
        
        # Retrieve state
        retrieved_state = agent_wrapper.state.get_agent_state(agent_wrapper.agent_id)
        assert retrieved_state["status"] == "testing"
        assert retrieved_state["fixture"] is True


if __name__ == "__main__":
    unittest.main()