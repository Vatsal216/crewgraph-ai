"""
Integration tests for CrewGraph AI workflows

Author: Vatsal216
Created: 2025-07-23 06:14:25 UTC
"""

import pytest
import unittest
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from crewgraph_ai.core import (
    AgentWrapper, TaskWrapper, GraphOrchestrator, 
    SharedState, StateManager
)
from crewgraph_ai.memory import DictMemory, create_memory
from crewgraph_ai.tools import ToolRegistry, BaseTool


class TestWorkflowIntegration(unittest.TestCase):
    """Test end-to-end workflow integration"""
    
    def setUp(self):
        """Setup test environment"""
        # Create memory backend
        self.memory = DictMemory(persistent=False)
        self.memory.connect()
        
        # Create shared state
        self.state = SharedState(memory_backend=self.memory)
        
        # Create orchestrator
        self.orchestrator = GraphOrchestrator(state=self.state)
        
        # Create tool registry
        self.tool_registry = ToolRegistry()
        
        # Setup mock components
        self.setup_mock_components()
    
    def tearDown(self):
        """Cleanup test environment"""
        try:
            self.state.clear()
            self.memory.disconnect()
            self.tool_registry.clear()
        except Exception:
            pass
    
    def setup_mock_components(self):
        """Setup mock agents, tasks, and tools"""
        # Create mock agents
        self.mock_agents = []
        agent_configs = [
            {"role": "Researcher", "goal": "Research information", "backstory": "Expert researcher"},
            {"role": "Analyst", "goal": "Analyze data", "backstory": "Data analysis expert"},
            {"role": "Writer", "goal": "Write content", "backstory": "Professional writer"}
        ]
        
        for config in agent_configs:
            agent = Mock()
            agent.role = config["role"]
            agent.goal = config["goal"]
            agent.backstory = config["backstory"]
            agent.tools = []
            agent.execute_task = Mock(return_value=f"Result from {config['role']}")
            self.mock_agents.append(agent)
        
        # Create mock tasks
        self.mock_tasks = []
        task_configs = [
            {"description": "Research the topic", "expected_output": "Research report"},
            {"description": "Analyze research data", "expected_output": "Analysis report"},
            {"description": "Write final report", "expected_output": "Written report"}
        ]
        
        for config in task_configs:
            task = Mock()
            task.description = config["description"]
            task.expected_output = config["expected_output"]
            task.tools = []
            task.execute = Mock(return_value=f"Output: {config['expected_output']}")
            self.mock_tasks.append(task)
        
        # Create mock tools
        self.mock_tools = []
        tool_configs = [
            {"name": "search_tool", "description": "Search for information"},
            {"name": "analysis_tool", "description": "Analyze data"},
            {"name": "writer_tool", "description": "Format and write content"}
        ]
        
        for config in tool_configs:
            tool = Mock(spec=BaseTool)
            tool.name = config["name"]
            tool.description = config["description"]
            tool.run = Mock(return_value=f"Result from {config['name']}")
            self.mock_tools.append(tool)
            self.tool_registry.register_tool(tool)
    
    def test_simple_linear_workflow(self):
        """Test simple linear workflow execution"""
        # Register agents and tasks
        agent_ids = []
        for agent in self.mock_agents:
            agent_id = self.orchestrator.register_agent(agent)
            agent_ids.append(agent_id)
        
        task_ids = []
        for task in self.mock_tasks:
            task_id = self.orchestrator.register_task(task)
            task_ids.append(task_id)
        
        # Define linear workflow
        workflow_config = {
            "nodes": (
                [{"id": aid, "type": "agent"} for aid in agent_ids] +
                [{"id": tid, "type": "task"} for tid in task_ids]
            ),
            "edges": [
                {"from": agent_ids[0], "to": task_ids[0]},
                {"from": task_ids[0], "to": task_ids[1]},
                {"from": agent_ids[1], "to": task_ids[1]},
                {"from": task_ids[1], "to": task_ids[2]},
                {"from": agent_ids[2], "to": task_ids[2]}
            ]
        }
        
        # Execute workflow
        self.orchestrator.define_workflow(workflow_config)
        results = self.orchestrator.execute_workflow({"input": "Test linear workflow"})
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertTrue(len(results) > 0)
        
        # Verify all tasks were executed
        for task in self.mock_tasks:
            task.execute.assert_called()
    
    def test_parallel_workflow_execution(self):
        """Test parallel workflow execution"""
        # Register components for parallel execution
        agent_ids = []
        for agent in self.mock_agents:
            agent_id = self.orchestrator.register_agent(agent)
            agent_ids.append(agent_id)
        
        task_ids = []
        for task in self.mock_tasks:
            task_id = self.orchestrator.register_task(task)
            task_ids.append(task_id)
        
        # Define parallel workflow (tasks can run simultaneously)
        workflow_config = {
            "nodes": (
                [{"id": aid, "type": "agent"} for aid in agent_ids] +
                [{"id": tid, "type": "task"} for tid in task_ids]
            ),
            "edges": [
                # Each agent works on a separate task in parallel
                {"from": agent_ids[0], "to": task_ids[0]},
                {"from": agent_ids[1], "to": task_ids[1]},
                {"from": agent_ids[2], "to": task_ids[2]}
            ]
        }
        
        # Execute with parallel mode
        self.orchestrator.define_workflow(workflow_config)
        results = self.orchestrator.execute_workflow(
            {"input": "Test parallel workflow"}, 
            parallel=True
        )
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(task_ids))
    
    def test_conditional_workflow_execution(self):
        """Test conditional workflow execution"""
        # Setup conditional logic
        condition_agent = self.mock_agents[0]
        success_task = self.mock_tasks[0]
        failure_task = self.mock_tasks[1]
        
        # Mock conditional result
        condition_agent.execute_task = Mock(return_value={"condition_met": True})
        
        # Register components
        condition_id = self.orchestrator.register_agent(condition_agent)
        success_id = self.orchestrator.register_task(success_task)
        failure_id = self.orchestrator.register_task(failure_task)
        
        # Define conditional workflow
        workflow_config = {
            "nodes": [
                {"id": condition_id, "type": "agent"},
                {"id": success_id, "type": "task"},
                {"id": failure_id, "type": "task"}
            ],
            "edges": [
                {
                    "from": condition_id,
                    "to": success_id,
                    "condition": lambda result: result.get("condition_met", False)
                },
                {
                    "from": condition_id,
                    "to": failure_id,
                    "condition": lambda result: not result.get("condition_met", False)
                }
            ]
        }
        
        # Execute conditional workflow
        self.orchestrator.define_workflow(workflow_config)
        results = self.orchestrator.execute_workflow({"input": "Test conditional"})
        
        # Verify correct path was taken
        self.assertIn(condition_id, results)
        self.assertIn(success_id, results)
        self.assertNotIn(failure_id, results)
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery in workflows"""
        # Setup error scenario
        error_task = Mock()
        error_task.description = "Task that fails"
        error_task.execute = Mock(side_effect=Exception("Simulated error"))
        
        recovery_task = Mock()
        recovery_task.description = "Recovery task"
        recovery_task.execute = Mock(return_value="Recovery successful")
        
        # Register components
        error_id = self.orchestrator.register_task(error_task)
        recovery_id = self.orchestrator.register_task(recovery_task)
        
        # Define workflow with error handling
        workflow_config = {
            "nodes": [
                {"id": error_id, "type": "task"},
                {"id": recovery_id, "type": "task"}
            ],
            "edges": [
                {"from": error_id, "to": recovery_id, "on_error": True}
            ]
        }
        
        # Execute with error handling enabled
        self.orchestrator.define_workflow(workflow_config)
        results = self.orchestrator.execute_workflow(
            {"input": "Test error handling"},
            error_handling=True
        )
        
        # Verify error was handled and recovery executed
        self.assertIsInstance(results, dict)
        self.assertIn(recovery_id, results)
    
    def test_state_persistence_across_workflow(self):
        """Test state persistence across workflow execution"""
        # Setup workflow that modifies state
        state_modifier = Mock()
        state_modifier.role = "State Modifier"
        state_modifier.execute_task = Mock(side_effect=self._modify_state)
        
        state_reader = Mock()
        state_reader.role = "State Reader"
        state_reader.execute_task = Mock(side_effect=self._read_state)
        
        # Register components
        modifier_id = self.orchestrator.register_agent(state_modifier)
        reader_id = self.orchestrator.register_agent(state_reader)
        
        # Define workflow
        workflow_config = {
            "nodes": [
                {"id": modifier_id, "type": "agent"},
                {"id": reader_id, "type": "agent"}
            ],
            "edges": [
                {"from": modifier_id, "to": reader_id}
            ]
        }
        
        # Execute workflow
        self.orchestrator.define_workflow(workflow_config)
        results = self.orchestrator.execute_workflow({"input": "Test state persistence"})
        
        # Verify state was persisted and accessible
        self.assertIn("persistent_data", results[reader_id])
    
    def _modify_state(self, *args, **kwargs):
        """Helper method to modify state"""
        self.state.set("persistent_data", "data_from_modifier")
        return "State modified"
    
    def _read_state(self, *args, **kwargs):
        """Helper method to read state"""
        data = self.state.get("persistent_data")
        return {"persistent_data": data}
    
    def test_tool_integration_in_workflow(self):
        """Test tool integration in workflow execution"""
        # Setup agent with tools
        tool_user = Mock()
        tool_user.role = "Tool User"
        tool_user.tools = self.mock_tools
        tool_user.execute_task = Mock(return_value="Tool result")
        
        # Create task that uses tools
        tool_task = Mock()
        tool_task.description = "Task using tools"
        tool_task.tools = self.mock_tools
        tool_task.execute = Mock(return_value="Task with tools completed")
        
        # Register components
        agent_id = self.orchestrator.register_agent(tool_user)
        task_id = self.orchestrator.register_task(tool_task)
        
        # Define workflow
        workflow_config = {
            "nodes": [
                {"id": agent_id, "type": "agent"},
                {"id": task_id, "type": "task"}
            ],
            "edges": [
                {"from": agent_id, "to": task_id}
            ]
        }
        
        # Execute workflow
        self.orchestrator.define_workflow(workflow_config)
        results = self.orchestrator.execute_workflow({"input": "Test tool integration"})
        
        # Verify workflow executed successfully
        self.assertIsInstance(results, dict)
        self.assertIn(agent_id, results)
        self.assertIn(task_id, results)


# Pytest-style integration tests using fixtures
class TestWorkflowIntegrationWithFixtures:
    """Test workflow integration using pytest fixtures"""
    
    @pytest.mark.integration
    def test_complete_workflow_with_fixtures(self, orchestrator, sample_workflow_data):
        """Test complete workflow using fixtures"""
        workflow_data = sample_workflow_data
        
        # Setup agents
        agent_ids = {}
        for agent_config in workflow_data["agents"]:
            mock_agent = Mock()
            mock_agent.role = agent_config["role"]
            mock_agent.goal = agent_config["goal"]
            mock_agent.backstory = agent_config["backstory"]
            mock_agent.execute_task = Mock(return_value=f"Result from {agent_config['role']}")
            
            agent_id = orchestrator.register_agent(mock_agent)
            agent_ids[agent_config["id"]] = agent_id
        
        # Setup tasks
        task_ids = {}
        for task_config in workflow_data["tasks"]:
            mock_task = Mock()
            mock_task.description = task_config["description"]
            mock_task.expected_output = task_config["expected_output"]
            mock_task.execute = Mock(return_value=task_config["expected_output"])
            
            task_id = orchestrator.register_task(mock_task)
            task_ids[task_config["id"]] = task_id
        
        # Define workflow with dependencies
        workflow_config = {
            "nodes": (
                [{"id": agent_ids[aid], "type": "agent"} for aid in agent_ids] +
                [{"id": task_ids[tid], "type": "task"} for tid in task_ids]
            ),
            "edges": [
                {"from": agent_ids["agent1"], "to": task_ids["task1"]},
                {"from": task_ids["task1"], "to": task_ids["task2"]},
                {"from": agent_ids["agent2"], "to": task_ids["task2"]}
            ]
        }
        
        # Execute workflow
        orchestrator.define_workflow(workflow_config)
        results = orchestrator.execute_workflow({"input": "Integration test data"})
        
        # Verify results
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Verify workflow completed successfully
        for task_id in task_ids.values():
            assert task_id in results
    
    @pytest.mark.integration
    def test_memory_backend_integration(self, shared_state, generate_test_data):
        """Test memory backend integration in workflows"""
        # Generate test data
        test_data = generate_test_data(10, 100)
        
        # Store data in shared state
        for key, value in test_data.items():
            shared_state.set(key, value)
        
        # Verify data persistence
        for key, expected_value in test_data.items():
            actual_value = shared_state.get(key)
            assert actual_value == expected_value
        
        # Test bulk operations
        all_keys = shared_state.memory_backend.list_keys()
        assert len(all_keys) >= len(test_data)
        
        # Test state serialization
        serialized = shared_state.serialize()
        assert serialized is not None
        
        # Clear and restore
        shared_state.clear()
        assert len(shared_state.memory_backend.list_keys()) == 0
        
        shared_state.deserialize(serialized)
        restored_keys = shared_state.memory_backend.list_keys()
        assert len(restored_keys) >= len(test_data)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_workflow_execution(self, orchestrator, performance_config):
        """Test large workflow execution"""
        config = performance_config
        
        # Create large workflow
        num_agents = 10
        num_tasks = 20
        
        # Register many agents
        agent_ids = []
        for i in range(num_agents):
            agent = Mock()
            agent.role = f"Agent {i}"
            agent.goal = f"Goal {i}"
            agent.execute_task = Mock(return_value=f"Result {i}")
            agent_id = orchestrator.register_agent(agent)
            agent_ids.append(agent_id)
        
        # Register many tasks
        task_ids = []
        for i in range(num_tasks):
            task = Mock()
            task.description = f"Task {i}"
            task.execute = Mock(return_value=f"Output {i}")
            task_id = orchestrator.register_task(task)
            task_ids.append(task_id)
        
        # Create complex workflow
        workflow_config = {
            "nodes": (
                [{"id": aid, "type": "agent"} for aid in agent_ids] +
                [{"id": tid, "type": "task"} for tid in task_ids]
            ),
            "edges": []
        }
        
        # Create connections (each agent connected to 2 tasks)
        for i, agent_id in enumerate(agent_ids):
            task1_idx = (i * 2) % len(task_ids)
            task2_idx = (i * 2 + 1) % len(task_ids)
            workflow_config["edges"].extend([
                {"from": agent_id, "to": task_ids[task1_idx]},
                {"from": agent_id, "to": task_ids[task2_idx]}
            ])
        
        # Execute large workflow
        orchestrator.define_workflow(workflow_config)
        
        import time
        start_time = time.time()
        results = orchestrator.execute_workflow({"input": "Large workflow test"})
        execution_time = time.time() - start_time
        
        # Verify execution completed
        assert isinstance(results, dict)
        assert execution_time < config["timeout_seconds"]
        
        # Check that reasonable number of components executed
        assert len(results) > 0


if __name__ == "__main__":
    unittest.main()