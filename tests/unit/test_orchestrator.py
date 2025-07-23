"""
Unit tests for GraphOrchestrator and WorkflowBuilder classes

Author: Vatsal216
Created: 2025-07-23 06:14:25 UTC
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from crewgraph_ai.core.orchestrator import GraphOrchestrator, WorkflowBuilder
from crewgraph_ai.core.state import SharedState
from crewgraph_ai.memory import DictMemory


class TestGraphOrchestrator(unittest.TestCase):
    """Test GraphOrchestrator functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.memory = DictMemory(persistent=False)
        self.memory.connect()
        self.state = SharedState(memory_backend=self.memory)
        
        self.orchestrator = GraphOrchestrator(state=self.state)
    
    def tearDown(self):
        """Cleanup test environment"""
        try:
            self.state.clear()
            self.memory.disconnect()
        except Exception:
            pass
    
    def test_orchestrator_initialization(self):
        """Test GraphOrchestrator initialization"""
        self.assertIsNotNone(self.orchestrator)
        self.assertEqual(self.orchestrator.state, self.state)
        self.assertIsNotNone(self.orchestrator.graph)
        self.assertIsNotNone(self.orchestrator.metrics)
    
    def test_agent_registration(self):
        """Test agent registration with orchestrator"""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.role = "Test Agent"
        mock_agent.goal = "Test goal"
        mock_agent.backstory = "Test backstory"
        
        # Register agent
        agent_id = self.orchestrator.register_agent(mock_agent)
        self.assertIsNotNone(agent_id)
        
        # Verify agent is registered
        registered_agents = self.orchestrator.list_agents()
        self.assertEqual(len(registered_agents), 1)
        self.assertEqual(registered_agents[0]["agent_id"], agent_id)
    
    def test_task_registration(self):
        """Test task registration with orchestrator"""
        # Create mock task
        mock_task = Mock()
        mock_task.description = "Test task"
        mock_task.expected_output = "Test output"
        
        # Register task
        task_id = self.orchestrator.register_task(mock_task)
        self.assertIsNotNone(task_id)
        
        # Verify task is registered
        registered_tasks = self.orchestrator.list_tasks()
        self.assertEqual(len(registered_tasks), 1)
        self.assertEqual(registered_tasks[0]["task_id"], task_id)
    
    def test_workflow_definition(self):
        """Test workflow definition and validation"""
        # Register agents and tasks
        agent1 = Mock()
        agent1.role = "Agent 1"
        agent_id1 = self.orchestrator.register_agent(agent1)
        
        task1 = Mock()
        task1.description = "Task 1"
        task_id1 = self.orchestrator.register_task(task1)
        
        # Define workflow
        workflow_config = {
            "nodes": [
                {"id": agent_id1, "type": "agent"},
                {"id": task_id1, "type": "task"}
            ],
            "edges": [
                {"from": agent_id1, "to": task_id1}
            ]
        }
        
        success = self.orchestrator.define_workflow(workflow_config)
        self.assertTrue(success)
    
    def test_workflow_execution(self):
        """Test workflow execution"""
        # Setup workflow
        agent = Mock()
        agent.role = "Test Agent"
        agent.execute_task = Mock(return_value="agent result")
        agent_id = self.orchestrator.register_agent(agent)
        
        task = Mock()
        task.description = "Test task"
        task.execute = Mock(return_value="task result")
        task_id = self.orchestrator.register_task(task)
        
        # Define simple workflow
        workflow_config = {
            "nodes": [
                {"id": agent_id, "type": "agent"},
                {"id": task_id, "type": "task"}
            ],
            "edges": [
                {"from": agent_id, "to": task_id}
            ]
        }
        
        self.orchestrator.define_workflow(workflow_config)
        
        # Execute workflow
        results = self.orchestrator.execute_workflow({"input": "test"})
        self.assertIsInstance(results, dict)
    
    def test_parallel_execution(self):
        """Test parallel execution capabilities"""
        # Setup multiple independent tasks
        tasks = []
        task_ids = []
        
        for i in range(3):
            task = Mock()
            task.description = f"Parallel task {i}"
            task.execute = Mock(return_value=f"result {i}")
            task_id = self.orchestrator.register_task(task)
            tasks.append(task)
            task_ids.append(task_id)
        
        # Define parallel workflow
        workflow_config = {
            "nodes": [{"id": tid, "type": "task"} for tid in task_ids],
            "edges": []  # No dependencies - all can run in parallel
        }
        
        self.orchestrator.define_workflow(workflow_config)
        
        # Execute with parallel mode
        results = self.orchestrator.execute_workflow(
            {"input": "test"}, 
            parallel=True
        )
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)
    
    def test_conditional_execution(self):
        """Test conditional execution logic"""
        # Setup conditional workflow
        condition_task = Mock()
        condition_task.description = "Condition check"
        condition_task.execute = Mock(return_value={"condition": True})
        condition_id = self.orchestrator.register_task(condition_task)
        
        success_task = Mock()
        success_task.description = "Success path"
        success_task.execute = Mock(return_value="success")
        success_id = self.orchestrator.register_task(success_task)
        
        failure_task = Mock()
        failure_task.description = "Failure path"
        failure_task.execute = Mock(return_value="failure")
        failure_id = self.orchestrator.register_task(failure_task)
        
        # Define conditional workflow
        workflow_config = {
            "nodes": [
                {"id": condition_id, "type": "task"},
                {"id": success_id, "type": "task"},
                {"id": failure_id, "type": "task"}
            ],
            "edges": [
                {
                    "from": condition_id, 
                    "to": success_id,
                    "condition": lambda result: result.get("condition", False)
                },
                {
                    "from": condition_id,
                    "to": failure_id,
                    "condition": lambda result: not result.get("condition", False)
                }
            ]
        }
        
        self.orchestrator.define_workflow(workflow_config)
        results = self.orchestrator.execute_workflow({"input": "test"})
        
        # Should execute condition and success path
        self.assertIn(condition_id, results)
        self.assertIn(success_id, results)
        self.assertNotIn(failure_id, results)
    
    def test_error_handling(self):
        """Test error handling during execution"""
        # Create task that raises an exception
        error_task = Mock()
        error_task.description = "Error task"
        error_task.execute = Mock(side_effect=Exception("Test error"))
        error_id = self.orchestrator.register_task(error_task)
        
        # Create recovery task
        recovery_task = Mock()
        recovery_task.description = "Recovery task"
        recovery_task.execute = Mock(return_value="recovered")
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
        
        self.orchestrator.define_workflow(workflow_config)
        
        # Execute with error handling
        results = self.orchestrator.execute_workflow(
            {"input": "test"}, 
            error_handling=True
        )
        
        self.assertIsInstance(results, dict)
    
    def test_orchestrator_metrics(self):
        """Test orchestrator metrics collection"""
        # Execute some operations
        agent = Mock()
        self.orchestrator.register_agent(agent)
        
        task = Mock()
        self.orchestrator.register_task(task)
        
        # Get metrics
        metrics = self.orchestrator.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_agents", metrics)
        self.assertIn("total_tasks", metrics)
        self.assertIn("workflows_executed", metrics)


class TestWorkflowBuilder(unittest.TestCase):
    """Test WorkflowBuilder functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.memory = DictMemory(persistent=False)
        self.memory.connect()
        self.state = SharedState(memory_backend=self.memory)
        
        self.builder = WorkflowBuilder(state=self.state)
    
    def tearDown(self):
        """Cleanup test environment"""
        try:
            self.state.clear()
            self.memory.disconnect()
        except Exception:
            pass
    
    def test_builder_initialization(self):
        """Test WorkflowBuilder initialization"""
        self.assertIsNotNone(self.builder)
        self.assertEqual(self.builder.state, self.state)
        self.assertIsNotNone(self.builder.workflow_config)
    
    def test_fluent_interface(self):
        """Test fluent interface for building workflows"""
        # Create mock components
        agent = Mock()
        agent.role = "Test Agent"
        
        task = Mock()
        task.description = "Test Task"
        
        # Build workflow using fluent interface
        workflow = (self.builder
                   .add_agent(agent, "agent1")
                   .add_task(task, "task1")
                   .connect("agent1", "task1")
                   .build())
        
        self.assertIsNotNone(workflow)
        self.assertIn("nodes", workflow)
        self.assertIn("edges", workflow)
    
    def test_workflow_validation(self):
        """Test workflow validation"""
        # Build invalid workflow (disconnected nodes)
        agent = Mock()
        task1 = Mock()
        task2 = Mock()
        
        workflow = (self.builder
                   .add_agent(agent, "agent1")
                   .add_task(task1, "task1")
                   .add_task(task2, "task2")
                   .connect("agent1", "task1")
                   # task2 is disconnected
                   .build())
        
        # Validate workflow
        is_valid, errors = self.builder.validate_workflow(workflow)
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)
    
    def test_workflow_templates(self):
        """Test predefined workflow templates"""
        # Test linear workflow template
        components = [
            {"type": "agent", "id": "agent1", "config": {}},
            {"type": "task", "id": "task1", "config": {}},
            {"type": "task", "id": "task2", "config": {}}
        ]
        
        linear_workflow = self.builder.create_linear_workflow(components)
        self.assertIsNotNone(linear_workflow)
        
        # Test parallel workflow template
        parallel_workflow = self.builder.create_parallel_workflow(components)
        self.assertIsNotNone(parallel_workflow)
    
    def test_workflow_optimization(self):
        """Test workflow optimization"""
        # Create workflow with optimization opportunities
        agents = [Mock() for _ in range(3)]
        tasks = [Mock() for _ in range(5)]
        
        builder = self.builder
        for i, agent in enumerate(agents):
            builder.add_agent(agent, f"agent{i}")
        
        for i, task in enumerate(tasks):
            builder.add_task(task, f"task{i}")
            
        # Create some connections
        workflow = (builder
                   .connect("agent0", "task0")
                   .connect("task0", "task1")
                   .connect("task1", "task2")
                   .connect("agent1", "task3")
                   .connect("agent2", "task4")
                   .build())
        
        # Optimize workflow
        optimized = self.builder.optimize_workflow(workflow)
        self.assertIsNotNone(optimized)
    
    def test_workflow_export_import(self):
        """Test workflow export and import"""
        # Build a simple workflow
        agent = Mock()
        task = Mock()
        
        workflow = (self.builder
                   .add_agent(agent, "agent1")
                   .add_task(task, "task1")
                   .connect("agent1", "task1")
                   .build())
        
        # Export workflow
        exported = self.builder.export_workflow(workflow, format="json")
        self.assertIsNotNone(exported)
        
        # Import workflow
        imported = self.builder.import_workflow(exported, format="json")
        self.assertEqual(len(imported["nodes"]), len(workflow["nodes"]))
        self.assertEqual(len(imported["edges"]), len(workflow["edges"]))


# Pytest-style tests using fixtures
class TestOrchestratorWithFixtures:
    """Test orchestrator using pytest fixtures"""
    
    def test_orchestrator_with_fixtures(self, shared_state):
        """Test orchestrator with pytest fixtures"""
        orchestrator = GraphOrchestrator(state=shared_state)
        
        assert orchestrator is not None
        assert orchestrator.state == shared_state
    
    def test_agent_task_integration(self, orchestrator, mock_agent, mock_task):
        """Test agent and task integration"""
        # Register components
        agent_id = orchestrator.register_agent(mock_agent)
        task_id = orchestrator.register_task(mock_task)
        
        assert agent_id is not None
        assert task_id is not None
        
        # Verify registration
        agents = orchestrator.list_agents()
        tasks = orchestrator.list_tasks()
        
        assert len(agents) == 1
        assert len(tasks) == 1
        assert agents[0]["agent_id"] == agent_id
        assert tasks[0]["task_id"] == task_id
    
    @pytest.mark.integration
    def test_workflow_execution_integration(self, orchestrator, sample_workflow_data):
        """Test complete workflow execution"""
        # Setup workflow from sample data
        agent_configs = sample_workflow_data["agents"]
        task_configs = sample_workflow_data["tasks"]
        
        # Register agents
        agent_ids = {}
        for agent_config in agent_configs:
            mock_agent = Mock()
            mock_agent.role = agent_config["role"]
            mock_agent.goal = agent_config["goal"]
            mock_agent.backstory = agent_config["backstory"]
            agent_id = orchestrator.register_agent(mock_agent)
            agent_ids[agent_config["id"]] = agent_id
        
        # Register tasks
        task_ids = {}
        for task_config in task_configs:
            mock_task = Mock()
            mock_task.description = task_config["description"]
            mock_task.expected_output = task_config["expected_output"]
            mock_task.execute = Mock(return_value=f"Result for {task_config['id']}")
            task_id = orchestrator.register_task(mock_task)
            task_ids[task_config["id"]] = task_id
        
        # Define workflow
        workflow_config = {
            "nodes": (
                [{"id": agent_ids[aid], "type": "agent"} for aid in agent_ids] +
                [{"id": task_ids[tid], "type": "task"} for tid in task_ids]
            ),
            "edges": []
        }
        
        orchestrator.define_workflow(workflow_config)
        
        # Execute workflow
        results = orchestrator.execute_workflow({"input": "test integration"})
        assert isinstance(results, dict)


if __name__ == "__main__":
    unittest.main()