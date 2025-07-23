"""
Unit tests for TaskWrapper and TaskChain classes

Author: Vatsal216
Created: 2025-07-23 06:14:25 UTC
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from crewgraph_ai.core.tasks import TaskWrapper, TaskChain
from crewgraph_ai.core.state import SharedState
from crewgraph_ai.memory import DictMemory


class TestTaskWrapper(unittest.TestCase):
    """Test TaskWrapper functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.memory = DictMemory(persistent=False)
        self.memory.connect()
        self.state = SharedState(memory_backend=self.memory)
        
        # Create mock task
        self.mock_task = Mock()
        self.mock_task.description = "Test task description"
        self.mock_task.expected_output = "Test expected output"
        self.mock_task.tools = []
        
        self.task_wrapper = TaskWrapper(
            task=self.mock_task,
            state=self.state,
            task_id="test-task-001"
        )
    
    def tearDown(self):
        """Cleanup test environment"""
        try:
            self.state.clear()
            self.memory.disconnect()
        except Exception:
            pass
    
    def test_task_wrapper_initialization(self):
        """Test TaskWrapper initialization"""
        self.assertIsNotNone(self.task_wrapper)
        self.assertEqual(self.task_wrapper.task_id, "test-task-001")
        self.assertEqual(self.task_wrapper.task, self.mock_task)
        self.assertEqual(self.task_wrapper.state, self.state)
        self.assertIsNotNone(self.task_wrapper.metrics)
    
    def test_task_metadata_handling(self):
        """Test task metadata handling"""
        metadata = {
            "priority": "high",
            "category": "research",
            "estimated_duration": 30
        }
        
        self.task_wrapper.set_metadata(metadata)
        retrieved = self.task_wrapper.get_metadata()
        
        self.assertEqual(retrieved["priority"], "high")
        self.assertEqual(retrieved["category"], "research")
        self.assertEqual(retrieved["estimated_duration"], 30)
    
    def test_task_dependency_management(self):
        """Test task dependency management"""
        # Add dependencies
        dependencies = ["task-001", "task-002"]
        self.task_wrapper.add_dependencies(dependencies)
        
        # Check dependencies
        deps = self.task_wrapper.get_dependencies()
        self.assertEqual(len(deps), 2)
        self.assertIn("task-001", deps)
        self.assertIn("task-002", deps)
        
        # Remove dependency
        self.task_wrapper.remove_dependency("task-001")
        deps = self.task_wrapper.get_dependencies()
        self.assertEqual(len(deps), 1)
        self.assertNotIn("task-001", deps)
    
    def test_task_execution_tracking(self):
        """Test task execution tracking"""
        # Mock task execution
        self.mock_task.execute = Mock(return_value="task result")
        
        # Execute task
        context = {"input_data": "test input"}
        result = self.task_wrapper.execute(context)
        
        self.assertEqual(result, "task result")
        self.mock_task.execute.assert_called_once()
        
        # Check execution state
        execution_state = self.task_wrapper.get_execution_state()
        self.assertIn("status", execution_state)
        self.assertIn("start_time", execution_state)
    
    def test_task_status_management(self):
        """Test task status management"""
        # Test status transitions
        self.task_wrapper.set_status("pending")
        self.assertEqual(self.task_wrapper.get_status(), "pending")
        
        self.task_wrapper.set_status("running")
        self.assertEqual(self.task_wrapper.get_status(), "running")
        
        self.task_wrapper.set_status("completed")
        self.assertEqual(self.task_wrapper.get_status(), "completed")
    
    def test_task_result_handling(self):
        """Test task result handling"""
        result_data = {
            "output": "Task completed successfully",
            "artifacts": ["report.pdf", "data.csv"],
            "metadata": {"duration": 45}
        }
        
        self.task_wrapper.set_result(result_data)
        retrieved_result = self.task_wrapper.get_result()
        
        self.assertEqual(retrieved_result["output"], "Task completed successfully")
        self.assertEqual(len(retrieved_result["artifacts"]), 2)
        self.assertEqual(retrieved_result["metadata"]["duration"], 45)
    
    def test_task_metrics_collection(self):
        """Test task metrics collection"""
        # Simulate task execution
        self.task_wrapper.set_status("running")
        self.task_wrapper.set_status("completed")
        self.task_wrapper.set_result({"output": "test"})
        
        metrics = self.task_wrapper.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("executions", metrics)
        self.assertIn("status_changes", metrics)


class TestTaskChain(unittest.TestCase):
    """Test TaskChain functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.memory = DictMemory(persistent=False)
        self.memory.connect()
        self.state = SharedState(memory_backend=self.memory)
        
        self.task_chain = TaskChain(state=self.state)
        
        # Create mock tasks
        self.mock_tasks = []
        for i in range(3):
            task = Mock()
            task.description = f"Task {i} description"
            task.expected_output = f"Task {i} output"
            task.tools = []
            task.execute = Mock(return_value=f"Result {i}")
            self.mock_tasks.append(task)
    
    def tearDown(self):
        """Cleanup test environment"""
        try:
            self.state.clear()
            self.memory.disconnect()
        except Exception:
            pass
    
    def test_task_chain_initialization(self):
        """Test TaskChain initialization"""
        self.assertIsNotNone(self.task_chain)
        self.assertEqual(len(self.task_chain.tasks), 0)
        self.assertEqual(self.task_chain.state, self.state)
    
    def test_add_task_to_chain(self):
        """Test adding tasks to chain"""
        # Add first task
        task_id = self.task_chain.add_task(self.mock_tasks[0])
        self.assertIsNotNone(task_id)
        self.assertEqual(len(self.task_chain.tasks), 1)
        
        # Add task with custom ID and dependencies
        custom_id = "custom-task-001"
        result_id = self.task_chain.add_task(
            self.mock_tasks[1], 
            task_id=custom_id,
            dependencies=[task_id]
        )
        self.assertEqual(result_id, custom_id)
        self.assertEqual(len(self.task_chain.tasks), 2)
    
    def test_remove_task_from_chain(self):
        """Test removing tasks from chain"""
        # Add tasks
        task_id1 = self.task_chain.add_task(self.mock_tasks[0])
        task_id2 = self.task_chain.add_task(self.mock_tasks[1])
        
        # Remove task
        success = self.task_chain.remove_task(task_id1)
        self.assertTrue(success)
        self.assertEqual(len(self.task_chain.tasks), 1)
        
        # Try to remove non-existent task
        success = self.task_chain.remove_task("non-existent")
        self.assertFalse(success)
    
    def test_task_dependency_validation(self):
        """Test task dependency validation"""
        # Add tasks with dependencies
        task_id1 = self.task_chain.add_task(self.mock_tasks[0])
        task_id2 = self.task_chain.add_task(self.mock_tasks[1], dependencies=[task_id1])
        task_id3 = self.task_chain.add_task(self.mock_tasks[2], dependencies=[task_id2])
        
        # Validate dependencies
        is_valid = self.task_chain.validate_dependencies()
        self.assertTrue(is_valid)
        
        # Test circular dependency detection
        task_wrapper = self.task_chain.get_task(task_id1)
        task_wrapper.add_dependencies([task_id3])  # Creates circular dependency
        
        is_valid = self.task_chain.validate_dependencies()
        self.assertFalse(is_valid)
    
    def test_task_execution_order(self):
        """Test task execution order calculation"""
        # Add tasks with dependencies
        task_id1 = self.task_chain.add_task(self.mock_tasks[0])
        task_id2 = self.task_chain.add_task(self.mock_tasks[1], dependencies=[task_id1])
        task_id3 = self.task_chain.add_task(self.mock_tasks[2], dependencies=[task_id1])
        
        # Get execution order
        execution_order = self.task_chain.get_execution_order()
        
        self.assertEqual(len(execution_order), 3)
        self.assertEqual(execution_order[0], task_id1)  # First task has no dependencies
        self.assertIn(task_id2, execution_order[1:])    # Other tasks come after
        self.assertIn(task_id3, execution_order[1:])
    
    def test_chain_execution(self):
        """Test chain execution"""
        # Add tasks
        task_id1 = self.task_chain.add_task(self.mock_tasks[0])
        task_id2 = self.task_chain.add_task(self.mock_tasks[1], dependencies=[task_id1])
        
        # Execute chain
        results = self.task_chain.execute()
        
        self.assertIsInstance(results, dict)
        self.assertIn(task_id1, results)
        self.assertIn(task_id2, results)
        
        # Verify tasks were executed
        self.mock_tasks[0].execute.assert_called_once()
        self.mock_tasks[1].execute.assert_called_once()
    
    def test_parallel_execution_capability(self):
        """Test parallel execution of independent tasks"""
        # Add independent tasks
        task_id1 = self.task_chain.add_task(self.mock_tasks[0])
        task_id2 = self.task_chain.add_task(self.mock_tasks[1])  # No dependencies
        task_id3 = self.task_chain.add_task(self.mock_tasks[2])  # No dependencies
        
        # Get parallel execution groups
        parallel_groups = self.task_chain.get_parallel_groups()
        
        self.assertIsInstance(parallel_groups, list)
        self.assertTrue(len(parallel_groups) > 0)
        
        # First group should contain all independent tasks
        first_group = parallel_groups[0]
        self.assertIn(task_id1, first_group)
        self.assertIn(task_id2, first_group)
        self.assertIn(task_id3, first_group)
    
    def test_chain_metrics(self):
        """Test chain metrics collection"""
        # Add and execute tasks
        task_id1 = self.task_chain.add_task(self.mock_tasks[0])
        task_id2 = self.task_chain.add_task(self.mock_tasks[1], dependencies=[task_id1])
        
        self.task_chain.execute()
        
        # Get metrics
        metrics = self.task_chain.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_tasks", metrics)
        self.assertIn("executed_tasks", metrics)
        self.assertIn("execution_time", metrics)


# Pytest-style tests using fixtures
class TestTaskWrapperWithFixtures:
    """Test TaskWrapper using pytest fixtures"""
    
    def test_task_wrapper_with_mock(self, mock_task, shared_state):
        """Test TaskWrapper with pytest fixtures"""
        wrapper = TaskWrapper(
            task=mock_task,
            state=shared_state,
            task_id="fixture-test-task"
        )
        
        assert wrapper is not None
        assert wrapper.task_id == "fixture-test-task"
        assert wrapper.task == mock_task
        assert wrapper.state == shared_state
    
    def test_task_execution_with_fixtures(self, task_wrapper):
        """Test task execution using fixtures"""
        # Mock the task's execute method
        task_wrapper.task.execute = Mock(return_value="fixture result")
        
        result = task_wrapper.execute({"input": "test"})
        assert result == "fixture result"
        task_wrapper.task.execute.assert_called_once()
    
    def test_task_dependency_with_fixtures(self, task_wrapper):
        """Test task dependencies using fixtures"""
        dependencies = ["dep1", "dep2", "dep3"]
        task_wrapper.add_dependencies(dependencies)
        
        retrieved_deps = task_wrapper.get_dependencies()
        assert len(retrieved_deps) == 3
        assert "dep1" in retrieved_deps
        assert "dep2" in retrieved_deps
        assert "dep3" in retrieved_deps
    
    @pytest.mark.integration
    def test_task_state_integration(self, task_wrapper):
        """Test task state integration"""
        # Set task status and verify it's stored in shared state
        task_wrapper.set_status("testing")
        
        # Retrieve from shared state directly
        task_state = task_wrapper.state.get_task_state(task_wrapper.task_id)
        assert task_state is not None
        assert task_state.get("status") == "testing"


if __name__ == "__main__":
    unittest.main()