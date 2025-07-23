"""
Unit tests for SharedState and StateManager classes

Author: Vatsal216
Created: 2025-07-23 06:14:25 UTC
"""

import pytest
import unittest
from unittest.mock import Mock, patch
from typing import Dict, Any

from crewgraph_ai.core.state import SharedState, StateManager
from crewgraph_ai.memory import DictMemory


class TestSharedState(unittest.TestCase):
    """Test SharedState functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.memory = DictMemory(persistent=False)
        self.memory.connect()
        self.shared_state = SharedState(memory_backend=self.memory)
    
    def tearDown(self):
        """Cleanup test environment"""
        try:
            self.shared_state.clear()
            self.memory.disconnect()
        except Exception:
            pass
    
    def test_shared_state_initialization(self):
        """Test SharedState initialization"""
        self.assertIsNotNone(self.shared_state)
        self.assertEqual(self.shared_state.memory_backend, self.memory)
        self.assertIsNotNone(self.shared_state.metrics)
    
    def test_basic_state_operations(self):
        """Test basic state operations"""
        # Set state
        self.shared_state.set("test_key", "test_value")
        
        # Get state
        value = self.shared_state.get("test_key")
        self.assertEqual(value, "test_value")
        
        # Update state
        self.shared_state.update("test_key", "updated_value")
        updated_value = self.shared_state.get("test_key")
        self.assertEqual(updated_value, "updated_value")
        
        # Delete state
        self.shared_state.delete("test_key")
        deleted_value = self.shared_state.get("test_key")
        self.assertIsNone(deleted_value)
    
    def test_nested_state_operations(self):
        """Test nested state operations"""
        # Set nested state
        nested_data = {
            "user": {
                "name": "John Doe",
                "preferences": {
                    "theme": "dark",
                    "language": "en"
                }
            }
        }
        
        self.shared_state.set("user_data", nested_data)
        
        # Get nested value
        user_name = self.shared_state.get_nested("user_data.user.name")
        self.assertEqual(user_name, "John Doe")
        
        # Update nested value
        self.shared_state.set_nested("user_data.user.preferences.theme", "light")
        theme = self.shared_state.get_nested("user_data.user.preferences.theme")
        self.assertEqual(theme, "light")
    
    def test_agent_state_management(self):
        """Test agent-specific state management"""
        agent_id = "agent-001"
        agent_state = {
            "status": "active",
            "current_task": "task-001",
            "metadata": {"role": "researcher"}
        }
        
        # Set agent state
        self.shared_state.set_agent_state(agent_id, agent_state)
        
        # Get agent state
        retrieved_state = self.shared_state.get_agent_state(agent_id)
        self.assertEqual(retrieved_state["status"], "active")
        self.assertEqual(retrieved_state["current_task"], "task-001")
        
        # Update agent status
        self.shared_state.update_agent_status(agent_id, "idle")
        updated_state = self.shared_state.get_agent_state(agent_id)
        self.assertEqual(updated_state["status"], "idle")
    
    def test_task_state_management(self):
        """Test task-specific state management"""
        task_id = "task-001"
        task_state = {
            "status": "pending",
            "assigned_agent": "agent-001",
            "progress": 0.0,
            "metadata": {"priority": "high"}
        }
        
        # Set task state
        self.shared_state.set_task_state(task_id, task_state)
        
        # Get task state
        retrieved_state = self.shared_state.get_task_state(task_id)
        self.assertEqual(retrieved_state["status"], "pending")
        self.assertEqual(retrieved_state["assigned_agent"], "agent-001")
        
        # Update task progress
        self.shared_state.update_task_progress(task_id, 0.5)
        updated_state = self.shared_state.get_task_state(task_id)
        self.assertEqual(updated_state["progress"], 0.5)
    
    def test_workflow_state_management(self):
        """Test workflow-specific state management"""
        workflow_id = "workflow-001"
        workflow_state = {
            "status": "running",
            "current_step": 2,
            "total_steps": 5,
            "start_time": "2025-07-23T06:14:25Z"
        }
        
        # Set workflow state
        self.shared_state.set_workflow_state(workflow_id, workflow_state)
        
        # Get workflow state
        retrieved_state = self.shared_state.get_workflow_state(workflow_id)
        self.assertEqual(retrieved_state["status"], "running")
        self.assertEqual(retrieved_state["current_step"], 2)
        
        # Update workflow step
        self.shared_state.update_workflow_step(workflow_id, 3)
        updated_state = self.shared_state.get_workflow_state(workflow_id)
        self.assertEqual(updated_state["current_step"], 3)
    
    def test_state_history_tracking(self):
        """Test state history tracking"""
        key = "tracked_value"
        
        # Enable history tracking
        self.shared_state.enable_history(key)
        
        # Make several updates
        values = ["value1", "value2", "value3"]
        for value in values:
            self.shared_state.set(key, value)
        
        # Get history
        history = self.shared_state.get_history(key)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[-1]["value"], "value3")  # Latest value
        self.assertEqual(history[0]["value"], "value1")   # Oldest value
    
    def test_state_subscriptions(self):
        """Test state change subscriptions"""
        callback_called = False
        callback_data = None
        
        def state_callback(key, old_value, new_value):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = {"key": key, "old": old_value, "new": new_value}
        
        # Subscribe to state changes
        subscription_id = self.shared_state.subscribe("monitored_key", state_callback)
        self.assertIsNotNone(subscription_id)
        
        # Update monitored key
        self.shared_state.set("monitored_key", "initial_value")
        self.shared_state.set("monitored_key", "updated_value")
        
        # Verify callback was called
        self.assertTrue(callback_called)
        self.assertEqual(callback_data["key"], "monitored_key")
        self.assertEqual(callback_data["old"], "initial_value")
        self.assertEqual(callback_data["new"], "updated_value")
        
        # Unsubscribe
        success = self.shared_state.unsubscribe(subscription_id)
        self.assertTrue(success)
    
    def test_state_serialization(self):
        """Test state serialization and deserialization"""
        # Set various types of data
        test_data = {
            "string": "test",
            "number": 42,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        
        for key, value in test_data.items():
            self.shared_state.set(key, value)
        
        # Serialize state
        serialized = self.shared_state.serialize()
        self.assertIsInstance(serialized, (str, bytes))
        
        # Clear state
        self.shared_state.clear()
        
        # Deserialize state
        self.shared_state.deserialize(serialized)
        
        # Verify data is restored
        for key, expected_value in test_data.items():
            actual_value = self.shared_state.get(key)
            self.assertEqual(actual_value, expected_value)
    
    def test_concurrent_access(self):
        """Test concurrent access to shared state"""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    key = f"worker_{worker_id}_item_{i}"
                    value = f"value_{worker_id}_{i}"
                    self.shared_state.set(key, value)
                    retrieved = self.shared_state.get(key)
                    results.append((worker_id, i, retrieved == value))
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Create multiple threads
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        
        # Verify all operations succeeded
        successful_operations = sum(1 for _, _, success in results if success)
        self.assertEqual(successful_operations, 50)  # 5 workers * 10 operations


class TestStateManager(unittest.TestCase):
    """Test StateManager functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.memory = DictMemory(persistent=False)
        self.memory.connect()
        self.state_manager = StateManager(memory_backend=self.memory)
    
    def tearDown(self):
        """Cleanup test environment"""
        try:
            self.state_manager.clear_all()
            self.memory.disconnect()
        except Exception:
            pass
    
    def test_state_manager_initialization(self):
        """Test StateManager initialization"""
        self.assertIsNotNone(self.state_manager)
        self.assertEqual(self.state_manager.memory_backend, self.memory)
    
    def test_multiple_state_contexts(self):
        """Test managing multiple state contexts"""
        # Create different state contexts
        contexts = ["workflow1", "workflow2", "global"]
        
        for context in contexts:
            state = self.state_manager.get_state(context)
            self.assertIsNotNone(state)
            
            # Set context-specific data
            state.set("context_id", context)
            state.set("data", f"data_for_{context}")
        
        # Verify isolation between contexts
        for context in contexts:
            state = self.state_manager.get_state(context)
            self.assertEqual(state.get("context_id"), context)
            self.assertEqual(state.get("data"), f"data_for_{context}")
    
    def test_state_context_switching(self):
        """Test state context switching"""
        # Setup contexts
        context1 = "context1"
        context2 = "context2"
        
        # Set data in first context
        self.state_manager.switch_context(context1)
        current_state = self.state_manager.get_current_state()
        current_state.set("active_context", context1)
        
        # Switch to second context
        self.state_manager.switch_context(context2)
        current_state = self.state_manager.get_current_state()
        current_state.set("active_context", context2)
        
        # Verify context isolation
        state1 = self.state_manager.get_state(context1)
        state2 = self.state_manager.get_state(context2)
        
        self.assertEqual(state1.get("active_context"), context1)
        self.assertEqual(state2.get("active_context"), context2)
    
    def test_state_sharing_between_contexts(self):
        """Test state sharing between contexts"""
        context1 = "shared_context1"
        context2 = "shared_context2"
        
        # Set shared data
        shared_key = "shared_data"
        shared_value = "shared_value"
        
        self.state_manager.set_shared(shared_key, shared_value)
        
        # Access shared data from different contexts
        state1 = self.state_manager.get_state(context1)
        state2 = self.state_manager.get_state(context2)
        
        value1 = self.state_manager.get_shared(shared_key, context1)
        value2 = self.state_manager.get_shared(shared_key, context2)
        
        self.assertEqual(value1, shared_value)
        self.assertEqual(value2, shared_value)
    
    def test_state_persistence(self):
        """Test state persistence across manager instances"""
        context = "persistent_context"
        test_data = {"persistent": True, "value": 123}
        
        # Set data in first manager instance
        state = self.state_manager.get_state(context)
        state.set("test_data", test_data)
        
        # Create new manager instance with same memory backend
        new_manager = StateManager(memory_backend=self.memory)
        
        # Retrieve data from new manager
        new_state = new_manager.get_state(context)
        retrieved_data = new_state.get("test_data")
        
        self.assertEqual(retrieved_data, test_data)
    
    def test_state_cleanup(self):
        """Test state cleanup operations"""
        # Create multiple contexts with data
        contexts = ["cleanup1", "cleanup2", "cleanup3"]
        
        for context in contexts:
            state = self.state_manager.get_state(context)
            state.set("temp_data", f"data_for_{context}")
        
        # Verify contexts exist
        active_contexts = self.state_manager.list_contexts()
        for context in contexts:
            self.assertIn(context, active_contexts)
        
        # Cleanup specific context
        success = self.state_manager.cleanup_context("cleanup1")
        self.assertTrue(success)
        
        # Verify context was removed
        active_contexts = self.state_manager.list_contexts()
        self.assertNotIn("cleanup1", active_contexts)
        self.assertIn("cleanup2", active_contexts)
        self.assertIn("cleanup3", active_contexts)
        
        # Cleanup all contexts
        self.state_manager.clear_all()
        active_contexts = self.state_manager.list_contexts()
        self.assertEqual(len(active_contexts), 0)


# Pytest-style tests using fixtures
class TestSharedStateWithFixtures:
    """Test SharedState using pytest fixtures"""
    
    def test_shared_state_with_fixture(self, shared_state):
        """Test SharedState with pytest fixture"""
        assert shared_state is not None
        
        # Test basic operations
        shared_state.set("fixture_key", "fixture_value")
        value = shared_state.get("fixture_key")
        assert value == "fixture_value"
    
    def test_agent_state_with_fixtures(self, shared_state, mock_agent):
        """Test agent state management with fixtures"""
        agent_id = "fixture-agent-001"
        
        # Set agent state
        agent_state = {
            "status": "active",
            "role": mock_agent.role,
            "current_task": None
        }
        
        shared_state.set_agent_state(agent_id, agent_state)
        
        # Retrieve and verify
        retrieved = shared_state.get_agent_state(agent_id)
        assert retrieved["status"] == "active"
        assert retrieved["role"] == mock_agent.role
    
    def test_task_state_with_fixtures(self, shared_state, mock_task):
        """Test task state management with fixtures"""
        task_id = "fixture-task-001"
        
        # Set task state
        task_state = {
            "status": "pending",
            "description": mock_task.description,
            "progress": 0.0
        }
        
        shared_state.set_task_state(task_id, task_state)
        
        # Retrieve and verify
        retrieved = shared_state.get_task_state(task_id)
        assert retrieved["status"] == "pending"
        assert retrieved["description"] == mock_task.description
        assert retrieved["progress"] == 0.0
    
    @pytest.mark.integration
    def test_state_integration_workflow(self, shared_state, sample_workflow_data):
        """Test state management in workflow context"""
        workflow_data = sample_workflow_data
        workflow_id = "integration-workflow-001"
        
        # Set workflow state
        workflow_state = {
            "status": "initializing",
            "agents": workflow_data["agents"],
            "tasks": workflow_data["tasks"],
            "progress": 0.0
        }
        
        shared_state.set_workflow_state(workflow_id, workflow_state)
        
        # Update progress
        shared_state.update_workflow_step(workflow_id, 1)
        
        # Verify state updates
        updated_state = shared_state.get_workflow_state(workflow_id)
        assert updated_state["current_step"] == 1
        assert len(updated_state["agents"]) == 2
        assert len(updated_state["tasks"]) == 2


if __name__ == "__main__":
    unittest.main()