#!/usr/bin/env python3
"""
Test script to validate core production readiness fixes
"""

import os
import time
import tempfile
from pathlib import Path

def test_config_functions():
    """Test configuration utility functions"""
    print("🧪 Testing configuration functions...")
    
    # Test dynamic user function
    os.environ["CREWGRAPH_SYSTEM_USER"] = "test_user"
    from crewgraph_ai.config import get_current_user, get_formatted_timestamp, get_current_timestamp
    
    user = get_current_user()
    assert user == "test_user", f"Expected 'test_user', got '{user}'"
    print(f"✅ User configuration: {user}")
    
    # Test dynamic timestamps
    time1 = get_formatted_timestamp()
    time.sleep(1)
    time2 = get_formatted_timestamp()
    assert time1 != time2, "Timestamps should be dynamic, not hardcoded"
    print(f"✅ Dynamic timestamps: {time1} → {time2}")
    
    # Test ISO timestamp
    iso_time = get_current_timestamp()
    assert "T" in iso_time, "ISO timestamp should contain 'T'"
    print(f"✅ ISO timestamp: {iso_time}")


def test_memory_backend():
    """Test enhanced memory backend with persistence"""
    print("\n🧪 Testing memory backend...")
    
    # Import memory components from the main package
    from crewgraph_ai.memory.dict_memory import DictMemory
    
    # Test memory with persistence
    with tempfile.TemporaryDirectory() as temp_dir:
        persistence_file = Path(temp_dir) / "test_memory.json"
        
        # Create memory backend
        memory = DictMemory(persistent=True, persistence_file=str(persistence_file))
        memory.connect()
        print("✅ Memory backend connected")
        
        # Test save/load operations
        memory.save("test_key", "test_value")
        memory.save("ttl_key", "ttl_value", ttl=60)  # 60 second TTL
        
        value = memory.load("test_key")
        assert value == "test_value", f"Expected 'test_value', got '{value}'"
        print(f"✅ Memory save/load: {value}")
        
        # Test TTL functionality
        ttl_value = memory.load("ttl_key")
        assert ttl_value == "ttl_value", f"Expected 'ttl_value', got '{ttl_value}'"
        print(f"✅ Memory TTL: {ttl_value}")
        
        # Test health check
        health = memory.get_health()
        assert health["status"] == "healthy", f"Expected 'healthy', got '{health['status']}'"
        print(f"✅ Memory health: {health['status']}")
        
        # Test persistence
        memory.disconnect()
        assert persistence_file.exists(), "Persistence file should be created"
        print("✅ Memory persistence file created")
        
        # Test loading from persistence
        memory2 = DictMemory(persistent=True, persistence_file=str(persistence_file))
        memory2.connect()
        loaded_value = memory2.load("test_key")
        assert loaded_value == "test_value", f"Expected 'test_value', got '{loaded_value}'"
        print("✅ Memory persistence loading")
        
        memory2.disconnect()


def test_agent_wrapper():
    """Test agent wrapper with fallback execution"""
    print("\n🧪 Testing agent wrapper...")
    
    # Import agent components from the main package
    from crewgraph_ai.core.agents import AgentWrapper
    from crewgraph_ai.memory.dict_memory import DictMemory
    
    # Create memory backend
    memory = DictMemory()
    memory.connect()
    
    # Create agent wrapper (should work without crewai)
    agent = AgentWrapper(
        name="test_agent",
        role="Test Agent",
        goal="Execute tests",
        backstory="I am a test agent",
        memory=memory
    )
    
    print(f"✅ Agent created: {agent.name}")
    
    # Test task execution (should use fallback since crewai not available)
    result = agent.execute_task(
        task_name="test_task",
        prompt="Process this test prompt",
        context={"test": True}
    )
    
    assert result["success"] == True, f"Task should succeed, got: {result}"
    print(f"✅ Agent task execution: {result['success']}")
    
    # Test metrics
    metrics = agent.get_metrics()
    assert metrics.tasks_completed == 1, f"Expected 1 task completed, got {metrics.tasks_completed}"
    print(f"✅ Agent metrics: {metrics.tasks_completed} tasks completed")
    
    memory.disconnect()


def main():
    """Run all tests"""
    print("🚀 Testing CrewGraph AI Core Production Fixes")
    print("=" * 50)
    
    try:
        test_config_functions()
        test_memory_backend()
        test_agent_wrapper()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED! Core production fixes are working.")
        print("🎉 The hardcoded values have been successfully replaced with configurable system.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)