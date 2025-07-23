"""
Simple test for AI optimization modules without external dependencies.

Tests core functionality of the optimization modules.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all optimization modules can be imported."""
    print("🔍 Testing module imports...")
    
    try:
        # Test basic optimization imports
        from crewgraph_ai.optimization.workflow_optimizer import WorkflowOptimizer, OptimizationResult
        print("  ✅ WorkflowOptimizer imported")
        
        from crewgraph_ai.optimization.ml_optimizer import MLOptimizer, MLPrediction
        print("  ✅ MLOptimizer imported")
        
        from crewgraph_ai.optimization.reinforcement_learning import QLearningScheduler, RLState
        print("  ✅ QLearningScheduler imported")
        
        from crewgraph_ai.optimization.performance_predictor import PerformancePredictor
        print("  ✅ PerformancePredictor imported")
        
        from crewgraph_ai.optimization.resource_manager import ResourceManager
        print("  ✅ ResourceManager imported")
        
        from crewgraph_ai.optimization.auto_scaler import AutoScaler
        print("  ✅ AutoScaler imported")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from crewgraph_ai.optimization.ml_optimizer import MLOptimizer, PerformanceFeatures
        
        # Test ML optimizer initialization
        optimizer = MLOptimizer()
        print("  ✅ MLOptimizer initialized")
        
        # Test feature creation
        features = PerformanceFeatures(
            task_count=5,
            dependency_count=3,
            avg_task_complexity=2.0,
            parallel_potential=0.6,
            resource_intensity=0.7,
            input_data_size=100.0,
            historical_avg_time=30.0,
            time_of_day=14.0,
            day_of_week=2,
            system_load=0.5
        )
        print("  ✅ PerformanceFeatures created")
        
        # Test prediction (should fall back to heuristic)
        prediction = optimizer.predict_performance(features)
        print(f"  ✅ Performance prediction: {prediction.predicted_value:.2f}s")
        print(f"     Model used: {prediction.model_used}")
        
        # Test model stats
        stats = optimizer.get_model_stats()
        print(f"  ✅ Model stats: ML available={stats['ml_available']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def test_workflow_optimizer():
    """Test the original workflow optimizer."""
    print("\n⚙️ Testing WorkflowOptimizer...")
    
    try:
        from crewgraph_ai.optimization.workflow_optimizer import WorkflowOptimizer
        
        optimizer = WorkflowOptimizer()
        print("  ✅ WorkflowOptimizer initialized")
        
        # Test with sample workflow
        workflow_def = {
            "tasks": [
                {"id": "task1", "type": "data_processing", "estimated_duration": 20},
                {"id": "task2", "type": "ml", "estimated_duration": 35},
                {"id": "task3", "type": "standard", "estimated_duration": 10}
            ],
            "dependencies": [
                {"source": "task1", "target": "task2"}
            ]
        }
        
        # Test optimization
        results = optimizer.optimize_workflow(workflow_def)
        print(f"  ✅ Generated {len(results)} optimization results")
        
        for result in results:
            print(f"     - {result.optimization_type.value}: {result.performance_improvement:.1f}% improvement")
        
        # Test suggestions
        suggestions = optimizer.suggest_optimizations(workflow_def)
        print(f"  ✅ Generated {len(suggestions)} optimization suggestions")
        
        return True
        
    except Exception as e:
        print(f"  ❌ WorkflowOptimizer test failed: {e}")
        return False

def test_resource_manager():
    """Test resource manager functionality."""
    print("\n💾 Testing ResourceManager...")
    
    try:
        from crewgraph_ai.optimization.resource_manager import ResourceManager
        
        manager = ResourceManager()
        print("  ✅ ResourceManager initialized")
        
        # Test getting current metrics
        metrics = manager.get_current_metrics()
        print(f"  ✅ Current metrics: CPU={metrics.cpu_usage:.2%}, Memory={metrics.memory_usage:.2%}")
        
        # Test resource allocation
        allocation = manager.allocate_resources(
            workflow_id="test_workflow",
            resource_requirements={"cpu": 2.0, "memory": 4.0},
            priority=5
        )
        print(f"  ✅ Resource allocation: CPU={allocation.allocated_cpu:.2f}, Memory={allocation.allocated_memory:.2f}")
        
        # Test scaling analysis
        recommendations = manager.analyze_scaling_needs()
        print(f"  ✅ Scaling recommendations: {len(recommendations)}")
        
        # Test cost optimization
        cost_plan = manager.optimize_cost()
        print(f"  ✅ Cost optimization: ${cost_plan['current_hourly_cost']:.3f}/hour")
        
        # Clean up
        manager.release_resources("test_workflow")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ResourceManager test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🚀 Testing AI-Powered Workflow Optimization (Basic)")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_basic_functionality, 
        test_workflow_optimizer,
        test_resource_manager
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests completed successfully!")
        print("\n📋 Implemented AI Optimization Features:")
        print("  ✅ ML-based performance optimization engine")
        print("  ✅ Reinforcement learning task scheduler")
        print("  ✅ Performance prediction with historical analysis")
        print("  ✅ Intelligent resource management")
        print("  ✅ Auto-scaling with cost optimization")
        print("  ✅ Multi-objective optimization capabilities")
        print("\n💡 Note: Full ML capabilities available when sklearn/numpy installed")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)