"""
Test script for AI-Powered Workflow Optimization features

Tests the ML optimizer, reinforcement learning scheduler, performance predictor,
resource manager, and auto-scaler components.

Author: Vatsal216
Created: 2025-07-23 17:45:00 UTC
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from crewgraph_ai.optimization import (
        MLOptimizer,
        QLearningScheduler, 
        PerformancePredictor,
        ResourceManager,
        AutoScaler,
        PerformanceFeatures,
        RLState,
        RLAction,
        HistoricalExecution,
        ResourceType,
        CostOptimizationSettings
    )
    print("✅ Successfully imported optimization modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_ml_optimizer():
    """Test ML-based optimization engine."""
    print("\n🤖 Testing ML Optimizer...")
    
    # Initialize ML optimizer
    optimizer = MLOptimizer()
    
    # Create sample workflow features
    features = PerformanceFeatures(
        task_count=8,
        dependency_count=5,
        avg_task_complexity=2.5,
        parallel_potential=0.6,
        resource_intensity=0.7,
        input_data_size=100.0,
        historical_avg_time=45.0,
        time_of_day=14.5,
        day_of_week=2,
        system_load=0.6
    )
    
    # Test performance prediction
    prediction = optimizer.predict_performance(features)
    print(f"  Performance prediction: {prediction.predicted_value:.2f}s")
    print(f"  Confidence: {prediction.confidence_score:.2f}")
    print(f"  Model used: {prediction.model_used}")
    
    # Test optimization recommendations
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
    
    recommendations = optimizer.recommend_optimizations(workflow_def)
    print(f"  Generated {len(recommendations)} optimization recommendations")
    
    for i, rec in enumerate(recommendations[:2]):
        print(f"    {i+1}. {rec.optimization_type}: {rec.expected_improvement:.1f}% improvement")
    
    # Test model stats
    stats = optimizer.get_model_stats()
    print(f"  ML available: {stats['ml_available']}")
    print(f"  Training samples: {stats['training_samples']}")
    
    print("✅ ML Optimizer test completed")


def test_rl_scheduler():
    """Test reinforcement learning scheduler."""
    print("\n🎯 Testing RL Scheduler...")
    
    # Initialize RL scheduler
    scheduler = QLearningScheduler(
        learning_rate=0.1,
        exploration_rate=0.2
    )
    
    # Create sample RL state
    state = RLState(
        pending_tasks=["task1", "task2", "task3"],
        running_tasks=["task4"],
        completed_tasks=["task0"],
        available_resources={"cpu": 0.6, "memory": 0.8},
        current_time=100.0,
        workflow_progress=0.3,
        resource_utilization=0.65
    )
    
    # Create sample actions
    actions = [
        RLAction(action_type="schedule", task_id="task1"),
        RLAction(action_type="delay"),
        RLAction(action_type="parallel"),
        RLAction(action_type="resource_adjust", resource_allocation={"cpu": 0.8})
    ]
    
    # Test action selection
    decision = scheduler.select_action(state, actions, training_mode=True)
    print(f"  Selected action: {decision.recommended_action.action_type}")
    print(f"  Confidence: {decision.confidence_score:.2f}")
    print(f"  Expected reward: {decision.expected_reward:.3f}")
    print(f"  Reasoning: {decision.reasoning}")
    
    # Test reward calculation
    execution_metrics = {
        "execution_time": 25.0,
        "cpu_usage": 0.75,
        "memory_usage": 0.80
    }
    
    reward = scheduler.calculate_reward(state, decision.recommended_action, state, execution_metrics)
    print(f"  Calculated reward: {reward.total_reward:.3f}")
    
    # Update with reward
    scheduler.update_reward(reward, is_terminal=True)
    
    # Get learning stats
    stats = scheduler.get_learning_stats()
    print(f"  Episodes completed: {stats['episodes_completed']}")
    print(f"  Q-table size: {stats['q_table_size']}")
    
    print("✅ RL Scheduler test completed")


def test_performance_predictor():
    """Test performance prediction system."""
    print("\n📊 Testing Performance Predictor...")
    
    # Initialize predictor
    predictor = PerformancePredictor()
    
    # Create sample workflow
    workflow_def = {
        "tasks": [
            {"id": "task1", "type": "data_processing", "estimated_duration": 15},
            {"id": "task2", "type": "compute_intensive", "estimated_duration": 30},
            {"id": "task3", "type": "io_bound", "estimated_duration": 10}
        ],
        "dependencies": [
            {"source": "task1", "target": "task2"}
        ]
    }
    
    # Test performance prediction
    prediction = predictor.predict_performance(workflow_def)
    print(f"  Predicted execution time: {prediction.predicted_execution_time:.2f}s")
    print(f"  Prediction accuracy: {prediction.prediction_accuracy:.2f}")
    print(f"  Model used: {prediction.model_used}")
    
    cpu_usage = prediction.predicted_resource_usage.get("cpu", 0.5)
    memory_usage = prediction.predicted_resource_usage.get("memory", 0.5)
    print(f"  Predicted CPU usage: {cpu_usage:.2f}")
    print(f"  Predicted memory usage: {memory_usage:.2f}")
    
    # Test bottleneck prediction
    bottlenecks = predictor.predict_bottlenecks(workflow_def, prediction)
    print(f"  Identified {len(bottlenecks)} potential bottlenecks")
    
    for bottleneck in bottlenecks[:2]:
        print(f"    - {bottleneck.bottleneck_type}: {bottleneck.severity:.2f} severity")
    
    # Test with historical data
    historical_data = HistoricalExecution(
        workflow_id="test_workflow",
        execution_id="exec_001",
        execution_time=28.5,
        resource_usage={"cpu": 0.72, "memory": 0.65},
        task_count=3,
        dependency_count=1,
        input_size=50.0,
        timestamp=datetime.now(),
        success=True,
        error_type=None,
        metadata={}
    )
    
    predictor.record_execution(historical_data)
    
    # Validate prediction
    accuracy = predictor.validate_prediction(prediction, historical_data)
    print(f"  Validation accuracy: {accuracy:.2f}")
    
    # Get stats
    stats = predictor.get_performance_stats()
    print(f"  History size: {stats['execution_history_size']}")
    print(f"  Models available: {stats['models_available']}")
    
    print("✅ Performance Predictor test completed")


def test_resource_manager():
    """Test intelligent resource manager."""
    print("\n💾 Testing Resource Manager...")
    
    # Initialize resource manager
    manager = ResourceManager(
        max_cpu_cores=8,
        max_memory_gb=16.0,
        cost_per_cpu_hour=0.10,
        cost_per_gb_memory_hour=0.02
    )
    
    # Test resource allocation
    requirements = {
        "cpu": 2.0,
        "memory": 4.0,
        "storage": 10.0
    }
    
    allocation = manager.allocate_resources(
        workflow_id="test_workflow",
        resource_requirements=requirements,
        priority=7
    )
    
    print(f"  Allocated CPU: {allocation.allocated_cpu:.2f}")
    print(f"  Allocated memory: {allocation.allocated_memory:.2f}GB")
    print(f"  Priority: {allocation.priority}")
    
    # Test current metrics
    metrics = manager.get_current_metrics()
    print(f"  Current CPU usage: {metrics.cpu_usage:.2%}")
    print(f"  Current memory usage: {metrics.memory_usage:.2%}")
    
    # Test scaling analysis
    scaling_recommendations = manager.analyze_scaling_needs()
    print(f"  Scaling recommendations: {len(scaling_recommendations)}")
    
    for rec in scaling_recommendations[:2]:
        print(f"    - {rec.resource_type.value}: {rec.scaling_direction.value}")
        print(f"      Confidence: {rec.confidence:.2f}")
    
    # Test usage prediction
    forecast = manager.predict_future_usage(horizon_minutes=30)
    print(f"  30-min forecast accuracy: {forecast.forecast_accuracy:.2f}")
    
    if forecast.peak_usage_time:
        print(f"  Peak usage predicted at: {forecast.peak_usage_time.strftime('%H:%M')}")
    
    # Test cost optimization
    cost_plan = manager.optimize_cost(target_cost_reduction=0.20)
    print(f"  Current cost: ${cost_plan['current_hourly_cost']:.3f}/hour")
    print(f"  Potential savings: ${cost_plan['potential_savings']:.3f}/hour")
    
    # Release resources
    manager.release_resources("test_workflow")
    
    # Get stats
    stats = manager.get_resource_stats()
    print(f"  Active allocations: {stats['active_allocations']}")
    
    print("✅ Resource Manager test completed")


def test_auto_scaler():
    """Test auto-scaling system."""
    print("\n⚡ Testing Auto Scaler...")
    
    # Initialize resource manager for auto-scaler
    resource_manager = ResourceManager()
    
    # Initialize auto-scaler with cost settings
    cost_settings = CostOptimizationSettings(
        max_hourly_cost=5.0,
        cost_efficiency_target=0.08,
        enable_spot_instances=True,
        enable_time_shifting=True,
        off_peak_hours=[0, 1, 2, 3, 22, 23],
        weekend_discount_factor=0.8
    )
    
    auto_scaler = AutoScaler(resource_manager, cost_settings)
    
    # Test scaling predictions
    predictions = auto_scaler.predict_scaling_needs(horizon_minutes=60)
    print(f"  Scaling predictions: {len(predictions)}")
    
    for pred in predictions[:2]:
        print(f"    - {pred['resource_type']}: {pred['recommended_action']}")
        print(f"      Confidence: {pred['confidence']:.2f}")
    
    # Test cost vs performance analysis
    analysis = auto_scaler.analyze_cost_vs_performance()
    
    if analysis["analysis_available"]:
        scaling_summary = analysis["scaling_summary"]
        print(f"  Total scaling events: {scaling_summary['total_events']}")
        print(f"  Success rate: {scaling_summary['success_rate']:.2f}")
        
        cost_analysis = analysis["cost_analysis"]
        print(f"  Current cost: ${cost_analysis['current_hourly_cost']:.3f}/hour")
        print(f"  Cost efficiency: {cost_analysis['cost_efficiency']:.3f}")
        
        recommendations = analysis["recommendations"]
        print(f"  Recommendations: {len(recommendations)}")
        for rec in recommendations[:2]:
            print(f"    - {rec}")
    else:
        print(f"  Analysis not available: {analysis['reason']}")
    
    # Get scaling metrics
    metrics = auto_scaler.get_scaling_metrics()
    print(f"  Total scaling events: {metrics.total_scaling_events}")
    print(f"  Success rate: {metrics.successful_scaling_events}/{metrics.total_scaling_events}")
    print(f"  Cost savings: {metrics.cost_savings_percentage:.1f}%")
    print(f"  Stability score: {metrics.stability_score:.2f}")
    
    print("✅ Auto Scaler test completed")


def main():
    """Run all optimization tests."""
    print("🚀 Starting AI-Powered Workflow Optimization Tests")
    print("=" * 60)
    
    try:
        test_ml_optimizer()
        test_rl_scheduler()
        test_performance_predictor()
        test_resource_manager()
        test_auto_scaler()
        
        print("\n" + "=" * 60)
        print("🎉 All optimization tests completed successfully!")
        print("\n📋 Summary of implemented features:")
        print("  ✅ ML-based performance optimization engine")
        print("  ✅ Reinforcement learning scheduler")
        print("  ✅ Performance prediction with historical analysis")
        print("  ✅ Intelligent resource management")
        print("  ✅ Auto-scaling with cost optimization")
        print("  ✅ Multi-objective optimization capabilities")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)