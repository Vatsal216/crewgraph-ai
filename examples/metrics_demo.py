"""
CrewGraph AI Built-in Metrics Demonstration
Author: Vatsal216
Date: 2025-07-22 11:25:03 UTC
"""

import sys
import os
import time
import json
from crewai import Agent, Tool

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crewgraph_ai import CrewGraph
from crewgraph_ai.utils.metrics import get_metrics_collector

def research_tool(query: str) -> str:
    """Research simulation with metrics tracking"""
    start_time = time.time()
    
    # Simulate different execution times
    import random
    execution_time = random.uniform(0.5, 2.0)
    time.sleep(execution_time)
    
    result = f"📊 Research completed for '{query}' in {execution_time:.2f}s"
    
    # Manual tool metrics (optional - framework handles most)
    metrics = get_metrics_collector()
    metrics.record_duration("tool_execution_research", execution_time)
    metrics.increment_counter("tool_calls_total", labels={"tool": "research"})
    
    return result

def analysis_tool(data: str) -> str:
    """Analysis simulation with metrics"""
    start_time = time.time()
    
    # Simulate processing
    time.sleep(1.0)
    
    result = f"🔍 Analysis completed: {len(data)} characters processed"
    
    # Track analysis metrics
    metrics = get_metrics_collector()
    metrics.record_duration("tool_execution_analysis", 1.0)
    metrics.record_gauge("data_size_processed", len(data))
    
    return result

def main():
    """Comprehensive metrics demonstration"""
    print("🎉 CrewGraph AI - Built-in Metrics Demo")
    print("📅 Created by: Vatsal216")
    print("⏰ Date: 2025-07-22 11:25:03 UTC")
    print("="*60)
    
    # Get global metrics collector
    metrics = get_metrics_collector()
    
    # Create agents with tools
    researcher = Agent(
        role='Research Analyst',
        goal='Conduct thorough research with metrics tracking',
        backstory='Expert researcher with built-in performance monitoring',
        tools=[Tool(name="research", func=research_tool, description="Research with metrics")],
        verbose=True
    )
    
    analyst = Agent(
        role='Data Analyst', 
        goal='Analyze data with comprehensive metrics',
        backstory='Data analyst with automatic performance tracking',
        tools=[Tool(name="analyze", func=analysis_tool, description="Analysis with metrics")],
        verbose=True
    )
    
    # Create CrewGraph with built-in metrics
    print("\n🏗️  Creating CrewGraph with Built-in Metrics...")
    workflow = CrewGraph("metrics_demo_workflow")
    
    # Add components (automatically tracked)
    workflow.add_agent(researcher, "researcher")
    workflow.add_agent(analyst, "analyst")
    
    workflow.add_task(
        name="research_task",
        description="Research AI metrics and monitoring",
        agent="researcher"
    )
    
    workflow.add_task(
        name="analysis_task", 
        description="Analyze research findings",
        agent="analyst"
    )
    
    # Show initial metrics
    print("\n📊 Initial Metrics:")
    print("-" * 40)
    initial_metrics = workflow.get_metrics()
    print(f"Agents created: {initial_metrics['component_counts']['agents']}")
    print(f"Tasks created: {initial_metrics['component_counts']['tasks']}")
    print(f"Total executions: {initial_metrics['execution_metrics']['total_executions']}")
    
    # Execute workflow multiple times to show metrics
    print("\n🚀 Executing Workflow Multiple Times...")
    
    for i in range(3):
        print(f"\n--- Execution {i+1}/3 ---")
        
        result = workflow.execute({
            "research_topic": f"AI Metrics Study #{i+1}",
            "analysis_depth": "comprehensive",
            "iteration": i+1
        })
        
        print(f"✅ Execution {i+1} Status: {result['status']}")
        print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")
        
        # Show live metrics after each execution
        current_metrics = workflow.get_metrics()
        print(f"📈 Success Rate: {current_metrics['execution_metrics']['success_rate']:.2%}")
        print(f"📊 Avg Execution Time: {current_metrics['execution_metrics']['average_execution_time']:.2f}s")
    
    # Comprehensive metrics report
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE METRICS REPORT")
    print("="*60)
    
    final_metrics = workflow.get_metrics()
    
    # Workflow Info
    print("\n🏷️  Workflow Information:")
    print(f"   Name: {final_metrics['workflow_info']['name']}")
    print(f"   Created by: {final_metrics['workflow_info']['created_by']}")
    print(f"   Created at: {final_metrics['workflow_info']['created_at']}")
    
    # Component Metrics
    print("\n🧩 Component Metrics:")
    components = final_metrics['component_counts']
    print(f"   Agents: {components['agents']}")
    print(f"   Tasks: {components['tasks']}")
    print(f"   Tools: {components['tools']}")
    
    # Execution Metrics
    print("\n📈 Execution Metrics:")
    exec_metrics = final_metrics['execution_metrics']
    print(f"   Total Executions: {exec_metrics['total_executions']}")
    print(f"   Successful: {exec_metrics['successful_executions']}")
    print(f"   Failed: {exec_metrics['failed_executions']}")
    print(f"   Success Rate: {exec_metrics['success_rate']:.2%}")
    print(f"   Average Time: {exec_metrics['average_execution_time']:.2f}s")
    
    # System Metrics
    print("\n🖥️  System Metrics:")
    system_metrics = final_metrics['system_metrics']
    print(f"   Total Metrics Tracked: {system_metrics['total_metrics']}")
    print(f"   Total Samples: {system_metrics['total_samples']}")
    
    # Performance Metrics
    print("\n⚡ Performance Metrics:")
    perf_metrics = final_metrics['performance_metrics']
    print(f"   Active Operations: {perf_metrics['active_operations']}")
    
    # Export Prometheus metrics
    print("\n📤 Prometheus Metrics Export:")
    print("-" * 40)
    prometheus_metrics = workflow.get_prometheus_metrics()
    print(prometheus_metrics[:500] + "..." if len(prometheus_metrics) > 500 else prometheus_metrics)
    
    # Global metrics summary
    print("\n🌍 Global Metrics Summary:")
    print("-" * 40)
    global_metrics = metrics.get_summary()
    for key, value in global_metrics.items():
        print(f"   {key}: {value}")
    
    # Save metrics to file
    print("\n💾 Saving Metrics to File...")
    metrics_file = "crewgraph_metrics_report.json"
    with open(metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=2, default=str)
    print(f"   Metrics saved to: {metrics_file}")
    
    # Show how to access specific metrics
    print("\n🔍 Accessing Specific Metrics:")
    print("-" * 40)
    print("# Get workflow execution count:")
    print(f"workflow.get_metrics()['execution_metrics']['total_executions'] = {exec_metrics['total_executions']}")
    
    print("\n# Get success rate:")
    print(f"workflow.get_metrics()['execution_metrics']['success_rate'] = {exec_metrics['success_rate']:.2%}")
    
    print("\n# Get Prometheus metrics:")
    print("workflow.get_prometheus_metrics()")
    
    print("\n# Get live system metrics:")
    print("from crewgraph_ai.utils.metrics import get_metrics_collector")
    print("metrics = get_metrics_collector()")
    print("metrics.get_all_metrics()")
    
    print("\n" + "="*60)
    print("🎉 Built-in Metrics Demo Completed Successfully!")
    print("📊 All operations were automatically tracked and monitored")
    print("🔗 Repository: https://github.com/Vatsal216/crewgraph-ai")
    print("📅 Created by: Vatsal216 on 2025-07-22 11:25:03 UTC")
    print("="*60)

if __name__ == "__main__":
    main()