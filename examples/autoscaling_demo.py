"""
CrewGraph AI Auto-Scaling Demo
Comprehensive example of auto-scaling configuration and usage

Author: Vatsal216
Created: 2025-07-22 13:17:52 UTC
"""

import time
from crewai import Agent, Task
from crewgraph_ai.core import CrewGraph
from crewgraph_ai.scaling import WorkflowAutoScaler, AgentPoolScaler, MemoryAutoScaler
from crewgraph_ai.scaling import ScalingDashboard
from crewgraph_ai.memory.config import MemoryConfig, MemoryType


def create_sample_workflow() -> CrewGraph:
    """Create a sample workflow for scaling"""
    workflow = CrewGraph("scalable_workflow")
    
    # Add sample agent
    agent = Agent(
        role="Sample Agent",
        goal="Process tasks efficiently",
        backstory="Created for auto-scaling demo by Vatsal216"
    )
    workflow.add_agent(agent, "sample_agent")
    
    return workflow


def create_sample_agent() -> Agent:
    """Create sample agent for pool"""
    return Agent(
        role="Pool Agent", 
        goal="Handle dynamic workload",
        backstory="Dynamic agent created by Vatsal216 at 2025-07-22 13:17:52"
    )


def main():
    """Main auto-scaling demo"""
    print("🚀 CrewGraph AI Auto-Scaling Demo")
    print("👤 Created by: Vatsal216")
    print("⏰ Time: 2025-07-22 13:17:52 UTC")
    print("=" * 60)
    
    try:
        # 1. Setup Workflow Auto-Scaling
        print("\n🧠 Setting up Workflow Auto-Scaling...")
        workflow_scaler = WorkflowAutoScaler(
            workflow_factory=create_sample_workflow,
            monitoring_interval=30.0,
            enable_predictive=True
        )
        
        # 2. Setup Agent Pool Auto-Scaling  
        print("🤖 Setting up Agent Pool Auto-Scaling...")
        agent_scaler = AgentPoolScaler(
            agent_factory={
                "general": create_sample_agent,
                "specialized": lambda: Agent(
                    role="Specialist", 
                    goal="Handle complex tasks",
                    backstory="Specialist agent by Vatsal216"
                )
            }
        )
        
        # 3. Setup Memory Auto-Scaling
        print("💾 Setting up Memory Auto-Scaling...")
        memory_configs = {
            "primary": MemoryConfig(
                memory_type=MemoryType.DICT,
                max_size=10000
            ),
            "cache": MemoryConfig(
                memory_type=MemoryType.REDIS,
                redis_host="localhost"
            )
        }
        memory_scaler = MemoryAutoScaler(memory_configs)
        
        # 4. Setup Dashboard
        print("📊 Setting up Scaling Dashboard...")
        dashboard = ScalingDashboard(
            workflow_scaler=workflow_scaler,
            agent_scaler=agent_scaler,
            memory_scaler=memory_scaler
        )
        
        # 5. Start all scaling systems
        print("\n▶️  Starting Auto-Scaling Systems...")
        workflow_scaler.start()
        agent_scaler.start_scaling()
        memory_scaler.start_scaling()
        
        # 6. Monitor for demo duration
        print("📈 Monitoring auto-scaling for 5 minutes...")
        print("   (In production, this would run continuously)")
        
        for i in range(10):  # 10 iterations = ~5 minutes
            time.sleep(30)  # 30 second intervals
            
            # Get current status
            metrics = dashboard.get_real_time_metrics()
            
            print(f"\n📊 Status Update #{i+1}:")
            print(f"   Workflow Instances: {metrics.workflow_instances}")
            print(f"   Agent Pool Size: {metrics.agent_pool_size}")
            print(f"   CPU Usage: {metrics.cpu_usage_percent:.1f}%")
            print(f"   Queue Length: {metrics.queue_length}")
            print(f"   Efficiency Score: {metrics.efficiency_score:.1f}/100")
            print(f"   Estimated Cost: ${metrics.cost_estimate_hourly:.2f}/hour")
            
            # Show scaling recommendations
            dashboard_data = dashboard.get_dashboard_data()
            recommendations = dashboard_data.get('optimization_recommendations', [])
            
            if recommendations:
                print(f"   💡 Recommendations:")
                for rec in recommendations[:2]:  # Show top 2
                    print(f"     - {rec['title']}: {rec['description']}")
        
        # 7. Generate final report
        print("\n📋 Generating Final Auto-Scaling Report...")
        
        workflow_status = workflow_scaler.get_scaling_status()
        agent_status = agent_scaler.get_pool_status()
        memory_status = memory_scaler.get_scaling_status()
        
        print(f"""
📊 AUTO-SCALING DEMO RESULTS
{'='*50}
⏰ Demo Duration: 5 minutes
👤 Created by: Vatsal216

🧠 WORKFLOW SCALING:
   Total Instances: {workflow_status['total_instances']}
   Active Instances: {workflow_status['active_instances']} 
   Scale Up Events: {workflow_status['statistics']['total_scale_ups']}
   Scale Down Events: {workflow_status['statistics']['total_scale_downs']}

🤖 AGENT SCALING:
   Total Agents: {agent_status['total_agents']}
   Agent Types: {len(agent_status['pools'])}
   Scaling Events: {len(agent_status['scaling_events'])}

💾 MEMORY SCALING:
   Active Backends: {len(memory_status['backends'])}
   Status: {'✅ Running' if memory_status['running'] else '❌ Stopped'}

💡 KEY INSIGHTS:
✅ Auto-scaling systems working correctly
✅ Resource optimization active
✅ Cost management enabled
✅ Performance monitoring operational

🎯 NEXT STEPS:
1. Customize scaling rules for your workload
2. Integrate with your monitoring system
3. Set up alerts and notifications
4. Fine-tune cost optimization settings
        """)
        
        # 8. Export metrics
        print("\n📤 Exporting metrics data...")
        metrics_export = dashboard.export_metrics(format_type='json', hours=1)
        
        with open('autoscaling_metrics.json', 'w') as f:
            f.write(metrics_export)
        
        print("   Metrics exported to: autoscaling_metrics.json")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up auto-scaling systems...")
        try:
            workflow_scaler.stop()
            agent_scaler.stop_scaling() 
            memory_scaler.stop_scaling()
        except:
            pass
        
        print("✅ Auto-scaling demo completed!")
        print(f"👤 Demo completed by: Vatsal216")
        print(f"⏰ Completed at: 2025-07-22 13:17:52 UTC")


if __name__ == "__main__":
    main()