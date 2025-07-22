"""
Enhanced CrewGraph with Built-in Metrics
Author: Vatsal216
Date: 2025-07-22 11:25:03 UTC
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Union
from crewai import Agent, Task, Tool, Crew
from langgraph.graph import StateGraph, END

# Import metrics system
from ..utils.metrics import get_metrics_collector, PerformanceMonitor
from ..utils.decorators import monitor, retry

# Initialize global metrics
metrics = get_metrics_collector()
performance_monitor = PerformanceMonitor()

class CrewGraph:
    """
    CrewGraph with Built-in Metrics and Monitoring
    All operations are automatically tracked and monitored
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """Initialize CrewGraph with metrics tracking"""
        self.name = name
        self.id = str(uuid.uuid4())
        self.config = config or {}
        
        # CrewAI components
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.tools: Dict[str, Tool] = {}
        
        # LangGraph components
        self._state_graph: Optional[StateGraph] = None
        self._compiled_graph = None
        
        # Execution tracking
        self.execution_history: List[Dict[str, Any]] = []
        
        # Record workflow creation
        metrics.increment_counter(
            "crewgraph_workflows_created_total",
            labels={"workflow_name": name, "created_by": "Vatsal216"}
        )
        
        print(f"✅ CrewGraph '{name}' initialized with built-in metrics")
        print(f"📊 Metrics tracking enabled for user: Vatsal216")
        print(f"⏰ Created at: 2025-07-22 11:25:03 UTC")
    
    @monitor(operation_name="agent_addition")
    def add_agent(self, agent: Agent, name: str) -> None:
        """Add CrewAI agent with metrics tracking"""
        self.agents[name] = agent
        
        # Record agent metrics
        metrics.increment_counter(
            "crewgraph_agents_added_total",
            labels={
                "workflow_name": self.name,
                "agent_name": name,
                "created_by": "Vatsal216"
            }
        )
        
        print(f"✅ Agent '{name}' added with metrics tracking")
    
    @monitor(operation_name="task_addition")
    def add_task(self, name: str, description: str, agent: str, 
                 dependencies: Optional[List[str]] = None) -> Task:
        """Add task with metrics tracking"""
        if agent not in self.agents:
            # Record error metric
            metrics.increment_counter(
                "crewgraph_task_addition_errors_total",
                labels={
                    "workflow_name": self.name,
                    "error_type": "agent_not_found",
                    "task_name": name
                }
            )
            raise ValueError(f"Agent '{agent}' not found")
        
        # Create CrewAI task
        task = Task(
            description=description,
            agent=self.agents[agent]
        )
        
        self.tasks[name] = task
        
        # Record task metrics
        metrics.increment_counter(
            "crewgraph_tasks_added_total",
            labels={
                "workflow_name": self.name,
                "task_name": name,
                "agent_name": agent,
                "has_dependencies": str(bool(dependencies)),
                "created_by": "Vatsal216"
            }
        )
        
        print(f"✅ Task '{name}' added with metrics tracking")
        return task
    
    @monitor(operation_name="tool_addition")
    def add_tool(self, name: str, func, description: str) -> Tool:
        """Add tool with metrics tracking"""
        tool = Tool(name=name, func=func, description=description)
        self.tools[name] = tool
        
        # Record tool metrics
        metrics.increment_counter(
            "crewgraph_tools_added_total",
            labels={
                "workflow_name": self.name,
                "tool_name": name,
                "created_by": "Vatsal216"
            }
        )
        
        print(f"✅ Tool '{name}' added with metrics tracking")
        return tool
    
    @retry(max_attempts=3, delay=1.0)
    @monitor(operation_name="workflow_execution")
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with comprehensive metrics tracking"""
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        
        # Track execution start
        metrics.increment_counter(
            "crewgraph_workflow_executions_started_total",
            labels={
                "workflow_name": self.name,
                "execution_id": execution_id,
                "user": "Vatsal216"
            }
        )
        
        print(f"🚀 Executing CrewGraph '{self.name}' with built-in metrics")
        print(f"📊 Execution ID: {execution_id}")
        print(f"📈 Input size: {len(str(input_data))} characters")
        print(f"⏰ Started at: 2025-07-22 11:25:03 UTC")
        
        try:
            with performance_monitor.track_execution(execution_id):
                # Create CrewAI crew from agents and tasks
                crew = Crew(
                    agents=list(self.agents.values()),
                    tasks=list(self.tasks.values()),
                    verbose=True
                )
                
                # Track crew creation
                metrics.record_gauge(
                    "crewgraph_crew_size_agents",
                    len(self.agents),
                    labels={"workflow_name": self.name}
                )
                
                metrics.record_gauge(
                    "crewgraph_crew_size_tasks",
                    len(self.tasks),
                    labels={"workflow_name": self.name}
                )
                
                # Execute with CrewAI and track execution
                result = crew.kickoff()
                
                execution_time = time.time() - start_time
                
                # Record successful execution metrics
                metrics.record_duration(
                    "crewgraph_workflow_execution_duration_seconds",
                    execution_time,
                    labels={
                        "workflow_name": self.name,
                        "status": "success",
                        "user": "Vatsal216"
                    }
                )
                
                metrics.increment_counter(
                    "crewgraph_workflow_executions_completed_total",
                    labels={
                        "workflow_name": self.name,
                        "status": "success",
                        "user": "Vatsal216"
                    }
                )
                
                # Track result size
                result_size = len(str(result))
                metrics.record_gauge(
                    "crewgraph_execution_result_size_bytes",
                    result_size,
                    labels={"workflow_name": self.name}
                )
                
                # Prepare response with metrics
                response = {
                    'execution_id': execution_id,
                    'workflow_name': self.name,
                    'status': 'completed',
                    'result': result,
                    'input_data': input_data,
                    'execution_time': execution_time,
                    'result_size_bytes': result_size,
                    'agents_count': len(self.agents),
                    'tasks_count': len(self.tasks),
                    'tools_count': len(self.tools),
                    'completed_at': '2025-07-22 11:25:03',
                    'created_by': 'Vatsal216',
                    'version': '1.0.0',
                    'metrics_enabled': True
                }
                
                # Store execution history
                self.execution_history.append(response)
                
                # Track execution history size
                metrics.record_gauge(
                    "crewgraph_execution_history_size",
                    len(self.execution_history),
                    labels={"workflow_name": self.name}
                )
                
                print(f"✅ Workflow completed successfully in {execution_time:.2f} seconds")
                print(f"📊 Result size: {result_size} bytes")
                print(f"📈 Total executions: {len(self.execution_history)}")
                
                return response
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failure metrics
            metrics.record_duration(
                "crewgraph_workflow_execution_duration_seconds",
                execution_time,
                labels={
                    "workflow_name": self.name,
                    "status": "failure",
                    "error_type": type(e).__name__,
                    "user": "Vatsal216"
                }
            )
            
            metrics.increment_counter(
                "crewgraph_workflow_executions_completed_total",
                labels={
                    "workflow_name": self.name,
                    "status": "failure",
                    "error_type": type(e).__name__,
                    "user": "Vatsal216"
                }
            )
            
            error_response = {
                'execution_id': execution_id,
                'workflow_name': self.name,
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'input_data': input_data,
                'execution_time': execution_time,
                'failed_at': '2025-07-22 11:25:03',
                'created_by': 'Vatsal216',
                'metrics_enabled': True
            }
            
            print(f"❌ Workflow failed after {execution_time:.2f} seconds: {e}")
            
            return error_response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive workflow metrics"""
        current_time = time.time()
        
        # Calculate success rate
        total_executions = len(self.execution_history)
        successful_executions = sum(
            1 for exec in self.execution_history 
            if exec.get('status') == 'completed'
        )
        success_rate = successful_executions / total_executions if total_executions > 0 else 0
        
        # Calculate average execution time
        execution_times = [
            exec.get('execution_time', 0) 
            for exec in self.execution_history 
            if exec.get('execution_time')
        ]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        workflow_metrics = {
            'workflow_info': {
                'name': self.name,
                'id': self.id,
                'created_by': 'Vatsal216',
                'created_at': '2025-07-22 11:25:03',
                'current_time': current_time
            },
            'component_counts': {
                'agents': len(self.agents),
                'tasks': len(self.tasks),
                'tools': len(self.tools)
            },
            'execution_metrics': {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'failed_executions': total_executions - successful_executions,
                'success_rate': success_rate,
                'average_execution_time': avg_execution_time
            },
            'system_metrics': metrics.get_all_metrics(),
            'performance_metrics': performance_monitor.get_stats()
        }
        
        return workflow_metrics
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return metrics.export_prometheus_format()
    
    def get_status(self) -> Dict[str, Any]:
        """Get workflow status with metrics summary"""
        status = {
            'name': self.name,
            'id': self.id,
            'agents_count': len(self.agents),
            'tasks_count': len(self.tasks),
            'tools_count': len(self.tools),
            'executions_count': len(self.execution_history),
            'metrics_enabled': True,
            'created_by': 'Vatsal216',
            'version': '1.0.0',
            'timestamp': '2025-07-22 11:25:03'
        }
        
        # Add metrics summary
        if self.execution_history:
            recent_execution = self.execution_history[-1]
            status['last_execution'] = {
                'status': recent_execution.get('status'),
                'execution_time': recent_execution.get('execution_time'),
                'completed_at': recent_execution.get('completed_at')
            }
        
        return status
    
    def __repr__(self) -> str:
        metrics_info = f"metrics=enabled, executions={len(self.execution_history)}"
        return f"CrewGraph(name='{self.name}', agents={len(self.agents)}, tasks={len(self.tasks)}, {metrics_info})"