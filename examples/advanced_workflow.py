"""
Advanced CrewGraph AI Example
Demonstrates complex workflows with conditional branching, parallel execution,
and dynamic planning
"""

import asyncio
import time
from typing import Dict, Any, List
from crewai import Agent, Task, Tool, Crew
from crewgraph_ai import (
    CrewGraph, CrewGraphConfig, AgentWrapper, TaskWrapper,
    DynamicPlanner, PlannerConfig, GraphOrchestrator
)
from crewgraph_ai.memory import RedisMemory, DictMemory
from crewgraph_ai.tools import ToolRegistry, ToolWrapper
from crewgraph_ai.planning import OptimalStrategy
from crewgraph_ai.utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Advanced tools with error handling
def analyze_data(data: str, analysis_type: str = "sentiment") -> Dict[str, Any]:
    """Advanced data analysis tool"""
    logger.info(f"Analyzing data with type: {analysis_type}")
    
    # Simulate analysis
    time.sleep(2)
    
    if analysis_type == "sentiment":
        return {
            "sentiment": "positive",
            "confidence": 0.85,
            "keywords": ["innovation", "growth", "success"]
        }
    elif analysis_type == "topics":
        return {
            "topics": ["AI", "Machine Learning", "Automation"],
            "relevance_scores": [0.9, 0.8, 0.7]
        }
    else:
        return {"analysis": "completed", "type": analysis_type}

def generate_report(analysis_results: Dict[str, Any], format_type: str = "markdown") -> str:
    """Generate formatted report"""
    logger.info(f"Generating report in format: {format_type}")
    
    if format_type == "markdown":
        return f"""
# Analysis Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Results
- Analysis Type: {analysis_results.get('type', 'unknown')}
- Confidence: {analysis_results.get('confidence', 'N/A')}
- Key Findings: {analysis_results.get('keywords', [])}
"""
    else:
        return str(analysis_results)

def send_notification(message: str, channel: str = "email") -> bool:
    """Send notification"""
    logger.info(f"Sending notification via {channel}: {message}")
    return True

class AdvancedWorkflowExample:
    """Advanced workflow demonstration"""
    
    def __init__(self):
        """Initialize advanced workflow"""
        
        # Enhanced configuration
        self.config = CrewGraphConfig(
            memory_backend=DictMemory(),  # Use Redis in production
            enable_planning=True,
            max_concurrent_tasks=5,
            task_timeout=300.0,
            enable_logging=True,
            log_level="INFO"
        )
        
        # Planning configuration
        self.planner_config = PlannerConfig(
            strategy="optimal",
            max_planning_time=30.0,
            enable_parallel_execution=True,
            max_parallel_tasks=3,
            learning_enabled=True,
            optimization_enabled=True
        )
        
        # Initialize components
        self.workflow = CrewGraph("advanced_workflow", self.config)
        self.tool_registry = ToolRegistry()
        self.planner = DynamicPlanner(self.planner_config)
        
        self._setup_tools()
        self._setup_agents()
        self._setup_tasks()
    
    def _setup_tools(self):
        """Setup advanced tool registry"""
        logger.info("Setting up advanced tool registry")
        
        # Register tools with metadata
        self.tool_registry.register_tool(ToolWrapper(
            name="analyze_data",
            func=analyze_data,
            description="Advanced data analysis with multiple algorithms",
            tool_type="ai_model",
            version="2.0.0",
            author="Vatsal216",
            tags=["analysis", "ml", "data"],
            timeout=30.0
        ))
        
        self.tool_registry.register_tool(ToolWrapper(
            name="generate_report",
            func=generate_report,
            description="Generate formatted reports from analysis results",
            tool_type="utility",
            version="1.5.0",
            author="Vatsal216",
            tags=["reporting", "format", "output"]
        ))
        
        self.tool_registry.register_tool(ToolWrapper(
            name="send_notification",
            func=send_notification,
            description="Send notifications via multiple channels",
            tool_type="api",
            version="1.0.0",
            author="Vatsal216",
            tags=["notification", "communication"]
        ))
        
        # Add tools to workflow
        self.workflow.add_tool("analyze_data", analyze_data, "Advanced data analysis")
        self.workflow.add_tool("generate_report", generate_report, "Report generation")
        self.workflow.add_tool("send_notification", send_notification, "Notification service")
    
    def _setup_agents(self):
        """Setup specialized agents"""
        logger.info("Setting up specialized agents")
        
        # Data Analyst Agent
        analyst_agent = Agent(
            role='Senior Data Analyst',
            goal='Perform comprehensive data analysis and extract insights',
            backstory='''You are a senior data analyst with expertise in statistical analysis,
                        machine learning, and data visualization. You excel at finding patterns
                        and extracting meaningful insights from complex datasets.''',
            tools=[
                Tool(name="analyze_data", func=analyze_data, description="Analyze data with various algorithms")
            ],
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
        
        # Report Generator Agent  
        reporter_agent = Agent(
            role='Technical Report Writer',
            goal='Create comprehensive and professional reports from analysis results',
            backstory='''You are an expert technical writer who specializes in translating
                        complex data analysis results into clear, actionable reports for
                        different audiences.''',
            tools=[
                Tool(name="generate_report", func=generate_report, description="Generate formatted reports")
            ],
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
        
        # Quality Assurance Agent
        qa_agent = Agent(
            role='Quality Assurance Specialist',
            goal='Review and validate analysis results and reports for accuracy',
            backstory='''You are a meticulous QA specialist with deep expertise in data
                        validation, statistical methods, and quality control processes.''',
            tools=[
                Tool(name="send_notification", func=send_notification, description="Send notifications")
            ],
            verbose=True,
            allow_delegation=False
        )
        
        # Add agents to workflow
        self.workflow.add_agent(analyst_agent, name="data_analyst")
        self.workflow.add_agent(reporter_agent, name="report_writer")
        self.workflow.add_agent(qa_agent, name="qa_specialist")
    
    def _setup_tasks(self):
        """Setup complex task dependencies"""
        logger.info("Setting up complex task workflow")
        
        # Analysis tasks (can run in parallel)
        sentiment_task = self.workflow.add_task(
            name="sentiment_analysis",
            description="Perform sentiment analysis on input data",
            agent="data_analyst",
            tools=["analyze_data"]
        )
        
        topic_task = self.workflow.add_task(
            name="topic_analysis", 
            description="Extract and analyze topics from input data",
            agent="data_analyst",
            tools=["analyze_data"]
        )
        
        # Report generation (depends on analysis)
        report_task = self.workflow.add_task(
            name="generate_report",
            description="Generate comprehensive analysis report",
            agent="report_writer",
            tools=["generate_report"],
            dependencies=["sentiment_analysis", "topic_analysis"]
        )
        
        # Quality assurance (depends on report)
        qa_task = self.workflow.add_task(
            name="quality_review",
            description="Review and validate the generated report",
            agent="qa_specialist",
            dependencies=["generate_report"]
        )
        
        # Notification (depends on QA)
        notification_task = self.workflow.add_task(
            name="send_notification",
            description="Send completion notification",
            agent="qa_specialist", 
            tools=["send_notification"],
            dependencies=["quality_review"]
        )
    
    def run_basic_workflow(self) -> Dict[str, Any]:
        """Run basic workflow with standard execution"""
        logger.info("Running basic workflow")
        
        initial_state = {
            "input_data": "Sample data for analysis containing positive sentiment about AI innovations",
            "analysis_types": ["sentiment", "topics"],
            "report_format": "markdown",
            "notification_channel": "email"
        }
        
        return self.workflow.execute(initial_state)
    
    async def run_async_workflow(self) -> Dict[str, Any]:
        """Run workflow asynchronously"""
        logger.info("Running async workflow")
        
        initial_state = {
            "input_data": "Large dataset requiring async processing",
            "analysis_types": ["sentiment", "topics", "entities"],
            "report_format": "json",
            "notification_channel": "slack"
        }
        
        return self.workflow.execute(initial_state, async_mode=True)
    
    def run_with_dynamic_planning(self) -> Dict[str, Any]:
        """Run workflow with dynamic planning and optimization"""
        logger.info("Running workflow with dynamic planning")
        
        # Get all tasks
        tasks = [self.workflow.get_task(name) for name in self.workflow.list_tasks()]
        
        # Create optimal execution plan
        execution_plan = self.planner.create_plan(
            tasks=tasks,
            state=self.workflow.get_state(),
            constraints={
                "max_execution_time": 600,  # 10 minutes
                "max_parallel_tasks": 3,
                "resource_limits": {
                    "cpu": 4.0,
                    "memory": 2048  # MB
                }
            }
        )
        
        logger.info(f"Execution plan created: {execution_plan.name}")
        logger.info(f"Estimated duration: {execution_plan.estimated_total_duration:.2f}s")
        
        # Execute with plan
        initial_state = {
            "input_data": "Complex dataset requiring optimized processing",
            "analysis_types": ["sentiment", "topics", "entities", "keywords"],
            "report_format": "html",
            "notification_channel": "teams"
        }
        
        return self.workflow.execute(initial_state)
    
    def run_with_error_recovery(self) -> Dict[str, Any]:
        """Run workflow with error handling and recovery"""
        logger.info("Running workflow with error recovery")
        
        # Add error handling callbacks
        def on_task_error(task, error):
            logger.error(f"Task {task.name} failed: {error}")
            # Could implement custom recovery logic here
        
        def on_workflow_complete(workflow, result):
            logger.info(f"Workflow {workflow.name} completed with result: {result}")
        
        # Set callbacks on agents
        for agent_name in self.workflow.list_agents():
            agent = self.workflow.get_agent(agent_name)
            agent.set_callbacks(
                on_error=on_task_error,
                on_task_complete=lambda a, t, r: logger.info(f"Task {t} completed by {a.name}")
            )
        
        # Simulate some errors by using invalid data
        initial_state = {
            "input_data": "",  # Empty data to trigger handling
            "analysis_types": ["invalid_type"],  # Invalid type
            "report_format": "unknown_format",
            "notification_channel": "invalid_channel"
        }
        
        try:
            return self.workflow.execute(initial_state)
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def demonstrate_langgraph_access(self):
        """Demonstrate direct LangGraph access"""
        logger.info("Demonstrating direct LangGraph access")
        
        # Get the underlying LangGraph orchestrator
        orchestrator = GraphOrchestrator("direct_langgraph_demo")
        
        # Create a LangGraph StateGraph directly
        from langgraph.graph import StateGraph, END
        from typing_extensions import TypedDict
        
        class GraphState(TypedDict):
            messages: List[str]
            current_step: str
            results: Dict[str, Any]
        
        # Create StateGraph with full LangGraph features
        graph = orchestrator.create_state_graph(GraphState)
        
        # Add nodes with LangGraph syntax
        def analysis_node(state: GraphState) -> GraphState:
            logger.info("Executing analysis node")
            state["messages"].append("Analysis completed")
            state["current_step"] = "analysis"
            state["results"]["analysis"] = "sample_result"
            return state
        
        def reporting_node(state: GraphState) -> GraphState:
            logger.info("Executing reporting node")
            state["messages"].append("Report generated")
            state["current_step"] = "reporting"
            state["results"]["report"] = "sample_report"
            return state
        
        # Add nodes using LangGraph API
        graph.add_node("analysis", analysis_node)
        graph.add_node("reporting", reporting_node)
        
        # Add edges using LangGraph API
        graph.add_edge("analysis", "reporting")
        graph.add_edge("reporting", END)
        
        # Set entry point
        graph.set_entry_point("analysis")
        
        # Compile and execute with full LangGraph features
        orchestrator.build_graph()
        
        result = orchestrator.execute({
            "messages": [],
            "current_step": "start",
            "results": {}
        })
        
        logger.info(f"LangGraph direct execution result: {result}")
        return result
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Get comprehensive analytics report"""
        return {
            "workflow_analytics": self.workflow.get_state().get_analytics() if hasattr(self.workflow.get_state(), 'get_analytics') else {},
            "tool_analytics": self.tool_registry.get_analytics_report(),
            "planner_analytics": self.planner.get_planner_stats(),
            "agent_metrics": {
                name: agent.get_metrics() if hasattr(agent, 'get_metrics') else {}
                for name, agent in [(name, self.workflow.get_agent(name)) for name in self.workflow.list_agents()]
            }
        }

def main():
    """Main execution function"""
    logger.info("Starting Advanced CrewGraph AI Workflow Example")
    logger.info(f"Executed by user: Vatsal216 at 2025-07-22 10:19:38")
    
    # Create advanced workflow
    advanced_workflow = AdvancedWorkflowExample()
    
    print("\n" + "="*60)
    print("1. Running Basic Workflow")
    print("="*60)
    basic_result = advanced_workflow.run_basic_workflow()
    print(f"Basic workflow result: {basic_result}")
    
    print("\n" + "="*60)
    print("2. Running Async Workflow")
    print("="*60)
    async_result = asyncio.run(advanced_workflow.run_async_workflow())
    print(f"Async workflow result: {async_result}")
    
    print("\n" + "="*60)
    print("3. Running with Dynamic Planning")
    print("="*60)
    planned_result = advanced_workflow.run_with_dynamic_planning()
    print(f"Planned workflow result: {planned_result}")
    
    print("\n" + "="*60)
    print("4. Demonstrating Direct LangGraph Access")
    print("="*60)
    langgraph_result = advanced_workflow.demonstrate_langgraph_access()
    print(f"LangGraph direct result: {langgraph_result}")
    
    print("\n" + "="*60)
    print("5. Running with Error Recovery")
    print("="*60)
    error_result = advanced_workflow.run_with_error_recovery()
    print(f"Error recovery result: {error_result}")
    
    print("\n" + "="*60)
    print("6. Analytics Report")
    print("="*60)
    analytics = advanced_workflow.get_analytics_report()
    print(f"Analytics: {analytics}")
    
    logger.info("Advanced workflow examples completed successfully!")

if __name__ == "__main__":
    main()