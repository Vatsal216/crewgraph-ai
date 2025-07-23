"""
Main CrewGraph class - High-level interface for the library
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass

from .core.agents import AgentWrapper, AgentPool
from .core.tasks import TaskWrapper, TaskChain  
from .core.orchestrator import GraphOrchestrator
from .core.state import SharedState
from .memory.base import BaseMemory
from .memory.dict_memory import DictMemory
from .tools.registry import ToolRegistry
from .planning.planner import DynamicPlanner
from .utils.logging import get_logger
from .utils.exceptions import CrewGraphError, ValidationError, ExecutionError

logger = get_logger(__name__)


@dataclass
class CrewGraphConfig:
    """
    Configuration for CrewGraph instance.
    
    Provides comprehensive configuration options for workflow behavior,
    performance tuning, and feature enablement.
    
    Attributes:
        memory_backend: Memory backend for state persistence. Defaults to DictMemory.
        enable_planning: Whether to enable dynamic planning optimization.
        max_concurrent_tasks: Maximum number of tasks to execute in parallel.
        task_timeout: Maximum time (seconds) for individual task execution.
        enable_logging: Whether to enable structured logging.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        enable_visualization: Whether to enable visualization features.
        visualization_output_dir: Directory for saving visualization outputs.
        visualization_format: Default format for visualizations.
        enable_real_time_monitoring: Whether to enable real-time execution monitoring.
        enable_performance_tracking: Whether to track detailed performance metrics.
        checkpoint_interval: Interval (seconds) for automatic state checkpointing.
        
    Example:
        ```python
        from crewgraph_ai import CrewGraphConfig
        from crewgraph_ai.memory import RedisMemory
        
        config = CrewGraphConfig(
            memory_backend=RedisMemory(host="localhost", port=6379),
            enable_planning=True,
            max_concurrent_tasks=5,
            task_timeout=600.0,
            enable_visualization=True,
            visualization_output_dir="./workflow_viz",
            enable_real_time_monitoring=True
        )
        ```
    """
    memory_backend: Optional[BaseMemory] = None
    enable_planning: bool = True
    max_concurrent_tasks: int = 10
    task_timeout: float = 300.0  # 5 minutes
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Visualization settings
    enable_visualization: bool = True
    visualization_output_dir: str = "visualizations"
    visualization_format: str = "html"  # html, png, svg, pdf
    
    # Monitoring settings
    enable_real_time_monitoring: bool = False
    enable_performance_tracking: bool = True
    
    # Advanced settings
    checkpoint_interval: float = 30.0  # seconds
    enable_debug_mode: bool = False


class CrewGraph:
    """
    Main interface for CrewGraph AI library.
    
    This class provides a high-level API for creating and managing agent workflows
    that combine CrewAI agents with LangGraph orchestration. It handles the complete
    lifecycle of workflow definition, execution, and monitoring.
    
    The CrewGraph class serves as the primary entry point for users who want to:
    - Define multi-agent workflows with dependencies
    - Execute workflows with state management
    - Monitor execution progress and performance
    - Visualize workflow structure and execution flow
    
    Attributes:
        name: The workflow name for identification and logging
        config: Configuration object containing workflow settings
        
    Example:
        Basic workflow creation and execution:
        
        ```python
        from crewgraph_ai import CrewGraph, CrewGraphConfig
        
        # Create workflow with custom configuration
        config = CrewGraphConfig(
            enable_planning=True,
            max_concurrent_tasks=5,
            task_timeout=300.0
        )
        workflow = CrewGraph("data_processing", config)
        
        # Add agents and tasks
        workflow.add_agent(data_collector_agent, "collector")
        workflow.add_agent(data_analyzer_agent, "analyzer")
        
        workflow.add_task("collect_data", "Collect data from sources", 
                         agent="collector")
        workflow.add_task("analyze_data", "Analyze collected data", 
                         agent="analyzer", dependencies=["collect_data"])
        
        # Execute workflow
        result = workflow.execute({"source_url": "https://api.example.com"})
        print(f"Workflow completed: {result}")
        ```
        
        Advanced workflow with custom tools:
        
        ```python
        # Add custom tools
        def custom_processor(data):
            return {"processed": data}
            
        workflow.add_tool("processor", custom_processor, 
                         "Process data with custom logic")
        
        # Create task chains for sequential execution
        chain = workflow.create_chain("collect_data", "analyze_data", "report")
        
        # Execute with async mode
        result = workflow.execute(async_mode=True)
        ```
    """
    
    def __init__(self, 
                 name: str = "default_workflow",
                 config: Optional[CrewGraphConfig] = None) -> None:
        """
        Initialize CrewGraph instance.
        
        Sets up the core components needed for workflow execution including
        state management, agent pool, tool registry, and orchestrator.
        
        Args:
            name: Unique name for the workflow. Used for logging, monitoring,
                and identification. Should be descriptive and unique across
                your application.
            config: Configuration object specifying workflow behavior. If None,
                uses default configuration with dictionary memory backend and
                basic settings.
                
        Raises:
            CrewGraphError: If initialization fails due to invalid configuration
                or missing dependencies.
                
        Example:
            ```python
            # Basic initialization
            workflow = CrewGraph("my_workflow")
            
            # With custom configuration
            from crewgraph_ai.memory import RedisMemory
            
            config = CrewGraphConfig(
                memory_backend=RedisMemory(host="localhost"),
                enable_planning=True,
                max_concurrent_tasks=10,
                enable_logging=True,
                log_level="DEBUG"
            )
            workflow = CrewGraph("advanced_workflow", config)
            ```
        """
        self.name = name
        self.config = config or CrewGraphConfig()
        
        # Initialize core components
        self._state = SharedState(memory=self.config.memory_backend or DictMemory())
        self._agent_pool = AgentPool()
        self._tool_registry = ToolRegistry()
        self._orchestrator = GraphOrchestrator(name=name)
        self._planner = DynamicPlanner() if self.config.enable_planning else None
        
        # Track workflow components
        self._agents: Dict[str, AgentWrapper] = {}
        self._tasks: Dict[str, TaskWrapper] = {}
        self._task_chains: List[TaskChain] = []
        
        logger.info(f"CrewGraph '{name}' initialized with config: {self.config}")
    
    def add_agent(self, 
                  agent: Union[AgentWrapper, Any], 
                  name: Optional[str] = None) -> AgentWrapper:
        """
        Add an agent to the workflow.
        
        Registers an agent for use in workflow tasks. The agent can be either
        a pre-wrapped AgentWrapper instance or a raw CrewAI agent that will
        be automatically wrapped.
        
        Args:
            agent: Agent to add to the workflow. Can be:
                - AgentWrapper: Pre-configured wrapper with state access
                - CrewAI Agent: Raw agent that will be wrapped automatically
                - Any object with execute() method: Custom agent implementation
            name: Optional name for the agent. Required if agent is not an
                AgentWrapper. Must be unique within the workflow.
                
        Returns:
            AgentWrapper instance that can be used for task assignment and
            execution. The wrapper provides state access and monitoring.
            
        Raises:
            ValidationError: If name is required but not provided, or if an
                agent with the same name already exists.
            CrewGraphError: If agent initialization fails.
            
        Example:
            ```python
            from crewai import Agent
            
            # Add raw CrewAI agent
            crew_agent = Agent(
                role="Data Analyst",
                goal="Analyze data patterns",
                backstory="Expert in statistical analysis"
            )
            wrapper = workflow.add_agent(crew_agent, "analyst")
            
            # Add pre-wrapped agent
            from crewgraph_ai import AgentWrapper
            
            wrapped_agent = AgentWrapper(
                name="researcher", 
                crew_agent=research_agent,
                state=shared_state
            )
            workflow.add_agent(wrapped_agent)
            
            # Verify agent was added
            print(f"Added agent: {wrapper.name}")
            print(f"Total agents: {len(workflow.list_agents())}")
            ```
            
        Note:
            Once added, agents can be assigned to tasks and will have access
            to the shared workflow state for data exchange between tasks.
        """
        if isinstance(agent, AgentWrapper):
            wrapper = agent
        else:
            if not name:
                raise ValidationError("Name required when adding raw CrewAI agent")
            wrapper = AgentWrapper(name=name, crew_agent=agent, state=self._state)
            
        self._agents[wrapper.name] = wrapper
        self._agent_pool.add_agent(wrapper)
        
        logger.info(f"Added agent '{wrapper.name}' to workflow")
        return wrapper
    
    def add_task(self, 
                 name: str,
                 description: str = "",
                 agent: Optional[str] = None,
                 tools: Optional[List[str]] = None,
                 dependencies: Optional[List[str]] = None) -> TaskWrapper:
        """
        Add a task to the workflow.
        
        Creates and registers a new task within the workflow. Tasks represent
        discrete units of work that can be executed by agents with optional
        tool access and dependency relationships.
        
        Args:
            name: Unique identifier for the task within the workflow. Used for
                dependency references and execution tracking. Must not conflict
                with existing task names.
            description: Human-readable description of what the task should
                accomplish. Used by agents to understand task requirements and
                by the planning system for optimization. Should be detailed
                enough for agent execution.
            agent: Optional name of the agent to assign this task to. Must
                correspond to an agent previously added with add_agent(). If
                None, task can be assigned later or executed without a specific
                agent.
            tools: Optional list of tool names to make available during task
                execution. Tools must be previously registered in the tool
                registry. Agents can access these tools during task execution.
            dependencies: Optional list of task names that must complete before
                this task can execute. Creates execution order constraints and
                enables data flow between tasks. All dependency tasks must
                exist in the workflow.
                
        Returns:
            TaskWrapper instance that encapsulates the task configuration and
            provides execution interface. Can be used for further configuration
            or direct execution.
            
        Raises:
            ValidationError: If task name already exists, if specified agent
                doesn't exist, or if dependency tasks don't exist.
            CrewGraphError: If task initialization fails.
            
        Example:
            ```python
            # Basic task creation
            task1 = workflow.add_task(
                name="data_collection",
                description="Collect data from external APIs",
                agent="collector"
            )
            
            # Task with tools and dependencies
            task2 = workflow.add_task(
                name="data_analysis", 
                description="Analyze collected data for patterns",
                agent="analyzer",
                tools=["pandas_processor", "statistical_analyzer"],
                dependencies=["data_collection"]
            )
            
            # Complex task with multiple dependencies
            workflow.add_task(
                name="generate_report",
                description="Generate comprehensive analysis report",
                agent="reporter", 
                tools=["report_generator", "chart_creator"],
                dependencies=["data_collection", "data_analysis"]
            )
            
            print(f"Added {len(workflow.list_tasks())} tasks to workflow")
            ```
            
        Note:
            Tasks with dependencies will automatically wait for their
            dependencies to complete before execution. The workflow engine
            handles proper sequencing and data flow between tasks.
        """
        task = TaskWrapper(
            name=name,
            description=description,
            state=self._state,
            tool_registry=self._tool_registry
        )
        
        if agent and agent in self._agents:
            task.assign_agent(self._agents[agent])
        
        if tools:
            for tool_name in tools:
                tool = self._tool_registry.get_tool(tool_name)
                if tool:
                    task.add_tool(tool)
        
        if dependencies:
            task.dependencies = dependencies
            
        self._tasks[name] = task
        logger.info(f"Added task '{name}' to workflow")
        return task
    
    def create_chain(self, *task_names: str) -> TaskChain:
        """
        Create a sequential chain of tasks.
        
        Args:
            task_names: Names of tasks in execution order
            
        Returns:
            TaskChain instance
        """
        tasks = []
        for name in task_names:
            if name not in self._tasks:
                raise ValidationError(f"Task '{name}' not found")
            tasks.append(self._tasks[name])
        
        chain = TaskChain(tasks)
        self._task_chains.append(chain)
        
        logger.info(f"Created task chain: {' -> '.join(task_names)}")
        return chain
    
    def add_tool(self, name: str, func: Callable, description: str = "") -> None:
        """
        Add a tool to the registry.
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
        """
        from .tools.wrapper import ToolWrapper
        tool = ToolWrapper(name=name, func=func, description=description)
        self._tool_registry.register_tool(tool)
        logger.info(f"Added tool '{name}' to registry")
    
    def build_workflow(self) -> None:
        """Build the workflow graph from tasks and dependencies."""
        try:
            # Add tasks to orchestrator
            for task in self._tasks.values():
                self._orchestrator.add_node(task.name, task.execute)
            
            # Add task chains
            for chain in self._task_chains:
                self._orchestrator.add_chain(chain)
            
            # Build the graph
            self._orchestrator.build_graph()
            logger.info("Workflow graph built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build workflow: {e}")
            raise CrewGraphError(f"Workflow build failed: {e}")
    
    def execute(self, 
                initial_state: Optional[Dict[str, Any]] = None,
                async_mode: bool = False) -> Dict[str, Any]:
        """
        Execute the workflow.
        
        Args:
            initial_state: Initial state values
            async_mode: Whether to run asynchronously
            
        Returns:
            Final workflow state
        """
        if not self._orchestrator.is_built:
            self.build_workflow()
        
        if initial_state:
            self._state.update(initial_state)
        
        try:
            if async_mode:
                return asyncio.run(self._execute_async())
            else:
                return self._execute_sync()
                
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise ExecutionError(f"Workflow execution failed: {e}")
    
    def _execute_sync(self) -> Dict[str, Any]:
        """Execute workflow synchronously."""
        logger.info(f"Starting synchronous execution of workflow '{self.name}'")
        
        if self._planner:
            execution_plan = self._planner.create_plan(
                list(self._tasks.values()),
                self._state
            )
            result = self._orchestrator.execute_with_plan(execution_plan, self._state)
        else:
            result = self._orchestrator.execute(self._state)
        
        logger.info("Workflow execution completed")
        return result
    
    async def _execute_async(self) -> Dict[str, Any]:
        """Execute workflow asynchronously."""
        logger.info(f"Starting asynchronous execution of workflow '{self.name}'")
        
        if self._planner:
            execution_plan = await self._planner.create_plan_async(
                list(self._tasks.values()),
                self._state
            )
            result = await self._orchestrator.execute_with_plan_async(execution_plan, self._state)
        else:
            result = await self._orchestrator.execute_async(self._state)
        
        logger.info("Async workflow execution completed")
        return result
    
    def get_state(self) -> SharedState:
        """Get the current workflow state."""
        return self._state
    
    def get_agent(self, name: str) -> Optional[AgentWrapper]:
        """Get an agent by name."""
        return self._agents.get(name)
    
    def get_task(self, name: str) -> Optional[TaskWrapper]:
        """Get a task by name."""
        return self._tasks.get(name)
    
    def list_agents(self) -> List[str]:
        """List all agent names."""
        return list(self._agents.keys())
    
    def list_tasks(self) -> List[str]:
        """List all task names."""
        return list(self._tasks.keys())
    
    def reset(self) -> None:
        """Reset the workflow state."""
        self._state.reset()
        self._orchestrator.reset()
        logger.info("Workflow state reset")
    
    def save_state(self, filename: str) -> None:
        """Save workflow state to file."""
        self._state.save(filename)
        logger.info(f"State saved to {filename}")
    
    def load_state(self, filename: str) -> None:
        """Load workflow state from file."""
        self._state.load(filename)
        logger.info(f"State loaded from {filename}")
    
    # ============= VISUALIZATION AND DEBUG METHODS =============
    
    def visualize_workflow(self, 
                          output_path: Optional[str] = None,
                          format: str = "html",
                          show_details: bool = True) -> str:
        """
        Generate visual representation of the workflow.
        
        Creates an interactive or static visualization of the workflow structure,
        showing agents, tasks, dependencies, and execution status.
        
        Args:
            output_path: Optional custom path for output file. If None, uses
                the configured visualization output directory.
            format: Output format ('html', 'png', 'svg', 'pdf'). Interactive
                features only available with 'html' format.
            show_details: Whether to include detailed information like task
                descriptions, agent assignments, and execution statistics.
                
        Returns:
            Path to the generated visualization file.
            
        Raises:
            CrewGraphError: If visualization dependencies are not available
                or if visualization generation fails.
                
        Example:
            ```python
            # Generate interactive HTML visualization
            viz_path = workflow.visualize_workflow(format="html")
            print(f"Workflow visualization saved to: {viz_path}")
            
            # Generate static PNG with custom path
            workflow.visualize_workflow(
                output_path="./reports/workflow.png",
                format="png", 
                show_details=False
            )
            ```
            
        Note:
            Requires visualization dependencies. Install with:
            pip install crewgraph-ai[visualization]
        """
        if not self.config.enable_visualization:
            raise CrewGraphError("Visualization is disabled in configuration")
        
        try:
            return self._orchestrator.visualize_workflow(
                output_path=output_path or self.config.visualization_output_dir,
                format=format
            )
        except Exception as e:
            logger.error(f"Failed to visualize workflow: {e}")
            raise CrewGraphError(f"Workflow visualization failed: {e}")
    
    def start_real_time_monitoring(self) -> str:
        """
        Start real-time monitoring of workflow execution.
        
        Enables live tracking of task execution, performance metrics, and
        system resource usage during workflow execution.
        
        Returns:
            Session ID for the monitoring session. Use this to stop monitoring
            or retrieve monitoring data.
            
        Raises:
            CrewGraphError: If monitoring is disabled or initialization fails.
            
        Example:
            ```python
            # Start monitoring before execution
            session_id = workflow.start_real_time_monitoring()
            
            # Execute workflow with monitoring
            result = workflow.execute({"input": "data"})
            
            # Get monitoring report
            monitoring_data = workflow.get_monitoring_report(session_id)
            
            # Stop monitoring
            workflow.stop_real_time_monitoring(session_id)
            ```
        """
        if not self.config.enable_real_time_monitoring:
            raise CrewGraphError("Real-time monitoring is disabled in configuration")
        
        try:
            from crewgraph_ai.visualization.execution_tracer import ExecutionTracer
            
            if not hasattr(self, '_execution_tracer'):
                self._execution_tracer = ExecutionTracer(
                    workflow_name=self.name,
                    output_dir=self.config.visualization_output_dir
                )
            
            session_id = self._execution_tracer.start_workflow_trace()
            logger.info(f"Started real-time monitoring with session: {session_id}")
            return session_id
            
        except ImportError:
            raise CrewGraphError(
                "Visualization dependencies not available. "
                "Install with: pip install crewgraph-ai[visualization]"
            )
    
    def stop_real_time_monitoring(self, session_id: str) -> Dict[str, Any]:
        """
        Stop real-time monitoring and get final report.
        
        Args:
            session_id: Session ID from start_real_time_monitoring()
            
        Returns:
            Final monitoring report with execution summary and metrics.
            
        Example:
            ```python
            session_id = workflow.start_real_time_monitoring()
            # ... execute workflow ...
            final_report = workflow.stop_real_time_monitoring(session_id)
            print(f"Workflow completed in {final_report['total_duration']:.2f}s")
            ```
        """
        if hasattr(self, '_execution_tracer'):
            return self._execution_tracer.end_workflow_trace(session_id)
        else:
            return {"error": "No active monitoring session"}
    
    def generate_debug_report(self, include_visualizations: bool = True) -> str:
        """
        Generate comprehensive debug report for the workflow.
        
        Creates a detailed analysis of the workflow including validation
        issues, performance bottlenecks, dependency analysis, and
        configuration review.
        
        Args:
            include_visualizations: Whether to generate visual diagrams
                in addition to the textual report.
                
        Returns:
            Path to the generated debug report file (HTML format).
            
        Example:
            ```python
            # Generate full debug report
            report_path = workflow.generate_debug_report()
            print(f"Debug report available at: {report_path}")
            
            # Generate report without visualizations (faster)
            workflow.generate_debug_report(include_visualizations=False)
            ```
        """
        try:
            return self._orchestrator.generate_debug_report()
        except Exception as e:
            logger.error(f"Failed to generate debug report: {e}")
            raise CrewGraphError(f"Debug report generation failed: {e}")
    
    def export_execution_trace(self, format: str = "json") -> str:
        """
        Export detailed execution trace for analysis.
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Path to exported trace file.
            
        Example:
            ```python
            # Export as JSON for detailed analysis
            trace_path = workflow.export_execution_trace("json")
            
            # Export as CSV for spreadsheet analysis  
            workflow.export_execution_trace("csv")
            ```
        """
        try:
            trace_data = self._orchestrator.export_execution_trace(include_memory=True)
            
            if hasattr(self, '_execution_tracer'):
                return self._execution_tracer.export_trace_data(format)
            else:
                # Export basic trace data
                import json
                import os
                from datetime import datetime
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"execution_trace_{self.name}_{timestamp}.json"
                filepath = os.path.join(self.config.visualization_output_dir, filename)
                
                os.makedirs(self.config.visualization_output_dir, exist_ok=True)
                
                with open(filepath, 'w') as f:
                    json.dump(trace_data, f, indent=2, default=str)
                
                return filepath
                
        except Exception as e:
            logger.error(f"Failed to export execution trace: {e}")
            raise CrewGraphError(f"Execution trace export failed: {e}")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze workflow performance and identify bottlenecks.
        
        Returns:
            Dictionary containing performance analysis results including
            execution times, resource usage, and optimization recommendations.
            
        Example:
            ```python
            # Run performance analysis
            perf_analysis = workflow.analyze_performance()
            
            print(f"Total execution time: {perf_analysis['total_time']:.2f}s")
            print(f"Bottlenecks found: {len(perf_analysis['bottlenecks'])}")
            
            for bottleneck in perf_analysis['bottlenecks']:
                print(f"- {bottleneck['component']}: {bottleneck['issue']}")
            ```
        """
        try:
            if not self.config.enable_performance_tracking:
                logger.warning("Performance tracking is disabled")
                return {"error": "Performance tracking disabled"}
            
            # Get execution data from orchestrator
            execution_data = self._orchestrator.export_execution_trace(include_memory=False)
            
            # Analyze performance with debug tools
            from crewgraph_ai.visualization.debug_tools import DebugTools
            debug_tools = DebugTools()
            
            bottlenecks = debug_tools.identify_performance_bottlenecks(
                execution_data=execution_data,
                threshold_percentile=0.9
            )
            
            return {
                "total_execution_time": execution_data.get("performance_metrics", {}).get("total_execution_time", 0),
                "bottlenecks": [
                    {
                        "component": b.component_id,
                        "type": b.bottleneck_type,
                        "severity": b.severity,
                        "issue": b.impact_description,
                        "recommendations": b.recommendations
                    }
                    for b in bottlenecks
                ],
                "recommendations": debug_tools._generate_overall_recommendations(),
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze performance: {e}")
            return {"error": str(e)}
    
    def validate_workflow(self) -> Dict[str, Any]:
        """
        Validate workflow configuration and structure.
        
        Performs comprehensive validation of agents, tasks, dependencies,
        and configuration to identify potential issues before execution.
        
        Returns:
            Validation report with issues categorized by severity.
            
        Example:
            ```python
            # Validate before execution
            validation = workflow.validate_workflow()
            
            if validation['summary']['errors'] > 0:
                print("❌ Workflow has critical errors:")
                for issue in validation['issues_by_severity']['error']:
                    print(f"  - {issue.message}")
            else:
                print("✅ Workflow validation passed")
            ```
        """
        try:
            from crewgraph_ai.visualization.debug_tools import DebugTools
            
            debug_tools = DebugTools()
            
            return debug_tools.validate_workflow(
                agents=self._agents,
                tasks=self._tasks,
                state=self._state,
                tool_registry=self._tool_registry
            )
            
        except Exception as e:
            logger.error(f"Failed to validate workflow: {e}")
            return {
                "summary": {"total_issues": 1, "errors": 1, "warnings": 0, "info": 0},
                "validation_error": str(e)
            }
    
    def __repr__(self) -> str:
        return f"CrewGraph(name='{self.name}', agents={len(self._agents)}, tasks={len(self._tasks)})"