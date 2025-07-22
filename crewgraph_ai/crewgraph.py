"""
Main CrewGraph class - High-level interface for the library
"""

import asyncio
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
from .utils.exceptions import CrewGraphError, ValidationError

logger = get_logger(__name__)


@dataclass
class CrewGraphConfig:
    """Configuration for CrewGraph instance"""
    memory_backend: Optional[BaseMemory] = None
    enable_planning: bool = True
    max_concurrent_tasks: int = 10
    task_timeout: float = 300.0  # 5 minutes
    enable_logging: bool = True
    log_level: str = "INFO"


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
    
    def __repr__(self) -> str:
        return f"CrewGraph(name='{self.name}', agents={len(self._agents)}, tasks={len(self._tasks)})"