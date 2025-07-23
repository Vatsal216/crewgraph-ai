"""
CrewGraph Agent Components for Langflow Integration

This module provides Langflow components that bridge to CrewGraph agents.

Created by: Vatsal216
Date: 2025-07-23
"""

from typing import Any, Dict, List, Optional

from .base import LangflowComponent, ComponentInput, ComponentOutput, ComponentMetadata

# Import CrewGraph agent functionality
try:
    from ....core.agents import AgentWrapper, AgentPool
    from ....core.tasks import TaskWrapper
    from ....memory.base import BaseMemory
    from ....memory.dict_memory import DictMemory
except ImportError:
    # Fallback for development
    AgentWrapper = None
    AgentPool = None
    TaskWrapper = None
    BaseMemory = None
    DictMemory = None


class CrewGraphAgentComponent(LangflowComponent):
    """
    Langflow component for CrewGraph AI agents
    
    This component allows users to create and configure CrewGraph agents
    in the Langflow visual interface.
    """
    
    def _get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="CrewGraphAgent",
            display_name="CrewGraph Agent",
            description="A CrewGraph AI agent that can perform tasks with specific roles and goals",
            category="CrewGraph AI",
            tags=["agent", "ai", "crewgraph", "automation"],
            icon="🤖",
            documentation_url="https://github.com/Vatsal216/crewgraph-ai/docs/agents",
        )
    
    def _get_inputs(self) -> List[ComponentInput]:
        return [
            ComponentInput(
                name="role",
                display_name="Agent Role",
                input_type="str",
                required=True,
                description="The role this agent will play (e.g., 'Data Analyst', 'Writer', 'Researcher')",
                default_value="Assistant"
            ),
            ComponentInput(
                name="goal",
                display_name="Agent Goal",
                input_type="str",
                required=True,
                multiline=True,
                description="The primary goal or objective for this agent",
                default_value="Help users accomplish their tasks effectively"
            ),
            ComponentInput(
                name="backstory",
                display_name="Agent Backstory",
                input_type="str",
                required=False,
                multiline=True,
                description="Background story and context for the agent's expertise",
                default_value=""
            ),
            ComponentInput(
                name="llm_model",
                display_name="LLM Model",
                input_type="str",
                required=False,
                description="Language model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')",
                options=["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-haiku"],
                default_value="gpt-3.5-turbo"
            ),
            ComponentInput(
                name="max_iter",
                display_name="Max Iterations",
                input_type="int",
                required=False,
                description="Maximum number of iterations for task execution",
                default_value=5,
                min_value=1,
                max_value=100
            ),
            ComponentInput(
                name="max_execution_time",
                display_name="Max Execution Time (seconds)",
                input_type="int",
                required=False,
                description="Maximum execution time in seconds",
                default_value=300,
                min_value=30,
                max_value=3600
            ),
            ComponentInput(
                name="verbose",
                display_name="Verbose Output",
                input_type="bool",
                required=False,
                description="Enable verbose output for debugging",
                default_value=True
            ),
            ComponentInput(
                name="allow_delegation",
                display_name="Allow Delegation",
                input_type="bool",
                required=False,
                description="Allow this agent to delegate tasks to other agents",
                default_value=False
            ),
            ComponentInput(
                name="tools",
                display_name="Tools",
                input_type="list",
                required=False,
                description="List of tools available to this agent (JSON array of tool names)",
                default_value=[]
            ),
            ComponentInput(
                name="memory_backend",
                display_name="Memory Backend",
                input_type="str",
                required=False,
                description="Memory backend to use for this agent",
                options=["dict", "redis", "faiss", "sql"],
                default_value="dict"
            ),
            ComponentInput(
                name="system_template",
                display_name="System Template",
                input_type="str",
                required=False,
                multiline=True,
                description="Custom system prompt template for the agent",
                default_value=""
            )
        ]
    
    def _get_outputs(self) -> List[ComponentOutput]:
        return [
            ComponentOutput(
                name="agent",
                display_name="Agent Instance",
                output_type="object",
                description="The configured CrewGraph agent instance"
            ),
            ComponentOutput(
                name="agent_id",
                display_name="Agent ID",
                output_type="str",
                description="Unique identifier for the created agent"
            ),
            ComponentOutput(
                name="configuration",
                display_name="Agent Configuration",
                output_type="dict",
                description="The complete agent configuration"
            ),
            ComponentOutput(
                name="status",
                display_name="Creation Status",
                output_type="str",
                description="Status of agent creation (success/error)"
            )
        ]
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent component - creates and configures a CrewGraph agent
        
        Args:
            inputs: Validated input parameters
            
        Returns:
            Dictionary containing the created agent and metadata
        """
        try:
            # Extract inputs
            role = inputs["role"]
            goal = inputs["goal"]
            backstory = inputs.get("backstory", "")
            llm_model = inputs.get("llm_model", "gpt-3.5-turbo")
            max_iter = inputs.get("max_iter", 5)
            max_execution_time = inputs.get("max_execution_time", 300)
            verbose = inputs.get("verbose", True)
            allow_delegation = inputs.get("allow_delegation", False)
            tools = inputs.get("tools", [])
            memory_backend = inputs.get("memory_backend", "dict")
            system_template = inputs.get("system_template", "")
            
            # Initialize memory backend
            memory = self._create_memory_backend(memory_backend)
            
            # Create agent configuration
            agent_config = {
                "role": role,
                "goal": goal,
                "backstory": backstory,
                "llm_model": llm_model,
                "max_iter": max_iter,
                "max_execution_time": max_execution_time,
                "verbose": verbose,
                "allow_delegation": allow_delegation,
                "tools": tools,
                "memory_backend": memory_backend,
                "system_template": system_template
            }
            
            # Create agent wrapper
            if AgentWrapper:
                # In a real implementation, this would create a proper CrewAI agent
                # For now, we'll create a mock agent wrapper
                agent = self._create_mock_agent(agent_config, memory)
            else:
                # Fallback for development
                agent = self._create_mock_agent(agent_config, memory)
            
            # Generate agent ID
            import uuid
            agent_id = str(uuid.uuid4())
            
            self.logger.info(f"Created CrewGraph agent '{role}' with ID: {agent_id}")
            
            return {
                "agent": agent,
                "agent_id": agent_id,
                "configuration": agent_config,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create agent: {e}")
            return {
                "agent": None,
                "agent_id": "",
                "configuration": inputs,
                "status": f"error: {str(e)}"
            }
    
    def _create_memory_backend(self, backend_type: str) -> Optional[BaseMemory]:
        """Create memory backend based on type"""
        try:
            if backend_type == "dict":
                return DictMemory() if DictMemory else None
            elif backend_type == "redis":
                # In real implementation, create RedisMemory
                return DictMemory() if DictMemory else None  # Fallback
            elif backend_type == "faiss":
                # In real implementation, create FAISSMemory
                return DictMemory() if DictMemory else None  # Fallback
            elif backend_type == "sql":
                # In real implementation, create SQLMemory
                return DictMemory() if DictMemory else None  # Fallback
            else:
                return DictMemory() if DictMemory else None
        except Exception as e:
            self.logger.warning(f"Failed to create {backend_type} memory backend: {e}")
            return DictMemory() if DictMemory else None
    
    def _create_mock_agent(self, config: Dict[str, Any], memory: Optional[BaseMemory]) -> Dict[str, Any]:
        """Create a mock agent for development/testing"""
        return {
            "type": "CrewGraphAgent",
            "role": config["role"],
            "goal": config["goal"],
            "backstory": config["backstory"],
            "configuration": config,
            "memory": memory,
            "status": "initialized",
            "created_at": "2025-07-23T20:00:00Z"
        }


class CrewGraphTaskComponent(LangflowComponent):
    """
    Langflow component for CrewGraph AI tasks
    
    This component allows users to create and configure tasks that can be
    assigned to CrewGraph agents.
    """
    
    def _get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="CrewGraphTask",
            display_name="CrewGraph Task",
            description="A task that can be executed by CrewGraph agents",
            category="CrewGraph AI",
            tags=["task", "workflow", "crewgraph"],
            icon="📋",
            documentation_url="https://github.com/Vatsal216/crewgraph-ai/docs/tasks",
        )
    
    def _get_inputs(self) -> List[ComponentInput]:
        return [
            ComponentInput(
                name="description",
                display_name="Task Description",
                input_type="str",
                required=True,
                multiline=True,
                description="Detailed description of what the task should accomplish"
            ),
            ComponentInput(
                name="expected_output",
                display_name="Expected Output",
                input_type="str",
                required=False,
                multiline=True,
                description="Description of the expected output format and content",
                default_value=""
            ),
            ComponentInput(
                name="agent",
                display_name="Assigned Agent",
                input_type="object",
                required=False,
                description="The agent responsible for executing this task (from CrewGraphAgent component)"
            ),
            ComponentInput(
                name="tools",
                display_name="Required Tools",
                input_type="list",
                required=False,
                description="List of tools required for this task",
                default_value=[]
            ),
            ComponentInput(
                name="context",
                display_name="Task Context",
                input_type="dict",
                required=False,
                description="Additional context data for the task",
                default_value={}
            ),
            ComponentInput(
                name="async_execution",
                display_name="Async Execution",
                input_type="bool",
                required=False,
                description="Whether to execute this task asynchronously",
                default_value=False
            ),
            ComponentInput(
                name="callback_function",
                display_name="Callback Function",
                input_type="str",
                required=False,
                description="Name of callback function to execute on completion",
                default_value=""
            ),
            ComponentInput(
                name="human_input",
                display_name="Require Human Input",
                input_type="bool",
                required=False,
                description="Whether this task requires human input/approval",
                default_value=False
            )
        ]
    
    def _get_outputs(self) -> List[ComponentOutput]:
        return [
            ComponentOutput(
                name="task",
                display_name="Task Instance",
                output_type="object",
                description="The configured CrewGraph task instance"
            ),
            ComponentOutput(
                name="task_id",
                display_name="Task ID",
                output_type="str",
                description="Unique identifier for the created task"
            ),
            ComponentOutput(
                name="configuration",
                display_name="Task Configuration",
                output_type="dict",
                description="The complete task configuration"
            ),
            ComponentOutput(
                name="status",
                display_name="Creation Status",
                output_type="str",
                description="Status of task creation (success/error)"
            )
        ]
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the task component - creates and configures a CrewGraph task
        
        Args:
            inputs: Validated input parameters
            
        Returns:
            Dictionary containing the created task and metadata
        """
        try:
            # Extract inputs
            description = inputs["description"]
            expected_output = inputs.get("expected_output", "")
            agent = inputs.get("agent")
            tools = inputs.get("tools", [])
            context = inputs.get("context", {})
            async_execution = inputs.get("async_execution", False)
            callback_function = inputs.get("callback_function", "")
            human_input = inputs.get("human_input", False)
            
            # Create task configuration
            task_config = {
                "description": description,
                "expected_output": expected_output,
                "agent": agent,
                "tools": tools,
                "context": context,
                "async_execution": async_execution,
                "callback_function": callback_function,
                "human_input": human_input
            }
            
            # Create task wrapper
            if TaskWrapper:
                # In a real implementation, this would create a proper CrewAI task
                task = self._create_mock_task(task_config)
            else:
                # Fallback for development
                task = self._create_mock_task(task_config)
            
            # Generate task ID
            import uuid
            task_id = str(uuid.uuid4())
            
            self.logger.info(f"Created CrewGraph task with ID: {task_id}")
            
            return {
                "task": task,
                "task_id": task_id,
                "configuration": task_config,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create task: {e}")
            return {
                "task": None,
                "task_id": "",
                "configuration": inputs,
                "status": f"error: {str(e)}"
            }
    
    def _create_mock_task(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a mock task for development/testing"""
        return {
            "type": "CrewGraphTask",
            "description": config["description"],
            "expected_output": config["expected_output"],
            "configuration": config,
            "status": "initialized",
            "created_at": "2025-07-23T20:00:00Z"
        }