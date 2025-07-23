"""
Production-ready agent wrapper with full CrewAI integration
"""

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from crewai import Agent
from pydantic import BaseModel, Field

from ..memory.base import BaseMemory
from ..utils.exceptions import CrewGraphError, ValidationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AgentStatus(Enum):
    """Agent execution status"""

    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentMetrics:
    """Agent performance metrics"""

    tasks_completed: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    success_rate: float = 1.0
    errors: List[str] = field(default_factory=list)


class AgentWrapper:
    """
    Production-ready wrapper around CrewAI Agent with enhanced capabilities.

    Features:
    - Memory persistence
    - Performance monitoring
    - Error handling and recovery
    - Async/sync execution
    - State management
    - Tool integration
    """

    def __init__(
        self,
        name: str,
        role: str = "",
        crew_agent: Optional[Agent] = None,
        state: Optional[Any] = None,
        memory: Optional[BaseMemory] = None,
        max_retries: int = 3,
        timeout: float = 300.0,
    ):
        """
        Initialize agent wrapper.

        Args:
            name: Unique agent identifier
            role: Agent role description
            crew_agent: CrewAI Agent instance
            state: Shared state manager
            memory: Memory backend
            max_retries: Maximum retry attempts
            timeout: Task timeout in seconds
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.crew_agent = crew_agent
        self.state = state
        self.memory = memory
        self.max_retries = max_retries
        self.timeout = timeout

        # Status and metrics
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()
        self.current_task: Optional[str] = None

        # Task queue and execution
        self._task_queue: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Callbacks
        self._on_task_start: Optional[Callable] = None
        self._on_task_complete: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

        # Communication
        self._message_handler: Optional[Callable] = None

        logger.info(f"Agent '{name}' initialized with ID: {self.id}")

    def set_callbacks(
        self,
        on_task_start: Optional[Callable] = None,
        on_task_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        """Set event callbacks."""
        self._on_task_start = on_task_start
        self._on_task_complete = on_task_complete
        self._on_error = on_error

    def add_to_queue(self, task: Dict[str, Any]) -> None:
        """
        Add task to agent's execution queue.

        Args:
            task: Task dictionary with 'name', 'prompt', 'context'
        """
        with self._lock:
            task["id"] = str(uuid.uuid4())
            task["queued_at"] = asyncio.get_event_loop().time()
            self._task_queue.append(task)

        logger.debug(f"Task '{task.get('name', 'unnamed')}' added to agent '{self.name}' queue")

    def get_queue_size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._task_queue)

    def execute_task(
        self,
        task_name: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single task synchronously.

        Args:
            task_name: Task identifier
            prompt: Task prompt/instruction
            context: Additional context data
            tools: Available tools

        Returns:
            Task execution result
        """
        start_time = asyncio.get_event_loop().time()
        self.status = AgentStatus.RUNNING
        self.current_task = task_name

        if self._on_task_start:
            self._on_task_start(self, task_name)

        try:
            # Get memory context
            memory_context = self._get_memory_context(task_name)

            # Prepare full context
            full_context = {
                **(context or {}),
                **memory_context,
                "agent_name": self.name,
                "task_name": task_name,
                "previous_results": self._get_previous_results(),
            }

            # Execute with CrewAI agent
            if self.crew_agent:
                result = self._execute_with_crew_agent(prompt, full_context, tools)
            else:
                result = self._execute_default(prompt, full_context)

            # Store result in memory
            if self.memory:
                self.memory.save(f"{self.name}:task:{task_name}:result", result)

            # Update metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(execution_time, True)

            # Update state
            if self.state:
                self.state.set(f"{self.name}:last_result", result)
                self.state.set(f"{self.name}:last_task", task_name)

            self.status = AgentStatus.COMPLETED

            if self._on_task_complete:
                self._on_task_complete(self, task_name, result)

            logger.info(
                f"Agent '{self.name}' completed task '{task_name}' in {execution_time:.2f}s"
            )

            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "agent": self.name,
                "task": task_name,
            }

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(execution_time, False)
            self.status = AgentStatus.ERROR

            error_msg = f"Agent '{self.name}' failed task '{task_name}': {str(e)}"
            logger.error(error_msg)

            if self._on_error:
                self._on_error(self, task_name, e)

            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "agent": self.name,
                "task": task_name,
            }
        finally:
            self.current_task = None

    async def execute_task_async(
        self,
        task_name: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Execute task asynchronously."""
        loop = asyncio.get_event_loop()

        # Run in thread pool to avoid blocking
        result = await loop.run_in_executor(
            self._executor, self.execute_task, task_name, prompt, context, tools
        )

        return result

    def _execute_with_crew_agent(
        self, prompt: str, context: Dict[str, Any], tools: Optional[List[Any]] = None
    ) -> Any:
        """Execute using CrewAI agent."""
        try:
            # Update agent with current tools if provided
            if tools and hasattr(self.crew_agent, "tools"):
                self.crew_agent.tools = tools

            # Execute the agent
            if hasattr(self.crew_agent, "execute"):
                return self.crew_agent.execute(prompt, context=context)
            else:
                # Fallback to direct call
                return self.crew_agent(prompt)

        except Exception as e:
            logger.error(f"CrewAI agent execution failed: {e}")
            raise

    def _execute_default(self, prompt: str, context: Dict[str, Any]) -> str:
        """Default execution when no CrewAI agent is provided."""
        logger.warning(f"No CrewAI agent provided for '{self.name}', using default execution")

        # Simple template-based execution
        result = f"Agent '{self.name}' processed: {prompt}"
        if context:
            result += f"\nContext: {context}"

        return result

    def _get_memory_context(self, task_name: str) -> Dict[str, Any]:
        """Get relevant context from memory."""
        if not self.memory:
            return {}

        context = {}

        # Get previous results for this agent
        prev_result = self.memory.load(f"{self.name}:last_result")
        if prev_result:
            context["previous_result"] = prev_result

        # Get specific task history
        task_history = self.memory.load(f"{self.name}:task:{task_name}:history")
        if task_history:
            context["task_history"] = task_history

        return context

    def _get_previous_results(self) -> Dict[str, Any]:
        """Get previous task results from state."""
        if not self.state:
            return {}

        return self.state.get(f"{self.name}:results", {})

    def _update_metrics(self, execution_time: float, success: bool):
        """Update agent performance metrics."""
        self.metrics.tasks_completed += 1
        self.metrics.total_execution_time += execution_time
        self.metrics.average_execution_time = (
            self.metrics.total_execution_time / self.metrics.tasks_completed
        )

        if success:
            # Update success rate
            total_tasks = self.metrics.tasks_completed
            current_successes = total_tasks - len(self.metrics.errors)
            self.metrics.success_rate = current_successes / total_tasks

    def get_metrics(self) -> AgentMetrics:
        """Get agent performance metrics."""
        return self.metrics

    def reset_metrics(self):
        """Reset agent metrics."""
        self.metrics = AgentMetrics()
        logger.info(f"Metrics reset for agent '{self.name}'")

    def save_state(self) -> Dict[str, Any]:
        """Save agent state."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "status": self.status.value,
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "total_execution_time": self.metrics.total_execution_time,
                "average_execution_time": self.metrics.average_execution_time,
                "success_rate": self.metrics.success_rate,
                "errors": self.metrics.errors.copy(),
            },
            "queue_size": self.get_queue_size(),
        }

    # Communication Protocol Implementation
    def receive_message(self, message: Any) -> None:
        """
        Handle incoming message from communication hub.

        Args:
            message: Message object from communication system
        """
        try:
            # Store message for later processing or handle immediately
            if hasattr(message, "content"):
                logger.info(f"Agent {self.name} received message: {message.content[:100]}...")

                # Trigger message handler if available
                if hasattr(self, "_message_handler") and self._message_handler:
                    self._message_handler(message)
                else:
                    # Default handling - just log for now
                    logger.debug(f"Agent {self.name} processing message from {message.sender_id}")

        except Exception as e:
            logger.error(f"Error processing message in agent {self.name}: {e}")

    def get_agent_id(self) -> str:
        """Get agent identifier for communication."""
        return self.id

    def set_message_handler(self, handler: Callable[[Any], None]):
        """Set custom message handler for this agent."""
        self._message_handler = handler
        logger.info(f"Message handler set for agent {self.name}")

    def __repr__(self) -> str:
        return (
            f"AgentWrapper(name='{self.name}', role='{self.role}', "
            f"status={self.status.value}, tasks_completed={self.metrics.tasks_completed})"
        )


class AgentPool:
    """
    Manages a pool of agents with load balancing and coordination.
    """

    def __init__(self, max_concurrent_agents: int = 5):
        """
        Initialize agent pool.

        Args:
            max_concurrent_agents: Maximum agents running concurrently
        """
        self.max_concurrent_agents = max_concurrent_agents
        self._agents: Dict[str, AgentWrapper] = {}
        self._active_agents: Dict[str, AgentWrapper] = {}
        self._lock = threading.Lock()

        logger.info(f"AgentPool initialized with max_concurrent_agents={max_concurrent_agents}")

    def add_agent(self, agent: AgentWrapper) -> None:
        """Add agent to pool."""
        with self._lock:
            self._agents[agent.name] = agent
        logger.info(f"Agent '{agent.name}' added to pool")

    def remove_agent(self, name: str) -> None:
        """Remove agent from pool."""
        with self._lock:
            if name in self._agents:
                del self._agents[name]
            if name in self._active_agents:
                del self._active_agents[name]
        logger.info(f"Agent '{name}' removed from pool")

    def get_agent(self, name: str) -> Optional[AgentWrapper]:
        """Get agent by name."""
        return self._agents.get(name)

    def get_available_agent(self, role: Optional[str] = None) -> Optional[AgentWrapper]:
        """
        Get an available agent, optionally filtered by role.

        Args:
            role: Required agent role (optional)

        Returns:
            Available agent or None
        """
        with self._lock:
            for agent in self._agents.values():
                # Check if agent is idle and matches role if specified
                if (
                    agent.status == AgentStatus.IDLE
                    and (not role or agent.role == role)
                    and agent.name not in self._active_agents
                ):
                    return agent

        return None

    def assign_task(
        self,
        task_name: str,
        prompt: str,
        agent_name: Optional[str] = None,
        role: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Assign task to an agent.

        Args:
            task_name: Task identifier
            prompt: Task prompt
            agent_name: Specific agent name (optional)
            role: Required role (optional)
            context: Task context

        Returns:
            Assigned agent name or None
        """
        if agent_name and agent_name in self._agents:
            agent = self._agents[agent_name]
        else:
            agent = self.get_available_agent(role)

        if not agent:
            logger.warning(f"No available agent found for task '{task_name}'")
            return None

        # Mark agent as active
        with self._lock:
            self._active_agents[agent.name] = agent

        # Add task to agent's queue
        task_data = {"name": task_name, "prompt": prompt, "context": context or {}}
        agent.add_to_queue(task_data)

        logger.info(f"Task '{task_name}' assigned to agent '{agent.name}'")
        return agent.name

    def get_pool_status(self) -> Dict[str, Any]:
        """Get overall pool status."""
        with self._lock:
            return {
                "total_agents": len(self._agents),
                "active_agents": len(self._active_agents),
                "idle_agents": len(self._agents) - len(self._active_agents),
                "agents": {
                    name: {
                        "status": agent.status.value,
                        "current_task": agent.current_task,
                        "queue_size": agent.get_queue_size(),
                        "metrics": {
                            "tasks_completed": agent.metrics.tasks_completed,
                            "success_rate": agent.metrics.success_rate,
                        },
                    }
                    for name, agent in self._agents.items()
                },
            }

    def list_agents(self) -> List[str]:
        """List all agent names."""
        return list(self._agents.keys())

    def shutdown(self):
        """Shutdown agent pool and cleanup resources."""
        with self._lock:
            for agent in self._agents.values():
                if hasattr(agent, "_executor"):
                    agent._executor.shutdown(wait=True)

        logger.info("AgentPool shutdown completed")
