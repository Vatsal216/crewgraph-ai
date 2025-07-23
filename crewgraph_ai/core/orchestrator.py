"""
Production-ready graph orchestrator with full LangGraph integration
"""

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

# from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessageGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ..types import (
    AfterExecutionCallback,
    AsyncAfterExecutionCallback,
    AsyncBeforeExecutionCallback,
    AsyncErrorCallback,
    BeforeExecutionCallback,
    ErrorCallback,
    ExecutionEvent,
    ExecutionId,
    ExecutionResult,
    NodeId,
    NodeStatus,
    PerformanceMetrics,
    StateDict,
    VisualizationFormat,
    WorkflowData,
    WorkflowId,
    WorkflowStatus,
)
from ..utils.exceptions import CrewGraphError, ExecutionError, ValidationError
from ..utils.logging import get_logger
from .state import SharedState
from .tasks import TaskChain, TaskResult, TaskStatus, TaskWrapper

logger = get_logger(__name__)


class DefaultState(TypedDict):
    """Enhanced state schema with LangChain message support"""

    messages: List[BaseMessage]  # ← Enhanced with proper message types
    current_node: str
    results: Dict[str, Any]
    errors: List[str]
    metadata: Dict[str, Any]
    conversation_id: Optional[str]
    task_context: Dict[str, Any]


class WorkflowStatus(Enum):
    """Workflow execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class WorkflowResult:
    """Workflow execution result with complete type annotations."""

    workflow_id: WorkflowId
    workflow_name: str
    success: bool
    results: StateDict = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphOrchestrator:
    """
    Production-ready orchestrator that provides full LangGraph access while adding
    advanced workflow management, error handling, and state persistence.

    Features:
    - Complete LangGraph StateGraph integration
    - Dynamic workflow modification at runtime
    - Parallel and sequential execution
    - Advanced error handling and recovery
    - Workflow persistence and resume
    - Real-time monitoring and metrics
    """

    def __init__(
        self,
        name: str = "default_workflow",
        max_concurrent_nodes: int = 5,
        enable_checkpoints: bool = True,
        checkpoint_interval: int = 5,
    ) -> None:
        """
        Initialize graph orchestrator.

        Args:
            name: Workflow identifier
            max_concurrent_nodes: Maximum parallel node execution
            enable_checkpoints: Enable workflow checkpointing
            checkpoint_interval: Checkpoint save interval (seconds)
        """
        self.name: str = name
        self.id: WorkflowId = str(uuid.uuid4())
        self.max_concurrent_nodes: int = max_concurrent_nodes
        self.enable_checkpoints: bool = enable_checkpoints
        self.checkpoint_interval: int = checkpoint_interval

        # LangGraph integration
        self._state_graph: Optional[StateGraph] = None
        self._message_graph: Optional[MessageGraph] = None  # ← New MessageGraph support
        self._compiled_graph: Optional[Any] = None
        self._compiled_message_graph: Optional[Any] = None  # ← Compiled MessageGraph
        self._checkpointer = MemorySaver() if enable_checkpoints else None
        self._workflow_mode: str = "state"  # "state" or "message"

        # Workflow components
        self._nodes: Dict[NodeId, Callable[..., Any]] = {}
        self._edges: List[Tuple[NodeId, NodeId]] = []
        self._conditional_edges: List[Dict[str, Any]] = []
        self._node_metadata: Dict[NodeId, Dict[str, Any]] = {}

        # Execution tracking
        self.status: WorkflowStatus = WorkflowStatus.PENDING
        self.shared_state: Optional[SharedState] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._execution_history: List[ExecutionEvent] = []
        self._node_status: Dict[NodeId, NodeStatus] = {}

        # Callbacks
        self._before_execution_callbacks: List[BeforeExecutionCallback] = []
        self._after_execution_callbacks: List[AfterExecutionCallback] = []
        self._error_callbacks: List[ErrorCallback] = []

        # Async callbacks
        self._async_before_execution_callbacks: List[AsyncBeforeExecutionCallback] = []
        self._async_after_execution_callbacks: List[AsyncAfterExecutionCallback] = []
        self._async_error_callbacks: List[AsyncErrorCallback] = []

        # Performance tracking
        self._performance_metrics: PerformanceMetrics = {}
        self._node_execution_times: Dict[NodeId, List[float]] = {}

        # Threading for parallel execution
        self._executor: Optional[ThreadPoolExecutor] = None
        self._active_executions: Set[NodeId] = set()
        self._execution_lock = threading.Lock()

        # Graph state
        self.is_built: bool = False
        self._entry_point: Optional[NodeId] = None
        self._finish_points: Set[NodeId] = set()

        logger.info(f"GraphOrchestrator '{name}' initialized with ID: {self.id}")
        self.is_built = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.current_thread_id: Optional[str] = None

        # Task management
        self._tasks: Dict[str, TaskWrapper] = {}
        self._task_chains: List[TaskChain] = []
        self._execution_results: Dict[str, TaskResult] = {}

        # Concurrency control
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_nodes)
        self._lock = threading.Lock()

        # Callbacks
        self._on_node_start: Optional[Callable] = None
        self._on_node_complete: Optional[Callable] = None
        self._on_workflow_complete: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

        logger.info(f"GraphOrchestrator '{name}' initialized with ID: {self.id}")

    def get_langgraph(self) -> Optional[StateGraph]:
        """
        Get direct access to the underlying LangGraph StateGraph.

        Returns:
            StateGraph instance for full LangGraph feature access
        """
        return self._state_graph

    def get_message_graph(self) -> Optional[MessageGraph]:
        """
        Get direct access to the underlying LangGraph MessageGraph.

        Returns:
            MessageGraph instance for conversation-based workflows
        """
        return self._message_graph

    def get_compiled_graph(self) -> Optional[Any]:
        """
        Get the compiled LangGraph for execution.

        Returns:
            Compiled graph instance
        """
        return self._compiled_graph

    def get_compiled_message_graph(self) -> Optional[Any]:
        """
        Get the compiled MessageGraph for execution.

        Returns:
            Compiled MessageGraph instance
        """
        return self._compiled_message_graph

    def create_state_graph(self, state_schema: Optional[Type[TypedDict]] = None) -> StateGraph:
        """
        Create a new LangGraph StateGraph with full feature access.

        Args:
            state_schema: Custom state schema (optional)

        Returns:
            StateGraph instance with full LangGraph capabilities
        """
        if state_schema:
            self._state_graph = StateGraph(state_schema)
        else:
            # Use enhanced default state schema with message support
            self._state_graph = StateGraph(DefaultState)

        logger.info(f"StateGraph created for workflow '{self.name}' with message support")
        return self._state_graph

    def add_node(
        self, name: str, func: Callable, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add node to workflow with full LangGraph integration.

        Args:
            name: Node name
            func: Node function
            metadata: Node metadata
        """
        if not self._state_graph:
            self.create_state_graph()

        # Check if node already exists
        if name in self._nodes:
            logger.debug(f"Node '{name}' already exists, skipping addition")
            return

        # Wrap function for enhanced monitoring
        wrapped_func = self._wrap_node_function(name, func)

        # Add to LangGraph
        self._state_graph.add_node(name, wrapped_func)

        # Store locally
        self._nodes[name] = func
        self._node_metadata[name] = metadata or {}

        logger.info(f"Node '{name}' added to workflow '{self.name}'")

    def add_edge(self, from_node: str, to_node: str) -> None:
        """
        Add edge between nodes with full LangGraph integration.

        Args:
            from_node: Source node name
            to_node: Target node name
        """
        if not self._state_graph:
            self.create_state_graph()

        self._state_graph.add_edge(from_node, to_node)
        self._edges.append((from_node, to_node))

        logger.info(f"Edge added: {from_node} -> {to_node}")

    def add_conditional_edges(
        self, source: str, condition: Callable, mapping: Dict[str, str], then: Optional[str] = None
    ) -> None:
        """
        Add conditional edges with full LangGraph support.

        Args:
            source: Source node name
            condition: Condition function
            mapping: Condition -> target node mapping
            then: Default target node
        """
        if not self._state_graph:
            self.create_state_graph()

        # Enhanced condition wrapper for monitoring
        wrapped_condition = self._wrap_condition_function(source, condition)

        if then:
            self._state_graph.add_conditional_edges(source, wrapped_condition, mapping, then)
        else:
            self._state_graph.add_conditional_edges(source, wrapped_condition, mapping)

        self._conditional_edges.append(
            {"source": source, "condition": condition, "mapping": mapping, "then": then}
        )

        logger.info(f"Conditional edges added from '{source}' with {len(mapping)} conditions")

    def set_entry_point(self, node: str) -> None:
        """Set workflow entry point."""
        if not self._state_graph:
            self.create_state_graph()

        self._state_graph.set_entry_point(node)
        logger.info(f"Entry point set to '{node}'")

    def set_finish_point(self, node: str) -> None:
        """Set workflow finish point."""
        if not self._state_graph:
            self.create_state_graph()

        self._state_graph.add_edge(node, END)
        logger.info(f"Finish point set to '{node}'")

    def add_task_node(self, task: TaskWrapper) -> None:
        """
        Add TaskWrapper as a workflow node.

        Args:
            task: TaskWrapper instance
        """
        self._tasks[task.name] = task

        # Create node function from task
        def task_node(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = task.execute(state)

                # Update state with task result
                new_state = state.copy()
                new_state["results"] = new_state.get("results", {})
                new_state["results"][task.name] = result.result if result.success else None

                if result.success:
                    new_state["current_node"] = task.name
                else:
                    new_state["errors"] = new_state.get("errors", [])
                    new_state["errors"].append(f"Task '{task.name}' failed: {result.error}")

                self._execution_results[task.name] = result
                return new_state

            except Exception as e:
                logger.error(f"Task node '{task.name}' failed: {e}")
                state["errors"] = state.get("errors", [])
                state["errors"].append(f"Task '{task.name}' exception: {str(e)}")
                return state

        self.add_node(
            task.name,
            task_node,
            {"type": "task", "task_id": task.id, "description": task.description},
        )

        logger.info(f"Task '{task.name}' added as workflow node")

    def add_chain(self, chain: TaskChain) -> None:
        """
        Add TaskChain as sequential nodes.

        Args:
            chain: TaskChain instance
        """
        self._task_chains.append(chain)

        # Add each task as a node
        for task in chain.tasks:
            if task.name not in self._tasks:
                self.add_task_node(task)

        # Add sequential edges
        for i in range(len(chain.tasks) - 1):
            current_task = chain.tasks[i]
            next_task = chain.tasks[i + 1]
            self.add_edge(current_task.name, next_task.name)

        # Connect first task to START (entry point)
        if chain.tasks:
            first_task = chain.tasks[0]
            if hasattr(self._state_graph, 'set_entry_point'):
                self._state_graph.set_entry_point(first_task.name)
            else:
                from langgraph.graph import START
                self._state_graph.add_edge(START, first_task.name)

        logger.info(f"TaskChain '{chain.name}' added with {len(chain.tasks)} sequential nodes")

    def build_graph(self) -> None:
        """
        Build and compile the workflow graph (supports both StateGraph and MessageGraph).
        """
        try:
            if self._workflow_mode == "message":
                # Build MessageGraph workflow
                if not self._message_graph:
                    raise CrewGraphError("No MessageGraph created. Enable message mode first.")

                # Compile the MessageGraph with checkpointing if enabled
                if self._checkpointer:
                    self._compiled_message_graph = self._message_graph.compile(
                        checkpointer=self._checkpointer
                    )
                else:
                    self._compiled_message_graph = self._message_graph.compile()

                logger.info(f"MessageGraph workflow '{self.name}' compiled successfully")

            else:
                # Build StateGraph workflow (default)
                if not self._state_graph:
                    raise CrewGraphError("No StateGraph created. Add nodes first.")

                # Compile the graph with checkpointing if enabled
                if self._checkpointer:
                    self._compiled_graph = self._state_graph.compile(
                        checkpointer=self._checkpointer
                    )
                else:
                    self._compiled_graph = self._state_graph.compile()

                logger.info(f"StateGraph workflow '{self.name}' compiled successfully")

            self.is_built = True

        except Exception as e:
            logger.error(f"Failed to build workflow '{self.name}': {e}")
            raise CrewGraphError(f"Workflow build failed: {e}")

    def execute(
        self, state: Union[SharedState, Dict[str, Any]], config: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Execute workflow with full LangGraph integration.

        Args:
            state: Initial state (SharedState or dict)
            config: Execution configuration

        Returns:
            WorkflowResult with execution details
        """
        if not self.is_built:
            self.build_graph()

        if not self._compiled_graph:
            raise ExecutionError("No compiled graph available")

        self.status = WorkflowStatus.RUNNING
        self.start_time = time.time()
        self.current_thread_id = str(uuid.uuid4())

        try:
            # Convert SharedState to dict if needed
            if hasattr(state, "to_dict"):
                initial_state = state.to_dict()
            else:
                initial_state = dict(state)

            # Add default state fields
            initial_state.setdefault("messages", [])
            initial_state.setdefault("current_node", "")
            initial_state.setdefault("results", {})
            initial_state.setdefault("errors", [])
            initial_state.setdefault(
                "metadata",
                {"workflow_id": self.id, "workflow_name": self.name, "start_time": self.start_time},
            )

            # Prepare execution config
            exec_config = config or {}
            if self._checkpointer:
                exec_config["configurable"] = {"thread_id": self.current_thread_id}

            logger.info(
                f"Starting workflow execution '{self.name}' with thread_id: {self.current_thread_id}"
            )

            # Execute with LangGraph
            final_state = None
            for step_result in self._compiled_graph.stream(initial_state, config=exec_config):
                final_state = step_result

                # Real-time monitoring
                if isinstance(final_state, dict):
                    current_node = final_state.get("current_node", "")
                    if current_node:
                        logger.debug(f"Workflow step completed: {current_node}")

            self.end_time = time.time()
            execution_time = self.end_time - self.start_time

            # Process results
            if final_state and isinstance(final_state, dict):
                success = len(final_state.get("errors", [])) == 0
                tasks_completed = len([r for r in self._execution_results.values() if r.success])
                tasks_failed = len([r for r in self._execution_results.values() if not r.success])

                self.status = WorkflowStatus.COMPLETED if success else WorkflowStatus.FAILED

                result = WorkflowResult(
                    workflow_id=self.id,
                    workflow_name=self.name,
                    success=success,
                    results=final_state.get("results", {}),
                    error=(
                        "; ".join(final_state.get("errors", []))
                        if final_state.get("errors")
                        else None
                    ),
                    execution_time=execution_time,
                    tasks_completed=tasks_completed,
                    tasks_failed=tasks_failed,
                    metadata={
                        "thread_id": self.current_thread_id,
                        "nodes_executed": len(self._execution_results),
                        "final_state": final_state,
                    },
                )

                # Update original state if SharedState
                if hasattr(state, "update"):
                    state.update(final_state.get("results", {}))

                if self._on_workflow_complete:
                    self._on_workflow_complete(self, result)

                logger.info(f"Workflow '{self.name}' completed in {execution_time:.2f}s")
                return result

            else:
                raise ExecutionError("Invalid final state from workflow execution")

        except Exception as e:
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time if self.start_time else 0
            self.status = WorkflowStatus.FAILED

            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)

            if self._on_error:
                self._on_error(self, e)

            return WorkflowResult(
                workflow_id=self.id,
                workflow_name=self.name,
                success=False,
                error=error_msg,
                execution_time=execution_time,
            )

    # ============= MESSAGE HANDLING METHODS =============

    def add_human_message(self, content: str, **kwargs) -> None:
        """Add HumanMessage to current workflow state"""
        message = HumanMessage(content=content, **kwargs)
        # Store for next execution if state is available
        if hasattr(self, "_current_state") and self._current_state:
            self._current_state.setdefault("messages", []).append(message)
        logger.debug(f"Added HumanMessage: {content[:100]}...")

    def add_ai_message(self, content: str, **kwargs) -> None:
        """Add AIMessage to current workflow state"""
        message = AIMessage(content=content, **kwargs)
        # Store for next execution if state is available
        if hasattr(self, "_current_state") and self._current_state:
            self._current_state.setdefault("messages", []).append(message)
        logger.debug(f"Added AIMessage: {content[:100]}...")

    def get_message_history(self) -> List[BaseMessage]:
        """Get complete message history from workflow"""
        if hasattr(self, "_current_state") and self._current_state:
            return self._current_state.get("messages", [])
        return []

    def execute_with_messages(
        self,
        initial_messages: List[BaseMessage],
        state: Optional[Dict] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """Execute workflow with LangChain message context"""
        if not state:
            state = {}

        # Initialize enhanced state with messages
        enhanced_state = state.copy()
        enhanced_state["messages"] = initial_messages
        enhanced_state.setdefault("conversation_id", str(uuid.uuid4()))
        enhanced_state.setdefault("task_context", {})

        # Store current state for message tracking
        self._current_state = enhanced_state

        logger.info(f"Executing workflow with {len(initial_messages)} initial messages")
        return self.execute(enhanced_state, config)

    def execute_conversation(
        self,
        conversation_id: str,
        new_message: Union[str, BaseMessage],
        state: Optional[Dict] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """Execute workflow in conversation mode"""
        # Convert string to HumanMessage if needed
        if isinstance(new_message, str):
            message = HumanMessage(content=new_message)
        else:
            message = new_message

        # Initialize state with conversation context
        if not state:
            state = {}

        enhanced_state = state.copy()
        enhanced_state.setdefault("messages", []).append(message)
        enhanced_state["conversation_id"] = conversation_id
        enhanced_state.setdefault("task_context", {})

        # Store current state for message tracking
        self._current_state = enhanced_state

        logger.info(f"Executing conversation {conversation_id} with new message")
        return self.execute(enhanced_state, config)

    # ============= END MESSAGE HANDLING METHODS =============

    async def execute_async(
        self, state: Union[SharedState, Dict[str, Any]], config: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """Execute workflow asynchronously."""
        loop = asyncio.get_event_loop()

        # Run in thread pool to avoid blocking
        result = await loop.run_in_executor(None, self.execute, state, config)
        return result

    def execute_with_plan(
        self, execution_plan: Any, state: Union[SharedState, Dict[str, Any]]
    ) -> WorkflowResult:
        """
        Execute workflow with dynamic planning.

        Args:
            execution_plan: Execution plan from DynamicPlanner
            state: Initial state

        Returns:
            WorkflowResult
        """
        # Implementation for planned execution
        # This integrates with the planning module
        logger.info(f"Executing workflow '{self.name}' with dynamic plan")

        # For now, delegate to regular execution
        # Enhanced planning integration will be added
        return self.execute(state)

    async def execute_with_plan_async(
        self, execution_plan: Any, state: Union[SharedState, Dict[str, Any]]
    ) -> WorkflowResult:
        """Execute workflow with plan asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_with_plan, execution_plan, state)

    def pause_execution(self) -> None:
        """Pause workflow execution."""
        self.status = WorkflowStatus.PAUSED
        logger.info(f"Workflow '{self.name}' paused")

    def resume_execution(
        self, state: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """Resume paused workflow execution."""
        if not self.current_thread_id:
            raise ExecutionError("No thread ID available for resume")

        logger.info(f"Resuming workflow '{self.name}' with thread_id: {self.current_thread_id}")

        # Resume from checkpoint
        resume_config = config or {}
        resume_config["configurable"] = {"thread_id": self.current_thread_id}

        return self.execute(state or {}, resume_config)

    def cancel_execution(self) -> None:
        """Cancel workflow execution."""
        self.status = WorkflowStatus.CANCELLED
        logger.info(f"Workflow '{self.name}' cancelled")

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get workflow execution history from checkpoints."""
        if not self._checkpointer or not self.current_thread_id:
            return []

        try:
            # Get checkpoint history
            config = {"configurable": {"thread_id": self.current_thread_id}}
            history = list(self._compiled_graph.get_state_history(config))

            return [
                {
                    "step": i,
                    "timestamp": (
                        checkpoint.created_at if hasattr(checkpoint, "created_at") else None
                    ),
                    "state": checkpoint.values if hasattr(checkpoint, "values") else checkpoint,
                }
                for i, checkpoint in enumerate(history)
            ]

        except Exception as e:
            logger.error(f"Failed to get execution history: {e}")
            return []

    def _wrap_node_function(self, name: str, func: Callable) -> Callable:
        """Wrap node function for monitoring and error handling."""

        def wrapped_func(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()

            if self._on_node_start:
                self._on_node_start(self, name)

            try:
                logger.debug(f"Executing node '{name}'")
                result = func(state)

                execution_time = time.time() - start_time
                logger.debug(f"Node '{name}' completed in {execution_time:.2f}s")

                if self._on_node_complete:
                    self._on_node_complete(self, name, result)

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Node '{name}' failed after {execution_time:.2f}s: {e}")

                # Add error to state
                if isinstance(state, dict):
                    state["errors"] = state.get("errors", [])
                    state["errors"].append(f"Node '{name}' failed: {str(e)}")

                raise

        return wrapped_func

    def _wrap_condition_function(self, source: str, condition: Callable) -> Callable:
        """Wrap condition function for monitoring."""

        def wrapped_condition(state: Dict[str, Any]) -> str:
            try:
                logger.debug(f"Evaluating condition from '{source}'")
                result = condition(state)
                logger.debug(f"Condition from '{source}' returned: {result}")
                return result

            except Exception as e:
                logger.error(f"Condition from '{source}' failed: {e}")
                raise

        return wrapped_condition

    def set_callbacks(
        self,
        on_node_start: Optional[Callable] = None,
        on_node_complete: Optional[Callable] = None,
        on_workflow_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        """Set execution callbacks."""
        self._on_node_start = on_node_start
        self._on_node_complete = on_node_complete
        self._on_workflow_complete = on_workflow_complete
        self._on_error = on_error

    def reset(self) -> None:
        """Reset workflow state."""
        self.status = WorkflowStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.current_thread_id = None
        self._execution_results.clear()

        logger.info(f"Workflow '{self.name}' reset")

    def get_workflow_state(self) -> Dict[str, Any]:
        """Get current workflow state."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "is_built": self.is_built,
            "nodes": list(self._nodes.keys()),
            "edges": self._edges,
            "conditional_edges": len(self._conditional_edges),
            "execution_results": {
                name: result.to_dict() if hasattr(result, "to_dict") else str(result)
                for name, result in self._execution_results.items()
            },
            "current_thread_id": self.current_thread_id,
            "execution_time": (
                self.end_time - self.start_time if self.start_time and self.end_time else None
            ),
        }

    def visualize_graph(self) -> str:
        """Generate visual representation of the workflow graph."""
        if not self._compiled_graph:
            return "Graph not compiled yet"

        try:
            # Use LangGraph's built-in visualization
            from langgraph.graph.graph import draw_mermaid

            return draw_mermaid(self._compiled_graph.get_graph())
        except ImportError:
            # Fallback to text representation
            lines = [f"Workflow: {self.name}"]
            lines.append(f"Nodes: {', '.join(self._nodes.keys())}")

            for from_node, to_node in self._edges:
                lines.append(f"  {from_node} -> {to_node}")

            for edge_info in self._conditional_edges:
                source = edge_info["source"]
                mapping = edge_info["mapping"]
                lines.append(f"  {source} -> [conditional] -> {list(mapping.values())}")

            return "\n".join(lines)

    def shutdown(self):
        """Shutdown orchestrator and cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)

        logger.info(f"GraphOrchestrator '{self.name}' shutdown completed")

    def __repr__(self) -> str:
        return (
            f"GraphOrchestrator(name='{self.name}', nodes={len(self._nodes)}, "
            f"status={self.status.value}, built={self.is_built})"
        )


class WorkflowBuilder:
    """
    Fluent interface for building complex workflows with LangGraph integration.
    """

    def __init__(self, name: str = "workflow"):
        """Initialize workflow builder."""
        self.orchestrator = GraphOrchestrator(name)
        self._current_node: Optional[str] = None

    def add_node(self, name: str, func: Callable, metadata: Optional[Dict] = None):
        """Add node and return self for chaining."""
        self.orchestrator.add_node(name, func, metadata)
        self._current_node = name
        return self

    def add_task(self, task: TaskWrapper):
        """Add task as node and return self for chaining."""
        self.orchestrator.add_task_node(task)
        self._current_node = task.name
        return self

    def then(self, name: str, func: Callable, metadata: Optional[Dict] = None):
        """Add node and connect from current node."""
        if self._current_node:
            self.orchestrator.add_node(name, func, metadata)
            self.orchestrator.add_edge(self._current_node, name)
            self._current_node = name
        else:
            self.add_node(name, func, metadata)
        return self

    def branch(self, condition: Callable, mapping: Dict[str, str]):
        """Add conditional branching from current node."""
        if self._current_node:
            self.orchestrator.add_conditional_edges(self._current_node, condition, mapping)
        return self

    def parallel(self, *nodes: Tuple[str, Callable]):
        """Add parallel execution nodes."""
        if self._current_node:
            for name, func in nodes:
                self.orchestrator.add_node(name, func)
                self.orchestrator.add_edge(self._current_node, name)
        return self

    # ============= MESSAGEGRAPH WORKFLOW METHODS =============

    def enable_message_mode(self) -> None:
        """
        Switch orchestrator to message-based workflow mode using MessageGraph.

        This enables conversation-aware workflows where agents communicate
        through structured LangChain messages instead of generic state.
        """
        self._workflow_mode = "message"
        self._message_graph = MessageGraph()
        logger.info(f"Enabled MessageGraph mode for workflow '{self.name}'")

    def create_message_graph(self) -> MessageGraph:
        """
        Create and return a new MessageGraph for conversation-based workflows.

        Returns:
            MessageGraph instance configured with checkpointing
        """
        if not self._message_graph:
            self._message_graph = MessageGraph()
            self._workflow_mode = "message"

        logger.info(f"Created MessageGraph for workflow '{self.name}'")
        return self._message_graph

    def add_message_node(self, name: str, agent_func: Callable) -> "GraphOrchestrator":
        """
        Add a node that processes messages in MessageGraph workflow.

        Args:
            name: Node identifier
            agent_func: Function that takes List[BaseMessage] and returns BaseMessage or str

        Returns:
            Self for method chaining
        """
        if not self._message_graph:
            self.create_message_graph()

        def message_wrapper(messages: List[BaseMessage]) -> List[BaseMessage]:
            """Wrapper to ensure proper message handling"""
            try:
                # Call the agent function with messages
                response = agent_func(messages, {})  # Empty state for message mode

                # Convert response to proper message type
                if isinstance(response, str):
                    new_message = AIMessage(content=response)
                elif isinstance(response, BaseMessage):
                    new_message = response
                else:
                    # Handle TaskResult or other types
                    content = str(response)
                    if hasattr(response, "result"):
                        content = str(response.result)
                    new_message = AIMessage(content=content)

                # Return updated message list
                return messages + [new_message]

            except Exception as e:
                logger.error(f"Error in message node '{name}': {e}")
                error_message = AIMessage(
                    content=f"Error in {name}: {str(e)}",
                    additional_kwargs={"error": True, "node": name},
                )
                return messages + [error_message]

        self._message_graph.add_node(name, message_wrapper)
        self._nodes[name] = message_wrapper

        logger.info(f"Added message node '{name}' to MessageGraph")
        return self

    def add_conversation_agent(self, name: str, agent_func: Callable) -> "GraphOrchestrator":
        """
        Add a conversation-aware agent to the MessageGraph workflow.

        Args:
            name: Agent identifier
            agent_func: Function that processes conversation context

        Returns:
            Self for method chaining
        """
        if not self._message_graph:
            self.create_message_graph()

        def conversation_wrapper(messages: List[BaseMessage]) -> List[BaseMessage]:
            """Enhanced wrapper for conversation-aware agents"""
            try:
                # Extract conversation context
                conversation_context = {
                    "messages": messages,
                    "message_count": len(messages),
                    "latest_human_message": None,
                    "latest_ai_message": None,
                    "conversation_summary": "",
                }

                # Get latest messages for context
                for msg in reversed(messages):
                    if (
                        isinstance(msg, HumanMessage)
                        and not conversation_context["latest_human_message"]
                    ):
                        conversation_context["latest_human_message"] = msg.content
                    elif (
                        isinstance(msg, AIMessage) and not conversation_context["latest_ai_message"]
                    ):
                        conversation_context["latest_ai_message"] = msg.content

                    if (
                        conversation_context["latest_human_message"]
                        and conversation_context["latest_ai_message"]
                    ):
                        break

                # Create conversation summary
                human_count = sum(1 for msg in messages if isinstance(msg, HumanMessage))
                ai_count = sum(1 for msg in messages if isinstance(msg, AIMessage))
                conversation_context["conversation_summary"] = (
                    f"Conversation with {human_count} human messages and {ai_count} AI responses"
                )

                # Call agent with enhanced context
                response = agent_func(messages, conversation_context)

                # Process response
                if isinstance(response, BaseMessage):
                    new_message = response
                elif isinstance(response, str):
                    new_message = AIMessage(
                        content=response,
                        additional_kwargs={
                            "agent": name,
                            "conversation_turn": len(messages) + 1,
                            "timestamp": time.time(),
                        },
                    )
                else:
                    # Handle complex responses
                    content = str(response)
                    if hasattr(response, "result"):
                        content = str(response.result)

                    new_message = AIMessage(
                        content=content,
                        additional_kwargs={
                            "agent": name,
                            "conversation_turn": len(messages) + 1,
                            "timestamp": time.time(),
                            "response_type": type(response).__name__,
                        },
                    )

                return messages + [new_message]

            except Exception as e:
                logger.error(f"Error in conversation agent '{name}': {e}")
                error_message = AIMessage(
                    content=f"I encountered an error while processing your request: {str(e)}",
                    additional_kwargs={"error": True, "agent": name, "timestamp": time.time()},
                )
                return messages + [error_message]

        self._message_graph.add_node(name, conversation_wrapper)
        self._nodes[name] = conversation_wrapper

        logger.info(f"Added conversation agent '{name}' to MessageGraph")
        return self

    def execute_conversation(
        self, initial_messages: List[BaseMessage], config: Optional[Dict[str, Any]] = None
    ) -> List[BaseMessage]:
        """
        Execute conversation-based workflow using MessageGraph.

        Args:
            initial_messages: Starting messages for the conversation
            config: Additional configuration for execution

        Returns:
            Complete message history after workflow execution
        """
        if not self._message_graph:
            raise CrewGraphError("MessageGraph not created. Call create_message_graph() first.")

        if not self.is_built:
            raise CrewGraphError("MessageGraph not built. Call build() first.")

        try:
            start_time = time.time()
            self.status = WorkflowStatus.RUNNING

            logger.info(f"Executing conversation with {len(initial_messages)} initial messages")

            # Compile the MessageGraph if not already compiled
            if not self._compiled_message_graph:
                self._compiled_message_graph = self._message_graph.compile(
                    checkpointer=self._checkpointer
                )

            # Execute the conversation
            result_messages = self._compiled_message_graph.invoke(
                initial_messages, config=config or {}
            )

            self.status = WorkflowStatus.COMPLETED
            execution_time = time.time() - start_time

            logger.info(
                f"Conversation completed in {execution_time:.2f}s with {len(result_messages)} messages"
            )

            return result_messages

        except Exception as e:
            self.status = WorkflowStatus.FAILED
            logger.error(f"Conversation execution failed: {e}")
            raise ExecutionError(f"Conversation execution failed: {e}") from e

    def add_message_edge(self, from_node: str, to_node: str) -> "GraphOrchestrator":
        """
        Add edge between nodes in MessageGraph.

        Args:
            from_node: Source node name
            to_node: Target node name

        Returns:
            Self for method chaining
        """
        if not self._message_graph:
            self.create_message_graph()

        self._message_graph.add_edge(from_node, to_node)
        self._edges.append((from_node, to_node))

        logger.info(f"Added message edge: {from_node} -> {to_node}")
        return self

    def set_message_entry_point(self, node_name: str) -> "GraphOrchestrator":
        """
        Set entry point for MessageGraph workflow.

        Args:
            node_name: Name of the entry node

        Returns:
            Self for method chaining
        """
        if not self._message_graph:
            self.create_message_graph()

        self._message_graph.set_entry_point(node_name)
        logger.info(f"Set message entry point: {node_name}")
        return self

    def set_message_finish_point(self, node_name: str) -> "GraphOrchestrator":
        """
        Set finish point for MessageGraph workflow.

        Args:
            node_name: Name of the finish node

        Returns:
            Self for method chaining
        """
        if not self._message_graph:
            self.create_message_graph()

        self._message_graph.set_finish_point(node_name)
        logger.info(f"Set message finish point: {node_name}")
        return self

    # ============= END MESSAGEGRAPH WORKFLOW METHODS =============

    def with_conversation(self, messages: List[BaseMessage]):
        """Initialize workflow with conversation context"""
        # Store initial messages for the workflow
        self._initial_messages = messages
        return self

    def add_conversation_node(self, name: str, agent_func: Callable):
        """Add node that handles conversation messages"""

        def conversation_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state.get("messages", [])

            # Execute agent function with message context
            try:
                result = agent_func(messages, state)

                # Handle different result types
                if isinstance(result, str):
                    # Create AI message from string result
                    ai_message = AIMessage(
                        content=result,
                        additional_kwargs={"node_name": name, "timestamp": time.time()},
                    )
                    state["messages"].append(ai_message)
                elif isinstance(result, BaseMessage):
                    # Add message directly
                    state["messages"].append(result)
                elif isinstance(result, dict):
                    # Update state with result dict
                    state.update(result)
                    # If result contains a message, add it
                    if "message" in result:
                        if isinstance(result["message"], str):
                            ai_message = AIMessage(content=result["message"])
                            state["messages"].append(ai_message)
                        elif isinstance(result["message"], BaseMessage):
                            state["messages"].append(result["message"])

                # Update current node
                state["current_node"] = name

                return state

            except Exception as e:
                logger.error(f"Conversation node '{name}' failed: {e}")
                # Add error message
                error_message = AIMessage(
                    content=f"Error in {name}: {str(e)}",
                    additional_kwargs={"error": True, "node_name": name},
                )
                state["messages"].append(error_message)
                state.setdefault("errors", []).append(f"Node '{name}' failed: {str(e)}")
                return state

        return self.add_node(name, conversation_wrapper)

    def add_human_input_node(self, name: str, prompt: str = "Enter your message:"):
        """Add node that waits for human input"""

        def human_input_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            # In a real implementation, this would handle user input
            # For now, we'll check if there's pending human input in state
            human_input = state.get("pending_human_input", "")

            if human_input:
                human_message = HumanMessage(
                    content=human_input,
                    additional_kwargs={"node_name": name, "timestamp": time.time()},
                )
                state["messages"].append(human_message)
                # Clear pending input
                state.pop("pending_human_input", None)

            state["current_node"] = name
            return state

        return self.add_node(name, human_input_wrapper)

    def add_memory_node(self, name: str, memory_backend, conversation_id: str):
        """Add node that saves/loads conversation from memory"""

        def memory_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Save current conversation to memory
                messages = state.get("messages", [])
                if messages:
                    success = memory_backend.save_conversation(conversation_id, messages)
                    if success:
                        logger.info(
                            f"Saved {len(messages)} messages to conversation {conversation_id}"
                        )
                    else:
                        logger.warning(f"Failed to save conversation {conversation_id}")

                state["current_node"] = name
                return state

            except Exception as e:
                logger.error(f"Memory node '{name}' failed: {e}")
                state.setdefault("errors", []).append(f"Memory operation failed: {str(e)}")
                return state

        return self.add_node(name, memory_wrapper)

    def with_message_flow(self):
        """Configure workflow for message-based execution using MessageGraph"""
        # Enable message mode
        self.enable_message_mode()

        # Create default conversation state if using StateGraph
        if self._workflow_mode == "state":
            if not hasattr(self, "_initial_messages"):
                self._initial_messages = []

            # Set entry point to message initialization if no entry point set
            def init_messages(state: Dict[str, Any]) -> Dict[str, Any]:
                if "messages" not in state:
                    state["messages"] = getattr(self, "_initial_messages", [])
                return state

            # Add initialization node for StateGraph
            self.add_node("__init_messages__", init_messages)

        logger.info(
            f"Configured message flow for workflow '{self.name}' in {self._workflow_mode} mode"
        )
        return self

    def add_conversation_node(self, name: str, agent_func: Callable):
        """Add conversation-aware node (works with both StateGraph and MessageGraph)"""
        if self._workflow_mode == "message":
            # Use MessageGraph approach
            return self.add_conversation_agent(name, agent_func)
        else:
            # Use StateGraph approach with message context
            def message_context_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    messages = state.get("messages", [])

                    # Call agent with message context
                    response = agent_func(messages, state)

                    # Convert response to message and add to state
                    if isinstance(response, str):
                        new_message = AIMessage(
                            content=response,
                            additional_kwargs={"agent": name, "timestamp": time.time()},
                        )
                    elif isinstance(response, BaseMessage):
                        new_message = response
                    else:
                        # Handle complex responses
                        content = str(response)
                        if hasattr(response, "result"):
                            content = str(response.result)

                        new_message = AIMessage(
                            content=content,
                            additional_kwargs={
                                "agent": name,
                                "timestamp": time.time(),
                                "response_type": type(response).__name__,
                            },
                        )

                    # Update state with new message
                    updated_messages = messages + [new_message]
                    state["messages"] = updated_messages
                    state["current_node"] = name

                    return state

                except Exception as e:
                    logger.error(f"Conversation node '{name}' failed: {e}")
                    error_message = AIMessage(
                        content=f"Error in {name}: {str(e)}",
                        additional_kwargs={"error": True, "agent": name, "timestamp": time.time()},
                    )

                    messages = state.get("messages", [])
                    state["messages"] = messages + [error_message]
                    state.setdefault("errors", []).append(f"Node '{name}' failed: {str(e)}")

                    return state

            return self.add_node(name, message_context_wrapper)

        def init_messages(state):
            """Initialize conversation state with default values"""
            state.setdefault("conversation_id", str(uuid.uuid4()))
            state.setdefault("task_context", {})
            return state

        self.orchestrator.add_node("__init_messages__", init_messages)
        self.orchestrator.set_entry_point("__init_messages__")
        self._current_node = "__init_messages__"

        return self

    # ============= END MESSAGE-AWARE WORKFLOW BUILDING =============

    def build(self) -> GraphOrchestrator:
        """Build and return the orchestrator."""
        self.orchestrator.build_graph()
        return self.orchestrator

    # ============= VISUALIZATION METHODS =============

    def visualize_workflow(
        self, output_path: Optional[str] = None, format: VisualizationFormat = "html"
    ) -> str:
        """
        Generate visual representation of workflow graph.

        Args:
            output_path: Optional custom output path
            format: Output format ('html', 'png', 'svg', 'pdf')

        Returns:
            Path to generated visualization file

        Example:
            ```python
            orchestrator = GraphOrchestrator("my_workflow")
            # ... add nodes and edges ...
            viz_path = orchestrator.visualize_workflow(format="html")
            print(f"Visualization saved to: {viz_path}")
            ```
        """
        try:
            from ..visualization.workflow_visualizer import WorkflowVisualizer

            visualizer = WorkflowVisualizer(output_dir=output_path or "visualizations")

            # Convert orchestrator state to visualization data
            workflow_data = self._extract_workflow_data()

            return visualizer.visualize_workflow_graph(
                workflow_data=workflow_data,
                title=f"Workflow: {self.name}",
                format=format,
                show_details=True,
            )

        except ImportError:
            raise CrewGraphError(
                "Visualization dependencies not available. "
                "Install with: pip install crewgraph-ai[visualization]"
            )
        except Exception as e:
            logger.error(f"Failed to visualize workflow: {e}")
            raise CrewGraphError(f"Workflow visualization failed: {e}")

    def export_execution_trace(self, include_memory: bool = True) -> StateDict:
        """
        Export detailed execution trace for debugging.

        Args:
            include_memory: Whether to include memory state information

        Returns:
            Dictionary containing detailed execution trace

        Example:
            ```python
            trace = orchestrator.export_execution_trace(include_memory=True)
            print(f"Executed {len(trace['execution_events'])} events")
            ```
        """
        try:
            trace_data = {
                "workflow_id": self.id,
                "workflow_name": self.name,
                "export_timestamp": time.time(),
                "workflow_status": self.status.value,
                "execution_events": self._collect_execution_events(),
                "node_statistics": self._calculate_node_statistics(),
                "performance_metrics": self._gather_performance_metrics(),
                "error_summary": self._summarize_errors(),
            }

            if include_memory and hasattr(self, "shared_state") and self.shared_state:
                try:
                    from ..visualization.memory_inspector import MemoryInspector

                    inspector = MemoryInspector(self.shared_state.memory)
                    trace_data["memory_analysis"] = inspector.get_memory_usage_report()
                except Exception as e:
                    logger.warning(f"Failed to include memory analysis: {e}")
                    trace_data["memory_analysis"] = {"error": str(e)}

            return trace_data

        except Exception as e:
            logger.error(f"Failed to export execution trace: {e}")
            raise CrewGraphError(f"Execution trace export failed: {e}")

    def dump_memory_state(self, backend_details: bool = False) -> StateDict:
        """
        Dump current memory state for inspection.

        Args:
            backend_details: Whether to include detailed backend information

        Returns:
            Dictionary containing memory state dump

        Example:
            ```python
            memory_dump = orchestrator.dump_memory_state(backend_details=True)
            print(f"Memory backend: {memory_dump['backend_type']}")
            ```
        """
        try:
            from ..visualization.memory_inspector import MemoryInspector

            if not hasattr(self, "shared_state") or not self.shared_state:
                return {"error": "No shared state available", "timestamp": time.time()}

            inspector = MemoryInspector(self.shared_state.memory)

            return inspector.dump_memory_state(
                include_backend_details=backend_details,
                include_gc_objects=False,  # Avoid performance impact
            )

        except ImportError:
            raise CrewGraphError(
                "Visualization dependencies not available. "
                "Install with: pip install crewgraph-ai[visualization]"
            )
        except Exception as e:
            logger.error(f"Failed to dump memory state: {e}")
            raise CrewGraphError(f"Memory state dump failed: {e}")

    def generate_debug_report(self) -> StateDict:
        """
        Generate comprehensive debug report.

        Returns:
            Dictionary containing comprehensive debugging information

        Example:
            ```python
            debug_report = orchestrator.generate_debug_report()
            print(f"Found {len(debug_report['validation_issues'])} issues")
            ```
        """
        try:
            from ..visualization.debug_tools import DebugTools

            debug_tools = DebugTools()

            # Collect workflow components for analysis
            workflow_data = {
                "orchestrator": self,
                "nodes": getattr(self, "_nodes", {}),
                "edges": getattr(self, "_edges", []),
                "state": getattr(self, "shared_state", None),
                "status": self.status.value,
                "execution_history": getattr(self, "_execution_history", []),
            }

            # Generate comprehensive debug report
            report_path = debug_tools.generate_debug_report(
                workflow_data=workflow_data, include_visualizations=True
            )

            return {
                "report_generated": True,
                "report_path": report_path,
                "workflow_id": self.id,
                "workflow_name": self.name,
                "timestamp": time.time(),
            }

        except ImportError:
            raise CrewGraphError(
                "Visualization dependencies not available. "
                "Install with: pip install crewgraph-ai[visualization]"
            )
        except Exception as e:
            logger.error(f"Failed to generate debug report: {e}")
            raise CrewGraphError(f"Debug report generation failed: {e}")

    # ============= VISUALIZATION HELPER METHODS =============

    def _extract_workflow_data(self) -> WorkflowData:
        """Extract workflow data for visualization."""
        nodes = []
        edges = []

        # Extract nodes information
        if hasattr(self, "_nodes"):
            for node_id, node_info in self._nodes.items():
                node_data = {
                    "id": node_id,
                    "name": node_id,  # Use ID as name if no better name available
                    "status": self._get_node_status(node_id),
                    "type": "task",  # Default type
                }

                # Add additional node information if available
                if isinstance(node_info, dict):
                    node_data.update(node_info)

                nodes.append(node_data)

        # Extract edges information
        if hasattr(self, "_edges"):
            for edge in self._edges:
                if isinstance(edge, dict):
                    edges.append(edge)
                elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    edges.append({"source": edge[0], "target": edge[1], "type": "dependency"})

        # If no nodes/edges found, try to extract from LangGraph
        if not nodes and self._state_graph:
            try:
                # This would require introspection of the LangGraph structure
                # Implementation depends on LangGraph's internal API
                pass
            except Exception:
                pass

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "workflow_id": self.id,
                "workflow_name": self.name,
                "status": self.status.value,
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        }

    def _get_node_status(self, node_id: NodeId) -> NodeStatus:
        """Get the current status of a node."""
        # This would check the node's execution status
        # Implementation depends on how node status is tracked
        if hasattr(self, "_node_status"):
            return self._node_status.get(node_id, "pending")
        return "pending"

    def _collect_execution_events(self) -> List[ExecutionEvent]:
        """Collect execution events for trace export."""
        events = []

        if hasattr(self, "_execution_history"):
            events.extend(self._execution_history)

        # Add workflow-level events
        events.append(
            {
                "event_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "event_type": "workflow_info",
                "node_id": "__workflow__",
                "message": f"Workflow {self.name} status: {self.status.value}",
                "metadata": {"workflow_id": self.id, "workflow_name": self.name},
            }
        )

        return events

    def _calculate_node_statistics(self) -> Dict[NodeId, Dict[str, Any]]:
        """Calculate statistics for each node."""
        stats = {}

        if hasattr(self, "_nodes"):
            for node_id in self._nodes.keys():
                stats[node_id] = {
                    "execution_count": 0,
                    "total_duration": 0.0,
                    "average_duration": 0.0,
                    "error_count": 0,
                    "last_execution": None,
                }

                # Calculate actual statistics from execution history
                if hasattr(self, "_execution_history"):
                    node_events = [
                        e for e in self._execution_history if e.get("node_id") == node_id
                    ]

                    stats[node_id]["execution_count"] = len(node_events)

                    # Calculate duration and error statistics
                    durations = [
                        e.get("duration", 0) for e in node_events if e.get("duration") is not None
                    ]
                    if durations:
                        stats[node_id]["total_duration"] = sum(durations)
                        stats[node_id]["average_duration"] = sum(durations) / len(durations)

                    error_events = [e for e in node_events if e.get("event_type") == "error"]
                    stats[node_id]["error_count"] = len(error_events)

                    if node_events:
                        stats[node_id]["last_execution"] = max(
                            e.get("timestamp", 0) for e in node_events
                        )

        return stats

    def _gather_performance_metrics(self) -> PerformanceMetrics:
        """Gather performance metrics."""
        metrics = {
            "workflow_start_time": getattr(self, "start_time", None),
            "workflow_end_time": getattr(self, "end_time", None),
            "total_execution_time": 0.0,
            "node_count": len(getattr(self, "_nodes", {})),
            "edge_count": len(getattr(self, "_edges", [])),
            "concurrent_executions": 0,
            "memory_usage": {},
        }

        # Calculate total execution time
        if hasattr(self, "start_time") and hasattr(self, "end_time"):
            if self.start_time and self.end_time:
                metrics["total_execution_time"] = self.end_time - self.start_time

        return metrics

    def _summarize_errors(self) -> Dict[str, Any]:
        """Summarize errors that occurred during execution."""
        error_summary = {
            "total_errors": 0,
            "errors_by_node": {},
            "errors_by_type": {},
            "recent_errors": [],
        }

        if hasattr(self, "_execution_history"):
            error_events = [e for e in self._execution_history if e.get("event_type") == "error"]

            error_summary["total_errors"] = len(error_events)

            # Group errors by node
            for error in error_events:
                node_id = error.get("node_id", "unknown")
                error_summary["errors_by_node"][node_id] = (
                    error_summary["errors_by_node"].get(node_id, 0) + 1
                )

            # Keep recent errors (last 10)
            error_summary["recent_errors"] = error_events[-10:] if error_events else []

        return error_summary
