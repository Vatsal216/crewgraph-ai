"""
Production-ready graph orchestrator with full LangGraph integration
"""

import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import MessageGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

from .tasks import TaskWrapper, TaskChain, TaskResult, TaskStatus
from .state import SharedState
from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError, ValidationError, ExecutionError

logger = get_logger(__name__)


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
    """Workflow execution result"""
    workflow_id: str
    workflow_name: str
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
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
    
    def __init__(self,
                 name: str = "default_workflow",
                 max_concurrent_nodes: int = 5,
                 enable_checkpoints: bool = True,
                 checkpoint_interval: int = 5):
        """
        Initialize graph orchestrator.
        
        Args:
            name: Workflow identifier
            max_concurrent_nodes: Maximum parallel node execution
            enable_checkpoints: Enable workflow checkpointing
            checkpoint_interval: Checkpoint save interval (seconds)
        """
        self.name = name
        self.id = str(uuid.uuid4())
        self.max_concurrent_nodes = max_concurrent_nodes
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_interval = checkpoint_interval
        
        # LangGraph integration
        self._state_graph: Optional[StateGraph] = None
        self._compiled_graph: Optional[Any] = None
        self._checkpointer = MemorySaver() if enable_checkpoints else None
        
        # Workflow components
        self._nodes: Dict[str, Callable] = {}
        self._edges: List[Tuple[str, str]] = []
        self._conditional_edges: List[Dict[str, Any]] = []
        self._node_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Execution tracking
        self.status = WorkflowStatus.PENDING
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
    
    def get_compiled_graph(self) -> Optional[Any]:
        """
        Get the compiled LangGraph for execution.
        
        Returns:
            Compiled graph instance
        """
        return self._compiled_graph
    
    def create_state_graph(self, state_schema: Optional[Any] = None) -> StateGraph:
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
            # Default state schema
            from typing_extensions import TypedDict
            
            class DefaultState(TypedDict):
                messages: List[str]
                current_node: str
                results: Dict[str, Any]
                errors: List[str]
                metadata: Dict[str, Any]
            
            self._state_graph = StateGraph(DefaultState)
        
        logger.info(f"StateGraph created for workflow '{self.name}'")
        return self._state_graph
    
    def add_node(self, 
                 name: str, 
                 func: Callable,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add node to workflow with full LangGraph integration.
        
        Args:
            name: Node name
            func: Node function
            metadata: Node metadata
        """
        if not self._state_graph:
            self.create_state_graph()
        
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
    
    def add_conditional_edges(self,
                             source: str,
                             condition: Callable,
                             mapping: Dict[str, str],
                             then: Optional[str] = None) -> None:
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
        
        self._conditional_edges.append({
            'source': source,
            'condition': condition,
            'mapping': mapping,
            'then': then
        })
        
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
                new_state['results'] = new_state.get('results', {})
                new_state['results'][task.name] = result.result if result.success else None
                
                if result.success:
                    new_state['current_node'] = task.name
                else:
                    new_state['errors'] = new_state.get('errors', [])
                    new_state['errors'].append(f"Task '{task.name}' failed: {result.error}")
                
                self._execution_results[task.name] = result
                return new_state
                
            except Exception as e:
                logger.error(f"Task node '{task.name}' failed: {e}")
                state['errors'] = state.get('errors', [])
                state['errors'].append(f"Task '{task.name}' exception: {str(e)}")
                return state
        
        self.add_node(task.name, task_node, {
            'type': 'task',
            'task_id': task.id,
            'description': task.description
        })
        
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
        
        logger.info(f"TaskChain '{chain.name}' added with {len(chain.tasks)} sequential nodes")
    
    def build_graph(self) -> None:
        """
        Build and compile the workflow graph.
        """
        try:
            if not self._state_graph:
                raise CrewGraphError("No StateGraph created. Add nodes first.")
            
            # Compile the graph with checkpointing if enabled
            if self._checkpointer:
                self._compiled_graph = self._state_graph.compile(checkpointer=self._checkpointer)
            else:
                self._compiled_graph = self._state_graph.compile()
            
            self.is_built = True
            logger.info(f"Workflow '{self.name}' compiled successfully")
            
        except Exception as e:
            logger.error(f"Failed to build workflow '{self.name}': {e}")
            raise CrewGraphError(f"Workflow build failed: {e}")
    
    def execute(self, 
                state: Union[SharedState, Dict[str, Any]],
                config: Optional[Dict[str, Any]] = None) -> WorkflowResult:
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
            if hasattr(state, 'to_dict'):
                initial_state = state.to_dict()
            else:
                initial_state = dict(state)
            
            # Add default state fields
            initial_state.setdefault('messages', [])
            initial_state.setdefault('current_node', '')
            initial_state.setdefault('results', {})
            initial_state.setdefault('errors', [])
            initial_state.setdefault('metadata', {
                'workflow_id': self.id,
                'workflow_name': self.name,
                'start_time': self.start_time
            })
            
            # Prepare execution config
            exec_config = config or {}
            if self._checkpointer:
                exec_config['configurable'] = {'thread_id': self.current_thread_id}
            
            logger.info(f"Starting workflow execution '{self.name}' with thread_id: {self.current_thread_id}")
            
            # Execute with LangGraph
            final_state = None
            for step_result in self._compiled_graph.stream(initial_state, config=exec_config):
                final_state = step_result
                
                # Real-time monitoring
                if isinstance(final_state, dict):
                    current_node = final_state.get('current_node', '')
                    if current_node:
                        logger.debug(f"Workflow step completed: {current_node}")
            
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            
            # Process results
            if final_state and isinstance(final_state, dict):
                success = len(final_state.get('errors', [])) == 0
                tasks_completed = len([r for r in self._execution_results.values() if r.success])
                tasks_failed = len([r for r in self._execution_results.values() if not r.success])
                
                self.status = WorkflowStatus.COMPLETED if success else WorkflowStatus.FAILED
                
                result = WorkflowResult(
                    workflow_id=self.id,
                    workflow_name=self.name,
                    success=success,
                    results=final_state.get('results', {}),
                    error='; '.join(final_state.get('errors', [])) if final_state.get('errors') else None,
                    execution_time=execution_time,
                    tasks_completed=tasks_completed,
                    tasks_failed=tasks_failed,
                    metadata={
                        'thread_id': self.current_thread_id,
                        'nodes_executed': len(self._execution_results),
                        'final_state': final_state
                    }
                )
                
                # Update original state if SharedState
                if hasattr(state, 'update'):
                    state.update(final_state.get('results', {}))
                
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
                execution_time=execution_time
            )
    
    async def execute_async(self,
                           state: Union[SharedState, Dict[str, Any]],
                           config: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Execute workflow asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool to avoid blocking
        result = await loop.run_in_executor(None, self.execute, state, config)
        return result
    
    def execute_with_plan(self,
                         execution_plan: Any,
                         state: Union[SharedState, Dict[str, Any]]) -> WorkflowResult:
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
    
    async def execute_with_plan_async(self,
                                     execution_plan: Any,
                                     state: Union[SharedState, Dict[str, Any]]) -> WorkflowResult:
        """Execute workflow with plan asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_with_plan, execution_plan, state)
    
    def pause_execution(self) -> None:
        """Pause workflow execution."""
        self.status = WorkflowStatus.PAUSED
        logger.info(f"Workflow '{self.name}' paused")
    
    def resume_execution(self,
                        state: Optional[Dict[str, Any]] = None,
                        config: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Resume paused workflow execution."""
        if not self.current_thread_id:
            raise ExecutionError("No thread ID available for resume")
        
        logger.info(f"Resuming workflow '{self.name}' with thread_id: {self.current_thread_id}")
        
        # Resume from checkpoint
        resume_config = config or {}
        resume_config['configurable'] = {'thread_id': self.current_thread_id}
        
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
            config = {'configurable': {'thread_id': self.current_thread_id}}
            history = list(self._compiled_graph.get_state_history(config))
            
            return [
                {
                    'step': i,
                    'timestamp': checkpoint.created_at if hasattr(checkpoint, 'created_at') else None,
                    'state': checkpoint.values if hasattr(checkpoint, 'values') else checkpoint
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
                    state['errors'] = state.get('errors', [])
                    state['errors'].append(f"Node '{name}' failed: {str(e)}")
                
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
    
    def set_callbacks(self,
                      on_node_start: Optional[Callable] = None,
                      on_node_complete: Optional[Callable] = None,
                      on_workflow_complete: Optional[Callable] = None,
                      on_error: Optional[Callable] = None):
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
            'id': self.id,
            'name': self.name,
            'status': self.status.value,
            'is_built': self.is_built,
            'nodes': list(self._nodes.keys()),
            'edges': self._edges,
            'conditional_edges': len(self._conditional_edges),
            'execution_results': {
                name: result.to_dict() if hasattr(result, 'to_dict') else str(result)
                for name, result in self._execution_results.items()
            },
            'current_thread_id': self.current_thread_id,
            'execution_time': (
                self.end_time - self.start_time 
                if self.start_time and self.end_time 
                else None
            )
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
                source = edge_info['source']
                mapping = edge_info['mapping']
                lines.append(f"  {source} -> [conditional] -> {list(mapping.values())}")
            
            return "\n".join(lines)
    
    def shutdown(self):
        """Shutdown orchestrator and cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
        
        logger.info(f"GraphOrchestrator '{self.name}' shutdown completed")
    
    def __repr__(self) -> str:
        return (f"GraphOrchestrator(name='{self.name}', nodes={len(self._nodes)}, "
                f"status={self.status.value}, built={self.is_built})")


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
    
    def build(self) -> GraphOrchestrator:
        """Build and return the orchestrator."""
        self.orchestrator.build_graph()
        return self.orchestrator