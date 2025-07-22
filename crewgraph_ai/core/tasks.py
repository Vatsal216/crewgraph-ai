"""
Production-ready task management system with full CrewAI Task integration
"""

import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading

from crewai import Task as CrewTask
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError, ValidationError, ExecutionError
from ..memory.base import BaseMemory

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    task_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    agent_name: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskWrapper:
    """
    Enhanced wrapper around CrewAI Task with advanced orchestration features.
    
    Maintains 100% CrewAI Task compatibility while adding:
    - State management and persistence
    - Advanced retry logic with exponential backoff
    - Dynamic tool injection
    - Result caching and validation
    - Dependency management
    - Performance monitoring
    """
    
    def __init__(self,
                 name: str,
                 description: str = "",
                 crew_task: Optional[CrewTask] = None,
                 state: Optional[Any] = None,
                 tool_registry: Optional[Any] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: float = 300.0,
                 cache_results: bool = True):
        """
        Initialize task wrapper.
        
        Args:
            name: Task identifier
            description: Task description
            crew_task: Original CrewAI Task instance
            state: Shared state manager
            tool_registry: Available tools
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries (exponential backoff)
            timeout: Task timeout in seconds
            cache_results: Whether to cache task results
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.crew_task = crew_task  # Original CrewAI Task
        self.state = state
        self.tool_registry = tool_registry
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.cache_results = cache_results
        
        # Task configuration
        self.assigned_agent: Optional[Any] = None
        self.dependencies: List[str] = []
        self.tools: List[Any] = []
        self.prompt_template: Optional[str] = None
        self.output_parser: Optional[Callable] = None
        self.validators: List[Callable] = []
        
        # Execution tracking
        self.status = TaskStatus.PENDING
        self.retry_count = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.last_result: Optional[TaskResult] = None
        
        # Callbacks
        self._on_start: Optional[Callable] = None
        self._on_complete: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        self._on_retry: Optional[Callable] = None
        
        logger.info(f"TaskWrapper '{name}' initialized with ID: {self.id}")
    
    @classmethod
    def from_crew_task(cls, crew_task: CrewTask, name: Optional[str] = None, **kwargs):
        """
        Create TaskWrapper from existing CrewAI Task.
        
        Args:
            crew_task: CrewAI Task instance
            name: Override task name
            **kwargs: Additional TaskWrapper parameters
            
        Returns:
            TaskWrapper instance with full CrewAI compatibility
        """
        task_name = name or getattr(crew_task, 'description', f"task_{uuid.uuid4().hex[:8]}")
        
        wrapper = cls(
            name=task_name,
            description=getattr(crew_task, 'description', ''),
            crew_task=crew_task,
            **kwargs
        )
        
        # Preserve CrewAI Task attributes
        if hasattr(crew_task, 'tools'):
            wrapper.tools = crew_task.tools
        if hasattr(crew_task, 'agent'):
            wrapper.assigned_agent = crew_task.agent
            
        logger.info(f"TaskWrapper created from CrewAI Task: {task_name}")
        return wrapper
    
    def assign_agent(self, agent: Any) -> None:
        """
        Assign agent to task (supports both AgentWrapper and CrewAI Agent).
        
        Args:
            agent: Agent instance
        """
        self.assigned_agent = agent
        
        # If we have a CrewAI task, update its agent too
        if self.crew_task and hasattr(self.crew_task, 'agent'):
            # Extract CrewAI agent if wrapped
            crew_agent = getattr(agent, 'crew_agent', agent)
            self.crew_task.agent = crew_agent
        
        agent_name = getattr(agent, 'name', str(agent))
        logger.info(f"Agent '{agent_name}' assigned to task '{self.name}'")
    
    def add_tool(self, tool: Any) -> None:
        """Add tool to task."""
        self.tools.append(tool)
        
        # Update CrewAI task tools if available
        if self.crew_task and hasattr(self.crew_task, 'tools'):
            if not self.crew_task.tools:
                self.crew_task.tools = []
            self.crew_task.tools.append(tool)
        
        tool_name = getattr(tool, 'name', str(tool))
        logger.debug(f"Tool '{tool_name}' added to task '{self.name}'")
    
    def set_prompt_template(self, template: str) -> None:
        """Set dynamic prompt template."""
        self.prompt_template = template
        logger.debug(f"Prompt template set for task '{self.name}'")
    
    def add_validator(self, validator: Callable[[Any], bool]) -> None:
        """Add result validator function."""
        self.validators.append(validator)
        logger.debug(f"Validator added to task '{self.name}'")
    
    def set_callbacks(self,
                      on_start: Optional[Callable] = None,
                      on_complete: Optional[Callable] = None,
                      on_error: Optional[Callable] = None,
                      on_retry: Optional[Callable] = None):
        """Set execution callbacks."""
        self._on_start = on_start
        self._on_complete = on_complete
        self._on_error = on_error
        self._on_retry = on_retry
    
    def execute(self, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        """
        Execute task with full error handling and retry logic.
        
        Args:
            context: Execution context
            
        Returns:
            TaskResult with execution details
        """
        # Check cache first
        if self.cache_results and self.last_result and self.last_result.success:
            logger.info(f"Returning cached result for task '{self.name}'")
            return self.last_result
        
        # Check dependencies
        if not self._check_dependencies():
            error_msg = f"Dependencies not met for task '{self.name}'"
            logger.error(error_msg)
            return TaskResult(
                task_id=self.id,
                task_name=self.name,
                success=False,
                error=error_msg
            )
        
        # Execute with retries
        for attempt in range(self.max_retries + 1):
            try:
                self.retry_count = attempt
                result = self._execute_single_attempt(context)
                
                if result.success:
                    self.last_result = result
                    return result
                
                # If not successful and not last attempt, retry
                if attempt < self.max_retries:
                    self._handle_retry(result.error, attempt)
                else:
                    return result
                    
            except Exception as e:
                error_msg = f"Task '{self.name}' failed on attempt {attempt + 1}: {str(e)}"
                logger.error(error_msg)
                
                if attempt < self.max_retries:
                    self._handle_retry(str(e), attempt)
                else:
                    return TaskResult(
                        task_id=self.id,
                        task_name=self.name,
                        success=False,
                        error=error_msg,
                        retry_count=attempt
                    )
        
        # Should never reach here, but safety fallback
        return TaskResult(
            task_id=self.id,
            task_name=self.name,
            success=False,
            error="Max retries exceeded",
            retry_count=self.max_retries
        )
    
    def _execute_single_attempt(self, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        """Execute single task attempt."""
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        
        if self._on_start:
            self._on_start(self)
        
        try:
            # Prepare execution context
            exec_context = self._prepare_context(context)
            
            # Execute based on available components
            if self.crew_task:
                # Use CrewAI Task execution
                result = self._execute_crew_task(exec_context)
            elif self.assigned_agent:
                # Use agent execution
                result = self._execute_with_agent(exec_context)
            else:
                # Default execution
                result = self._execute_default(exec_context)
            
            # Validate result
            if not self._validate_result(result):
                raise ValidationError("Result validation failed")
            
            # Parse result if parser provided
            if self.output_parser:
                result = self.output_parser(result)
            
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            self.status = TaskStatus.COMPLETED
            
            task_result = TaskResult(
                task_id=self.id,
                task_name=self.name,
                success=True,
                result=result,
                execution_time=execution_time,
                agent_name=getattr(self.assigned_agent, 'name', None),
                retry_count=self.retry_count
            )
            
            # Store in state
            if self.state:
                self.state.set(f"task:{self.name}:result", result)
                self.state.set(f"task:{self.name}:completed", True)
            
            if self._on_complete:
                self._on_complete(self, task_result)
            
            logger.info(f"Task '{self.name}' completed successfully in {execution_time:.2f}s")
            return task_result
            
        except Exception as e:
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time if self.start_time else 0
            self.status = TaskStatus.FAILED
            
            error_msg = str(e)
            logger.error(f"Task '{self.name}' failed: {error_msg}")
            
            if self._on_error:
                self._on_error(self, e)
            
            return TaskResult(
                task_id=self.id,
                task_name=self.name,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                agent_name=getattr(self.assigned_agent, 'name', None),
                retry_count=self.retry_count
            )
    
    # ============= MESSAGE HANDLING METHODS =============
    
    def execute_with_message_context(self, 
                                   messages: List[BaseMessage],
                                   state: Dict[str, Any]) -> TaskResult:
        """Execute task with message context and enhanced conversation awareness"""
        # Prepare enhanced state with messages
        enhanced_state = state.copy()
        enhanced_state['messages'] = messages
        
        # Extract conversation context from messages
        if messages:
            # Add latest human message content to context if available
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    enhanced_state['user_input'] = msg.content
                    break
        
        # Add conversation summary to context
        enhanced_state['conversation_context'] = self._create_conversation_summary(messages)
        
        logger.info(f"Executing task '{self.name}' with {len(messages)} message context")
        return self.execute(enhanced_state)
    
    def execute_as_message_agent(self, 
                                messages: List[BaseMessage],
                                state: Dict[str, Any] = None) -> BaseMessage:
        """
        Execute task as a message-based agent for MessageGraph integration.
        
        This method allows TaskWrapper to be used directly in MessageGraph workflows.
        Returns a BaseMessage instead of TaskResult for proper message flow.
        """
        try:
            # Execute with message context
            result = self.execute_with_message_context(messages, state or {})
            
            # Convert result to proper message
            if result.success:
                response_message = AIMessage(
                    content=str(result.result),
                    additional_kwargs={
                        'task_id': self.id,
                        'task_name': self.name,
                        'agent_name': result.agent_name,
                        'execution_time': result.execution_time,
                        'success': True,
                        'conversation_turn': len(messages) + 1,
                        'timestamp': time.time()
                    }
                )
            else:
                response_message = AIMessage(
                    content=f"I encountered an issue while executing the task: {result.error}",
                    additional_kwargs={
                        'task_id': self.id,
                        'task_name': self.name,
                        'execution_time': result.execution_time,
                        'success': False,
                        'error': result.error,
                        'conversation_turn': len(messages) + 1,
                        'timestamp': time.time()
                    }
                )
            
            return response_message
            
        except Exception as e:
            logger.error(f"Task '{self.name}' execution as message agent failed: {e}")
            return AIMessage(
                content=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                additional_kwargs={
                    'task_id': self.id,
                    'task_name': self.name,
                    'success': False,
                    'error': str(e),
                    'conversation_turn': len(messages) + 1,
                    'timestamp': time.time()
                }
            )
    
    def _create_conversation_summary(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Create a summary of the conversation for context"""
        human_count = sum(1 for msg in messages if isinstance(msg, HumanMessage))
        ai_count = sum(1 for msg in messages if isinstance(msg, AIMessage))
        
        # Get recent context (last 3 messages)
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        recent_context = []
        
        for msg in recent_messages:
            msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            recent_context.append(f"{msg_type}: {content_preview}")
        
        return {
            'total_messages': len(messages),
            'human_messages': human_count,
            'ai_messages': ai_count,
            'recent_context': recent_context,
            'conversation_length': sum(len(msg.content) for msg in messages)
        }
    
    def add_response_message(self, content: str, result: TaskResult) -> AIMessage:
        """Convert task result to AIMessage"""
        return AIMessage(
            content=content,
            additional_kwargs={
                'task_id': self.id,
                'task_name': self.name,
                'success': result.success,
                'execution_time': result.execution_time,
                'agent_name': result.agent_name,
                'timestamp': time.time()
            }
        )
    
    def create_message_from_result(self, result: TaskResult) -> AIMessage:
        """Create AIMessage from TaskResult"""
        if result.success:
            content = str(result.result) if result.result else f"Task '{self.name}' completed successfully"
        else:
            content = f"Task '{self.name}' failed: {result.error}"
        
        return self.add_response_message(content, result)
    
    def get_message_context(self, context: Dict[str, Any]) -> List[BaseMessage]:
        """Extract message context from execution context"""
        return context.get('messages', [])
    
    def update_context_with_messages(self, 
                                   context: Dict[str, Any], 
                                   messages: List[BaseMessage]) -> Dict[str, Any]:
        """Update execution context with messages"""
        enhanced_context = context.copy()
        enhanced_context['messages'] = messages
        
        # Add conversation summary
        if messages:
            enhanced_context['conversation_summary'] = self._summarize_messages(messages)
        
        return enhanced_context
    
    def _summarize_messages(self, messages: List[BaseMessage]) -> str:
        """Create a summary of the conversation messages"""
        if not messages:
            return ""
        
        summary_parts = []
        for i, msg in enumerate(messages[-5:]):  # Last 5 messages
            msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_parts.append(f"{msg_type}: {content_preview}")
        
        return " | ".join(summary_parts)
    
    # ============= END MESSAGE HANDLING METHODS =============
    
    def _execute_crew_task(self, context: Dict[str, Any]) -> Any:
        """Execute using original CrewAI Task."""
        if not self.crew_task:
            raise ExecutionError("No CrewAI task available")
        
        # CrewAI Task execution with full feature access
        if hasattr(self.crew_task, 'execute'):
            return self.crew_task.execute(context=context)
        else:
            # Fallback for different CrewAI versions
            from crewai import Crew
            
            # Create temporary crew if needed
            if self.assigned_agent and hasattr(self.assigned_agent, 'crew_agent'):
                crew = Crew(
                    agents=[self.assigned_agent.crew_agent],
                    tasks=[self.crew_task],
                    verbose=True
                )
                result = crew.kickoff()
                return result
            else:
                raise ExecutionError("Cannot execute CrewAI task without agent")
    
    def _execute_with_agent(self, context: Dict[str, Any]) -> Any:
        """Execute using assigned agent."""
        if not self.assigned_agent:
            raise ExecutionError("No agent assigned")
        
        # Prepare prompt
        prompt = self._prepare_prompt(context)
        
        # Execute with agent
        if hasattr(self.assigned_agent, 'execute_task'):
            # CrewGraph AgentWrapper
            result = self.assigned_agent.execute_task(
                task_name=self.name,
                prompt=prompt,
                context=context,
                tools=self.tools
            )
            return result.get('result')
        else:
            # Direct CrewAI agent
            return self.assigned_agent.execute(prompt, context=context)
    
    def _execute_default(self, context: Dict[str, Any]) -> str:
        """Default execution when no specific method available."""
        logger.warning(f"Using default execution for task '{self.name}'")
        
        prompt = self._prepare_prompt(context)
        return f"Task '{self.name}' executed with prompt: {prompt}"
    
    def _prepare_context(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare full execution context."""
        exec_context = context or {}
        
        # Add state context
        if self.state:
            state_context = self.state.get_context(f"task:{self.name}")
            exec_context.update(state_context)
        
        # Add dependency results
        for dep_name in self.dependencies:
            if self.state:
                dep_result = self.state.get(f"task:{dep_name}:result")
                if dep_result:
                    exec_context[f"dependency_{dep_name}"] = dep_result
        
        exec_context.update({
            'task_name': self.name,
            'task_description': self.description,
            'retry_count': self.retry_count
        })
        
        return exec_context
    
    def _prepare_prompt(self, context: Dict[str, Any]) -> str:
        """Prepare task prompt with template substitution."""
        if self.prompt_template:
            try:
                return self.prompt_template.format(**context)
            except KeyError as e:
                logger.warning(f"Template variable missing: {e}")
                return self.prompt_template
        
        # Default prompt construction
        prompt = f"Task: {self.name}\n"
        if self.description:
            prompt += f"Description: {self.description}\n"
        
        if context:
            prompt += f"Context: {context}\n"
        
        return prompt
    
    def _validate_result(self, result: Any) -> bool:
        """Validate task result using registered validators."""
        for validator in self.validators:
            try:
                if not validator(result):
                    return False
            except Exception as e:
                logger.error(f"Validator failed for task '{self.name}': {e}")
                return False
        
        return True
    
    def _check_dependencies(self) -> bool:
        """Check if all dependencies are satisfied."""
        if not self.dependencies or not self.state:
            return True
        
        for dep_name in self.dependencies:
            if not self.state.get(f"task:{dep_name}:completed", False):
                logger.warning(f"Dependency '{dep_name}' not completed for task '{self.name}'")
                return False
        
        return True
    
    def _handle_retry(self, error: str, attempt: int):
        """Handle retry logic with exponential backoff."""
        self.status = TaskStatus.RETRYING
        
        if self._on_retry:
            self._on_retry(self, error, attempt)
        
        # Exponential backoff
        delay = self.retry_delay * (2 ** attempt)
        logger.info(f"Retrying task '{self.name}' in {delay:.2f}s (attempt {attempt + 1})")
        
        time.sleep(delay)
    
    async def execute_async(self, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        """Execute task asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool to avoid blocking
        result = await loop.run_in_executor(None, self.execute, context)
        return result
    
    def cancel(self) -> None:
        """Cancel task execution."""
        self.status = TaskStatus.CANCELLED
        logger.info(f"Task '{self.name}' cancelled")
    
    def reset(self) -> None:
        """Reset task state."""
        self.status = TaskStatus.PENDING
        self.retry_count = 0
        self.start_time = None
        self.end_time = None
        self.last_result = None
        
        if self.state:
            self.state.delete(f"task:{self.name}:result")
            self.state.delete(f"task:{self.name}:completed")
        
        logger.info(f"Task '{self.name}' reset")
    
    def get_crew_task(self) -> Optional[CrewTask]:
        """Get the original CrewAI Task instance."""
        return self.crew_task
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'dependencies': self.dependencies,
            'retry_count': self.retry_count,
            'has_crew_task': self.crew_task is not None,
            'has_agent': self.assigned_agent is not None,
            'tools_count': len(self.tools),
            'execution_time': (
                self.end_time - self.start_time 
                if self.start_time and self.end_time 
                else None
            )
        }
    
    def __repr__(self) -> str:
        return (f"TaskWrapper(name='{self.name}', status={self.status.value}, "
                f"has_crew_task={self.crew_task is not None})")


class TaskChain:
    """
    Manages sequential execution of multiple tasks with dependency resolution.
    """
    
    def __init__(self, tasks: List[TaskWrapper], name: Optional[str] = None):
        """
        Initialize task chain.
        
        Args:
            tasks: List of tasks in execution order
            name: Chain identifier
        """
        self.name = name or f"chain_{uuid.uuid4().hex[:8]}"
        self.tasks = tasks
        self.current_index = 0
        self.results: List[TaskResult] = []
        self.status = TaskStatus.PENDING
        
        # Set up dependencies automatically
        for i, task in enumerate(tasks[1:], 1):
            prev_task = tasks[i-1]
            if prev_task.name not in task.dependencies:
                task.dependencies.append(prev_task.name)
        
        logger.info(f"TaskChain '{self.name}' created with {len(tasks)} tasks")
    
    def execute(self, context: Optional[Dict[str, Any]] = None) -> List[TaskResult]:
        """Execute all tasks in sequence."""
        self.status = TaskStatus.RUNNING
        self.results = []
        
        logger.info(f"Starting execution of TaskChain '{self.name}'")
        
        for i, task in enumerate(self.tasks):
            self.current_index = i
            
            logger.info(f"Executing task {i+1}/{len(self.tasks)}: '{task.name}'")
            
            # Add results from previous tasks to context
            task_context = context.copy() if context else {}
            for j, prev_result in enumerate(self.results):
                task_context[f"task_{j}_result"] = prev_result.result
                task_context[f"task_{self.tasks[j].name}_result"] = prev_result.result
            
            # Execute task
            result = task.execute(task_context)
            self.results.append(result)
            
            # Stop on failure if task doesn't allow continuation
            if not result.success:
                logger.error(f"TaskChain '{self.name}' failed at task '{task.name}'")
                self.status = TaskStatus.FAILED
                return self.results
        
        self.status = TaskStatus.COMPLETED
        logger.info(f"TaskChain '{self.name}' completed successfully")
        return self.results
    
    async def execute_async(self, context: Optional[Dict[str, Any]] = None) -> List[TaskResult]:
        """Execute task chain asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, context)
    
    def get_final_result(self) -> Optional[Any]:
        """Get the result of the last task."""
        if self.results and self.results[-1].success:
            return self.results[-1].result
        return None
    
    def __repr__(self) -> str:
        return f"TaskChain(name='{self.name}', tasks={len(self.tasks)}, status={self.status.value})"