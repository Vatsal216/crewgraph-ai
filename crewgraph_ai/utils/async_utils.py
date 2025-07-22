"""
Advanced async utilities for CrewGraph AI
"""

import asyncio
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum

from .logging import get_logger
from .exceptions import ExecutionError
from .metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()

T = TypeVar('T')


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AsyncTask:
    """Async task wrapper with metadata"""
    id: str
    coro: Awaitable[Any]
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncTaskManager:
    """Advanced async task management with priority queues and load balancing"""
    
    def __init__(self, 
                 max_concurrent_tasks: int = 10,
                 default_timeout: float = 300.0):
        """
        Initialize async task manager.
        
        Args:
            max_concurrent_tasks: Maximum concurrent tasks
            default_timeout: Default task timeout in seconds
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        
        # Task queues by priority
        self._priority_queues: Dict[TaskPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in TaskPriority
        }
        
        # Active tasks tracking
        self._active_tasks: Dict[str, AsyncTask] = {}
        self._completed_tasks: Dict[str, Dict[str, Any]] = {}
        self._failed_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Task execution control
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Metrics and monitoring
        self._task_counter = 0
        self._start_time = time.time()
        
        logger.info(f"AsyncTaskManager initialized with max_concurrent_tasks={max_concurrent_tasks}")
    
    async def start(self) -> None:
        """Start the task manager"""
        if self._running:
            logger.warning("AsyncTaskManager already running")
            return
        
        self._running = True
        
        # Start worker tasks for each priority level
        for priority in TaskPriority:
            worker_task = asyncio.create_task(
                self._worker_loop(priority),
                name=f"worker_{priority.name.lower()}"
            )
            self._worker_tasks.append(worker_task)
        
        logger.info("AsyncTaskManager started with priority-based workers")
    
    async def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the task manager"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        if timeout:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._worker_tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Some worker tasks did not finish within timeout")
        
        # Wait for active tasks to complete
        active_tasks = list(self._active_tasks.values())
        if active_tasks:
            logger.info(f"Waiting for {len(active_tasks)} active tasks to complete")
            await self._wait_for_active_tasks(timeout)
        
        self._worker_tasks.clear()
        logger.info("AsyncTaskManager stopped")
    
    async def submit_task(self, 
                         coro: Awaitable[T],
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: Optional[float] = None,
                         task_id: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit task for async execution.
        
        Args:
            coro: Coroutine to execute
            priority: Task priority
            timeout: Task timeout (uses default if None)
            task_id: Custom task ID
            metadata: Task metadata
            
        Returns:
            Task ID
        """
        if not self._running:
            await self.start()
        
        # Generate task ID
        if task_id is None:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}_{int(time.time() * 1000)}"
        
        # Create task wrapper
        task = AsyncTask(
            id=task_id,
            coro=coro,
            priority=priority,
            timeout=timeout or self.default_timeout,
            metadata=metadata or {}
        )
        
        # Add to appropriate priority queue
        await self._priority_queues[priority].put(task)
        
        logger.debug(f"Task submitted: {task_id} with priority {priority.name}")
        
        # Record metrics
        metrics.increment_counter(
            "async_tasks_submitted_total",
            labels={"priority": priority.name.lower()}
        )
        
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get result of completed task.
        
        Args:
            task_id: Task ID
            timeout: Wait timeout
            
        Returns:
            Task result
        """
        start_time = time.time()
        
        while True:
            # Check if task completed
            if task_id in self._completed_tasks:
                return self._completed_tasks[task_id]['result']
            
            # Check if task failed
            if task_id in self._failed_tasks:
                error_info = self._failed_tasks[task_id]
                raise ExecutionError(
                    f"Task {task_id} failed: {error_info['error']}",
                    details=error_info
                )
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise ExecutionError(f"Timeout waiting for task {task_id}")
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        if task_id in self._active_tasks:
            # Task is currently running - this is complex to handle
            logger.warning(f"Cannot cancel running task {task_id}")
            return False
        
        # Remove from queues
        for queue in self._priority_queues.values():
            # Note: asyncio.Queue doesn't support item removal
            # In production, you'd need a more sophisticated queue
            pass
        
        logger.info(f"Task {task_id} cancellation requested")
        return True
    
    async def _worker_loop(self, priority: TaskPriority) -> None:
        """Worker loop for specific priority level"""
        queue = self._priority_queues[priority]
        
        while self._running:
            try:
                # Get task from queue (with timeout to allow checking _running)
                try:
                    task = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Execute task with semaphore control
                async with self._semaphore:
                    await self._execute_task(task)
                
            except asyncio.CancelledError:
                logger.debug(f"Worker for priority {priority.name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker error in priority {priority.name}: {e}")
    
    async def _execute_task(self, task: AsyncTask) -> None:
        """Execute individual task"""
        task_id = task.id
        start_time = time.time()
        
        # Add to active tasks
        self._active_tasks[task_id] = task
        
        try:
            logger.debug(f"Executing task {task_id}")
            
            # Execute with timeout
            if task.timeout:
                result = await asyncio.wait_for(task.coro, timeout=task.timeout)
            else:
                result = await task.coro
            
            execution_time = time.time() - start_time
            
            # Store result
            self._completed_tasks[task_id] = {
                'result': result,
                'execution_time': execution_time,
                'completed_at': time.time(),
                'metadata': task.metadata
            }
            
            # Record metrics
            metrics.record_duration(
                "async_task_execution",
                execution_time,
                labels={"priority": task.priority.name.lower(), "status": "success"}
            )
            
            logger.info(f"Task {task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Handle retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.warning(f"Task {task_id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
                
                # Re-queue for retry
                await self._priority_queues[task.priority].put(task)
                return
            
            # Store failure
            self._failed_tasks[task_id] = {
                'error': str(e),
                'execution_time': execution_time,
                'failed_at': time.time(),
                'retry_count': task.retry_count,
                'metadata': task.metadata
            }
            
            # Record metrics
            metrics.record_duration(
                "async_task_execution",
                execution_time,
                labels={"priority": task.priority.name.lower(), "status": "failure"}
            )
            
            logger.error(f"Task {task_id} failed after {task.retry_count} retries: {e}")
            
        finally:
            # Remove from active tasks
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
    
    async def _wait_for_active_tasks(self, timeout: Optional[float] = None) -> None:
        """Wait for all active tasks to complete"""
        start_time = time.time()
        
        while self._active_tasks:
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for {len(self._active_tasks)} active tasks")
                break
            
            await asyncio.sleep(0.5)
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        total_queued = sum(queue.qsize() for queue in self._priority_queues.values())
        
        return {
            "running": self._running,
            "active_tasks": len(self._active_tasks),
            "queued_tasks": total_queued,
            "completed_tasks": len(self._completed_tasks),
            "failed_tasks": len(self._failed_tasks),
            "max_concurrent": self.max_concurrent_tasks,
            "uptime": time.time() - self._start_time,
            "queue_sizes": {
                priority.name.lower(): queue.qsize() 
                for priority, queue in self._priority_queues.items()
            }
        }


class BatchProcessor:
    """Process large batches of data efficiently with async processing"""
    
    def __init__(self, 
                 batch_size: int = 100,
                 max_concurrent_batches: int = 5,
                 retry_failed_items: bool = True):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Items per batch
            max_concurrent_batches: Maximum concurrent batches
            retry_failed_items: Whether to retry failed items
        """
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.retry_failed_items = retry_failed_items
        
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        logger.info(f"BatchProcessor initialized: batch_size={batch_size}, max_concurrent={max_concurrent_batches}")
    
    async def process_batch(self, 
                           items: List[Any],
                           processor_func: Callable[[Any], Awaitable[Any]],
                           progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """
        Process batch of items with async function.
        
        Args:
            items: Items to process
            processor_func: Async function to process each item
            progress_callback: Optional progress callback
            
        Returns:
            Processing results summary
        """
        total_items = len(items)
        logger.info(f"Starting batch processing of {total_items} items")
        
        # Split into batches
        batches = [
            items[i:i + self.batch_size] 
            for i in range(0, total_items, self.batch_size)
        ]
        
        results = []
        failed_items = []
        processed_count = 0
        
        # Process batches concurrently
        batch_tasks = []
        for batch_idx, batch in enumerate(batches):
            task = asyncio.create_task(
                self._process_single_batch(batch, batch_idx, processor_func),
                name=f"batch_{batch_idx}"
            )
            batch_tasks.append(task)
        
        # Wait for batches to complete
        for completed_task in asyncio.as_completed(batch_tasks):
            try:
                batch_result = await completed_task
                results.extend(batch_result['successful'])
                failed_items.extend(batch_result['failed'])
                processed_count += batch_result['processed']
                
                # Progress callback
                if progress_callback:
                    progress_callback(processed_count, total_items)
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
        
        # Retry failed items if enabled
        if self.retry_failed_items and failed_items:
            logger.info(f"Retrying {len(failed_items)} failed items")
            retry_results = await self._retry_failed_items(failed_items, processor_func)
            results.extend(retry_results['successful'])
            failed_items = retry_results['failed']
        
        processing_summary = {
            'total_items': total_items,
            'successful': len(results),
            'failed': len(failed_items),
            'success_rate': len(results) / total_items if total_items > 0 else 0,
            'results': results,
            'failed_items': failed_items
        }
        
        logger.info(f"Batch processing completed: {processing_summary['successful']}/{total_items} successful")
        return processing_summary
    
    async def _process_single_batch(self, 
                                   batch: List[Any],
                                   batch_idx: int,
                                   processor_func: Callable[[Any], Awaitable[Any]]) -> Dict[str, Any]:
        """Process single batch of items"""
        async with self._semaphore:
            logger.debug(f"Processing batch {batch_idx} with {len(batch)} items")
            
            successful = []
            failed = []
            
            # Process items in batch concurrently
            item_tasks = [
                asyncio.create_task(processor_func(item), name=f"item_{batch_idx}_{i}")
                for i, item in enumerate(batch)
            ]
            
            for i, item_task in enumerate(asyncio.as_completed(item_tasks)):
                try:
                    result = await item_task
                    successful.append({
                        'item': batch[i],
                        'result': result,
                        'batch_idx': batch_idx,
                        'item_idx': i
                    })
                except Exception as e:
                    failed.append({
                        'item': batch[i],
                        'error': str(e),
                        'batch_idx': batch_idx,
                        'item_idx': i
                    })
            
            return {
                'successful': successful,
                'failed': failed,
                'processed': len(batch)
            }
    
    async def _retry_failed_items(self, 
                                 failed_items: List[Dict[str, Any]],
                                 processor_func: Callable[[Any], Awaitable[Any]]) -> Dict[str, Any]:
        """Retry processing failed items"""
        successful = []
        still_failed = []
        
        for failed_item in failed_items:
            try:
                result = await processor_func(failed_item['item'])
                successful.append({
                    'item': failed_item['item'],
                    'result': result,
                    'retry': True
                })
            except Exception as e:
                still_failed.append({
                    'item': failed_item['item'],
                    'error': str(e),
                    'retry_failed': True
                })
        
        return {
            'successful': successful,
            'failed': still_failed
        }


# Convenience functions for common async patterns

async def run_with_timeout(coro: Awaitable[T], timeout: float) -> T:
    """Run coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise ExecutionError(f"Operation timed out after {timeout} seconds")


async def gather_with_concurrency_limit(coroutines: List[Awaitable[T]], 
                                       limit: int = 10) -> List[T]:
    """Run coroutines with concurrency limit"""
    semaphore = asyncio.Semaphore(limit)
    
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[sem_coro(coro) for coro in coroutines])


async def retry_async(coro_func: Callable[[], Awaitable[T]], 
                     max_retries: int = 3,
                     delay: float = 1.0,
                     backoff_factor: float = 2.0) -> T:
    """Retry async function with exponential backoff"""
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            return await coro_func()
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {current_delay}s: {e}")
                await asyncio.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
                raise ExecutionError(f"Max retries exceeded: {e}") from last_exception


# Global async task manager
_global_task_manager: Optional[AsyncTaskManager] = None


async def get_task_manager() -> AsyncTaskManager:
    """Get global async task manager"""
    global _global_task_manager
    
    if _global_task_manager is None:
        _global_task_manager = AsyncTaskManager()
        await _global_task_manager.start()
    
    return _global_task_manager