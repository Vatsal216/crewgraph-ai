"""
Utility decorators for CrewGraph AI
"""

import time
import functools
import asyncio
from typing import Any, Callable, Dict, Optional, Union
from contextlib import contextmanager

from .logging import get_logger
from .exceptions import ExecutionError
from .metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


def retry(max_attempts: int = 3,
          delay: float = 1.0,
          backoff_factor: float = 2.0,
          exceptions: tuple = (Exception,),
          on_retry: Optional[Callable] = None):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Callback function called on each retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    
                    # Record successful execution after retries
                    if attempt > 0:
                        metrics.increment_counter(
                            "retry_success_total",
                            labels={
                                "function": func.__name__,
                                "attempt": str(attempt + 1)
                            }
                        )
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    # Log retry attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}"
                    )
                    
                    # Record retry attempt
                    metrics.increment_counter(
                        "retry_attempt_total",
                        labels={
                            "function": func.__name__,
                            "attempt": str(attempt + 1),
                            "exception": type(e).__name__
                        }
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt + 1, e, args, kwargs)
                    
                    # Don't delay on the last attempt
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
            
            # Record final failure
            metrics.increment_counter(
                "retry_failure_total",
                labels={
                    "function": func.__name__,
                    "max_attempts": str(max_attempts)
                }
            )
            
            # All attempts failed
            raise ExecutionError(
                f"Function {func.__name__} failed after {max_attempts} attempts",
                original_error=last_exception
            )
        
        return wrapper
    return decorator


def timeout(seconds: float, error_message: Optional[str] = None):
    """
    Timeout decorator for functions.
    
    Args:
        seconds: Timeout in seconds
        error_message: Custom error message
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(
                    error_message or f"Function {func.__name__} timed out after {seconds} seconds"
                )
            
            # Set up signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful execution
                metrics.increment_counter(
                    "timeout_success_total",
                    labels={"function": func.__name__, "timeout": str(seconds)}
                )
                
                return result
                
            except TimeoutError as e:
                # Record timeout
                metrics.increment_counter(
                    "timeout_failure_total",
                    labels={"function": func.__name__, "timeout": str(seconds)}
                )
                
                logger.error(f"Function {func.__name__} timed out: {e}")
                raise ExecutionError(str(e))
                
            finally:
                # Restore signal handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


def cache(ttl: Optional[float] = None,
          max_size: int = 128,
          key_func: Optional[Callable] = None):
    """
    Caching decorator with TTL support.
    
    Args:
        ttl: Time to live in seconds (None for no expiration)
        max_size: Maximum cache size
        key_func: Function to generate cache key
    """
    def decorator(func: Callable) -> Callable:
        cache_dict = {}
        cache_times = {}
        access_order = []
        
        def default_key_func(*args, **kwargs):
            return str(args) + str(sorted(kwargs.items()))
        
        key_generator = key_func or default_key_func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = key_generator(*args, **kwargs)
            current_time = time.time()
            
            # Check if cached value exists and is valid
            if cache_key in cache_dict:
                if ttl is None or (current_time - cache_times[cache_key]) < ttl:
                    # Cache hit
                    metrics.increment_counter(
                        "cache_hit_total",
                        labels={"function": func.__name__}
                    )
                    
                    # Update access order
                    if cache_key in access_order:
                        access_order.remove(cache_key)
                    access_order.append(cache_key)
                    
                    return cache_dict[cache_key]
                else:
                    # Cache expired
                    del cache_dict[cache_key]
                    del cache_times[cache_key]
                    if cache_key in access_order:
                        access_order.remove(cache_key)
            
            # Cache miss - execute function
            metrics.increment_counter(
                "cache_miss_total",
                labels={"function": func.__name__}
            )
            
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_dict[cache_key] = result
            cache_times[cache_key] = current_time
            access_order.append(cache_key)
            
            # Enforce max size (LRU eviction)
            while len(cache_dict) > max_size:
                oldest_key = access_order.pop(0)
                del cache_dict[oldest_key]
                del cache_times[oldest_key]
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: (cache_dict.clear(), cache_times.clear(), access_order.clear())
        wrapper.cache_info = lambda: {
            "size": len(cache_dict),
            "max_size": max_size,
            "ttl": ttl
        }
        
        return wrapper
    return decorator


def monitor(operation_name: Optional[str] = None,
           include_args: bool = False,
           include_result: bool = False):
    """
    Monitoring decorator that tracks execution metrics.
    
    Args:
        operation_name: Custom operation name (defaults to function name)
        include_args: Whether to include arguments in logs
        include_result: Whether to include result in logs
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Log execution start
            log_data = {"operation": op_name, "start_time": start_time}
            if include_args:
                log_data["args"] = str(args)
                log_data["kwargs"] = str(kwargs)
            
            logger.info(f"Starting operation: {op_name}", **log_data)
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Record metrics
                metrics.record_duration(f"{op_name}_operation", execution_time)
                metrics.increment_counter(
                    f"{op_name}_operations_total",
                    labels={"status": "success"}
                )
                
                # Log successful completion
                completion_log = {
                    "operation": op_name,
                    "execution_time": execution_time,
                    "status": "success"
                }
                
                if include_result:
                    completion_log["result"] = str(result)
                
                logger.info(f"Operation completed: {op_name}", **completion_log)
                
                return result
                
            except Exception as e:
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Record failure metrics
                metrics.record_duration(f"{op_name}_operation", execution_time)
                metrics.increment_counter(
                    f"{op_name}_operations_total",
                    labels={"status": "failure", "error_type": type(e).__name__}
                )
                
                # Log failure
                logger.error(
                    f"Operation failed: {op_name}",
                    operation=op_name,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__
                )
                
                raise
        
        return wrapper
    return decorator


def async_retry(max_attempts: int = 3,
               delay: float = 1.0,
               backoff_factor: float = 2.0,
               exceptions: tuple = (Exception,)):
    """
    Async version of retry decorator.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    logger.warning(
                        f"Async attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}"
                    )
                    
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
            
            # All attempts failed
            raise ExecutionError(
                f"Async function {func.__name__} failed after {max_attempts} attempts",
                original_error=last_exception
            )
        
        return wrapper
    return decorator


def async_timeout(seconds: float):
    """
    Async timeout decorator.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
                else:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, func, *args, **kwargs),
                        timeout=seconds
                    )
                
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"Async function {func.__name__} timed out after {seconds} seconds"
                logger.error(error_msg)
                raise ExecutionError(error_msg)
        
        return wrapper
    return decorator


def rate_limit(calls_per_second: float = 1.0):
    """
    Rate limiting decorator.
    
    Args:
        calls_per_second: Maximum calls per second
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            elapsed = current_time - last_called[0]
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
            
            last_called[0] = time.time()
            
            # Record rate limit metrics
            metrics.increment_counter(
                "rate_limit_calls_total",
                labels={"function": func.__name__}
            )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Convenience function for creating monitored context
@contextmanager
def monitored_operation(operation_name: str, labels: Optional[Dict[str, str]] = None):
    """Context manager for monitoring operations"""
    start_time = time.time()
    
    logger.info(f"Starting monitored operation: {operation_name}")
    
    try:
        yield
        
        execution_time = time.time() - start_time
        success_labels = (labels or {}).copy()
        success_labels["status"] = "success"
        
        metrics.record_duration(f"{operation_name}_operation", execution_time, success_labels)
        metrics.increment_counter(f"{operation_name}_operations_total", labels=success_labels)
        
        logger.info(f"Monitored operation completed: {operation_name} in {execution_time:.2f}s")
        
    except Exception as e:
        execution_time = time.time() - start_time
        failure_labels = (labels or {}).copy()
        failure_labels["status"] = "failure"
        failure_labels["error_type"] = type(e).__name__
        
        metrics.record_duration(f"{operation_name}_operation", execution_time, failure_labels)
        metrics.increment_counter(f"{operation_name}_operations_total", labels=failure_labels)
        
        logger.error(f"Monitored operation failed: {operation_name} after {execution_time:.2f}s - {e}")
        raise