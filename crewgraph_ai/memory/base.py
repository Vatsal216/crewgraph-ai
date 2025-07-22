"""
Base Memory Interface for CrewGraph AI
Defines the contract for all memory backends

Author: Vatsal216
Created: 2025-07-22 12:01:02 UTC
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass

from ..utils.logging import get_logger
from ..utils.exceptions import MemoryError
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class MemoryOperation(Enum):
    """Memory operation types for tracking"""
    SAVE = "save"
    LOAD = "load"
    DELETE = "delete"
    EXISTS = "exists"
    CLEAR = "clear"
    LIST_KEYS = "list_keys"
    GET_SIZE = "get_size"
    SEARCH = "search"


@dataclass
class MemoryStats:
    """Memory backend statistics"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_keys: int = 0
    total_size_bytes: int = 0
    average_operation_time: float = 0.0
    last_operation_time: float = 0.0
    backend_type: str = ""
    created_at: str = "2025-07-22 12:01:02"
    created_by: str = "Vatsal216"


class BaseMemory(ABC):
    """
    Abstract base class for all memory backends in CrewGraph AI.
    
    This class defines the standard interface that all memory backends
    must implement, ensuring consistency across different storage systems.
    
    Created by: Vatsal216
    Date: 2025-07-22 12:01:02 UTC
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base memory backend.
        
        Args:
            config: Backend-specific configuration
        """
        self.config = config or {}
        self.stats = MemoryStats(backend_type=self.__class__.__name__)
        self._lock = threading.RLock()
        self._connected = False
        
        logger.info(f"Initializing {self.__class__.__name__} memory backend")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:01:02")
    
    def _record_operation(self, operation: MemoryOperation, success: bool, duration: float):
        """Record operation metrics and statistics"""
        with self._lock:
            self.stats.total_operations += 1
            self.stats.last_operation_time = time.time()
            
            if success:
                self.stats.successful_operations += 1
            else:
                self.stats.failed_operations += 1
            
            # Update average operation time
            if self.stats.total_operations > 0:
                total_time = self.stats.average_operation_time * (self.stats.total_operations - 1)
                self.stats.average_operation_time = (total_time + duration) / self.stats.total_operations
        
        # Record global metrics
        metrics.record_duration(
            f"memory_operation_{operation.value}_duration_seconds",
            duration,
            labels={
                "backend": self.__class__.__name__,
                "success": str(success),
                "user": "Vatsal216"
            }
        )
        
        metrics.increment_counter(
            f"memory_operations_total",
            labels={
                "backend": self.__class__.__name__,
                "operation": operation.value,
                "success": str(success),
                "user": "Vatsal216"
            }
        )
    
    def _execute_with_metrics(self, operation: MemoryOperation, func, *args, **kwargs):
        """Execute operation with automatic metrics recording"""
        start_time = time.time()
        success = False
        
        try:
            result = func(*args, **kwargs)
            success = True
            return result
        except Exception as e:
            logger.error(f"Memory operation {operation.value} failed: {e}")
            raise MemoryError(
                f"Memory operation failed: {operation.value}",
                operation=operation.value,
                backend=self.__class__.__name__,
                original_error=e
            )
        finally:
            duration = time.time() - start_time
            self._record_operation(operation, success, duration)
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the memory backend"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the memory backend"""
        pass
    
    @abstractmethod
    def save(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Save value to memory with optional TTL.
        
        Args:
            key: Storage key
            value: Value to store
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load(self, key: str) -> Any:
        """
        Load value from memory.
        
        Args:
            key: Storage key
            
        Returns:
            Stored value or None if not found
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete value from memory.
        
        Args:
            key: Storage key
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in memory.
        
        Args:
            key: Storage key
            
        Returns:
            True if key exists, False otherwise
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all data from memory.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        List all keys in memory, optionally filtered by pattern.
        
        Args:
            pattern: Optional pattern to filter keys
            
        Returns:
            List of keys
        """
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """
        Get total size of stored data in bytes.
        
        Returns:
            Size in bytes
        """
        pass
    
    def batch_save(self, data: Dict[str, Any], ttl: Optional[int] = None) -> Dict[str, bool]:
        """
        Save multiple key-value pairs in batch.
        
        Args:
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds
            
        Returns:
            Dictionary of key -> success status
        """
        def _batch_save():
            results = {}
            for key, value in data.items():
                try:
                    results[key] = self.save(key, value, ttl)
                except Exception as e:
                    logger.error(f"Batch save failed for key '{key}': {e}")
                    results[key] = False
            return results
        
        return self._execute_with_metrics(MemoryOperation.SAVE, _batch_save)
    
    def batch_load(self, keys: List[str]) -> Dict[str, Any]:
        """
        Load multiple values in batch.
        
        Args:
            keys: List of keys to load
            
        Returns:
            Dictionary of key -> value (None for missing keys)
        """
        def _batch_load():
            results = {}
            for key in keys:
                try:
                    results[key] = self.load(key)
                except Exception as e:
                    logger.error(f"Batch load failed for key '{key}': {e}")
                    results[key] = None
            return results
        
        return self._execute_with_metrics(MemoryOperation.LOAD, _batch_load)
    
    def batch_delete(self, keys: List[str]) -> Dict[str, bool]:
        """
        Delete multiple keys in batch.
        
        Args:
            keys: List of keys to delete
            
        Returns:
            Dictionary of key -> success status
        """
        def _batch_delete():
            results = {}
            for key in keys:
                try:
                    results[key] = self.delete(key)
                except Exception as e:
                    logger.error(f"Batch delete failed for key '{key}': {e}")
                    results[key] = False
            return results
        
        return self._execute_with_metrics(MemoryOperation.DELETE, _batch_delete)
    
    def get_stats(self) -> MemoryStats:
        """Get memory backend statistics"""
        with self._lock:
            # Update current stats
            self.stats.total_keys = len(self.list_keys())
            self.stats.total_size_bytes = self.get_size()
            return self.stats
    
    def get_health(self) -> Dict[str, Any]:
        """Get memory backend health status"""
        try:
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            test_value = "health_check_value"
            
            # Test save/load/delete
            save_success = self.save(test_key, test_value, ttl=60)
            load_success = self.load(test_key) == test_value
            delete_success = self.delete(test_key)
            
            healthy = save_success and load_success and delete_success
            
            return {
                "status": "healthy" if healthy else "unhealthy",
                "backend_type": self.__class__.__name__,
                "connected": self._connected,
                "operations_test": {
                    "save": save_success,
                    "load": load_success,
                    "delete": delete_success
                },
                "stats": self.get_stats().__dict__,
                "timestamp": "2025-07-22 12:01:02",
                "checked_by": "Vatsal216"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend_type": self.__class__.__name__,
                "connected": self._connected,
                "error": str(e),
                "timestamp": "2025-07-22 12:01:02",
                "checked_by": "Vatsal216"
            }
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(connected={self._connected}, keys={self.stats.total_keys})"