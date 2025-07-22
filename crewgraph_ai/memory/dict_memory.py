"""
In-memory dictionary-based memory implementation
"""

import threading
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .base import BaseMemory, MemoryConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryEntry:
    """Memory entry with metadata"""
    value: bytes
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class DictMemory(BaseMemory):
    """
    High-performance in-memory dictionary storage with TTL and LRU eviction.
    
    Features:
    - Thread-safe operations
    - TTL (Time To Live) support
    - LRU (Least Recently Used) eviction
    - Memory usage monitoring
    - Access statistics
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize dictionary memory.
        
        Args:
            config: Memory configuration
        """
        super().__init__(config)
        self._data: Dict[str, MemoryEntry] = {}
        self._lock = threading.RLock()
        self._access_order: List[str] = []  # For LRU tracking
        self._cleanup_interval = 60  # Cleanup every 60 seconds
        self._last_cleanup = time.time()
        
        logger.info("DictMemory initialized")
    
    def connect(self) -> None:
        """Connect to memory backend."""
        self._is_connected = True
        logger.debug("DictMemory connected")
    
    def disconnect(self) -> None:
        """Disconnect from memory backend."""
        with self._lock:
            self._data.clear()
            self._access_order.clear()
        
        self._is_connected = False
        logger.debug("DictMemory disconnected")
    
    def _get(self, key: str) -> Optional[bytes]:
        """Get raw value from backend."""
        with self._lock:
            self._cleanup_expired()
            
            entry = self._data.get(key)
            if entry is None:
                return None
            
            # Check if expired
            current_time = time.time()
            if entry.expires_at and current_time > entry.expires_at:
                del self._data[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = current_time
            
            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return entry.value
    
    def _set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        """Set raw value in backend."""
        with self._lock:
            current_time = time.time()
            
            # Calculate expiration time
            expires_at = None
            if ttl:
                expires_at = current_time + ttl
            
            # Create entry
            entry = MemoryEntry(
                value=value,
                created_at=current_time,
                expires_at=expires_at
            )
            
            # Check if we need to evict entries (LRU)
            if self.config.max_size and len(self._data) >= self.config.max_size:
                self._evict_lru()
            
            # Store entry
            self._data[key] = entry
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            self._cleanup_expired()
    
    def _delete(self, key: str) -> bool:
        """Delete key from backend."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False
    
    def _exists(self, key: str) -> bool:
        """Check if key exists in backend."""
        with self._lock:
            self._cleanup_expired()
            
            if key not in self._data:
                return False
            
            entry = self._data[key]
            current_time = time.time()
            
            # Check if expired
            if entry.expires_at and current_time > entry.expires_at:
                del self._data[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return False
            
            return True
    
    def _keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        import fnmatch
        
        with self._lock:
            self._cleanup_expired()
            
            if pattern == "*":
                return list(self._data.keys())
            
            return [key for key in self._data.keys() if fnmatch.fnmatch(key, pattern)]
    
    def _clear(self) -> None:
        """Clear all data from backend."""
        with self._lock:
            self._data.clear()
            self._access_order.clear()
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        
        # Only cleanup if interval has passed
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        expired_keys = []
        for key, entry in self._data.items():
            if entry.expires_at and current_time > entry.expires_at:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._data[key]
            if key in self._access_order:
                self._access_order.remove(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
        
        self._last_cleanup = current_time
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self._access_order:
            return
        
        # Remove oldest accessed key
        oldest_key = self._access_order[0]
        del self._data[oldest_key]
        self._access_order.remove(oldest_key)
        
        logger.debug(f"Evicted LRU entry: {oldest_key}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage statistics."""
        with self._lock:
            total_size = 0
            entry_count = len(self._data)
            expired_count = 0
            current_time = time.time()
            
            for entry in self._data.values():
                total_size += len(entry.value)
                if entry.expires_at and current_time > entry.expires_at:
                    expired_count += 1
            
            return {
                'total_entries': entry_count,
                'total_size_bytes': total_size,
                'expired_entries': expired_count,
                'max_size_limit': self.config.max_size,
                'cleanup_interval': self._cleanup_interval,
                'last_cleanup': self._last_cleanup
            }
    
    def get_access_stats(self) -> Dict[str, Any]:
        """Get access statistics for all entries."""
        with self._lock:
            stats = {}
            for key, entry in self._data.items():
                stats[key] = {
                    'access_count': entry.access_count,
                    'last_accessed': entry.last_accessed,
                    'created_at': entry.created_at,
                    'expires_at': entry.expires_at,
                    'size_bytes': len(entry.value)
                }
            return stats
    
    def optimize(self) -> Dict[str, Any]:
        """Optimize memory usage by cleaning up and reorganizing."""
        with self._lock:
            initial_count = len(self._data)
            
            # Force cleanup of expired entries
            self._cleanup_expired()
            
            # Rebuild access order to remove any inconsistencies
            valid_keys = set(self._data.keys())
            self._access_order = [key for key in self._access_order if key in valid_keys]
            
            final_count = len(self._data)
            cleaned_count = initial_count - final_count
            
            result = {
                'initial_entries': initial_count,
                'final_entries': final_count,
                'cleaned_entries': cleaned_count,
                'memory_usage': self.get_memory_usage()
            }
            
            logger.info(f"Memory optimization completed: {result}")
            return result
    
    def __len__(self) -> int:
        """Get number of entries."""
        with self._lock:
            return len(self._data)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.exists(key)
    
    def __repr__(self) -> str:
        return f"DictMemory(entries={len(self._data)}, max_size={self.config.max_size})"