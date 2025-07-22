"""
Memory Utilities for CrewGraph AI
Helper functions and utilities for memory management

Author: Vatsal216
Created: 2025-07-22 12:09:19 UTC
"""

import json
import pickle
import gzip
import hashlib
import time
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass
import threading

from .base import BaseMemory
from .dict_memory import DictMemory
from .config import MemoryConfig, MemoryType
from ..utils.logging import get_logger
from ..utils.exceptions import MemoryError

logger = get_logger(__name__)


@dataclass
class MemoryBenchmarkResult:
    """Memory backend benchmark results"""
    backend_type: str
    operation: str
    total_operations: int
    total_time: float
    operations_per_second: float
    average_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    success_rate: float
    memory_usage_mb: float
    created_by: str = "Vatsal216"
    created_at: str = "2025-07-22 12:09:19"


class MemorySerializer:
    """
    Advanced serialization utilities for memory backends.
    
    Provides multiple serialization formats with compression,
    encryption, and performance optimization.
    
    Created by: Vatsal216
    Date: 2025-07-22 12:09:19 UTC
    """
    
    def __init__(self, 
                 format_type: str = "pickle",
                 enable_compression: bool = False,
                 compression_threshold: int = 1024,
                 compression_level: int = 6):
        """
        Initialize memory serializer.
        
        Args:
            format_type: Serialization format ('pickle', 'json', 'msgpack')
            enable_compression: Enable compression for large objects
            compression_threshold: Minimum size in bytes to trigger compression
            compression_level: Compression level (1-9, higher = better compression)
        """
        self.format_type = format_type
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.compression_level = compression_level
        
        # Import optional dependencies
        self._msgpack = None
        if format_type == "msgpack":
            try:
                import msgpack
                self._msgpack = msgpack
            except ImportError:
                logger.warning("msgpack not available, falling back to pickle")
                self.format_type = "pickle"
        
        logger.info(f"MemorySerializer initialized: {self.format_type}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:09:19")
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object to bytes with optional compression"""
        try:
            # Serialize based on format
            if self.format_type == "json":
                serialized = json.dumps(obj, default=str).encode('utf-8')
            elif self.format_type == "msgpack" and self._msgpack:
                serialized = self._msgpack.packb(obj, default=str)
            else:
                # Default to pickle
                serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Apply compression if enabled and threshold met
            if self.enable_compression and len(serialized) >= self.compression_threshold:
                compressed = gzip.compress(serialized, compresslevel=self.compression_level)
                # Add compression marker
                return b'GZIP:' + compressed
            
            return serialized
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise MemoryError(f"Serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to object with decompression"""
        try:
            # Check for compression marker
            if data.startswith(b'GZIP:'):
                # Decompress
                compressed_data = data[5:]  # Remove 'GZIP:' prefix
                data = gzip.decompress(compressed_data)
            
            # Deserialize based on format
            if self.format_type == "json":
                return json.loads(data.decode('utf-8'))
            elif self.format_type == "msgpack" and self._msgpack:
                return self._msgpack.unpackb(data, raw=False)
            else:
                # Default to pickle
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise MemoryError(f"Deserialization failed: {e}")
    
    def get_compression_ratio(self, obj: Any) -> float:
        """Get compression ratio for an object"""
        try:
            original = self.serialize(obj)
            if original.startswith(b'GZIP:'):
                # Already compressed
                uncompressed_size = len(gzip.decompress(original[5:]))
                compressed_size = len(original) - 5  # Remove marker
                return uncompressed_size / compressed_size
            else:
                # Not compressed
                return 1.0
                
        except Exception:
            return 1.0


class MemoryUtils:
    """
    Utility functions for memory management and optimization.
    
    Provides helper functions for memory backend selection,
    performance testing, and optimization.
    
    Created by: Vatsal216
    Date: 2025-07-22 12:09:19 UTC
    """
    
    @staticmethod
    def create_memory_backend(config: Union[MemoryConfig, Dict[str, Any]]) -> BaseMemory:
        """
        Factory method to create memory backend from configuration.
        
        Args:
            config: Memory configuration
            
        Returns:
            Configured memory backend instance
        """
        if isinstance(config, dict):
            from .config import MemoryConfig
            config = MemoryConfig.from_dict(config)
        
        logger.info(f"Creating memory backend: {config.memory_type.value}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:09:19")
        
        if config.memory_type == MemoryType.DICT:
            backend_config = config.get_backend_config()
            return DictMemory(**backend_config)
        
        elif config.memory_type == MemoryType.REDIS:
            from .redis_memory import RedisMemory
            return RedisMemory(config)
        
        elif config.memory_type == MemoryType.FAISS:
            from .faiss_memory import FAISSMemory
            backend_config = config.get_backend_config()
            return FAISSMemory(**backend_config)
        
        elif config.memory_type == MemoryType.SQL:
            from .sql_memory import SQLMemory
            backend_config = config.get_backend_config()
            return SQLMemory(**backend_config)
        
        else:
            raise ValueError(f"Unsupported memory type: {config.memory_type}")
    
    @staticmethod
    def benchmark_memory_backend(backend: BaseMemory, 
                                test_data_size: int = 1000,
                                key_prefix: str = "benchmark_",
                                value_size: int = 1024) -> Dict[str, MemoryBenchmarkResult]:
        """
        Benchmark memory backend performance.
        
        Args:
            backend: Memory backend to benchmark
            test_data_size: Number of test items
            key_prefix: Prefix for test keys
            value_size: Size of test values in bytes
            
        Returns:
            Dictionary of benchmark results by operation
        """
        logger.info(f"Starting memory backend benchmark")
        logger.info(f"Backend: {backend.__class__.__name__}, Size: {test_data_size}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:09:19")
        
        results = {}
        
        # Generate test data
        test_data = {}
        test_value = 'x' * value_size
        
        for i in range(test_data_size):
            test_data[f"{key_prefix}{i}"] = f"{test_value}_{i}"
        
        # Benchmark save operations
        start_time = time.time()
        save_times = []
        save_successes = 0
        
        for key, value in test_data.items():
            op_start = time.time()
            try:
                if backend.save(key, value):
                    save_successes += 1
                save_times.append((time.time() - op_start) * 1000)  # Convert to ms
            except Exception as e:
                logger.error(f"Save failed for key {key}: {e}")
                save_times.append(float('inf'))
        
        total_save_time = time.time() - start_time
        
        results['save'] = MemoryBenchmarkResult(
            backend_type=backend.__class__.__name__,
            operation="save",
            total_operations=test_data_size,
            total_time=total_save_time,
            operations_per_second=test_data_size / total_save_time,
            average_latency_ms=sum(t for t in save_times if t != float('inf')) / len([t for t in save_times if t != float('inf')]),
            min_latency_ms=min(t for t in save_times if t != float('inf')),
            max_latency_ms=max(t for t in save_times if t != float('inf')),
            success_rate=save_successes / test_data_size,
            memory_usage_mb=backend.get_size() / (1024 * 1024)
        )
        
        # Benchmark load operations
        start_time = time.time()
        load_times = []
        load_successes = 0
        
        for key in test_data.keys():
            op_start = time.time()
            try:
                value = backend.load(key)
                if value is not None:
                    load_successes += 1
                load_times.append((time.time() - op_start) * 1000)
            except Exception as e:
                logger.error(f"Load failed for key {key}: {e}")
                load_times.append(float('inf'))
        
        total_load_time = time.time() - start_time
        
        results['load'] = MemoryBenchmarkResult(
            backend_type=backend.__class__.__name__,
            operation="load",
            total_operations=test_data_size,
            total_time=total_load_time,
            operations_per_second=test_data_size / total_load_time,
            average_latency_ms=sum(t for t in load_times if t != float('inf')) / len([t for t in load_times if t != float('inf')]),
            min_latency_ms=min(t for t in load_times if t != float('inf')),
            max_latency_ms=max(t for t in load_times if t != float('inf')),
            success_rate=load_successes / test_data_size,
            memory_usage_mb=backend.get_size() / (1024 * 1024)
        )
        
        # Benchmark delete operations
        start_time = time.time()
        delete_times = []
        delete_successes = 0
        
        for key in test_data.keys():
            op_start = time.time()
            try:
                if backend.delete(key):
                    delete_successes += 1
                delete_times.append((time.time() - op_start) * 1000)
            except Exception as e:
                logger.error(f"Delete failed for key {key}: {e}")
                delete_times.append(float('inf'))
        
        total_delete_time = time.time() - start_time
        
        results['delete'] = MemoryBenchmarkResult(
            backend_type=backend.__class__.__name__,
            operation="delete",
            total_operations=test_data_size,
            total_time=total_delete_time,
            operations_per_second=test_data_size / total_delete_time,
            average_latency_ms=sum(t for t in delete_times if t != float('inf')) / len([t for t in delete_times if t != float('inf')]),
            min_latency_ms=min(t for t in delete_times if t != float('inf')),
            max_latency_ms=max(t for t in delete_times if t != float('inf')),
            success_rate=delete_successes / test_data_size,
            memory_usage_mb=backend.get_size() / (1024 * 1024)
        )
        
        logger.info(f"Benchmark completed for {backend.__class__.__name__}")
        
        return results
    
    @staticmethod
    def compare_memory_backends(configs: List[MemoryConfig], 
                               test_size: int = 1000) -> Dict[str, Dict[str, MemoryBenchmarkResult]]:
        """
        Compare performance of multiple memory backends.
        
        Args:
            configs: List of memory configurations to compare
            test_size: Number of test operations
            
        Returns:
            Comparison results by backend and operation
        """
        logger.info(f"Comparing {len(configs)} memory backends")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:09:19")
        
        comparison_results = {}
        
        for config in configs:
            backend_name = f"{config.memory_type.value}_{config.environment}"
            
            try:
                # Create and connect backend
                backend = MemoryUtils.create_memory_backend(config)
                backend.connect()
                
                # Run benchmark
                results = MemoryUtils.benchmark_memory_backend(backend, test_size)
                comparison_results[backend_name] = results
                
                # Cleanup
                backend.clear()
                backend.disconnect()
                
                logger.info(f"Completed benchmark for {backend_name}")
                
            except Exception as e:
                logger.error(f"Benchmark failed for {backend_name}: {e}")
                comparison_results[backend_name] = {"error": str(e)}
        
        return comparison_results
    
    @staticmethod
    def optimize_memory_config(workload_pattern: Dict[str, Any]) -> MemoryConfig:
        """
        Optimize memory configuration based on workload patterns.
        
        Args:
            workload_pattern: Dictionary describing workload characteristics
            
        Returns:
            Optimized memory configuration
        """
        logger.info("Optimizing memory configuration based on workload")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:09:19")
        
        # Extract workload characteristics
        data_size = workload_pattern.get('data_size', 'medium')  # small, medium, large
        access_pattern = workload_pattern.get('access_pattern', 'random')  # random, sequential, hotspot
        read_write_ratio = workload_pattern.get('read_write_ratio', 0.8)  # 0.0 = all writes, 1.0 = all reads
        durability_required = workload_pattern.get('durability', False)
        vector_search = workload_pattern.get('vector_search', False)
        environment = workload_pattern.get('environment', 'development')
        
        # Choose optimal backend type
        if vector_search:
            memory_type = MemoryType.FAISS
        elif durability_required or data_size == 'large':
            memory_type = MemoryType.SQL if durability_required else MemoryType.REDIS
        elif data_size == 'small' and environment == 'development':
            memory_type = MemoryType.DICT
        else:
            memory_type = MemoryType.REDIS
        
        # Create base configuration
        config = MemoryConfig(
            memory_type=memory_type,
            environment=environment,
            created_by="Vatsal216",
            created_at="2025-07-22 12:09:19"
        )
        
        # Optimize based on data size
        if data_size == 'small':
            config.max_size = 1000
            config.compression = False
        elif data_size == 'medium':
            config.max_size = 10000
            config.compression = True
            config.compression_threshold = 1024
        else:  # large
            config.max_size = 100000
            config.compression = True
            config.compression_threshold = 512
        
        # Optimize based on access pattern
        if access_pattern == 'hotspot':
            # Favor faster backends for hot data
            if memory_type == MemoryType.REDIS:
                config.connection_pool_size = 20
        
        # Optimize based on read/write ratio
        if read_write_ratio > 0.9:
            # Read-heavy workload
            config.ttl = 3600  # Longer TTL for cached reads
        elif read_write_ratio < 0.1:
            # Write-heavy workload
            config.ttl = 300   # Shorter TTL to avoid stale data
        
        logger.info(f"Optimized configuration: {memory_type.value} for {environment}")
        
        return config
    
    @staticmethod
    def generate_cache_key(namespace: str, *args, **kwargs) -> str:
        """
        Generate consistent cache key from arguments.
        
        Args:
            namespace: Cache namespace
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Generated cache key
        """
        # Create deterministic string from arguments
        key_parts = [namespace]
        
        # Add positional arguments
        for arg in args:
            key_parts.append(str(arg))
        
        # Add keyword arguments (sorted for consistency)
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}={value}")
        
        # Create hash of the key parts
        key_string = ":".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"{namespace}:{key_hash}"
    
    @staticmethod
    def estimate_memory_usage(data: Any) -> int:
        """
        Estimate memory usage of data in bytes.
        
        Args:
            data: Data to estimate
            
        Returns:
            Estimated size in bytes
        """
        try:
            # Try pickle serialization for accurate size
            serialized = pickle.dumps(data)
            return len(serialized)
        except Exception:
            # Fallback to rough estimation
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (int, float)):
                return 8  # Rough estimate
            elif isinstance(data, (list, tuple)):
                return sum(MemoryUtils.estimate_memory_usage(item) for item in data)
            elif isinstance(data, dict):
                return sum(
                    MemoryUtils.estimate_memory_usage(k) + MemoryUtils.estimate_memory_usage(v) 
                    for k, v in data.items()
                )
            else:
                return 1024  # Default estimate


class MemoryPool:
    """
    Connection pool manager for memory backends.
    
    Manages multiple memory backend connections for improved
    performance and resource utilization.
    
    Created by: Vatsal216
    Date: 2025-07-22 12:09:19 UTC
    """
    
    def __init__(self, 
                 config: MemoryConfig,
                 pool_size: int = 5,
                 max_pool_size: int = 10,
                 timeout: float = 30.0):
        """
        Initialize memory connection pool.
        
        Args:
            config: Memory backend configuration
            pool_size: Initial pool size
            max_pool_size: Maximum pool size
            timeout: Connection timeout
        """
        self.config = config
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.timeout = timeout
        
        # Pool management
        self._pool: List[BaseMemory] = []
        self._active_connections: Dict[int, BaseMemory] = {}
        self._lock = threading.RLock()
        self._pool_created = False
        
        logger.info(f"MemoryPool initialized: {config.memory_type.value}")
        logger.info(f"Pool size: {pool_size}, Max: {max_pool_size}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:09:19")
    
    def _create_connection(self) -> BaseMemory:
        """Create new memory backend connection"""
        backend = MemoryUtils.create_memory_backend(self.config)
        backend.connect()
        return backend
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        if self._pool_created:
            return
        
        with self._lock:
            if self._pool_created:
                return
            
            for _ in range(self.pool_size):
                try:
                    connection = self._create_connection()
                    self._pool.append(connection)
                except Exception as e:
                    logger.error(f"Failed to create pool connection: {e}")
            
            self._pool_created = True
            logger.info(f"Memory pool initialized with {len(self._pool)} connections")
    
    def get_connection(self) -> BaseMemory:
        """Get connection from pool"""
        self._initialize_pool()
        
        with self._lock:
            # Try to get from pool
            if self._pool:
                connection = self._pool.pop()
                connection_id = id(connection)
                self._active_connections[connection_id] = connection
                return connection
            
            # Create new connection if under max limit
            if len(self._active_connections) < self.max_pool_size:
                connection = self._create_connection()
                connection_id = id(connection)
                self._active_connections[connection_id] = connection
                return connection
            
            # Pool exhausted
            raise MemoryError("Memory pool exhausted")
    
    def return_connection(self, connection: BaseMemory):
        """Return connection to pool"""
        with self._lock:
            connection_id = id(connection)
            
            if connection_id in self._active_connections:
                del self._active_connections[connection_id]
                
                # Return to pool if not full
                if len(self._pool) < self.pool_size:
                    self._pool.append(connection)
                else:
                    # Pool full, disconnect
                    try:
                        connection.disconnect()
                    except Exception as e:
                        logger.error(f"Error disconnecting connection: {e}")
    
    def close_pool(self):
        """Close all pool connections"""
        with self._lock:
            # Close active connections
            for connection in self._active_connections.values():
                try:
                    connection.disconnect()
                except Exception as e:
                    logger.error(f"Error closing active connection: {e}")
            
            # Close pooled connections
            for connection in self._pool:
                try:
                    connection.disconnect()
                except Exception as e:
                    logger.error(f"Error closing pool connection: {e}")
            
            self._active_connections.clear()
            self._pool.clear()
            self._pool_created = False
            
        logger.info("Memory pool closed")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "active_connections": len(self._active_connections),
                "max_pool_size": self.max_pool_size,
                "total_capacity": self.max_pool_size,
                "utilization": len(self._active_connections) / self.max_pool_size,
                "backend_type": self.config.memory_type.value,
                "created_by": "Vatsal216",
                "timestamp": "2025-07-22 12:09:19"
            }


# Factory function for easy memory backend creation
def create_memory(memory_type: str = "dict", **kwargs) -> BaseMemory:
    """
    Convenience function to create memory backend.
    
    Args:
        memory_type: Type of memory backend
        **kwargs: Configuration options
        
    Returns:
        Configured memory backend
    """
    from .config import create_memory_config
    
    config = create_memory_config(memory_type, **kwargs)
    backend = MemoryUtils.create_memory_backend(config)
    backend.connect()
    
    logger.info(f"Created {memory_type} memory backend")
    logger.info(f"User: Vatsal216, Time: 2025-07-22 12:09:19")
    
    return backend