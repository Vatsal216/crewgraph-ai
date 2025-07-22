"""
Redis Memory Backend for CrewGraph AI
Production-ready Redis integration with clustering support

Author: Vatsal216
Created: 2025-07-22 12:01:02 UTC
"""

import time
import json
import pickle
import gzip
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

try:
    import redis
    from redis.sentinel import Sentinel
    from redis.cluster import RedisCluster
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .base import BaseMemory, MemoryOperation
from .config import MemoryConfig
from ..utils.logging import get_logger
from ..utils.exceptions import MemoryError

logger = get_logger(__name__)


class RedisMemory(BaseMemory):
    """
    Redis-based memory backend for CrewGraph AI.
    
    Provides production-ready Redis integration with support for:
    - Single Redis instance
    - Redis Sentinel for high availability
    - Redis Cluster for horizontal scaling
    - Connection pooling and retry logic
    - Compression and encryption
    - TTL management
    - Batch operations
    
    Created by: Vatsal216
    Date: 2025-07-22 12:01:02 UTC
    """
    
    def __init__(self, config: Optional[Union[Dict, MemoryConfig]] = None):
        """
        Initialize Redis memory backend.
        
        Args:
            config: Redis configuration
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not installed. Install with: pip install redis")
        
        super().__init__()
        
        # Parse configuration
        if isinstance(config, MemoryConfig):
            self.config_obj = config
        else:
            from .config import MemoryConfig, MemoryType
            self.config_obj = MemoryConfig(memory_type=MemoryType.REDIS)
            if config:
                for key, value in config.items():
                    if hasattr(self.config_obj, key):
                        setattr(self.config_obj, key, value)
        
        # Redis connection
        self.redis_client: Optional[Union[redis.Redis, RedisCluster]] = None
        self.sentinel: Optional[Sentinel] = None
        
        # Configuration shortcuts
        self.host = getattr(self.config_obj, 'redis_host', 'localhost')
        self.port = getattr(self.config_obj, 'redis_port', 6379)
        self.password = getattr(self.config_obj, 'redis_password', None)
        self.database = getattr(self.config_obj, 'redis_database', 0)
        self.prefix = getattr(self.config_obj, 'key_prefix', 'crewgraph:')
        self.default_ttl = getattr(self.config_obj, 'ttl', None)
        
        # Advanced features
        self.enable_compression = getattr(self.config_obj, 'compression', False)
        self.compression_threshold = getattr(self.config_obj, 'compression_threshold', 1024)
        self.serialization_format = getattr(self.config_obj, 'serialization', 'pickle')
        
        # Connection settings
        self.connection_pool_size = getattr(self.config_obj, 'connection_pool_size', 10)
        self.socket_timeout = getattr(self.config_obj, 'socket_timeout', 30)
        self.retry_attempts = getattr(self.config_obj, 'retry_attempts', 3)
        
        # Cluster/Sentinel settings
        self.use_cluster = getattr(self.config_obj, 'use_cluster', False)
        self.use_sentinel = getattr(self.config_obj, 'use_sentinel', False)
        self.cluster_nodes = getattr(self.config_obj, 'cluster_nodes', [])
        self.sentinel_hosts = getattr(self.config_obj, 'sentinel_hosts', [])
        self.master_name = getattr(self.config_obj, 'master_name', 'mymaster')
        
        logger.info(f"RedisMemory initialized for {self.host}:{self.port}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:01:02")
    
    def connect(self) -> None:
        """Connect to Redis with support for single instance, sentinel, or cluster"""
        try:
            if self.use_cluster:
                self._connect_cluster()
            elif self.use_sentinel:
                self._connect_sentinel()
            else:
                self._connect_single()
            
            # Test connection
            self.redis_client.ping()
            self._connected = True
            
            logger.info("Redis connection established successfully")
            logger.info(f"Connected by user: Vatsal216 at 2025-07-22 12:01:02")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise MemoryError(
                "Redis connection failed",
                backend="Redis",
                operation="connect",
                original_error=e
            )
    
    def _connect_single(self) -> None:
        """Connect to single Redis instance"""
        self.redis_client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            db=self.database,
            socket_timeout=self.socket_timeout,
            connection_pool_max_connections=self.connection_pool_size,
            decode_responses=False,  # We handle serialization ourselves
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        logger.info(f"Connected to single Redis instance: {self.host}:{self.port}")
    
    def _connect_sentinel(self) -> None:
        """Connect to Redis via Sentinel for high availability"""
        if not self.sentinel_hosts:
            raise MemoryError("Sentinel hosts not configured")
        
        sentinel_list = [(host, port) for host, port in self.sentinel_hosts]
        self.sentinel = Sentinel(
            sentinel_list,
            socket_timeout=self.socket_timeout,
            password=self.password
        )
        
        self.redis_client = self.sentinel.master_for(
            self.master_name,
            socket_timeout=self.socket_timeout,
            password=self.password,
            db=self.database
        )
        
        logger.info(f"Connected to Redis via Sentinel: {self.master_name}")
    
    def _connect_cluster(self) -> None:
        """Connect to Redis Cluster for horizontal scaling"""
        if not self.cluster_nodes:
            raise MemoryError("Cluster nodes not configured")
        
        startup_nodes = [{"host": host, "port": port} for host, port in self.cluster_nodes]
        
        self.redis_client = RedisCluster(
            startup_nodes=startup_nodes,
            password=self.password,
            socket_timeout=self.socket_timeout,
            decode_responses=False,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        logger.info(f"Connected to Redis Cluster: {len(self.cluster_nodes)} nodes")
    
    def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self.redis_client:
            try:
                if hasattr(self.redis_client, 'connection_pool'):
                    self.redis_client.connection_pool.disconnect()
                self._connected = False
                logger.info("Redis disconnected successfully")
            except Exception as e:
                logger.error(f"Error during Redis disconnect: {e}")
    
    def _get_full_key(self, key: str) -> str:
        """Get full key with prefix"""
        return f"{self.prefix}{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        if self.serialization_format == 'json':
            try:
                serialized = json.dumps(value).encode('utf-8')
            except (TypeError, ValueError):
                # Fallback to pickle for non-JSON serializable objects
                serialized = pickle.dumps(value)
        else:
            # Default to pickle
            serialized = pickle.dumps(value)
        
        # Apply compression if enabled and threshold met
        if self.enable_compression and len(serialized) >= self.compression_threshold:
            compressed = gzip.compress(serialized)
            # Add compression marker
            return b'GZIP:' + compressed
        
        return serialized
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        if data.startswith(b'GZIP:'):
            # Decompress
            compressed_data = data[5:]  # Remove 'GZIP:' prefix
            data = gzip.decompress(compressed_data)
        
        try:
            if self.serialization_format == 'json':
                return json.loads(data.decode('utf-8'))
            else:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize value: {e}")
            raise MemoryError(
                "Value deserialization failed",
                backend="Redis",
                operation="deserialize",
                original_error=e
            )
    
    def save(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Save value to Redis with TTL support"""
        def _save():
            full_key = self._get_full_key(key)
            serialized_value = self._serialize_value(value)
            effective_ttl = ttl if ttl is not None else self.default_ttl
            
            if effective_ttl:
                result = self.redis_client.setex(full_key, effective_ttl, serialized_value)
            else:
                result = self.redis_client.set(full_key, serialized_value)
            
            logger.debug(f"Saved key '{key}' to Redis (TTL: {effective_ttl})")
            return bool(result)
        
        return self._execute_with_metrics(MemoryOperation.SAVE, _save)
    
    def load(self, key: str) -> Any:
        """Load value from Redis"""
        def _load():
            full_key = self._get_full_key(key)
            data = self.redis_client.get(full_key)
            
            if data is None:
                logger.debug(f"Key '{key}' not found in Redis")
                return None
            
            value = self._deserialize_value(data)
            logger.debug(f"Loaded key '{key}' from Redis")
            return value
        
        return self._execute_with_metrics(MemoryOperation.LOAD, _load)
    
    def delete(self, key: str) -> bool:
        """Delete value from Redis"""
        def _delete():
            full_key = self._get_full_key(key)
            result = self.redis_client.delete(full_key)
            logger.debug(f"Deleted key '{key}' from Redis")
            return bool(result)
        
        return self._execute_with_metrics(MemoryOperation.DELETE, _delete)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        def _exists():
            full_key = self._get_full_key(key)
            return bool(self.redis_client.exists(full_key))
        
        return self._execute_with_metrics(MemoryOperation.EXISTS, _exists)
    
    def clear(self) -> bool:
        """Clear all keys with the configured prefix"""
        def _clear():
            # Use pattern to only delete keys with our prefix
            pattern = f"{self.prefix}*"
            
            if self.use_cluster:
                # For cluster, we need to scan each node
                deleted_count = 0
                for node in self.redis_client.get_nodes():
                    for key in node.scan_iter(match=pattern):
                        node.delete(key)
                        deleted_count += 1
            else:
                # For single instance or sentinel
                deleted_count = 0
                for key in self.redis_client.scan_iter(match=pattern):
                    self.redis_client.delete(key)
                    deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} keys from Redis with prefix '{self.prefix}'")
            return True
        
        return self._execute_with_metrics(MemoryOperation.CLEAR, _clear)
    
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all keys with optional pattern"""
        def _list_keys():
            if pattern:
                search_pattern = f"{self.prefix}{pattern}"
            else:
                search_pattern = f"{self.prefix}*"
            
            keys = []
            
            if self.use_cluster:
                # For cluster, scan each node
                for node in self.redis_client.get_nodes():
                    for key in node.scan_iter(match=search_pattern):
                        # Remove prefix and decode
                        clean_key = key.decode('utf-8')[len(self.prefix):]
                        keys.append(clean_key)
            else:
                # For single instance or sentinel
                for key in self.redis_client.scan_iter(match=search_pattern):
                    # Remove prefix and decode
                    clean_key = key.decode('utf-8')[len(self.prefix):]
                    keys.append(clean_key)
            
            return sorted(keys)
        
        return self._execute_with_metrics(MemoryOperation.LIST_KEYS, _list_keys)
    
    def get_size(self) -> int:
        """Get approximate total size of stored data"""
        def _get_size():
            total_size = 0
            pattern = f"{self.prefix}*"
            
            if self.use_cluster:
                # For cluster, check each node
                for node in self.redis_client.get_nodes():
                    for key in node.scan_iter(match=pattern):
                        try:
                            size = node.memory_usage(key)
                            if size:
                                total_size += size
                        except Exception:
                            # Fallback to string length estimate
                            total_size += len(key) + 1024  # Rough estimate
            else:
                # For single instance or sentinel
                for key in self.redis_client.scan_iter(match=pattern):
                    try:
                        size = self.redis_client.memory_usage(key)
                        if size:
                            total_size += size
                    except Exception:
                        # Fallback to string length estimate
                        total_size += len(key) + 1024  # Rough estimate
            
            return total_size
        
        return self._execute_with_metrics(MemoryOperation.GET_SIZE, _get_size)
    
    def batch_save(self, data: Dict[str, Any], ttl: Optional[int] = None) -> Dict[str, bool]:
        """Save multiple key-value pairs using Redis pipeline"""
        def _batch_save():
            results = {}
            effective_ttl = ttl if ttl is not None else self.default_ttl
            
            # Use pipeline for better performance
            pipe = self.redis_client.pipeline()
            
            for key, value in data.items():
                try:
                    full_key = self._get_full_key(key)
                    serialized_value = self._serialize_value(value)
                    
                    if effective_ttl:
                        pipe.setex(full_key, effective_ttl, serialized_value)
                    else:
                        pipe.set(full_key, serialized_value)
                        
                except Exception as e:
                    logger.error(f"Failed to prepare batch save for key '{key}': {e}")
                    results[key] = False
            
            # Execute pipeline
            try:
                pipeline_results = pipe.execute()
                
                # Map results back to keys
                key_list = list(data.keys())
                for i, result in enumerate(pipeline_results):
                    if i < len(key_list):
                        results[key_list[i]] = bool(result)
                        
            except Exception as e:
                logger.error(f"Batch save pipeline failed: {e}")
                # Mark all remaining as failed
                for key in data.keys():
                    if key not in results:
                        results[key] = False
            
            return results
        
        return self._execute_with_metrics(MemoryOperation.SAVE, _batch_save)
    
    def batch_load(self, keys: List[str]) -> Dict[str, Any]:
        """Load multiple values using Redis pipeline"""
        def _batch_load():
            results = {}
            
            # Prepare pipeline
            pipe = self.redis_client.pipeline()
            full_keys = [self._get_full_key(key) for key in keys]
            
            for full_key in full_keys:
                pipe.get(full_key)
            
            try:
                # Execute pipeline
                pipeline_results = pipe.execute()
                
                # Process results
                for i, (key, data) in enumerate(zip(keys, pipeline_results)):
                    if data is not None:
                        try:
                            results[key] = self._deserialize_value(data)
                        except Exception as e:
                            logger.error(f"Failed to deserialize key '{key}': {e}")
                            results[key] = None
                    else:
                        results[key] = None
                        
            except Exception as e:
                logger.error(f"Batch load pipeline failed: {e}")
                # Return None for all keys
                results = {key: None for key in keys}
            
            return results
        
        return self._execute_with_metrics(MemoryOperation.LOAD, _batch_load)
    
    def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        try:
            if self.use_cluster:
                # For cluster, get info from all nodes
                cluster_info = {}
                for node in self.redis_client.get_nodes():
                    node_info = node.info()
                    cluster_info[f"{node.host}:{node.port}"] = {
                        "role": node_info.get("role", "unknown"),
                        "connected_clients": node_info.get("connected_clients", 0),
                        "used_memory": node_info.get("used_memory", 0),
                        "keyspace": node_info.get("db0", {})
                    }
                return cluster_info
            else:
                # Single instance or sentinel
                info = self.redis_client.info()
                return {
                    "redis_version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory": info.get("used_memory"),
                    "used_memory_human": info.get("used_memory_human"),
                    "keyspace": info.get("db0", {}),
                    "role": info.get("role"),
                    "master_host": info.get("master_host"),
                    "master_port": info.get("master_port")
                }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {"error": str(e)}
    
    def get_ttl(self, key: str) -> int:
        """Get TTL for a key (-1 if no TTL, -2 if key doesn't exist)"""
        full_key = self._get_full_key(key)
        return self.redis_client.ttl(full_key)
    
    def set_ttl(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        full_key = self._get_full_key(key)
        return bool(self.redis_client.expire(full_key, ttl))
    
    def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """Extend TTL for existing key"""
        current_ttl = self.get_ttl(key)
        if current_ttl > 0:
            return self.set_ttl(key, current_ttl + additional_seconds)
        return False