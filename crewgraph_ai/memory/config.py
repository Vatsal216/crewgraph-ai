"""
Memory Configuration for CrewGraph AI
Centralized configuration for all memory backends

Author: Vatsal216
Created: 2025-07-22 12:05:13 UTC
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MemoryType(Enum):
    """Supported memory backend types"""
    DICT = "dict"
    REDIS = "redis"
    FAISS = "faiss"
    SQL = "sql"


class SerializationFormat(Enum):
    """Supported serialization formats"""
    PICKLE = "pickle"
    JSON = "json"
    MSGPACK = "msgpack"


@dataclass
class MemoryConfig:
    """
    Comprehensive memory configuration for CrewGraph AI.
    
    Supports configuration for all memory backends with
    environment-specific settings and best practices.
    
    Created by: Vatsal216
    Date: 2025-07-22 12:05:13 UTC
    """
    
    # Basic configuration
    memory_type: MemoryType = MemoryType.DICT
    ttl: Optional[int] = None  # Default TTL in seconds
    max_size: int = 10000
    key_prefix: str = "crewgraph:"
    
    # Serialization and compression
    serialization: SerializationFormat = SerializationFormat.PICKLE
    compression: bool = False
    compression_threshold: int = 1024  # Bytes
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_database: int = 0
    redis_ssl: bool = False
    connection_pool_size: int = 10
    socket_timeout: int = 30
    retry_attempts: int = 3
    
    # Redis cluster/sentinel
    use_cluster: bool = False
    use_sentinel: bool = False
    cluster_nodes: List[Tuple[str, int]] = field(default_factory=list)
    sentinel_hosts: List[Tuple[str, int]] = field(default_factory=list)
    master_name: str = "mymaster"
    
    # SQL configuration  
    database_url: str = "sqlite:///crewgraph_memory.db"
    table_prefix: str = "crewgraph_"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    enable_ttl_cleanup: bool = True
    ttl_cleanup_interval: int = 3600
    
    # FAISS configuration
    vector_dimension: int = 384
    faiss_index_type: str = "Flat"  # Flat, IVF, HNSW, LSH
    faiss_metric: str = "L2"  # L2, IP
    use_gpu: bool = False
    nlist: int = 100  # For IVF index
    max_vectors: int = 1000000
    enable_metadata: bool = True
    
    # Security and encryption
    encryption_key: Optional[str] = None
    enable_encryption: bool = False
    
    # Performance settings
    batch_size: int = 100
    enable_async: bool = False
    timeout: float = 30.0
    
    # Monitoring and logging
    enable_metrics: bool = True
    enable_health_checks: bool = True
    log_operations: bool = False
    
    # Environment specific
    environment: str = "development"
    created_by: str = "Vatsal216"
    created_at: str = "2025-07-22 12:05:13"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        self._apply_environment_defaults()
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Basic validation
        if self.max_size <= 0:
            errors.append("max_size must be positive")
        
        if self.ttl is not None and self.ttl <= 0:
            errors.append("ttl must be positive or None")
        
        # Redis validation
        if self.memory_type == MemoryType.REDIS:
            if not (1 <= self.redis_port <= 65535):
                errors.append("redis_port must be between 1 and 65535")
            
            if self.use_cluster and not self.cluster_nodes:
                errors.append("cluster_nodes required when use_cluster is True")
            
            if self.use_sentinel and not self.sentinel_hosts:
                errors.append("sentinel_hosts required when use_sentinel is True")
        
        # FAISS validation
        if self.memory_type == MemoryType.FAISS:
            if self.vector_dimension <= 0:
                errors.append("vector_dimension must be positive")
            
            if self.faiss_index_type not in ["Flat", "IVF", "HNSW", "LSH"]:
                errors.append("Invalid faiss_index_type")
            
            if self.faiss_metric not in ["L2", "IP"]:
                errors.append("Invalid faiss_metric")
        
        # SQL validation
        if self.memory_type == MemoryType.SQL:
            if not self.database_url:
                errors.append("database_url is required for SQL backend")
            
            if self.pool_size <= 0:
                errors.append("pool_size must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    
    def _apply_environment_defaults(self):
        """Apply environment-specific defaults"""
        if self.environment == "production":
            # Production defaults
            self.compression = True
            self.enable_encryption = True
            self.enable_metrics = True
            self.enable_health_checks = True
            self.ttl_cleanup_interval = 1800  # 30 minutes
            
            # More conservative settings
            if self.memory_type == MemoryType.REDIS:
                self.connection_pool_size = 20
                self.retry_attempts = 5
            
        elif self.environment == "development":
            # Development defaults
            self.compression = False
            self.enable_encryption = False
            self.log_operations = True
            
            # Relaxed settings
            if self.memory_type == MemoryType.SQL:
                self.database_url = "sqlite:///crewgraph_dev.db"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], tuple):
                # Handle list of tuples (cluster_nodes, sentinel_hosts)
                result[key] = [list(item) for item in value]
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryConfig':
        """Create configuration from dictionary"""
        # Convert enum strings back to enums
        if 'memory_type' in data:
            data['memory_type'] = MemoryType(data['memory_type'])
        if 'serialization' in data:
            data['serialization'] = SerializationFormat(data['serialization'])
        
        # Convert list of lists back to list of tuples
        for key in ['cluster_nodes', 'sentinel_hosts']:
            if key in data and data[key]:
                data[key] = [tuple(item) for item in data[key]]
        
        return cls(**data)
    
    def get_backend_config(self) -> Dict[str, Any]:
        """Get backend-specific configuration"""
        if self.memory_type == MemoryType.DICT:
            return {
                'max_size': self.max_size,
                'default_ttl': self.ttl,
                'enable_compression': self.compression,
                'compression_threshold': self.compression_threshold
            }
        
        elif self.memory_type == MemoryType.REDIS:
            config = {
                'redis_host': self.redis_host,
                'redis_port': self.redis_port,
                'redis_password': self.redis_password,
                'redis_database': self.redis_database,
                'key_prefix': self.key_prefix,
                'ttl': self.ttl,
                'compression': self.compression,
                'compression_threshold': self.compression_threshold,
                'serialization': self.serialization.value,
                'connection_pool_size': self.connection_pool_size,
                'socket_timeout': self.socket_timeout,
                'retry_attempts': self.retry_attempts,
                'use_cluster': self.use_cluster,
                'use_sentinel': self.use_sentinel,
                'cluster_nodes': self.cluster_nodes,
                'sentinel_hosts': self.sentinel_hosts,
                'master_name': self.master_name
            }
            return config
        
        elif self.memory_type == MemoryType.FAISS:
            return {
                'dimension': self.vector_dimension,
                'index_type': self.faiss_index_type,
                'metric_type': self.faiss_metric,
                'use_gpu': self.use_gpu,
                'nlist': self.nlist,
                'max_vectors': self.max_vectors,
                'enable_metadata': self.enable_metadata,
                'enable_ttl': self.ttl is not None
            }
        
        elif self.memory_type == MemoryType.SQL:
            return {
                'database_url': self.database_url,
                'table_prefix': self.table_prefix,
                'enable_compression': self.compression,
                'compression_threshold': self.compression_threshold,
                'serialization_format': self.serialization.value,
                'pool_size': self.pool_size,
                'max_overflow': self.max_overflow,
                'pool_timeout': self.pool_timeout,
                'enable_ttl_cleanup': self.enable_ttl_cleanup,
                'ttl_cleanup_interval': self.ttl_cleanup_interval
            }
        
        return {}


def create_memory_config(memory_type: str, environment: str = "development", **kwargs) -> MemoryConfig:
    """
    Create memory configuration with best practices.
    
    Args:
        memory_type: Type of memory backend
        environment: Environment (development, staging, production)
        **kwargs: Additional configuration options
        
    Returns:
        Configured MemoryConfig instance
    """
    # Convert string to enum
    if isinstance(memory_type, str):
        memory_type = MemoryType(memory_type.lower())
    
    # Base configuration
    config = MemoryConfig(
        memory_type=memory_type,
        environment=environment,
        created_by="Vatsal216",
        created_at="2025-07-22 12:05:13"
    )
    
    # Apply custom settings
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# Pre-configured memory configurations for common scenarios
DEVELOPMENT_CONFIGS = {
    MemoryType.DICT: MemoryConfig(
        memory_type=MemoryType.DICT,
        max_size=1000,
        compression=False,
        environment="development"
    ),
    
    MemoryType.REDIS: MemoryConfig(
        memory_type=MemoryType.REDIS,
        redis_host="localhost",
        compression=False,
        environment="development"
    ),
    
    MemoryType.SQL: MemoryConfig(
        memory_type=MemoryType.SQL,
        database_url="sqlite:///crewgraph_dev.db",
        compression=False,
        environment="development"
    ),
    
    MemoryType.FAISS: MemoryConfig(
        memory_type=MemoryType.FAISS,
        vector_dimension=384,
        max_vectors=10000,
        environment="development"
    )
}

PRODUCTION_CONFIGS = {
    MemoryType.REDIS: MemoryConfig(
        memory_type=MemoryType.REDIS,
        redis_host="redis-cluster.internal",
        redis_port=6379,
        compression=True,
        compression_threshold=512,
        connection_pool_size=20,
        retry_attempts=5,
        use_cluster=True,
        environment="production",
        created_by="Vatsal216",
        created_at="2025-07-22 12:09:19"
    ),
    
    MemoryType.SQL: MemoryConfig(
        memory_type=MemoryType.SQL,
        database_url="postgresql://crewgraph:password@postgres-cluster.internal:5432/crewgraph_prod",
        compression=True,
        pool_size=20,
        max_overflow=30,
        enable_ttl_cleanup=True,
        ttl_cleanup_interval=1800,
        environment="production",
        created_by="Vatsal216",
        created_at="2025-07-22 12:09:19"
    ),
    
    MemoryType.FAISS: MemoryConfig(
        memory_type=MemoryType.FAISS,
        vector_dimension=768,
        faiss_index_type="IVF",
        max_vectors=10000000,
        use_gpu=True,
        environment="production",
        created_by="Vatsal216",
        created_at="2025-07-22 12:09:19"
    )
}