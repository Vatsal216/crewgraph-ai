"""
Base memory interface and configuration
"""

import json
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError

logger = get_logger(__name__)


class MemoryType(Enum):
    """Memory backend types"""
    DICT = "dict"
    REDIS = "redis"
    FAISS = "faiss"
    SQL = "sql"
    MONGODB = "mongodb"


class SerializationFormat(Enum):
    """Serialization formats"""
    JSON = "json"
    PICKLE = "pickle"
    RAW = "raw"


@dataclass
class MemoryConfig:
    """Memory configuration"""
    memory_type: MemoryType = MemoryType.DICT
    serialization: SerializationFormat = SerializationFormat.JSON
    ttl: Optional[int] = None  # Time to live in seconds
    max_size: Optional[int] = None  # Maximum number of entries
    compression: bool = False
    encryption_key: Optional[str] = None
    
    # Backend-specific configs
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    faiss_dimension: int = 384
    faiss_index_type: str = "IndexFlatL2"
    
    sql_url: str = "sqlite:///crewgraph_memory.db"
    sql_table_name: str = "memory_store"


class BaseMemory(ABC):
    """
    Abstract base class for memory backends.
    
    Provides a unified interface for different memory storage systems
    with support for serialization, TTL, and optional encryption.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory backend.
        
        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        self._serializer = self._get_serializer()
        self._is_connected = False
        
    @abstractmethod
    def connect(self) -> None:
        """Connect to memory backend."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from memory backend."""
        pass
    
    @abstractmethod
    def _get(self, key: str) -> Optional[bytes]:
        """Get raw value from backend."""
        pass
    
    @abstractmethod
    def _set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        """Set raw value in backend."""
        pass
    
    @abstractmethod
    def _delete(self, key: str) -> bool:
        """Delete key from backend."""
        pass
    
    @abstractmethod
    def _exists(self, key: str) -> bool:
        """Check if key exists in backend."""
        pass
    
    @abstractmethod
    def _keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        pass
    
    @abstractmethod
    def _clear(self) -> None:
        """Clear all data from backend."""
        pass
    
    def load(self, key: str) -> Optional[Any]:
        """
        Load value from memory.
        
        Args:
            key: Storage key
            
        Returns:
            Deserialized value or None if not found
        """
        try:
            if not self._is_connected:
                self.connect()
            
            raw_value = self._get(key)
            if raw_value is None:
                return None
            
            return self._deserialize(raw_value)
            
        except Exception as e:
            logger.error(f"Failed to load key '{key}': {e}")
            return None
    
    def save(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Save value to memory.
        
        Args:
            key: Storage key
            value: Value to store
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._is_connected:
                self.connect()
            
            serialized_value = self._serialize(value)
            effective_ttl = ttl or self.config.ttl
            
            self._set(key, serialized_value, effective_ttl)
            logger.debug(f"Saved key '{key}' to memory")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save key '{key}': {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from memory.
        
        Args:
            key: Storage key
            
        Returns:
            True if deleted, False if not found
        """
        try:
            if not self._is_connected:
                self.connect()
            
            result = self._delete(key)
            if result:
                logger.debug(f"Deleted key '{key}' from memory")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete key '{key}': {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in memory.
        
        Args:
            key: Storage key
            
        Returns:
            True if exists, False otherwise
        """
        try:
            if not self._is_connected:
                self.connect()
            
            return self._exists(key)
            
        except Exception as e:
            logger.error(f"Failed to check existence of key '{key}': {e}")
            return False
    
    def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching pattern.
        
        Args:
            pattern: Key pattern (supports wildcards)
            
        Returns:
            List of matching keys
        """
        try:
            if not self._is_connected:
                self.connect()
            
            return self._keys(pattern)
            
        except Exception as e:
            logger.error(f"Failed to get keys with pattern '{pattern}': {e}")
            return []
    
    def clear(self) -> bool:
        """
        Clear all data from memory.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._is_connected:
                self.connect()
            
            self._clear()
            logger.info("Memory cleared")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory backend statistics."""
        return {
            'backend_type': self.config.memory_type.value,
            'serialization': self.config.serialization.value,
            'connected': self._is_connected,
            'total_keys': len(self.keys())
        }
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value based on configuration."""
        try:
            if self.config.serialization == SerializationFormat.JSON:
                serialized = json.dumps(value, default=str).encode('utf-8')
            elif self.config.serialization == SerializationFormat.PICKLE:
                serialized = pickle.dumps(value)
            else:  # RAW
                if isinstance(value, (str, bytes)):
                    serialized = value.encode('utf-8') if isinstance(value, str) else value
                else:
                    raise ValueError(f"RAW serialization requires str or bytes, got {type(value)}")
            
            # Apply compression if enabled
            if self.config.compression:
                import zlib
                serialized = zlib.compress(serialized)
            
            # Apply encryption if configured
            if self.config.encryption_key:
                serialized = self._encrypt(serialized)
            
            return serialized
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise CrewGraphError(f"Failed to serialize value: {e}")
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value based on configuration."""
        try:
            # Apply decryption if configured
            if self.config.encryption_key:
                data = self._decrypt(data)
            
            # Apply decompression if enabled
            if self.config.compression:
                import zlib
                data = zlib.decompress(data)
            
            if self.config.serialization == SerializationFormat.JSON:
                return json.loads(data.decode('utf-8'))
            elif self.config.serialization == SerializationFormat.PICKLE:
                return pickle.loads(data)
            else:  # RAW
                return data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise CrewGraphError(f"Failed to deserialize value: {e}")
    
    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data using configured key."""
        try:
            from cryptography.fernet import Fernet
            import base64
            import hashlib
            
            # Create key from password
            key = base64.urlsafe_b64encode(
                hashlib.sha256(self.config.encryption_key.encode()).digest()
            )
            
            fernet = Fernet(key)
            return fernet.encrypt(data)
            
        except ImportError:
            logger.warning("cryptography package not available, skipping encryption")
            return data
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt data using configured key."""
        try:
            from cryptography.fernet import Fernet
            import base64
            import hashlib
            
            # Create key from password
            key = base64.urlsafe_b64encode(
                hashlib.sha256(self.config.encryption_key.encode()).digest()
            )
            
            fernet = Fernet(key)
            return fernet.decrypt(data)
            
        except ImportError:
            logger.warning("cryptography package not available, skipping decryption")
            return data
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return data
    
    def _get_serializer(self):
        """Get appropriate serializer based on configuration."""
        return {
            SerializationFormat.JSON: (json.dumps, json.loads),
            SerializationFormat.PICKLE: (pickle.dumps, pickle.loads),
            SerializationFormat.RAW: (str, str)
        }.get(self.config.serialization, (json.dumps, json.loads))
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()