"""
CrewGraph AI Memory Backend System
Comprehensive memory management with multiple backend support

Author: Vatsal216
Created: 2025-07-22 12:01:02 UTC
"""

from .base import BaseMemory, MemoryOperation
from .config import MemoryConfig, MemoryType
from .dict_memory import DictMemory
from .utils import MemorySerializer, MemoryUtils, create_memory

# Optional imports (gracefully handle missing dependencies)
try:
    from .redis_memory import RedisMemory

    REDIS_AVAILABLE = True
except ImportError:
    RedisMemory = None
    REDIS_AVAILABLE = False

try:
    from .faiss_memory import FAISSMemory

    FAISS_AVAILABLE = True
except ImportError:
    FAISSMemory = None
    FAISS_AVAILABLE = False

try:
    from .sql_memory import SQLMemory

    SQL_AVAILABLE = True
except ImportError:
    SQLMemory = None
    SQL_AVAILABLE = False


def create_memory(memory_type: str = "dict", **kwargs) -> BaseMemory:
    """
    Factory function to create memory backends

    Args:
        memory_type: Type of memory backend ('dict', 'redis', 'faiss', 'sql')
        **kwargs: Configuration parameters for the memory backend

    Returns:
        BaseMemory: Configured memory backend instance

    Raises:
        ValueError: If memory_type is invalid or backend not available
    """
    memory_type = memory_type.lower()

    if memory_type == "dict":
        return DictMemory(**kwargs)
    elif memory_type == "redis":
        if not REDIS_AVAILABLE:
            raise ValueError("Redis memory backend not available. Install with: pip install redis")
        return RedisMemory(**kwargs)
    elif memory_type == "faiss":
        if not FAISS_AVAILABLE:
            raise ValueError(
                "FAISS memory backend not available. Install with: pip install faiss-cpu"
            )
        return FAISSMemory(**kwargs)
    elif memory_type == "sql":
        if not SQL_AVAILABLE:
            raise ValueError(
                "SQL memory backend not available. Install with: pip install sqlalchemy"
            )
        return SQLMemory(**kwargs)
    else:
        available_types = ["dict"]
        if REDIS_AVAILABLE:
            available_types.append("redis")
        if FAISS_AVAILABLE:
            available_types.append("faiss")
        if SQL_AVAILABLE:
            available_types.append("sql")
        raise ValueError(f"Invalid memory_type '{memory_type}'. Available types: {available_types}")


__all__ = [
    # Base classes
    "BaseMemory",
    "MemoryOperation",
    # Memory backends (always available)
    "DictMemory",
    # Configuration
    "MemoryConfig",
    "MemoryType",
    # Utilities
    "MemoryUtils",
    "MemorySerializer",
    # Factory function
    "create_memory",
]

# Add optional backends to __all__ if available
if REDIS_AVAILABLE:
    __all__.append("RedisMemory")
if FAISS_AVAILABLE:
    __all__.append("FAISSMemory")
if SQL_AVAILABLE:
    __all__.append("SQLMemory")

# Version info
__version__ = "1.0.0"
__author__ = "Vatsal216"
__created__ = "2025-07-22 12:01:02"

print(f"🧠 CrewGraph AI Memory System v{__version__} loaded")
print(f"👤 Created by: {__author__}")
print(f"⏰ Timestamp: {__created__}")
print(f"🔧 Available backends: Dict, Redis, FAISS, SQL")
