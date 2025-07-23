"""
CrewGraph AI Memory Backend System
Comprehensive memory management with multiple backend support

Author: Vatsal216
Created: 2025-07-22 12:01:02 UTC
"""

from .base import BaseMemory, MemoryOperation
from .dict_memory import DictMemory
from .config import MemoryConfig, MemoryType
from .utils import MemoryUtils, MemorySerializer, create_memory

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