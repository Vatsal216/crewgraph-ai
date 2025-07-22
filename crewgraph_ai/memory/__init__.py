"""
CrewGraph AI Memory Backend System
Comprehensive memory management with multiple backend support

Author: Vatsal216
Created: 2025-07-22 12:01:02 UTC
"""

from .base import BaseMemory, MemoryOperation
from .dict_memory import DictMemory
from .redis_memory import RedisMemory
from .faiss_memory import FAISSMemory
from .sql_memory import SQLMemory
from .config import MemoryConfig, MemoryType
from .utils import MemoryUtils, MemorySerializer

__all__ = [
    # Base classes
    "BaseMemory",
    "MemoryOperation",
    
    # Memory backends
    "DictMemory",
    "RedisMemory", 
    "FAISSMemory",
    "SQLMemory",
    
    # Configuration
    "MemoryConfig",
    "MemoryType",
    
    # Utilities
    "MemoryUtils",
    "MemorySerializer",
]

# Version info
__version__ = "1.0.0"
__author__ = "Vatsal216"
__created__ = "2025-07-22 12:01:02"

print(f"🧠 CrewGraph AI Memory System v{__version__} loaded")
print(f"👤 Created by: {__author__}")
print(f"⏰ Timestamp: {__created__}")
print(f"🔧 Available backends: Dict, Redis, FAISS, SQL")