"""
Distributed Memory Backend for CrewGraph AI
Provides distributed memory storage with graceful degradation and failover support.

Author: Vatsal216  
Created: 2025-07-23
"""

import asyncio
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Callable, Set
from enum import Enum

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError
from .base import BaseMemory, MemoryOperation
from .dict_memory import DictMemory

logger = get_logger(__name__)


class MemoryBackendType(Enum):
    """Memory backend types"""
    DICT = "dict"
    REDIS = "redis"
    FAISS = "faiss"
    SQL = "sql"
    DISTRIBUTED = "distributed"


class HealthStatus(Enum):
    """Backend health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class BackendHealth:
    """Health information for a memory backend"""
    status: HealthStatus
    last_check: float
    error_count: int
    latency: float
    availability: float
    details: Dict[str, Any]


class MemoryBackendProxy(ABC):
    """Proxy interface for memory backends with health monitoring"""
    
    def __init__(self, backend_id: str, backend: BaseMemory):
        self.backend_id = backend_id
        self.backend = backend
        self.health = BackendHealth(
            status=HealthStatus.UNKNOWN,
            last_check=0.0,
            error_count=0,
            latency=0.0,
            availability=100.0,
            details={}
        )
        self._lock = threading.Lock()
    
    async def health_check(self) -> bool:
        """Perform health check on backend"""
        start_time = time.time()
        
        try:
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            await self.backend.store(test_key, {"test": True})
            result = await self.backend.retrieve(test_key)
            await self.backend.delete(test_key)
            
            # Update health status
            latency = time.time() - start_time
            
            with self._lock:
                self.health.status = HealthStatus.HEALTHY
                self.health.last_check = time.time()
                self.health.latency = latency
                self.health.error_count = max(0, self.health.error_count - 1)
                
                # Update availability (exponential moving average)
                self.health.availability = 0.9 * self.health.availability + 0.1 * 100.0
            
            return True
            
        except Exception as e:
            with self._lock:
                self.health.status = HealthStatus.UNHEALTHY
                self.health.last_check = time.time()
                self.health.error_count += 1
                self.health.details["last_error"] = str(e)
                
                # Update availability
                self.health.availability = 0.9 * self.health.availability + 0.1 * 0.0
            
            logger.warning(f"Health check failed for backend {self.backend_id}: {e}")
            return False
    
    async def execute_with_monitoring(self, operation: str, func: Callable, *args, **kwargs):
        """Execute operation with health monitoring"""
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # Update health on success
            latency = time.time() - start_time
            with self._lock:
                if self.health.status == HealthStatus.UNHEALTHY:
                    self.health.status = HealthStatus.DEGRADED
                elif self.health.status == HealthStatus.DEGRADED:
                    # Improve to healthy after successful operations
                    if self.health.error_count == 0:
                        self.health.status = HealthStatus.HEALTHY
                
                self.health.latency = 0.8 * self.health.latency + 0.2 * latency
                self.health.error_count = max(0, self.health.error_count - 1)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.health.error_count += 1
                self.health.details[f"error_{operation}"] = str(e)
                
                # Degrade health status
                if self.health.error_count > 5:
                    self.health.status = HealthStatus.UNHEALTHY
                elif self.health.error_count > 2:
                    self.health.status = HealthStatus.DEGRADED
            
            logger.error(f"Operation {operation} failed on backend {self.backend_id}: {e}")
            raise


class DistributedMemoryBackend(BaseMemory):
    """
    Distributed memory backend with automatic failover and load balancing.
    
    Features:
    - Multiple backend support with priority ordering
    - Automatic failover to healthy backends
    - Load balancing across healthy backends
    - Health monitoring and recovery
    - Graceful degradation when backends fail
    - Consistency management across backends
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.backends: Dict[str, MemoryBackendProxy] = {}
        self.backend_priorities: List[str] = []
        self.consistency_level = self.config.get("consistency_level", "eventual")
        self.replication_factor = self.config.get("replication_factor", 1)
        self.health_check_interval = self.config.get("health_check_interval", 30.0)
        
        self._lock = threading.Lock()
        self._health_check_task = None
        self._running = False
        
        # Statistics
        self.stats = {
            "operations": 0,
            "failures": 0,
            "backend_switches": 0,
            "start_time": time.time()
        }
        
        # Initialize backends
        self._initialize_backends()
        
        logger.info(f"DistributedMemoryBackend initialized with {len(self.backends)} backends")
    
    def _initialize_backends(self):
        """Initialize memory backends from configuration"""
        backend_configs = self.config.get("backends", [])
        
        if not backend_configs:
            # Default configuration - use dict memory as fallback
            self._add_fallback_backend()
            return
        
        for i, backend_config in enumerate(backend_configs):
            backend_type = backend_config.get("type", "dict")
            backend_id = backend_config.get("id", f"{backend_type}_{i}")
            priority = backend_config.get("priority", i)
            
            try:
                backend = self._create_backend(backend_type, backend_config)
                proxy = MemoryBackendProxy(backend_id, backend)
                
                self.backends[backend_id] = proxy
                self.backend_priorities.append(backend_id)
                
                logger.info(f"Added backend: {backend_id} ({backend_type}) with priority {priority}")
                
            except Exception as e:
                logger.error(f"Failed to initialize backend {backend_id}: {e}")
        
        # Sort by priority
        self.backend_priorities.sort(key=lambda x: backend_configs[self.backend_priorities.index(x)].get("priority", 999))
        
        # Add fallback if no backends available
        if not self.backends:
            self._add_fallback_backend()
    
    def _add_fallback_backend(self):
        """Add fallback dict memory backend"""
        logger.warning("No backends configured, adding fallback dict memory")
        fallback = DictMemory()
        proxy = MemoryBackendProxy("fallback_dict", fallback)
        self.backends["fallback_dict"] = proxy
        self.backend_priorities.append("fallback_dict")
    
    def _create_backend(self, backend_type: str, config: Dict[str, Any]) -> BaseMemory:
        """Create memory backend instance"""
        if backend_type == "dict":
            return DictMemory(config)
        
        elif backend_type == "redis":
            try:
                from .redis_memory import RedisMemory
                return RedisMemory(config)
            except ImportError:
                logger.warning("Redis not available, falling back to dict memory")
                return DictMemory(config)
        
        elif backend_type == "faiss":
            try:
                from .faiss_memory import FAISSMemory
                return FAISSMemory(config)
            except ImportError:
                logger.warning("FAISS not available, falling back to dict memory")
                return DictMemory(config)
        
        elif backend_type == "sql":
            try:
                from .sql_memory import SQLMemory
                return SQLMemory(config)
            except ImportError:
                logger.warning("SQL memory not available, falling back to dict memory")
                return DictMemory(config)
        
        else:
            logger.warning(f"Unknown backend type {backend_type}, falling back to dict memory")
            return DictMemory(config)
    
    def get_healthy_backends(self) -> List[str]:
        """Get list of healthy backend IDs"""
        healthy = []
        for backend_id in self.backend_priorities:
            proxy = self.backends.get(backend_id)
            if proxy and proxy.health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                healthy.append(backend_id)
        return healthy
    
    def get_primary_backend(self) -> Optional[MemoryBackendProxy]:
        """Get primary (highest priority) healthy backend"""
        healthy_backends = self.get_healthy_backends()
        if not healthy_backends:
            return None
        
        # Return highest priority healthy backend
        for backend_id in self.backend_priorities:
            if backend_id in healthy_backends:
                return self.backends[backend_id]
        
        return None
    
    def get_replication_backends(self, exclude: Optional[str] = None) -> List[MemoryBackendProxy]:
        """Get backends for replication"""
        healthy_backends = self.get_healthy_backends()
        if exclude and exclude in healthy_backends:
            healthy_backends.remove(exclude)
        
        # Return up to replication_factor backends
        result = []
        for backend_id in self.backend_priorities:
            if backend_id in healthy_backends and len(result) < self.replication_factor - 1:
                result.append(self.backends[backend_id])
        
        return result
    
    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value with replication"""
        self.stats["operations"] += 1
        
        primary = self.get_primary_backend()
        if not primary:
            self.stats["failures"] += 1
            raise CrewGraphError("No healthy backends available for storage")
        
        try:
            # Store in primary backend
            await primary.execute_with_monitoring(
                "store",
                primary.backend.store,
                key, value, ttl
            )
            
            # Replicate to other backends if configured
            if self.replication_factor > 1:
                replication_backends = self.get_replication_backends(exclude=primary.backend_id)
                
                # Fire-and-forget replication
                for backend_proxy in replication_backends:
                    asyncio.create_task(
                        self._replicate_store(backend_proxy, key, value, ttl)
                    )
            
            return True
            
        except Exception as e:
            self.stats["failures"] += 1
            logger.error(f"Failed to store key {key}: {e}")
            raise
    
    async def _replicate_store(self, backend_proxy: MemoryBackendProxy, key: str, value: Any, ttl: Optional[int]):
        """Replicate store operation to backend"""
        try:
            await backend_proxy.execute_with_monitoring(
                "store",
                backend_proxy.backend.store,
                key, value, ttl
            )
        except Exception as e:
            logger.warning(f"Replication failed for backend {backend_proxy.backend_id}: {e}")
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value with fallback"""
        self.stats["operations"] += 1
        
        # Try backends in priority order
        for backend_id in self.backend_priorities:
            proxy = self.backends.get(backend_id)
            if not proxy or proxy.health.status == HealthStatus.UNHEALTHY:
                continue
            
            try:
                result = await proxy.execute_with_monitoring(
                    "retrieve",
                    proxy.backend.retrieve,
                    key
                )
                
                if result is not None:
                    return result
                    
            except Exception as e:
                logger.warning(f"Retrieve failed on backend {backend_id}: {e}")
                continue
        
        # No backend could retrieve the value
        return None
    
    async def delete(self, key: str) -> bool:
        """Delete value from all backends"""
        self.stats["operations"] += 1
        
        success_count = 0
        total_attempts = 0
        
        # Delete from all healthy backends
        for backend_id in self.backend_priorities:
            proxy = self.backends.get(backend_id)
            if not proxy or proxy.health.status == HealthStatus.UNHEALTHY:
                continue
            
            total_attempts += 1
            
            try:
                await proxy.execute_with_monitoring(
                    "delete",
                    proxy.backend.delete,
                    key
                )
                success_count += 1
                
            except Exception as e:
                logger.warning(f"Delete failed on backend {backend_id}: {e}")
        
        # Return success if at least one backend succeeded
        return success_count > 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in any backend"""
        self.stats["operations"] += 1
        
        # Check backends in priority order
        for backend_id in self.backend_priorities:
            proxy = self.backends.get(backend_id)
            if not proxy or proxy.health.status == HealthStatus.UNHEALTHY:
                continue
            
            try:
                result = await proxy.execute_with_monitoring(
                    "exists",
                    proxy.backend.exists,
                    key
                )
                
                if result:
                    return True
                    
            except Exception as e:
                logger.warning(f"Exists check failed on backend {backend_id}: {e}")
                continue
        
        return False
    
    async def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List keys from primary backend"""
        self.stats["operations"] += 1
        
        primary = self.get_primary_backend()
        if not primary:
            return []
        
        try:
            return await primary.execute_with_monitoring(
                "list_keys",
                primary.backend.list_keys,
                pattern
            )
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            return []
    
    async def clear(self) -> bool:
        """Clear all backends"""
        self.stats["operations"] += 1
        
        success_count = 0
        
        for backend_id in self.backend_priorities:
            proxy = self.backends.get(backend_id)
            if not proxy:
                continue
            
            try:
                await proxy.execute_with_monitoring(
                    "clear",
                    proxy.backend.clear
                )
                success_count += 1
                
            except Exception as e:
                logger.warning(f"Clear failed on backend {backend_id}: {e}")
        
        return success_count > 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        backend_stats = {}
        
        for backend_id, proxy in self.backends.items():
            try:
                stats = await proxy.backend.get_stats()
                stats.update({
                    "health_status": proxy.health.status.value,
                    "error_count": proxy.health.error_count,
                    "latency": proxy.health.latency,
                    "availability": proxy.health.availability,
                    "last_check": proxy.health.last_check
                })
                backend_stats[backend_id] = stats
            except Exception as e:
                backend_stats[backend_id] = {"error": str(e)}
        
        uptime = time.time() - self.stats["start_time"]
        
        return {
            "distributed_stats": {
                "total_operations": self.stats["operations"],
                "total_failures": self.stats["failures"],
                "backend_switches": self.stats["backend_switches"],
                "uptime_seconds": uptime,
                "consistency_level": self.consistency_level,
                "replication_factor": self.replication_factor
            },
            "backend_count": len(self.backends),
            "healthy_backends": len(self.get_healthy_backends()),
            "backend_priorities": self.backend_priorities,
            "backends": backend_stats
        }
    
    async def start_health_monitoring(self):
        """Start background health monitoring"""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started health monitoring for distributed memory backend")
    
    async def stop_health_monitoring(self):
        """Stop background health monitoring"""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped health monitoring for distributed memory backend")
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all backends"""
        tasks = []
        
        for backend_id, proxy in self.backends.items():
            tasks.append(proxy.health_check())
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log health status changes
            for i, (backend_id, result) in enumerate(zip(self.backends.keys(), results)):
                if isinstance(result, Exception):
                    logger.error(f"Health check exception for {backend_id}: {result}")
                elif not result:
                    logger.warning(f"Backend {backend_id} is unhealthy")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get detailed health report"""
        healthy_backends = self.get_healthy_backends()
        
        backend_health = {}
        for backend_id, proxy in self.backends.items():
            backend_health[backend_id] = {
                "status": proxy.health.status.value,
                "last_check": proxy.health.last_check,
                "error_count": proxy.health.error_count,
                "latency_ms": round(proxy.health.latency * 1000, 2),
                "availability_pct": round(proxy.health.availability, 2),
                "details": proxy.health.details
            }
        
        return {
            "overall_status": "healthy" if healthy_backends else "unhealthy",
            "healthy_backends": len(healthy_backends),
            "total_backends": len(self.backends),
            "primary_backend": self.get_primary_backend().backend_id if self.get_primary_backend() else None,
            "backend_priorities": self.backend_priorities,
            "backends": backend_health,
            "configuration": {
                "consistency_level": self.consistency_level,
                "replication_factor": self.replication_factor,
                "health_check_interval": self.health_check_interval
            }
        }


# Factory function for creating distributed memory backend
def create_distributed_memory(config: Optional[Dict[str, Any]] = None) -> DistributedMemoryBackend:
    """Create distributed memory backend with configuration"""
    return DistributedMemoryBackend(config)


# Example configuration for distributed memory
def get_example_config() -> Dict[str, Any]:
    """Get example configuration for distributed memory backend"""
    return {
        "consistency_level": "eventual",
        "replication_factor": 2,
        "health_check_interval": 30.0,
        "backends": [
            {
                "id": "primary_redis",
                "type": "redis",
                "priority": 1,
                "url": "redis://localhost:6379/0"
            },
            {
                "id": "secondary_redis",
                "type": "redis", 
                "priority": 2,
                "url": "redis://localhost:6379/1"
            },
            {
                "id": "fallback_dict",
                "type": "dict",
                "priority": 3
            }
        ]
    }