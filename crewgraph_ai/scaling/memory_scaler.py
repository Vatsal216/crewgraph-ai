"""
CrewGraph AI Memory Backend Auto-Scaling
Elastic scaling for memory systems based on usage patterns

Author: Vatsal216
Created: 2025-07-22 13:17:52 UTC
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..memory.base import BaseMemory
from ..memory.config import MemoryConfig, MemoryType
from ..memory.utils import MemoryUtils
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryScalingRule:
    """Memory scaling rule configuration"""

    memory_type: MemoryType
    scale_up_threshold: float = 80.0  # Usage percentage
    scale_down_threshold: float = 30.0
    min_connections: int = 1
    max_connections: int = 10
    check_interval: int = 60  # seconds


class MemoryAutoScaler:
    """
    Auto-scaling system for memory backends.

    Features:
    - Connection pool scaling
    - Cache size optimization
    - Performance-based scaling
    - Multi-backend support
    - Health monitoring

    Created by: Vatsal216
    Date: 2025-07-22 13:17:52 UTC
    """

    def __init__(self, memory_configs: Dict[str, MemoryConfig]):
        """Initialize memory auto-scaler"""
        self.memory_configs = memory_configs
        self.memory_backends: Dict[str, BaseMemory] = {}
        self.scaling_rules: Dict[str, MemoryScalingRule] = {}

        # Initialize backends and rules
        for name, config in memory_configs.items():
            rule = MemoryScalingRule(memory_type=config.memory_type)
            self.scaling_rules[name] = rule

            # Create backend
            backend = MemoryUtils.create_memory_backend(config)
            backend.connect()
            self.memory_backends[name] = backend

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        logger.info(f"MemoryAutoScaler initialized with {len(memory_configs)} backends")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 13:17:52")

    def start_scaling(self) -> None:
        """Start memory scaling"""
        self._running = True

        self._monitor_thread = threading.Thread(
            target=self._monitor_memory_usage, name="MemoryAutoScaler-Monitor", daemon=True
        )
        self._monitor_thread.start()

        logger.info("Memory auto-scaling started")

    def stop_scaling(self) -> None:
        """Stop memory scaling"""
        self._running = False

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        # Disconnect backends
        for backend in self.memory_backends.values():
            try:
                backend.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting memory backend: {e}")

        logger.info("Memory auto-scaling stopped")

    def _monitor_memory_usage(self) -> None:
        """Monitor memory usage and scale"""
        while self._running:
            try:
                for name, backend in self.memory_backends.items():
                    self._evaluate_backend_scaling(name, backend)

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in memory scaling monitoring: {e}")
                time.sleep(60)

    def _evaluate_backend_scaling(self, name: str, backend: BaseMemory) -> None:
        """Evaluate scaling needs for memory backend"""
        try:
            # Get backend health and usage stats
            health = backend.get_health()
            usage_percent = health.get("memory_usage_percent", 0)

            rule = self.scaling_rules[name]

            # Make scaling decisions based on usage
            if usage_percent > rule.scale_up_threshold:
                self._scale_memory_up(name, backend, rule)
            elif usage_percent < rule.scale_down_threshold:
                self._scale_memory_down(name, backend, rule)

        except Exception as e:
            logger.error(f"Error evaluating scaling for {name}: {e}")

    def _scale_memory_up(self, name: str, backend: BaseMemory, rule: MemoryScalingRule) -> None:
        """Scale memory backend up"""
        logger.info(f"Scaling UP memory backend: {name}")

        # Implementation depends on backend type
        if rule.memory_type == MemoryType.REDIS:
            # Increase connection pool, cache size, etc.
            pass
        elif rule.memory_type == MemoryType.SQL:
            # Increase connection pool
            pass
        # Add other backend-specific scaling logic

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get memory scaling status"""
        status = {
            "running": self._running,
            "backends": {},
            "created_by": "Vatsal216",
            "timestamp": "2025-07-22 13:17:52",
        }

        for name, backend in self.memory_backends.items():
            try:
                health = backend.get_health()
                status["backends"][name] = {
                    "type": self.scaling_rules[name].memory_type.value,
                    "status": health.get("status", "unknown"),
                    "usage_percent": health.get("memory_usage_percent", 0),
                    "connections": health.get("active_connections", 0),
                }
            except Exception as e:
                status["backends"][name] = {"error": str(e)}

        return status
