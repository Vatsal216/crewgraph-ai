"""
Dictionary-based memory backend for CrewGraph AI
Simple in-memory storage with persistence options

Author: Vatsal216
Created: 2025-07-22 12:01:02 UTC
"""

import json
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.exceptions import MemoryError
from ..utils.logging import get_logger
from .base import BaseMemory, MemoryOperation, MemoryStats

logger = get_logger(__name__)


class DictMemory(BaseMemory):
    """
    Dictionary-based memory backend with optional persistence.

    Features:
    - Fast in-memory operations
    - Optional file persistence
    - TTL support with automatic cleanup
    - Thread-safe operations
    - JSON and pickle serialization

    Created by: Vatsal216
    Date: 2025-07-22 12:01:02 UTC
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        persistent: bool = False,
        persistence_file: Optional[str] = None,
        auto_save: bool = False,
        auto_save_interval: int = 60,
        max_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize dictionary memory backend.

        Args:
            config: Configuration options
            persistent: Enable file persistence
            persistence_file: File path for persistence
            auto_save: Enable automatic saving
            auto_save_interval: Auto-save interval in seconds
            max_size: Maximum number of items (optional)
            **kwargs: Additional keyword arguments (for compatibility)
        """
        super().__init__(config)

        self.persistent = persistent
        self.persistence_file = persistence_file or "crewgraph_memory.json"
        self.auto_save = auto_save
        self.auto_save_interval = auto_save_interval
        self.max_size = max_size

        # Core storage
        self._storage: Dict[str, Any] = {}
        self._ttl_storage: Dict[str, float] = {}  # key -> expiration time
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        logger.info(
            f"DictMemory initialized - persistent={persistent}, auto_save={auto_save}, max_size={max_size}"
        )

    def connect(self) -> None:
        """Connect to memory backend (load from file if persistent)"""
        if self._connected:
            return

        if self.persistent:
            self._load_from_file()

        # Start TTL cleanup thread
        self._start_cleanup_thread()

        # Start auto-save thread
        if self.auto_save:
            self._start_auto_save_thread()

        self._connected = True
        logger.info("DictMemory connected successfully")

    def disconnect(self) -> None:
        """Disconnect from memory backend (save to file if persistent)"""
        if not self._connected:
            return

        # Stop background threads
        self._shutdown.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1)

        # Save to file if persistent
        if self.persistent:
            self._save_to_file()

        self._connected = False
        logger.info("DictMemory disconnected")

    def save(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Save value with optional TTL"""
        return self._execute_with_metrics(MemoryOperation.SAVE, self._save_impl, key, value, ttl)

    def _save_impl(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Internal save implementation"""
        with self._lock:
            self._storage[key] = value

            if ttl is not None:
                self._ttl_storage[key] = time.time() + ttl
            elif key in self._ttl_storage:
                # Remove TTL if not specified
                del self._ttl_storage[key]

        return True

    def load(self, key: str, default: Any = None) -> Any:
        """Load value from memory"""
        return self._execute_with_metrics(MemoryOperation.LOAD, self._load_impl, key, default)

    def _load_impl(self, key: str, default: Any = None) -> Any:
        """Internal load implementation"""
        with self._lock:
            # Check if key has expired
            if key in self._ttl_storage:
                if time.time() > self._ttl_storage[key]:
                    # Key expired, clean up
                    self._storage.pop(key, None)
                    self._ttl_storage.pop(key, None)
                    return default

            return self._storage.get(key, default)

    def delete(self, key: str) -> bool:
        """Delete key from memory"""
        return self._execute_with_metrics(MemoryOperation.DELETE, self._delete_impl, key)

    def _delete_impl(self, key: str) -> bool:
        """Internal delete implementation"""
        with self._lock:
            existed = key in self._storage
            self._storage.pop(key, None)
            self._ttl_storage.pop(key, None)
            return existed

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return self._execute_with_metrics(MemoryOperation.EXISTS, self._exists_impl, key)

    def _exists_impl(self, key: str) -> bool:
        """Internal exists implementation"""
        with self._lock:
            if key not in self._storage:
                return False

            # Check TTL
            if key in self._ttl_storage:
                if time.time() > self._ttl_storage[key]:
                    # Expired, clean up
                    self._storage.pop(key, None)
                    self._ttl_storage.pop(key, None)
                    return False

            return True

    def clear(self) -> bool:
        """Clear all data"""
        return self._execute_with_metrics(MemoryOperation.CLEAR, self._clear_impl)

    def _clear_impl(self) -> bool:
        """Internal clear implementation"""
        with self._lock:
            self._storage.clear()
            self._ttl_storage.clear()
            return True

    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all keys, optionally matching pattern"""
        return self._execute_with_metrics(MemoryOperation.LIST_KEYS, self._list_keys_impl, pattern)

    def _list_keys_impl(self, pattern: Optional[str] = None) -> List[str]:
        """Internal list keys implementation"""
        import fnmatch

        with self._lock:
            # Clean expired keys first
            current_time = time.time()
            expired_keys = [
                key for key, exp_time in self._ttl_storage.items() if current_time > exp_time
            ]
            for key in expired_keys:
                self._storage.pop(key, None)
                self._ttl_storage.pop(key, None)

            keys = list(self._storage.keys())

            if pattern:
                keys = [key for key in keys if fnmatch.fnmatch(key, pattern)]

            return keys

    def get_size(self) -> int:
        """Get number of valid (non-expired) keys"""
        return self._execute_with_metrics(MemoryOperation.GET_SIZE, self._get_size_impl)

    def _get_size_impl(self) -> int:
        """Internal get size implementation"""
        return len(self.list_keys())  # This will clean expired keys too

    def search(self, query: str, limit: Optional[int] = None) -> List[Tuple[str, Any]]:
        """Search for keys and values containing query string"""
        results = []
        keys = self.list_keys()

        for key in keys:
            if query.lower() in key.lower():
                value = self.load(key)
                if value is not None:
                    results.append((key, value))
                    if limit and len(results) >= limit:
                        break

        return results

    def _cleanup_expired_keys(self):
        """Background thread to cleanup expired keys"""
        while not self._shutdown.wait(30):  # Check every 30 seconds
            try:
                with self._lock:
                    current_time = time.time()
                    expired_keys = [
                        key
                        for key, exp_time in self._ttl_storage.items()
                        if current_time > exp_time
                    ]

                    for key in expired_keys:
                        self._storage.pop(key, None)
                        self._ttl_storage.pop(key, None)

                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired keys")

            except Exception as e:
                logger.error(f"Error during TTL cleanup: {e}")

    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        if not self._cleanup_thread or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(target=self._cleanup_expired_keys, daemon=True)
            self._cleanup_thread.start()

    def _start_auto_save_thread(self):
        """Start auto-save thread"""

        def auto_save_loop():
            while not self._shutdown.wait(self.auto_save_interval):
                try:
                    if self.persistent:
                        self._save_to_file()
                        logger.debug("Auto-saved memory to file")
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")

        thread = threading.Thread(target=auto_save_loop, daemon=True)
        thread.start()

    def _save_to_file(self):
        """Save memory to persistent file"""
        try:
            with self._lock:
                # Clean expired keys before saving
                current_time = time.time()
                valid_storage = {
                    key: value
                    for key, value in self._storage.items()
                    if key not in self._ttl_storage or self._ttl_storage[key] > current_time
                }
                valid_ttl = {
                    key: exp_time
                    for key, exp_time in self._ttl_storage.items()
                    if exp_time > current_time
                }

                data = {
                    "storage": valid_storage,
                    "ttl_storage": valid_ttl,
                    "saved_at": time.time(),
                    "version": "1.0.0",
                }

            # Try JSON first (more readable)
            path = Path(self.persistence_file)
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
            except (TypeError, ValueError):
                # Fall back to pickle for non-serializable objects
                pickle_file = path.with_suffix(".pickle")
                with open(pickle_file, "wb") as f:
                    pickle.dump(data, f)

        except Exception as e:
            logger.error(f"Failed to save to file: {e}")
            raise MemoryError(f"Persistence save failed: {e}")

    def _load_from_file(self):
        """Load memory from persistent file"""
        path = Path(self.persistence_file)
        pickle_path = path.with_suffix(".pickle")

        try:
            # Try JSON first
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif pickle_path.exists():
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
            else:
                logger.info("No persistence file found, starting with empty memory")
                return

            with self._lock:
                self._storage = data.get("storage", {})
                self._ttl_storage = data.get("ttl_storage", {})

                # Clean expired keys
                current_time = time.time()
                expired_keys = [
                    key for key, exp_time in self._ttl_storage.items() if current_time > exp_time
                ]
                for key in expired_keys:
                    self._storage.pop(key, None)
                    self._ttl_storage.pop(key, None)

            logger.info(f"Loaded {len(self._storage)} keys from persistence file")

        except Exception as e:
            logger.error(f"Failed to load from file: {e}")
            # Don't raise error, just start with empty memory

    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information"""
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for exp_time in self._ttl_storage.values() if current_time > exp_time
            )

            return {
                "total_keys": len(self._storage),
                "expired_keys": expired_count,
                "valid_keys": len(self._storage) - expired_count,
                "has_ttl": len(self._ttl_storage),
                "persistent": self.persistent,
                "persistence_file": self.persistence_file,
                "auto_save": self.auto_save,
                "connected": self._connected,
            }
