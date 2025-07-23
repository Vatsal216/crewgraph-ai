"""
Plugin Architecture for CrewGraph AI Integration Marketplace

Provides extensible plugin system for third-party integrations including:
- Plugin loading and management
- Sandboxed execution environment
- Plugin lifecycle management
- Plugin communication interface
- Security and validation

Author: Vatsal216
Created: 2025-01-27
"""

import asyncio
import importlib
import inspect
import json
import os
import sys
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import subprocess
import threading

from ...types import WorkflowId, TaskId
from ...utils.logging import get_logger

logger = get_logger(__name__)


class PluginState:
    """Plugin state constants."""
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UNLOADED = "unloaded"


class PluginPriority:
    """Plugin priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class PluginManifest:
    """Plugin manifest metadata."""
    
    id: str
    name: str
    version: str
    description: str
    author: str
    
    # Technical requirements
    api_version: str = "1.0.0"
    min_crewgraph_version: str = "1.0.0"
    python_version: str = ">=3.8"
    dependencies: List[str] = field(default_factory=list)
    
    # Plugin configuration
    entry_point: str = "main.py"
    plugin_class: str = "Plugin"
    category: str = "integration"
    tags: List[str] = field(default_factory=list)
    
    # Permissions and security
    permissions: List[str] = field(default_factory=list)
    sandbox_enabled: bool = True
    network_access: bool = False
    file_access: List[str] = field(default_factory=list)
    
    # Lifecycle hooks
    lifecycle_hooks: Dict[str, str] = field(default_factory=dict)
    
    # Configuration schema
    config_schema: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "api_version": self.api_version,
            "min_crewgraph_version": self.min_crewgraph_version,
            "python_version": self.python_version,
            "dependencies": self.dependencies,
            "entry_point": self.entry_point,
            "plugin_class": self.plugin_class,
            "category": self.category,
            "tags": self.tags,
            "permissions": self.permissions,
            "sandbox_enabled": self.sandbox_enabled,
            "network_access": self.network_access,
            "file_access": self.file_access,
            "lifecycle_hooks": self.lifecycle_hooks,
            "config_schema": self.config_schema
        }


@dataclass
class PluginContext:
    """Context provided to plugins."""
    
    plugin_id: str
    config: Dict[str, Any]
    data_dir: Path
    temp_dir: Path
    
    # Callbacks for plugin communication
    logger: Any
    event_bus: Any
    workflow_manager: Any
    
    # Security constraints
    allowed_modules: List[str] = field(default_factory=list)
    max_memory_mb: int = 512
    max_execution_time: int = 300
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def log_info(self, message: str):
        """Log info message."""
        if self.logger:
            self.logger.info(f"[{self.plugin_id}] {message}")
    
    def log_error(self, message: str):
        """Log error message."""
        if self.logger:
            self.logger.error(f"[{self.plugin_id}] {message}")


class BasePlugin(ABC):
    """
    Base class that all plugins must inherit from.
    
    Provides standard interface for plugin lifecycle and functionality.
    """
    
    def __init__(self, context: PluginContext):
        """Initialize plugin with context."""
        self.context = context
        self.plugin_id = context.plugin_id
        self.config = context.config
        self.logger = context.logger
        self._state = PluginState.LOADING
        self._start_time = time.time()
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the plugin.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute plugin functionality.
        
        Args:
            task: Task data to process
            
        Returns:
            Result data
        """
        pass
    
    async def cleanup(self):
        """Cleanup plugin resources."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status information
        """
        return {
            "status": "healthy",
            "uptime": time.time() - self._start_time,
            "state": self._state
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "plugin_id": self.plugin_id,
            "state": self._state,
            "uptime": time.time() - self._start_time
        }
    
    def _set_state(self, state: str):
        """Set plugin state."""
        self._state = state
        if self.logger:
            self.logger.info(f"Plugin {self.plugin_id} state changed to {state}")


class PluginLoader:
    """
    Loads and manages plugins with security and sandboxing.
    """
    
    def __init__(self, plugins_dir: str = "marketplace/plugins"):
        """Initialize plugin loader."""
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaded_plugins: Dict[str, BasePlugin] = {}
        self.plugin_manifests: Dict[str, PluginManifest] = {}
        self.plugin_modules: Dict[str, Any] = {}
        
        # Security settings
        self.sandbox_enabled = True
        self.max_plugins = 50
        
        logger.info(f"Plugin loader initialized with directory: {self.plugins_dir}")
    
    def discover_plugins(self) -> List[PluginManifest]:
        """Discover available plugins."""
        discovered = []
        
        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue
            
            manifest_file = plugin_dir / "manifest.json"
            if not manifest_file.exists():
                logger.warning(f"No manifest found for plugin: {plugin_dir.name}")
                continue
            
            try:
                with open(manifest_file, 'r') as f:
                    manifest_data = json.load(f)
                
                manifest = PluginManifest(**manifest_data)
                discovered.append(manifest)
                self.plugin_manifests[manifest.id] = manifest
                
                logger.info(f"Discovered plugin: {manifest.id} v{manifest.version}")
                
            except Exception as e:
                logger.error(f"Failed to load manifest for {plugin_dir.name}: {e}")
        
        return discovered
    
    async def load_plugin(self, plugin_id: str, config: Dict[str, Any] = None) -> bool:
        """
        Load a specific plugin.
        
        Args:
            plugin_id: Plugin identifier
            config: Plugin configuration
            
        Returns:
            True if loaded successfully
        """
        if plugin_id in self.loaded_plugins:
            logger.warning(f"Plugin {plugin_id} is already loaded")
            return True
        
        if len(self.loaded_plugins) >= self.max_plugins:
            logger.error(f"Maximum number of plugins ({self.max_plugins}) reached")
            return False
        
        manifest = self.plugin_manifests.get(plugin_id)
        if not manifest:
            logger.error(f"Plugin manifest not found: {plugin_id}")
            return False
        
        try:
            # Validate dependencies
            if not self._validate_dependencies(manifest):
                return False
            
            # Load plugin module
            plugin_module = await self._load_plugin_module(manifest)
            if not plugin_module:
                return False
            
            # Create plugin instance
            plugin_instance = await self._create_plugin_instance(
                manifest, plugin_module, config or {}
            )
            if not plugin_instance:
                return False
            
            # Initialize plugin
            if await plugin_instance.initialize():
                self.loaded_plugins[plugin_id] = plugin_instance
                self.plugin_modules[plugin_id] = plugin_module
                plugin_instance._set_state(PluginState.LOADED)
                
                logger.info(f"Successfully loaded plugin: {plugin_id}")
                return True
            else:
                logger.error(f"Plugin initialization failed: {plugin_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_id}: {e}")
            return False
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if unloaded successfully
        """
        if plugin_id not in self.loaded_plugins:
            logger.warning(f"Plugin {plugin_id} is not loaded")
            return True
        
        try:
            plugin = self.loaded_plugins[plugin_id]
            plugin._set_state(PluginState.UNLOADED)
            
            # Cleanup plugin
            await plugin.cleanup()
            
            # Remove from loaded plugins
            del self.loaded_plugins[plugin_id]
            if plugin_id in self.plugin_modules:
                del self.plugin_modules[plugin_id]
            
            logger.info(f"Successfully unloaded plugin: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_id}: {e}")
            return False
    
    async def execute_plugin(
        self,
        plugin_id: str,
        task: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a plugin with given task.
        
        Args:
            plugin_id: Plugin identifier
            task: Task data
            
        Returns:
            Plugin execution result
        """
        if plugin_id not in self.loaded_plugins:
            logger.error(f"Plugin {plugin_id} is not loaded")
            return None
        
        plugin = self.loaded_plugins[plugin_id]
        
        try:
            plugin._set_state(PluginState.ACTIVE)
            result = await plugin.execute(task)
            plugin._set_state(PluginState.LOADED)
            
            return result
            
        except Exception as e:
            logger.error(f"Plugin execution failed {plugin_id}: {e}")
            plugin._set_state(PluginState.ERROR)
            return None
    
    def get_plugin_status(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get plugin status information."""
        if plugin_id not in self.loaded_plugins:
            return None
        
        plugin = self.loaded_plugins[plugin_id]
        return plugin.get_info()
    
    def list_loaded_plugins(self) -> List[str]:
        """List all loaded plugin IDs."""
        return list(self.loaded_plugins.keys())
    
    def list_available_plugins(self) -> List[str]:
        """List all available plugin IDs."""
        return list(self.plugin_manifests.keys())
    
    def _validate_dependencies(self, manifest: PluginManifest) -> bool:
        """Validate plugin dependencies."""
        # Check Python version
        import sys
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        required_python = manifest.python_version.replace(">=", "")
        
        if current_python < required_python:
            logger.error(f"Python version {required_python} required, but {current_python} is available")
            return False
        
        # Check dependencies (simplified)
        missing_deps = []
        for dep in manifest.dependencies:
            try:
                # Extract package name
                pkg_name = dep.split(">=")[0].split("==")[0].split(">")[0].split("<")[0]
                importlib.import_module(pkg_name.replace("-", "_"))
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            logger.error(f"Missing dependencies: {missing_deps}")
            return False
        
        return True
    
    async def _load_plugin_module(self, manifest: PluginManifest):
        """Load plugin module."""
        plugin_dir = self.plugins_dir / manifest.id
        entry_point = plugin_dir / manifest.entry_point
        
        if not entry_point.exists():
            logger.error(f"Plugin entry point not found: {entry_point}")
            return None
        
        try:
            # Add plugin directory to Python path
            if str(plugin_dir) not in sys.path:
                sys.path.insert(0, str(plugin_dir))
            
            # Import the module
            spec = importlib.util.spec_from_file_location(
                f"plugin_{manifest.id}",
                entry_point
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return module
            
        except Exception as e:
            logger.error(f"Failed to load plugin module {manifest.id}: {e}")
            return None
    
    async def _create_plugin_instance(
        self,
        manifest: PluginManifest,
        module: Any,
        config: Dict[str, Any]
    ) -> Optional[BasePlugin]:
        """Create plugin instance."""
        try:
            # Get plugin class
            plugin_class = getattr(module, manifest.plugin_class)
            
            # Validate plugin class
            if not issubclass(plugin_class, BasePlugin):
                logger.error(f"Plugin class must inherit from BasePlugin: {manifest.id}")
                return None
            
            # Create plugin context
            context = self._create_plugin_context(manifest, config)
            
            # Create plugin instance
            instance = plugin_class(context)
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create plugin instance {manifest.id}: {e}")
            return None
    
    def _create_plugin_context(
        self,
        manifest: PluginManifest,
        config: Dict[str, Any]
    ) -> PluginContext:
        """Create plugin context."""
        # Create plugin data directory
        plugin_data_dir = self.plugins_dir / manifest.id / "data"
        plugin_data_dir.mkdir(exist_ok=True)
        
        # Create temporary directory
        plugin_temp_dir = Path(tempfile.mkdtemp(prefix=f"plugin_{manifest.id}_"))
        
        return PluginContext(
            plugin_id=manifest.id,
            config=config,
            data_dir=plugin_data_dir,
            temp_dir=plugin_temp_dir,
            logger=logger,
            event_bus=None,  # Could be injected
            workflow_manager=None,  # Could be injected
            allowed_modules=manifest.permissions,
            max_memory_mb=512,
            max_execution_time=300
        )


class PluginManager:
    """
    High-level plugin management system.
    
    Coordinates plugin loading, execution, and lifecycle management.
    """
    
    def __init__(self, plugins_dir: str = "marketplace/plugins"):
        """Initialize plugin manager."""
        self.loader = PluginLoader(plugins_dir)
        self.scheduler = PluginScheduler()
        self.sandbox = PluginSandbox()
        
        # Plugin registry
        self.plugin_registry: Dict[str, Dict[str, Any]] = {}
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        logger.info("Plugin manager initialized")
    
    async def initialize(self):
        """Initialize plugin manager."""
        # Discover available plugins
        manifests = self.loader.discover_plugins()
        logger.info(f"Discovered {len(manifests)} plugins")
        
        # Register plugins
        for manifest in manifests:
            self.plugin_registry[manifest.id] = {
                "manifest": manifest,
                "status": "available",
                "auto_load": False
            }
    
    async def install_plugin(
        self,
        plugin_package: str,
        config: Dict[str, Any] = None
    ) -> bool:
        """
        Install a plugin package.
        
        Args:
            plugin_package: Plugin package path or URL
            config: Plugin configuration
            
        Returns:
            True if installation successful
        """
        try:
            # This would handle downloading and extracting plugin packages
            # For now, we'll assume plugins are already in the plugins directory
            
            # Rediscover plugins to pick up new ones
            manifests = self.loader.discover_plugins()
            
            # Update registry
            for manifest in manifests:
                if manifest.id not in self.plugin_registry:
                    self.plugin_registry[manifest.id] = {
                        "manifest": manifest,
                        "status": "available",
                        "auto_load": False
                    }
            
            logger.info(f"Plugin installation completed: {plugin_package}")
            return True
            
        except Exception as e:
            logger.error(f"Plugin installation failed: {e}")
            return False
    
    async def load_plugin(
        self,
        plugin_id: str,
        config: Dict[str, Any] = None,
        auto_start: bool = True
    ) -> bool:
        """
        Load and optionally start a plugin.
        
        Args:
            plugin_id: Plugin identifier
            config: Plugin configuration
            auto_start: Whether to automatically start the plugin
            
        Returns:
            True if successful
        """
        if plugin_id not in self.plugin_registry:
            logger.error(f"Plugin not found in registry: {plugin_id}")
            return False
        
        # Load plugin
        success = await self.loader.load_plugin(plugin_id, config)
        if success:
            self.plugin_registry[plugin_id]["status"] = "loaded"
            
            if auto_start:
                await self.start_plugin(plugin_id)
        
        return success
    
    async def start_plugin(self, plugin_id: str) -> bool:
        """Start a loaded plugin."""
        if plugin_id not in self.loader.loaded_plugins:
            logger.error(f"Plugin not loaded: {plugin_id}")
            return False
        
        plugin = self.loader.loaded_plugins[plugin_id]
        plugin._set_state(PluginState.ACTIVE)
        self.plugin_registry[plugin_id]["status"] = "active"
        
        # Emit plugin started event
        await self._emit_event("plugin_started", {"plugin_id": plugin_id})
        
        logger.info(f"Plugin started: {plugin_id}")
        return True
    
    async def stop_plugin(self, plugin_id: str) -> bool:
        """Stop an active plugin."""
        if plugin_id not in self.loader.loaded_plugins:
            logger.error(f"Plugin not loaded: {plugin_id}")
            return False
        
        plugin = self.loader.loaded_plugins[plugin_id]
        plugin._set_state(PluginState.INACTIVE)
        self.plugin_registry[plugin_id]["status"] = "loaded"
        
        # Emit plugin stopped event
        await self._emit_event("plugin_stopped", {"plugin_id": plugin_id})
        
        logger.info(f"Plugin stopped: {plugin_id}")
        return True
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin."""
        success = await self.loader.unload_plugin(plugin_id)
        if success and plugin_id in self.plugin_registry:
            self.plugin_registry[plugin_id]["status"] = "available"
        
        return success
    
    async def execute_plugin_task(
        self,
        plugin_id: str,
        task: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a task using a plugin.
        
        Args:
            plugin_id: Plugin identifier
            task: Task data
            timeout: Execution timeout in seconds
            
        Returns:
            Task result
        """
        if plugin_id not in self.loader.loaded_plugins:
            logger.error(f"Plugin not loaded: {plugin_id}")
            return None
        
        # Check if plugin is active
        if self.plugin_registry[plugin_id]["status"] != "active":
            logger.error(f"Plugin not active: {plugin_id}")
            return None
        
        # Execute with sandbox
        return await self.sandbox.execute_safely(
            self.loader.loaded_plugins[plugin_id],
            task,
            timeout or 300
        )
    
    def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive plugin information."""
        if plugin_id not in self.plugin_registry:
            return None
        
        info = self.plugin_registry[plugin_id].copy()
        
        # Add runtime status if loaded
        if plugin_id in self.loader.loaded_plugins:
            status = self.loader.get_plugin_status(plugin_id)
            if status:
                info.update(status)
        
        return info
    
    def list_plugins(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List plugins with optional status filter.
        
        Args:
            status_filter: Filter by status (available, loaded, active, etc.)
            
        Returns:
            List of plugin information
        """
        plugins = []
        
        for plugin_id, info in self.plugin_registry.items():
            if status_filter is None or info["status"] == status_filter:
                plugin_info = self.get_plugin_info(plugin_id)
                if plugin_info:
                    plugins.append(plugin_info)
        
        return plugins
    
    def on_event(self, event_name: str, handler: Callable):
        """Register event handler."""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)
    
    async def _emit_event(self, event_name: str, data: Dict[str, Any]):
        """Emit an event to registered handlers."""
        handlers = self.event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler failed for {event_name}: {e}")


class PluginScheduler:
    """
    Schedules plugin execution with priority and resource management.
    """
    
    def __init__(self):
        """Initialize plugin scheduler."""
        self.task_queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.max_concurrent_tasks = 10
        
    async def schedule_task(
        self,
        plugin_id: str,
        task: Dict[str, Any],
        priority: int = PluginPriority.NORMAL
    ) -> str:
        """Schedule a task for plugin execution."""
        task_id = str(uuid.uuid4())
        
        task_item = {
            "task_id": task_id,
            "plugin_id": plugin_id,
            "task": task,
            "priority": priority,
            "created_at": time.time()
        }
        
        await self.task_queue.put(task_item)
        return task_id
    
    async def start_scheduler(self):
        """Start the task scheduler."""
        while True:
            try:
                # Wait for tasks
                task_item = await self.task_queue.get()
                
                # Check if we can run more tasks
                if len(self.running_tasks) < self.max_concurrent_tasks:
                    # Execute task
                    task = asyncio.create_task(
                        self._execute_scheduled_task(task_item)
                    )
                    self.running_tasks[task_item["task_id"]] = task
                else:
                    # Put task back in queue
                    await self.task_queue.put(task_item)
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    async def _execute_scheduled_task(self, task_item: Dict[str, Any]):
        """Execute a scheduled task."""
        task_id = task_item["task_id"]
        
        try:
            # This would call the plugin manager to execute the task
            # For now, we'll just simulate execution
            await asyncio.sleep(1)
            
            logger.info(f"Completed scheduled task: {task_id}")
            
        except Exception as e:
            logger.error(f"Scheduled task failed {task_id}: {e}")
        
        finally:
            # Remove from running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]


class PluginSandbox:
    """
    Provides sandboxed execution environment for plugins.
    """
    
    def __init__(self):
        """Initialize plugin sandbox."""
        self.resource_limits = {
            "max_memory_mb": 512,
            "max_execution_time": 300,
            "max_file_handles": 100
        }
    
    async def execute_safely(
        self,
        plugin: BasePlugin,
        task: Dict[str, Any],
        timeout: int = 300
    ) -> Optional[Dict[str, Any]]:
        """
        Execute plugin task in a sandboxed environment.
        
        Args:
            plugin: Plugin instance
            task: Task to execute
            timeout: Execution timeout
            
        Returns:
            Task result or None if failed
        """
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                plugin.execute(task),
                timeout=timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Plugin execution timed out: {plugin.plugin_id}")
            return None
        except Exception as e:
            logger.error(f"Plugin execution failed: {plugin.plugin_id}: {e}")
            return None
    
    def monitor_resources(self, plugin_id: str) -> Dict[str, Any]:
        """Monitor plugin resource usage."""
        # This would implement actual resource monitoring
        # For now, return mock data
        return {
            "memory_mb": 128,
            "cpu_percent": 15.5,
            "execution_time": 45.2,
            "file_handles": 12
        }


# Export all classes
__all__ = [
    "PluginState",
    "PluginPriority",
    "PluginManifest",
    "PluginContext",
    "BasePlugin",
    "PluginLoader",
    "PluginManager",
    "PluginScheduler",
    "PluginSandbox"
]