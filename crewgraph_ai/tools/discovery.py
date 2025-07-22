"""
CrewGraph AI Tool Discovery System
Automatic discovery, registration, and management of tools from various sources

Author: Vatsal216
Created: 2025-07-22 12:29:42 UTC
"""

import os
import sys
import importlib
import inspect
import json
import ast
import pkgutil
from typing import Any, Dict, List, Optional, Callable, Type, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from enum import Enum

from crewai import Tool
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector
from ..utils.exceptions import ToolError

logger = get_logger(__name__)
metrics = get_metrics_collector()


class ToolSourceType(Enum):
    """Types of tool sources for discovery"""
    MODULE = "module"
    PACKAGE = "package"
    DIRECTORY = "directory"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    REGISTRY = "registry"
    REMOTE = "remote"


@dataclass
class ToolDefinition:
    """Definition of a discovered tool"""
    name: str
    description: str
    function: Optional[Callable] = None
    source_type: ToolSourceType = ToolSourceType.FUNCTION
    source_path: str = ""
    module_name: str = ""
    class_name: Optional[str] = None
    function_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    return_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: str = "Unknown"
    created_at: str = "2025-07-22 12:29:42"
    discovered_by: str = "Vatsal216"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'source_type': self.source_type.value,
            'source_path': self.source_path,
            'module_name': self.module_name,
            'class_name': self.class_name,
            'function_name': self.function_name,
            'parameters': self.parameters,
            'return_type': self.return_type,
            'metadata': self.metadata,
            'tags': self.tags,
            'version': self.version,
            'author': self.author,
            'created_at': self.created_at,
            'discovered_by': self.discovered_by
        }
    
    def to_crewai_tool(self) -> Tool:
        """Convert to CrewAI Tool instance"""
        if not self.function:
            raise ToolError(f"Cannot create Tool: function not available for {self.name}")
        
        return Tool(
            name=self.name,
            func=self.function,
            description=self.description
        )


@dataclass
class DiscoveryConfig:
    """Configuration for tool discovery"""
    search_paths: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["__pycache__", "*.pyc", "test*"])
    recursive: bool = True
    follow_imports: bool = False
    max_depth: int = 10
    timeout: float = 30.0
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_validation: bool = True
    auto_register: bool = False
    discovery_methods: List[str] = field(default_factory=lambda: ["function", "class", "decorator"])
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.search_paths:
            self.search_paths = [os.getcwd()]
        
        if self.max_depth < 1:
            self.max_depth = 1
        
        if self.timeout <= 0:
            self.timeout = 30.0


class ToolDiscovery:
    """
    Advanced tool discovery system for CrewGraph AI.
    
    Automatically discovers tools from various sources including:
    - Python modules and packages
    - Local directories and files
    - Class methods and standalone functions
    - Decorated functions with tool metadata
    - Remote tool registries
    - Configuration files and manifests
    
    Features:
    - Multi-source discovery with configurable patterns
    - Intelligent function signature analysis
    - Automatic parameter extraction and validation
    - Caching for performance optimization
    - Thread-safe operations
    - Comprehensive logging and metrics
    
    Created by: Vatsal216
    Date: 2025-07-22 12:29:42 UTC
    """
    
    def __init__(self, config: Optional[DiscoveryConfig] = None):
        """
        Initialize tool discovery system.
        
        Args:
            config: Discovery configuration
        """
        self.config = config or DiscoveryConfig()
        
        # Discovered tools storage
        self.discovered_tools: Dict[str, ToolDefinition] = {}
        self.discovery_cache: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Discovery statistics
        self.discovery_stats = {
            'total_discoveries': 0,
            'successful_discoveries': 0,
            'failed_discoveries': 0,
            'cache_hits': 0,
            'last_discovery_time': None,
            'discovery_sources': {}
        }
        
        # Tool validators (will be imported dynamically)
        self._validators = []
        
        logger.info("ToolDiscovery system initialized")
        logger.info(f"Search paths: {self.config.search_paths}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:29:42")
        
        # Record initialization metrics
        metrics.increment_counter(
            "crewgraph_tool_discovery_initialized_total",
            labels={"user": "Vatsal216"}
        )
    
    def discover_from_module(self, module_name: str) -> List[ToolDefinition]:
        """
        Discover tools from a Python module.
        
        Args:
            module_name: Name of the module to inspect
            
        Returns:
            List of discovered tool definitions
        """
        start_time = time.time()
        discovered = []
        
        try:
            logger.info(f"Discovering tools from module: {module_name}")
            
            # Check cache first
            cache_key = f"module:{module_name}"
            if self.config.enable_caching and cache_key in self.discovery_cache:
                cache_entry = self.discovery_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.config.cache_ttl:
                    self.discovery_stats['cache_hits'] += 1
                    logger.debug(f"Using cached discovery for module: {module_name}")
                    return [ToolDefinition(**tool_data) for tool_data in cache_entry['tools']]
            
            # Import the module
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                logger.error(f"Failed to import module {module_name}: {e}")
                return discovered
            
            # Discover functions in module
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj):
                    tool_def = self._analyze_function(obj, module_name, name)
                    if tool_def:
                        discovered.append(tool_def)
                
                elif inspect.isclass(obj):
                    # Discover methods in classes
                    class_tools = self._analyze_class(obj, module_name, name)
                    discovered.extend(class_tools)
            
            # Cache results
            if self.config.enable_caching:
                self.discovery_cache[cache_key] = {
                    'timestamp': time.time(),
                    'tools': [tool.to_dict() for tool in discovered]
                }
            
            # Update statistics
            self.discovery_stats['total_discoveries'] += len(discovered)
            self.discovery_stats['successful_discoveries'] += len(discovered)
            self.discovery_stats['last_discovery_time'] = time.time()
            
            if module_name not in self.discovery_stats['discovery_sources']:
                self.discovery_stats['discovery_sources'][module_name] = 0
            self.discovery_stats['discovery_sources'][module_name] += len(discovered)
            
            # Record metrics
            discovery_time = time.time() - start_time
            metrics.record_duration(
                "crewgraph_tool_discovery_duration_seconds",
                discovery_time,
                labels={
                    "source_type": "module",
                    "source": module_name,
                    "user": "Vatsal216"
                }
            )
            
            metrics.increment_counter(
                "crewgraph_tools_discovered_total",
                len(discovered),
                labels={
                    "source_type": "module",
                    "source": module_name,
                    "user": "Vatsal216"
                }
            )
            
            logger.info(f"Discovered {len(discovered)} tools from module: {module_name}")
            
        except Exception as e:
            logger.error(f"Error discovering tools from module {module_name}: {e}")
            self.discovery_stats['failed_discoveries'] += 1
        
        return discovered
    
    def discover_from_directory(self, directory_path: str) -> List[ToolDefinition]:
        """
        Discover tools from a directory of Python files.
        
        Args:
            directory_path: Path to directory to search
            
        Returns:
            List of discovered tool definitions
        """
        start_time = time.time()
        discovered = []
        
        try:
            logger.info(f"Discovering tools from directory: {directory_path}")
            
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                logger.error(f"Directory not found or not a directory: {directory_path}")
                return discovered
            
            # Check cache
            cache_key = f"directory:{directory_path}"
            if self.config.enable_caching and cache_key in self.discovery_cache:
                cache_entry = self.discovery_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.config.cache_ttl:
                    self.discovery_stats['cache_hits'] += 1
                    logger.debug(f"Using cached discovery for directory: {directory_path}")
                    return [ToolDefinition(**tool_data) for tool_data in cache_entry['tools']]
            
            # Find Python files
            python_files = []
            if self.config.recursive:
                python_files = list(directory.rglob("*.py"))
            else:
                python_files = list(directory.glob("*.py"))
            
            # Filter files based on patterns
            filtered_files = self._filter_files(python_files)
            
            # Discover tools from each file
            for file_path in filtered_files:
                try:
                    file_tools = self._discover_from_file(file_path)
                    discovered.extend(file_tools)
                except Exception as e:
                    logger.error(f"Error discovering tools from file {file_path}: {e}")
            
            # Cache results
            if self.config.enable_caching:
                self.discovery_cache[cache_key] = {
                    'timestamp': time.time(),
                    'tools': [tool.to_dict() for tool in discovered]
                }
            
            # Update statistics
            self.discovery_stats['total_discoveries'] += len(discovered)
            self.discovery_stats['successful_discoveries'] += len(discovered)
            self.discovery_stats['last_discovery_time'] = time.time()
            
            # Record metrics
            discovery_time = time.time() - start_time
            metrics.record_duration(
                "crewgraph_tool_discovery_duration_seconds",
                discovery_time,
                labels={
                    "source_type": "directory",
                    "source": directory_path,
                    "user": "Vatsal216"
                }
            )
            
            logger.info(f"Discovered {len(discovered)} tools from directory: {directory_path}")
            
        except Exception as e:
            logger.error(f"Error discovering tools from directory {directory_path}: {e}")
            self.discovery_stats['failed_discoveries'] += 1
        
        return discovered
    
    def discover_from_package(self, package_name: str) -> List[ToolDefinition]:
        """
        Discover tools from a Python package.
        
        Args:
            package_name: Name of the package to inspect
            
        Returns:
            List of discovered tool definitions
        """
        start_time = time.time()
        discovered = []
        
        try:
            logger.info(f"Discovering tools from package: {package_name}")
            
            # Import the package
            try:
                package = importlib.import_module(package_name)
            except ImportError as e:
                logger.error(f"Failed to import package {package_name}: {e}")
                return discovered
            
            # Get package path
            if hasattr(package, '__path__'):
                package_paths = package.__path__
            else:
                logger.warning(f"Package {package_name} has no __path__ attribute")
                return discovered
            
            # Discover submodules
            for importer, modname, ispkg in pkgutil.iter_modules(package_paths):
                full_module_name = f"{package_name}.{modname}"
                
                try:
                    module_tools = self.discover_from_module(full_module_name)
                    discovered.extend(module_tools)
                except Exception as e:
                    logger.error(f"Error discovering tools from submodule {full_module_name}: {e}")
            
            # Record metrics
            discovery_time = time.time() - start_time
            metrics.record_duration(
                "crewgraph_tool_discovery_duration_seconds",
                discovery_time,
                labels={
                    "source_type": "package",
                    "source": package_name,
                    "user": "Vatsal216"
                }
            )
            
            logger.info(f"Discovered {len(discovered)} tools from package: {package_name}")
            
        except Exception as e:
            logger.error(f"Error discovering tools from package {package_name}: {e}")
            self.discovery_stats['failed_discoveries'] += 1
        
        return discovered
    
    def discover_decorated_tools(self, search_paths: Optional[List[str]] = None) -> List[ToolDefinition]:
        """
        Discover tools marked with special decorators.
        
        Args:
            search_paths: Paths to search for decorated tools
            
        Returns:
            List of discovered tool definitions
        """
        discovered = []
        paths = search_paths or self.config.search_paths
        
        logger.info("Discovering decorated tools")
        
        for path in paths:
            try:
                if os.path.isdir(path):
                    dir_tools = self._discover_decorated_from_directory(path)
                    discovered.extend(dir_tools)
                elif os.path.isfile(path) and path.endswith('.py'):
                    file_tools = self._discover_decorated_from_file(path)
                    discovered.extend(file_tools)
            except Exception as e:
                logger.error(f"Error discovering decorated tools from {path}: {e}")
        
        logger.info(f"Discovered {len(discovered)} decorated tools")
        return discovered
    
    def discover_all(self, auto_register: Optional[bool] = None) -> List[ToolDefinition]:
        """
        Discover tools from all configured sources.
        
        Args:
            auto_register: Whether to auto-register discovered tools
            
        Returns:
            List of all discovered tool definitions
        """
        start_time = time.time()
        all_discovered = []
        
        logger.info("Starting comprehensive tool discovery")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:29:42")
        
        with self._lock:
            # Clear previous discoveries
            self.discovered_tools.clear()
            
            # Discover from all search paths
            for path in self.config.search_paths:
                try:
                    if os.path.isdir(path):
                        path_tools = self.discover_from_directory(path)
                        all_discovered.extend(path_tools)
                    elif os.path.isfile(path) and path.endswith('.py'):
                        file_tools = self._discover_from_file(path)
                        all_discovered.extend(file_tools)
                except Exception as e:
                    logger.error(f"Error discovering from path {path}: {e}")
            
            # Discover decorated tools
            if "decorator" in self.config.discovery_methods:
                decorated_tools = self.discover_decorated_tools()
                all_discovered.extend(decorated_tools)
            
            # Remove duplicates and validate
            unique_tools = self._deduplicate_tools(all_discovered)
            
            # Validate tools if enabled
            if self.config.enable_validation:
                validated_tools = self._validate_tools(unique_tools)
            else:
                validated_tools = unique_tools
            
            # Store discovered tools
            for tool in validated_tools:
                self.discovered_tools[tool.name] = tool
            
            # Auto-register if enabled
            if auto_register or self.config.auto_register:
                self._auto_register_tools(validated_tools)
        
        # Record comprehensive metrics
        discovery_time = time.time() - start_time
        metrics.record_duration(
            "crewgraph_tool_discovery_complete_duration_seconds",
            discovery_time,
            labels={"user": "Vatsal216"}
        )
        
        metrics.record_gauge(
            "crewgraph_tools_discovered_count",
            len(validated_tools),
            labels={"user": "Vatsal216"}
        )
        
        logger.info(f"Discovery complete: {len(validated_tools)} tools discovered in {discovery_time:.2f}s")
        
        return validated_tools
    
    def _analyze_function(self, func: Callable, module_name: str, func_name: str) -> Optional[ToolDefinition]:
        """Analyze a function to create tool definition"""
        try:
            # Skip private functions
            if func_name.startswith('_'):
                return None
            
            # Get function signature
            sig = inspect.signature(func)
            
            # Extract parameters
            parameters = {}
            for param_name, param in sig.parameters.items():
                param_info = {
                    'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                }
                parameters[param_name] = param_info
            
            # Get return type
            return_type = str(sig.return_annotation) if sig.return_annotation != inspect.Parameter.empty else None
            
            # Get docstring
            docstring = inspect.getdoc(func) or f"Function {func_name} from {module_name}"
            
            # Extract metadata from docstring or decorators
            metadata = self._extract_metadata(func)
            
            # Create tool definition
            tool_def = ToolDefinition(
                name=func_name,
                description=docstring,
                function=func,
                source_type=ToolSourceType.FUNCTION,
                source_path=inspect.getfile(func),
                module_name=module_name,
                function_name=func_name,
                parameters=parameters,
                return_type=return_type,
                metadata=metadata,
                discovered_by="Vatsal216"
            )
            
            # Extract tags from metadata or docstring
            tool_def.tags = self._extract_tags(func, docstring, metadata)
            
            return tool_def
            
        except Exception as e:
            logger.error(f"Error analyzing function {func_name}: {e}")
            return None
    
    def _analyze_class(self, cls: Type, module_name: str, class_name: str) -> List[ToolDefinition]:
        """Analyze a class to find tool methods"""
        discovered = []
        
        try:
            # Look for methods that could be tools
            for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if method_name.startswith('_'):
                    continue
                
                # Check if method is marked as a tool
                if hasattr(method, '_is_tool') or self._is_tool_method(method):
                    # Create a wrapper function
                    def create_tool_wrapper(cls_ref, method_ref, method_name_ref):
                        def wrapper(*args, **kwargs):
                            instance = cls_ref()
                            return getattr(instance, method_name_ref)(*args, **kwargs)
                        wrapper.__name__ = method_name_ref
                        wrapper.__doc__ = method_ref.__doc__
                        return wrapper
                    
                    tool_func = create_tool_wrapper(cls, method, method_name)
                    
                    tool_def = self._analyze_function(tool_func, module_name, f"{class_name}.{method_name}")
                    if tool_def:
                        tool_def.class_name = class_name
                        tool_def.source_type = ToolSourceType.CLASS
                        discovered.append(tool_def)
        
        except Exception as e:
            logger.error(f"Error analyzing class {class_name}: {e}")
        
        return discovered
    
    def _discover_from_file(self, file_path: Path) -> List[ToolDefinition]:
        """Discover tools from a single Python file"""
        discovered = []
        
        try:
            # Parse the file to extract functions and classes
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Find function and class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Try to extract function info from AST
                    func_info = self._extract_function_info_from_ast(node, str(file_path))
                    if func_info:
                        discovered.append(func_info)
                
                elif isinstance(node, ast.ClassDef):
                    # Extract class methods
                    class_tools = self._extract_class_info_from_ast(node, str(file_path))
                    discovered.extend(class_tools)
        
        except Exception as e:
            logger.error(f"Error discovering tools from file {file_path}: {e}")
        
        return discovered
    
    def _extract_function_info_from_ast(self, node: ast.FunctionDef, file_path: str) -> Optional[ToolDefinition]:
        """Extract function information from AST node"""
        try:
            if node.name.startswith('_'):
                return None
            
            # Extract docstring
            docstring = ast.get_docstring(node) or f"Function {node.name}"
            
            # Extract parameters (basic extraction)
            parameters = {}
            for arg in node.args.args:
                parameters[arg.arg] = {
                    'type': 'Any',
                    'required': True
                }
            
            # Check for tool decorators
            is_tool = any(
                isinstance(decorator, ast.Name) and decorator.id == 'tool'
                for decorator in node.decorator_list
            )
            
            if not is_tool and not self._looks_like_tool_function(node.name, docstring):
                return None
            
            return ToolDefinition(
                name=node.name,
                description=docstring,
                source_type=ToolSourceType.FILE,
                source_path=file_path,
                function_name=node.name,
                parameters=parameters,
                metadata={'from_ast': True},
                discovered_by="Vatsal216"
            )
            
        except Exception as e:
            logger.error(f"Error extracting function info from AST: {e}")
            return None
    
    def _extract_class_info_from_ast(self, node: ast.ClassDef, file_path: str) -> List[ToolDefinition]:
        """Extract class method information from AST node"""
        discovered = []
        
        try:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                    method_info = self._extract_function_info_from_ast(item, file_path)
                    if method_info:
                        method_info.class_name = node.name
                        method_info.source_type = ToolSourceType.CLASS
                        method_info.name = f"{node.name}.{item.name}"
                        discovered.append(method_info)
        
        except Exception as e:
            logger.error(f"Error extracting class info from AST: {e}")
        
        return discovered
    
    def _discover_decorated_from_directory(self, directory_path: str) -> List[ToolDefinition]:
        """Discover decorated tools from directory"""
        discovered = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_tools = self._discover_decorated_from_file(file_path)
                    discovered.extend(file_tools)
        
        return discovered
    
    def _discover_decorated_from_file(self, file_path: str) -> List[ToolDefinition]:
        """Discover decorated tools from a single file"""
        discovered = []
        
        try:
            # This would involve more complex AST parsing to find decorated functions
            # For now, we'll use a simplified approach
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for tool decorator patterns
            if '@tool' in content or '@crewai_tool' in content:
                # Parse file and extract decorated functions
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check for tool decorators
                        has_tool_decorator = any(
                            (isinstance(d, ast.Name) and d.id in ['tool', 'crewai_tool']) or
                            (isinstance(d, ast.Attribute) and d.attr in ['tool', 'crewai_tool'])
                            for d in node.decorator_list
                        )
                        
                        if has_tool_decorator:
                            tool_def = self._extract_function_info_from_ast(node, file_path)
                            if tool_def:
                                tool_def.metadata['decorated'] = True
                                discovered.append(tool_def)
        
        except Exception as e:
            logger.error(f"Error discovering decorated tools from {file_path}: {e}")
        
        return discovered
    
    def _filter_files(self, files: List[Path]) -> List[Path]:
        """Filter files based on include/exclude patterns"""
        filtered = []
        
        for file in files:
            file_str = str(file)
            
            # Check exclude patterns
            excluded = any(
                pattern in file_str.lower() for pattern in self.config.exclude_patterns
            )
            
            if excluded:
                continue
            
            # Check include patterns
            included = any(
                pattern == "*" or pattern in file_str.lower() 
                for pattern in self.config.include_patterns
            )
            
            if included:
                filtered.append(file)
        
        return filtered
    
    def _extract_metadata(self, func: Callable) -> Dict[str, Any]:
        """Extract metadata from function"""
        metadata = {}
        
        # Check for custom attributes
        if hasattr(func, '_tool_metadata'):
            metadata.update(func._tool_metadata)
        
        # Extract from docstring
        docstring = inspect.getdoc(func) or ""
        
        # Look for metadata patterns in docstring
        if "Author:" in docstring:
            try:
                author_line = [line for line in docstring.split('\n') if 'Author:' in line][0]
                metadata['author'] = author_line.split('Author:', 1)[1].strip()
            except:
                pass
        
        if "Version:" in docstring:
            try:
                version_line = [line for line in docstring.split('\n') if 'Version:' in line][0]
                metadata['version'] = version_line.split('Version:', 1)[1].strip()
            except:
                pass
        
        return metadata
    
    def _extract_tags(self, func: Callable, docstring: str, metadata: Dict[str, Any]) -> List[str]:
        """Extract tags from function, docstring, and metadata"""
        tags = []
        
        # From metadata
        if 'tags' in metadata:
            tags.extend(metadata['tags'])
        
        # From function attributes
        if hasattr(func, '_tool_tags'):
            tags.extend(func._tool_tags)
        
        # From docstring keywords
        docstring_lower = docstring.lower()
        
        # Common tool type tags
        if any(word in docstring_lower for word in ['search', 'find', 'query']):
            tags.append('search')
        
        if any(word in docstring_lower for word in ['file', 'read', 'write', 'save']):
            tags.append('file')
        
        if any(word in docstring_lower for word in ['api', 'http', 'request', 'web']):
            tags.append('api')
        
        if any(word in docstring_lower for word in ['analyze', 'analysis', 'process']):
            tags.append('analysis')
        
        if any(word in docstring_lower for word in ['calculate', 'math', 'compute']):
            tags.append('math')
        
        # Remove duplicates
        return list(set(tags))
    
    def _is_tool_method(self, method: Callable) -> bool:
        """Check if a method should be considered a tool"""
        # Check for tool attributes
        if hasattr(method, '_is_tool'):
            return method._is_tool
        
        # Check docstring for tool indicators
        docstring = inspect.getdoc(method) or ""
        tool_indicators = ['tool', 'function', 'utility', 'helper']
        
        return any(indicator in docstring.lower() for indicator in tool_indicators)
    
    def _looks_like_tool_function(self, func_name: str, docstring: str) -> bool:
        """Heuristic to determine if a function looks like a tool"""
        # Function name patterns
        tool_name_patterns = [
            'process_', 'analyze_', 'calculate_', 'search_', 'find_',
            'get_', 'fetch_', 'read_', 'write_', 'parse_', 'convert_'
        ]
        
        if any(func_name.startswith(pattern) for pattern in tool_name_patterns):
            return True
        
        # Docstring patterns
        tool_doc_patterns = [
            'tool', 'function', 'utility', 'process', 'analyze',
            'calculate', 'search', 'find', 'get', 'fetch'
        ]
        
        if any(pattern in docstring.lower() for pattern in tool_doc_patterns):
            return True
        
        return False
    
    def _deduplicate_tools(self, tools: List[ToolDefinition]) -> List[ToolDefinition]:
        """Remove duplicate tools based on name and signature"""
        seen = {}
        unique_tools = []
        
        for tool in tools:
            # Create signature for comparison
            signature = f"{tool.name}:{tool.source_path}:{tool.function_name}"
            
            if signature not in seen:
                seen[signature] = tool
                unique_tools.append(tool)
            else:
                # Keep the one with more complete information
                existing = seen[signature]
                if len(tool.parameters) > len(existing.parameters):
                    seen[signature] = tool
                    # Replace in unique_tools list
                    for i, t in enumerate(unique_tools):
                        if t.name == existing.name and t.source_path == existing.source_path:
                            unique_tools[i] = tool
                            break
        
        logger.info(f"Deduplicated {len(tools)} tools to {len(unique_tools)} unique tools")
        return unique_tools
    
    def _validate_tools(self, tools: List[ToolDefinition]) -> List[ToolDefinition]:
        """Validate discovered tools"""
        validated = []
        
        for tool in tools:
            try:
                # Basic validation
                if not tool.name:
                    logger.warning(f"Tool missing name: {tool}")
                    continue
                
                if not tool.description:
                    logger.warning(f"Tool missing description: {tool.name}")
                    continue
                
                # Function validation (if available)
                if tool.function:
                    try:
                        # Try to get signature
                        inspect.signature(tool.function)
                    except Exception as e:
                        logger.warning(f"Invalid function signature for tool {tool.name}: {e}")
                        continue
                
                validated.append(tool)
                
            except Exception as e:
                logger.error(f"Error validating tool {tool.name}: {e}")
        
        logger.info(f"Validated {len(validated)} out of {len(tools)} tools")
        return validated
    
    def _auto_register_tools(self, tools: List[ToolDefinition]) -> None:
        """Auto-register tools in the tool registry"""
        try:
            from .registry import ToolRegistry
            
            registry = ToolRegistry()
            
            for tool in tools:
                try:
                    if tool.function:
                        crewai_tool = tool.to_crewai_tool()
                        registry.register_tool(crewai_tool, tool.metadata)
                        logger.debug(f"Auto-registered tool: {tool.name}")
                except Exception as e:
                    logger.error(f"Failed to auto-register tool {tool.name}: {e}")
            
            logger.info(f"Auto-registered {len(tools)} tools")
            
        except ImportError:
            logger.warning("ToolRegistry not available for auto-registration")
        except Exception as e:
            logger.error(f"Error during auto-registration: {e}")
    
    def get_discovered_tools(self) -> Dict[str, ToolDefinition]:
        """Get all discovered tools"""
        with self._lock:
            return self.discovered_tools.copy()
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a specific discovered tool"""
        with self._lock:
            return self.discovered_tools.get(name)
    
    def search_tools(self, query: str) -> List[ToolDefinition]:
        """Search discovered tools by name, description, or tags"""
        query = query.lower()
        results = []
        
        with self._lock:
            for tool in self.discovered_tools.values():
                if (query in tool.name.lower() or
                    query in tool.description.lower() or
                    any(query in tag.lower() for tag in tool.tags)):
                    results.append(tool)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        with self._lock:
            stats = self.discovery_stats.copy()
            stats.update({
                'discovered_tools_count': len(self.discovered_tools),
                'cache_entries': len(self.discovery_cache),
                'config': {
                    'search_paths': self.config.search_paths,
                    'discovery_methods': self.config.discovery_methods,
                    'enable_caching': self.config.enable_caching,
                    'enable_validation': self.config.enable_validation
                },
                'created_by': 'Vatsal216',
                'timestamp': '2025-07-22 12:29:42'
            })
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear discovery cache"""
        with self._lock:
            self.discovery_cache.clear()
            logger.info("Discovery cache cleared")
    
    def export_tools(self, format_type: str = "json") -> str:
        """Export discovered tools in specified format"""
        with self._lock:
            tools_data = [tool.to_dict() for tool in self.discovered_tools.values()]
            
            if format_type.lower() == "json":
                return json.dumps(tools_data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
    
    def __repr__(self) -> str:
        return f"ToolDiscovery(tools={len(self.discovered_tools)}, sources={len(self.config.search_paths)})"


def create_tool_discovery(search_paths: Optional[List[str]] = None, **kwargs) -> ToolDiscovery:
    """
    Factory function to create tool discovery instance.
    
    Args:
        search_paths: Paths to search for tools
        **kwargs: Additional configuration options
        
    Returns:
        Configured ToolDiscovery instance
    """
    config = DiscoveryConfig(**kwargs)
    if search_paths:
        config.search_paths = search_paths
    
    logger.info("Creating ToolDiscovery instance")
    logger.info(f"User: Vatsal216, Time: 2025-07-22 12:29:42")
    
    return ToolDiscovery(config)