"""Import cycle detection and dependency resolution system."""

import sys
import importlib
import threading
import weakref
from typing import Dict, Set, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ImportDependency:
    """Represents an import dependency."""
    source_module: str
    target_module: str
    import_type: str  # 'direct', 'from', 'lazy'
    line_number: Optional[int] = None

class ImportCycleDetector:
    """Detects and prevents circular import issues."""
    
    def __init__(self):
        self._import_graph: Dict[str, Set[str]] = {}
        self._loading_modules: Set[str] = set()
        self._lock = threading.Lock()
        self._lazy_imports: Dict[str, Any] = {}
    
    def register_import(self, source: str, target: str) -> None:
        """Register an import dependency."""
        with self._lock:
            if source not in self._import_graph:
                self._import_graph[source] = set()
            self._import_graph[source].add(target)
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect circular import cycles."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self._import_graph.get(node, []):
                dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for module in self._import_graph:
            if module not in visited:
                dfs(module, [])
        
        return cycles
    
    def lazy_import(self, module_name: str, attribute: Optional[str] = None):
        """Implement lazy import to break circular dependencies."""
        def _import():
            if module_name not in self._lazy_imports:
                try:
                    self._lazy_imports[module_name] = importlib.import_module(module_name)
                except ImportError as e:
                    logger.error(f"Failed to lazy import {module_name}: {e}")
                    raise
            
            module = self._lazy_imports[module_name]
            if attribute:
                return getattr(module, attribute)
            return module
        
        return _import

# Global import manager
_import_manager = ImportCycleDetector()

def safe_import(module_name: str, attribute: Optional[str] = None, fallback: Any = None):
    """Safely import module with cycle detection."""
    try:
        # Check for potential cycles
        current_module = sys._getframe(1).f_globals.get('__name__', 'unknown')
        _import_manager.register_import(current_module, module_name)
        
        cycles = _import_manager.detect_cycles()
        if cycles:
            logger.warning(f"Potential import cycles detected: {cycles}")
            # Use lazy import for cycles
            return _import_manager.lazy_import(module_name, attribute)()
        
        # Normal import
        module = importlib.import_module(module_name)
        if attribute:
            return getattr(module, attribute, fallback)
        return module
        
    except ImportError as e:
        logger.error(f"Import failed for {module_name}: {e}")
        if fallback is not None:
            return fallback
        raise

def get_import_cycles() -> List[List[str]]:
    """Get detected import cycles."""
    return _import_manager.detect_cycles()