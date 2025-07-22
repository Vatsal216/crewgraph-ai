"""
Tool registry for centralized tool management and discovery
"""

import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union
from collections import defaultdict

from .base import BaseTool, ToolMetadata, ToolType
from .wrapper import ToolWrapper
from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError, ValidationError

logger = get_logger(__name__)


class ToolRegistry:
    """
    Centralized registry for tool management with advanced features:
    
    - Tool discovery and registration
    - Version management
    - Category-based organization
    - Usage analytics
    - Performance monitoring
    - Tool recommendations
    - Dependency resolution
    """
    
    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = defaultdict(list)
        self._tags: Dict[str, List[str]] = defaultdict(list)
        self._versions: Dict[str, List[str]] = defaultdict(list)
        self._dependencies: Dict[str, List[str]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Analytics
        self._usage_analytics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._performance_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Auto-discovery
        self._auto_discovery_enabled = True
        self._discovery_paths: List[str] = []
        
        logger.info("ToolRegistry initialized")
    
    def register_tool(self, tool: BaseTool, force: bool = False) -> bool:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
            force: Force registration even if tool exists
            
        Returns:
            True if registered successfully, False otherwise
        """
        with self._lock:
            tool_name = tool.metadata.name
            
            # Check if tool already exists
            if tool_name in self._tools and not force:
                logger.warning(f"Tool '{tool_name}' already registered. Use force=True to override.")
                return False
            
            try:
                # Validate tool
                if not self._validate_tool(tool):
                    logger.error(f"Tool '{tool_name}' validation failed")
                    return False
                
                # Register tool
                self._tools[tool_name] = tool
                
                # Update categorization
                if tool.metadata.category:
                    self._categories[tool.metadata.category].append(tool_name)
                
                # Update tags
                for tag in tool.metadata.tags:
                    self._tags[tag].append(tool_name)
                
                # Update versions
                self._versions[tool_name].append(tool.metadata.version)
                
                # Update dependencies
                if tool.metadata.dependencies:
                    self._dependencies[tool_name] = tool.metadata.dependencies
                
                # Initialize analytics
                self._usage_analytics[tool_name] = {
                    'registered_at': time.time(),
                    'total_calls': 0,
                    'total_errors': 0,
                    'last_used': None
                }
                
                self._performance_metrics[tool_name] = {
                    'avg_execution_time': 0.0,
                    'min_execution_time': float('inf'),
                    'max_execution_time': 0.0,
                    'success_rate': 1.0
                }
                
                logger.info(f"Tool '{tool_name}' registered successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register tool '{tool_name}': {e}")
                return False
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            name: Tool name
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        with self._lock:
            if name not in self._tools:
                logger.warning(f"Tool '{name}' not found in registry")
                return False
            
            try:
                tool = self._tools[name]
                
                # Cleanup tool
                if hasattr(tool, 'cleanup'):
                    tool.cleanup()
                
                # Remove from registry
                del self._tools[name]
                
                # Remove from categorization
                if tool.metadata.category:
                    category_tools = self._categories[tool.metadata.category]
                    if name in category_tools:
                        category_tools.remove(name)
                
                # Remove from tags
                for tag in tool.metadata.tags:
                    tag_tools = self._tags[tag]
                    if name in tag_tools:
                        tag_tools.remove(name)
                
                # Cleanup analytics
                if name in self._usage_analytics:
                    del self._usage_analytics[name]
                
                if name in self._performance_metrics:
                    del self._performance_metrics[name]
                
                logger.info(f"Tool '{name}' unregistered successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unregister tool '{name}': {e}")
                return False
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        with self._lock:
            return self._tools.get(name)
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """
        Execute tool by name with analytics tracking.
        
        Args:
            name: Tool name
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        tool = self.get_tool(name)
        if not tool:
            raise CrewGraphError(f"Tool '{name}' not found in registry")
        
        # Track usage
        start_time = time.time()
        success = False
        
        try:
            result = tool.execute(**kwargs)
            success = True
            
            # Update analytics
            self._update_tool_analytics(name, success, time.time() - start_time)
            
            return result
            
        except Exception as e:
            self._update_tool_analytics(name, success, time.time() - start_time)
            raise
    
    def list_tools(self, 
                   category: Optional[str] = None,
                   tag: Optional[str] = None,
                   tool_type: Optional[ToolType] = None) -> List[str]:
        """
        List tools with optional filtering.
        
        Args:
            category: Filter by category
            tag: Filter by tag
            tool_type: Filter by tool type
            
        Returns:
            List of tool names
        """
        with self._lock:
            tools = list(self._tools.keys())
            
            # Filter by category
            if category:
                tools = [name for name in tools if name in self._categories.get(category, [])]
            
            # Filter by tag
            if tag:
                tools = [name for name in tools if name in self._tags.get(tag, [])]
            
            # Filter by tool type
            if tool_type:
                tools = [
                    name for name in tools 
                    if self._tools[name].metadata.tool_type == tool_type
                ]
            
            return sorted(tools)
    
    def search_tools(self, 
                     query: str,
                     search_description: bool = True,
                     search_tags: bool = True) -> List[str]:
        """
        Search tools by query string.
        
        Args:
            query: Search query
            search_description: Search in descriptions
            search_tags: Search in tags
            
        Returns:
            List of matching tool names
        """
        with self._lock:
            query_lower = query.lower()
            matching_tools = []
            
            for name, tool in self._tools.items():
                # Search in name
                if query_lower in name.lower():
                    matching_tools.append(name)
                    continue
                
                # Search in description
                if search_description and query_lower in tool.metadata.description.lower():
                    matching_tools.append(name)
                    continue
                
                # Search in tags
                if search_tags:
                    for tag in tool.metadata.tags:
                        if query_lower in tag.lower():
                            matching_tools.append(name)
                            break
            
            return sorted(matching_tools)
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive tool information.
        
        Args:
            name: Tool name
            
        Returns:
            Tool information dictionary
        """
        tool = self.get_tool(name)
        if not tool:
            return None
        
        with self._lock:
            return {
                'metadata': tool.to_dict(),
                'analytics': self._usage_analytics.get(name, {}),
                'performance': self._performance_metrics.get(name, {}),
                'dependencies': self._dependencies.get(name, []),
                'versions': self._versions.get(name, [])
            }
    
    def get_recommendations(self, 
                           context: Dict[str, Any],
                           limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get tool recommendations based on context.
        
        Args:
            context: Context for recommendations
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended tools with scores
        """
        with self._lock:
            recommendations = []
            
            for name, tool in self._tools.items():
                score = self._calculate_recommendation_score(tool, context)
                
                if score > 0:
                    recommendations.append({
                        'name': name,
                        'score': score,
                        'description': tool.metadata.description,
                        'usage_count': self._usage_analytics[name].get('total_calls', 0),
                        'success_rate': self._performance_metrics[name].get('success_rate', 1.0)
                    })
            
            # Sort by score and limit
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:limit]
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics report.
        
        Returns:
            Analytics report dictionary
        """
        with self._lock:
            total_tools = len(self._tools)
            total_calls = sum(
                analytics.get('total_calls', 0) 
                for analytics in self._usage_analytics.values()
            )
            
            # Most used tools
            most_used = sorted(
                self._usage_analytics.items(),
                key=lambda x: x[1].get('total_calls', 0),
                reverse=True
            )[:10]
            
            # Best performing tools
            best_performing = sorted(
                self._performance_metrics.items(),
                key=lambda x: x[1].get('success_rate', 0),
                reverse=True
            )[:10]
            
            # Category distribution
            category_dist = {
                category: len(tools) 
                for category, tools in self._categories.items()
            }
            
            return {
                'summary': {
                    'total_tools': total_tools,
                    'total_calls': total_calls,
                    'categories': len(self._categories),
                    'tags': len(self._tags)
                },
                'most_used_tools': [
                    {'name': name, 'calls': data.get('total_calls', 0)}
                    for name, data in most_used
                ],
                'best_performing_tools': [
                    {'name': name, 'success_rate': data.get('success_rate', 0)}
                    for name, data in best_performing
                ],
                'category_distribution': category_dist,
                'performance_overview': {
                    'avg_success_rate': sum(
                        metrics.get('success_rate', 0) 
                        for metrics in self._performance_metrics.values()
                    ) / max(total_tools, 1),
                    'avg_execution_time': sum(
                        metrics.get('avg_execution_time', 0) 
                        for metrics in self._performance_metrics.values()
                    ) / max(total_tools, 1)
                }
            }
    
    def register_function(self, 
                         func: Callable,
                         name: Optional[str] = None,
                         description: Optional[str] = None,
                         **metadata_kwargs) -> bool:
        """
        Quick registration of Python function as tool.
        
        Args:
            func: Function to register
            name: Tool name (defaults to function name)
            description: Tool description
            **metadata_kwargs: Additional metadata
            
        Returns:
            True if registered successfully
        """
        tool = ToolWrapper.from_function(
            func=func,
            name=name,
            description=description,
            **metadata_kwargs
        )
        
        return self.register_tool(tool)
    
    def register_crew_tool(self, 
                          crew_tool: Any,
                          name: Optional[str] = None,
                          **metadata_kwargs) -> bool:
        """
        Quick registration of CrewAI tool.
        
        Args:
            crew_tool: CrewAI tool instance
            name: Tool name
            **metadata_kwargs: Additional metadata
            
        Returns:
            True if registered successfully
        """
        tool = ToolWrapper.from_crew_tool(
            crew_tool=crew_tool,
            name=name,
            **metadata_kwargs
        )
        
        return self.register_tool(tool)
    
    def _validate_tool(self, tool: BaseTool) -> bool:
        """Validate tool before registration."""
        try:
            # Check required metadata
            if not tool.metadata.name:
                logger.error("Tool name is required")
                return False
            
            if not tool.metadata.description:
                logger.warning(f"Tool '{tool.metadata.name}' has no description")
            
            # Check for conflicts
            if tool.metadata.name in self._tools:
                logger.warning(f"Tool '{tool.metadata.name}' already exists")
            
            # Validate dependencies
            for dep in tool.metadata.dependencies:
                if dep not in self._tools:
                    logger.warning(f"Dependency '{dep}' not found for tool '{tool.metadata.name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Tool validation failed: {e}")
            return False
    
    def _update_tool_analytics(self, name: str, success: bool, execution_time: float):
        """Update tool analytics and performance metrics."""
        with self._lock:
            # Update analytics
            analytics = self._usage_analytics[name]
            analytics['total_calls'] += 1
            analytics['last_used'] = time.time()
            
            if not success:
                analytics['total_errors'] += 1
            
            # Update performance metrics
            metrics = self._performance_metrics[name]
            
            # Update execution time statistics
            total_calls = analytics['total_calls']
            current_avg = metrics['avg_execution_time']
            metrics['avg_execution_time'] = (
                (current_avg * (total_calls - 1) + execution_time) / total_calls
            )
            
            metrics['min_execution_time'] = min(
                metrics['min_execution_time'], 
                execution_time
            )
            metrics['max_execution_time'] = max(
                metrics['max_execution_time'], 
                execution_time
            )
            
            # Update success rate
            total_successes = analytics['total_calls'] - analytics['total_errors']
            metrics['success_rate'] = total_successes / analytics['total_calls']
    
    def _calculate_recommendation_score(self, 
                                       tool: BaseTool, 
                                       context: Dict[str, Any]) -> float:
        """Calculate recommendation score for a tool based on context."""
        score = 0.0
        
        # Base score from usage statistics
        analytics = self._usage_analytics.get(tool.metadata.name, {})
        performance = self._performance_metrics.get(tool.metadata.name, {})
        
        # Usage popularity (normalized)
        usage_count = analytics.get('total_calls', 0)
        max_usage = max(
            (a.get('total_calls', 0) for a in self._usage_analytics.values()),
            default=1
        )
        score += (usage_count / max_usage) * 0.3
        
        # Success rate
        score += performance.get('success_rate', 1.0) * 0.3
        
        # Performance (inverse of execution time)
        avg_time = performance.get('avg_execution_time', 1.0)
        if avg_time > 0:
            score += (1.0 / avg_time) * 0.2
        
        # Context matching
        context_keywords = context.get('keywords', [])
        description_words = tool.metadata.description.lower().split()
        
        keyword_matches = sum(
            1 for keyword in context_keywords 
            if keyword.lower() in description_words
        )
        
        if keyword_matches > 0:
            score += (keyword_matches / len(context_keywords)) * 0.2
        
        return score
    
    def clear_registry(self) -> None:
        """Clear all tools from registry."""
        with self._lock:
            # Cleanup all tools
            for tool in self._tools.values():
                if hasattr(tool, 'cleanup'):
                    try:
                        tool.cleanup()
                    except Exception as e:
                        logger.warning(f"Failed to cleanup tool: {e}")
            
            # Clear all data structures
            self._tools.clear()
            self._categories.clear()
            self._tags.clear()
            self._versions.clear()
            self._dependencies.clear()
            self._usage_analytics.clear()
            self._performance_metrics.clear()
            
            logger.info("Tool registry cleared")
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools
    
    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.items())
    
    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self._tools)}, categories={len(self._categories)})"


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.
    
    Returns:
        Global ToolRegistry instance
    """
    global _global_registry
    
    if _global_registry is None:
        _global_registry = ToolRegistry(name="global_registry")
        logger.info("Global tool registry created")
    
    return _global_registry


def register_tool_globally(tool: "BaseTool") -> bool:
    """
    Register a tool in the global registry.
    
    Args:
        tool: Tool to register
        
    Returns:
        True if registered successfully
    """
    return get_global_registry().register_tool(tool)


def get_global_tool(name: str) -> Optional["BaseTool"]:
    """
    Get a tool from the global registry.
    
    Args:
        name: Tool name
        
    Returns:
        Tool instance or None
    """
    return get_global_registry().get_tool(name)