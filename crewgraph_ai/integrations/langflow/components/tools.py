"""
CrewGraph Tool Components for Langflow Integration

This module provides Langflow components that bridge to CrewGraph tools.

Created by: Vatsal216
Date: 2025-07-23
"""

from typing import Any, Dict, List, Optional

from .base import LangflowComponent, ComponentInput, ComponentOutput, ComponentMetadata

# Import CrewGraph tool functionality
try:
    from ....tools.registry import ToolRegistry
    from ....tools.wrapper import ToolWrapper
    from ....tools.builtin import BuiltinTools
except ImportError:
    # Fallback for development
    ToolRegistry = None
    ToolWrapper = None
    BuiltinTools = None


class CrewGraphToolComponent(LangflowComponent):
    """
    Langflow component for CrewGraph AI tools
    
    This component allows users to access and configure CrewGraph tools
    in the Langflow visual interface.
    """
    
    def _get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="CrewGraphTool",
            display_name="CrewGraph Tool",
            description="Access to CrewGraph AI tools for various tasks",
            category="CrewGraph AI",
            tags=["tool", "utility", "crewgraph"],
            icon="🔧",
            documentation_url="https://github.com/Vatsal216/crewgraph-ai/docs/tools",
        )
    
    def _get_inputs(self) -> List[ComponentInput]:
        return [
            ComponentInput(
                name="tool_name",
                display_name="Tool Name",
                input_type="str",
                required=True,
                description="Name of the CrewGraph tool to use",
                options=[
                    "text_processor",
                    "file_handler", 
                    "data_converter",
                    "hash_generator",
                    "timestamp_utility",
                    "math_calculator",
                    "list_processor",
                    "web_scraper",
                    "api_client",
                    "email_sender"
                ],
                default_value="text_processor"
            ),
            ComponentInput(
                name="tool_input",
                display_name="Tool Input",
                input_type="str",
                required=True,
                multiline=True,
                description="Input data for the tool (format depends on the specific tool)"
            ),
            ComponentInput(
                name="tool_config",
                display_name="Tool Configuration",
                input_type="dict",
                required=False,
                description="Additional configuration parameters for the tool",
                default_value={}
            ),
            ComponentInput(
                name="cache_results",
                display_name="Cache Results",
                input_type="bool",
                required=False,
                description="Whether to cache tool execution results",
                default_value=True
            ),
            ComponentInput(
                name="retry_on_failure",
                display_name="Retry on Failure",
                input_type="bool",
                required=False,
                description="Whether to retry tool execution on failure",
                default_value=True
            ),
            ComponentInput(
                name="max_retries",
                display_name="Max Retries",
                input_type="int",
                required=False,
                description="Maximum number of retry attempts",
                default_value=3,
                min_value=0,
                max_value=10
            ),
            ComponentInput(
                name="timeout_seconds",
                display_name="Timeout (seconds)",
                input_type="int",
                required=False,
                description="Timeout for tool execution in seconds",
                default_value=30,
                min_value=1,
                max_value=300
            )
        ]
    
    def _get_outputs(self) -> List[ComponentOutput]:
        return [
            ComponentOutput(
                name="result",
                display_name="Tool Result",
                output_type="str",
                description="The result of tool execution"
            ),
            ComponentOutput(
                name="tool_metadata",
                display_name="Tool Metadata",
                output_type="dict",
                description="Metadata about the tool and execution"
            ),
            ComponentOutput(
                name="execution_time",
                display_name="Execution Time",
                output_type="float",
                description="Time taken to execute the tool in seconds"
            ),
            ComponentOutput(
                name="status",
                display_name="Execution Status",
                output_type="str",
                description="Status of tool execution (success/error)"
            ),
            ComponentOutput(
                name="error_message",
                display_name="Error Message",
                output_type="str",
                description="Error message if execution failed"
            )
        ]
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool component - runs the specified CrewGraph tool
        
        Args:
            inputs: Validated input parameters
            
        Returns:
            Dictionary containing tool execution results
        """
        import time
        start_time = time.time()
        
        try:
            # Extract inputs
            tool_name = inputs["tool_name"]
            tool_input = inputs["tool_input"]
            tool_config = inputs.get("tool_config", {})
            cache_results = inputs.get("cache_results", True)
            retry_on_failure = inputs.get("retry_on_failure", True)
            max_retries = inputs.get("max_retries", 3)
            timeout_seconds = inputs.get("timeout_seconds", 30)
            
            # Get tool from registry
            tool = self._get_tool(tool_name, tool_config)
            if not tool:
                raise ValueError(f"Tool '{tool_name}' not found")
            
            # Execute tool with retries
            result = None
            last_error = None
            attempts = 0
            
            while attempts <= max_retries:
                try:
                    # Execute tool
                    result = await self._execute_tool(
                        tool, 
                        tool_input, 
                        timeout_seconds
                    )
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    last_error = e
                    attempts += 1
                    
                    if not retry_on_failure or attempts > max_retries:
                        raise e
                    
                    self.logger.warning(f"Tool execution failed (attempt {attempts}): {e}")
                    time.sleep(0.5 * attempts)  # Exponential backoff
            
            execution_time = time.time() - start_time
            
            # Get tool metadata
            tool_metadata = self._get_tool_metadata(tool_name, tool)
            
            self.logger.info(f"Tool '{tool_name}' executed successfully in {execution_time:.2f}s")
            
            return {
                "result": result,
                "tool_metadata": tool_metadata,
                "execution_time": execution_time,
                "status": "success",
                "error_message": ""
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Tool execution failed: {e}")
            
            return {
                "result": "",
                "tool_metadata": {"tool_name": inputs.get("tool_name", "unknown")},
                "execution_time": execution_time,
                "status": "error",
                "error_message": str(e)
            }
    
    def _get_tool(self, tool_name: str, config: Dict[str, Any]) -> Optional[Any]:
        """Get tool instance from CrewGraph tool registry"""
        try:
            if ToolRegistry:
                # In real implementation, get from tool registry
                registry = ToolRegistry()
                return registry.get_tool(tool_name)
            elif BuiltinTools:
                # Get from builtin tools
                builtin = BuiltinTools()
                return getattr(builtin, tool_name, None)
            else:
                # Fallback mock tools
                return self._get_mock_tool(tool_name)
                
        except Exception as e:
            self.logger.error(f"Failed to get tool '{tool_name}': {e}")
            return None
    
    def _get_mock_tool(self, tool_name: str) -> Dict[str, Any]:
        """Create mock tool for development/testing"""
        mock_tools = {
            "text_processor": {
                "name": "text_processor",
                "description": "Process and analyze text",
                "execute": lambda x: f"Processed text: {x[:100]}..."
            },
            "file_handler": {
                "name": "file_handler", 
                "description": "Handle file operations",
                "execute": lambda x: f"File operation result for: {x}"
            },
            "data_converter": {
                "name": "data_converter",
                "description": "Convert data between formats",
                "execute": lambda x: f"Converted data: {x}"
            },
            "hash_generator": {
                "name": "hash_generator",
                "description": "Generate hash values",
                "execute": lambda x: f"hash_{hash(x) % 1000000}"
            },
            "timestamp_utility": {
                "name": "timestamp_utility",
                "description": "Work with timestamps",
                "execute": lambda x: f"Timestamp: {time.time()}"
            },
            "math_calculator": {
                "name": "math_calculator",
                "description": "Perform mathematical calculations",
                "execute": lambda x: f"Calculation result: {eval(x) if x.replace('.', '').replace('-', '').replace('+', '').replace('*', '').replace('/', '').replace('(', '').replace(')', '').replace(' ', '').isdigit() else 'Invalid expression'}"
            },
            "list_processor": {
                "name": "list_processor",
                "description": "Process lists and arrays",
                "execute": lambda x: f"Processed list with {len(x.split(','))} items"
            }
        }
        
        return mock_tools.get(tool_name, {
            "name": tool_name,
            "description": f"Mock tool: {tool_name}",
            "execute": lambda x: f"Mock result for {tool_name}: {x}"
        })
    
    async def _execute_tool(self, tool: Any, tool_input: str, timeout: int) -> str:
        """Execute the tool with the given input"""
        import asyncio
        
        try:
            # Handle different tool interfaces
            if isinstance(tool, dict) and "execute" in tool:
                # Mock tool
                if asyncio.iscoroutinefunction(tool["execute"]):
                    result = await asyncio.wait_for(tool["execute"](tool_input), timeout=timeout)
                else:
                    result = tool["execute"](tool_input)
            elif hasattr(tool, "run"):
                # CrewGraph tool with run method
                if asyncio.iscoroutinefunction(tool.run):
                    result = await asyncio.wait_for(tool.run(tool_input), timeout=timeout)
                else:
                    result = tool.run(tool_input)
            elif hasattr(tool, "execute"):
                # Tool with execute method
                if asyncio.iscoroutinefunction(tool.execute):
                    result = await asyncio.wait_for(tool.execute(tool_input), timeout=timeout)
                else:
                    result = tool.execute(tool_input)
            elif callable(tool):
                # Callable tool
                if asyncio.iscoroutinefunction(tool):
                    result = await asyncio.wait_for(tool(tool_input), timeout=timeout)
                else:
                    result = tool(tool_input)
            else:
                raise ValueError(f"Unknown tool interface: {type(tool)}")
            
            # Ensure result is a string
            if not isinstance(result, str):
                result = str(result)
            
            return result
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool execution timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Tool execution failed: {e}")
    
    def _get_tool_metadata(self, tool_name: str, tool: Any) -> Dict[str, Any]:
        """Get metadata about the tool"""
        metadata = {
            "tool_name": tool_name,
            "tool_type": "crewgraph",
            "version": "1.0.0"
        }
        
        if isinstance(tool, dict):
            metadata.update({
                "description": tool.get("description", ""),
                "category": tool.get("category", "utility")
            })
        elif hasattr(tool, "__dict__"):
            metadata.update({
                "description": getattr(tool, "description", ""),
                "category": getattr(tool, "category", "utility"),
                "tool_id": getattr(tool, "id", ""),
                "created_by": getattr(tool, "created_by", "")
            })
        
        return metadata


class CrewGraphToolRegistryComponent(LangflowComponent):
    """
    Langflow component for browsing and discovering CrewGraph tools
    
    This component provides access to the CrewGraph tool registry for
    discovering available tools and their capabilities.
    """
    
    def _get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="CrewGraphToolRegistry",
            display_name="CrewGraph Tool Registry",
            description="Browse and discover available CrewGraph tools",
            category="CrewGraph AI",
            tags=["tools", "registry", "discovery", "crewgraph"],
            icon="📚",
        )
    
    def _get_inputs(self) -> List[ComponentInput]:
        return [
            ComponentInput(
                name="category_filter",
                display_name="Category Filter",
                input_type="str",
                required=False,
                description="Filter tools by category (leave empty for all)",
                options=["", "text", "file", "data", "math", "web", "communication"],
                default_value=""
            ),
            ComponentInput(
                name="search_query",
                display_name="Search Query",
                input_type="str",
                required=False,
                description="Search tools by name or description",
                default_value=""
            ),
            ComponentInput(
                name="include_custom",
                display_name="Include Custom Tools",
                input_type="bool",
                required=False,
                description="Include custom/user-defined tools in results",
                default_value=True
            )
        ]
    
    def _get_outputs(self) -> List[ComponentOutput]:
        return [
            ComponentOutput(
                name="available_tools",
                display_name="Available Tools",
                output_type="list",
                description="List of available tools matching the criteria"
            ),
            ComponentOutput(
                name="tool_count",
                display_name="Tool Count",
                output_type="int",
                description="Number of tools found"
            ),
            ComponentOutput(
                name="categories",
                display_name="Available Categories",
                output_type="list",
                description="List of available tool categories"
            ),
            ComponentOutput(
                name="registry_info",
                display_name="Registry Information",
                output_type="dict",
                description="Information about the tool registry"
            )
        ]
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool registry component
        
        Args:
            inputs: Validated input parameters
            
        Returns:
            Dictionary containing tool registry information
        """
        try:
            category_filter = inputs.get("category_filter", "")
            search_query = inputs.get("search_query", "")
            include_custom = inputs.get("include_custom", True)
            
            # Get available tools
            if ToolRegistry:
                # Real implementation would query the tool registry
                available_tools = self._get_registry_tools(category_filter, search_query, include_custom)
            else:
                # Mock tools for development
                available_tools = self._get_mock_tools(category_filter, search_query)
            
            # Get categories
            categories = list(set(tool.get("category", "utility") for tool in available_tools))
            categories.sort()
            
            # Registry info
            registry_info = {
                "total_tools": len(available_tools),
                "builtin_tools": len([t for t in available_tools if t.get("type") == "builtin"]),
                "custom_tools": len([t for t in available_tools if t.get("type") == "custom"]),
                "last_updated": "2025-07-23T20:00:00Z"
            }
            
            return {
                "available_tools": available_tools,
                "tool_count": len(available_tools),
                "categories": categories,
                "registry_info": registry_info
            }
            
        except Exception as e:
            self.logger.error(f"Registry query failed: {e}")
            return {
                "available_tools": [],
                "tool_count": 0,
                "categories": [],
                "registry_info": {"error": str(e)}
            }
    
    def _get_registry_tools(self, category: str, search: str, include_custom: bool) -> List[Dict[str, Any]]:
        """Get tools from the actual registry"""
        # This would be implemented to query the real ToolRegistry
        return self._get_mock_tools(category, search)
    
    def _get_mock_tools(self, category: str, search: str) -> List[Dict[str, Any]]:
        """Get mock tools for development"""
        all_tools = [
            {
                "name": "text_processor",
                "display_name": "Text Processor",
                "description": "Process and analyze text content",
                "category": "text",
                "type": "builtin",
                "version": "1.0.0"
            },
            {
                "name": "file_handler",
                "display_name": "File Handler",
                "description": "Handle file operations like read, write, and convert",
                "category": "file",
                "type": "builtin",
                "version": "1.0.0"
            },
            {
                "name": "data_converter",
                "display_name": "Data Converter",
                "description": "Convert data between different formats",
                "category": "data",
                "type": "builtin",
                "version": "1.0.0"
            },
            {
                "name": "math_calculator",
                "display_name": "Math Calculator",
                "description": "Perform mathematical calculations and operations",
                "category": "math",
                "type": "builtin",
                "version": "1.0.0"
            },
            {
                "name": "web_scraper",
                "display_name": "Web Scraper",
                "description": "Extract data from web pages",
                "category": "web",
                "type": "custom",
                "version": "1.0.0"
            }
        ]
        
        # Apply filters
        filtered_tools = all_tools
        
        if category:
            filtered_tools = [t for t in filtered_tools if t["category"] == category]
        
        if search:
            search_lower = search.lower()
            filtered_tools = [
                t for t in filtered_tools
                if search_lower in t["name"].lower() or search_lower in t["description"].lower()
            ]
        
        return filtered_tools