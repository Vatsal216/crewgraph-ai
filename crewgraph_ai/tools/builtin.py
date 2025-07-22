"""
Built-in Tools Collection for CrewGraph AI
Ready-to-use tools for common workflow tasks

Author: Vatsal216
Created: 2025-07-22 12:40:39 UTC
"""

import os
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from crewai.tools import BaseTool as CrewAIBaseTool
from .base import ToolMetadata, ToolCategory, ToolStatus
from .wrapper import ToolWrapper, tool_decorator
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BuiltinTools:
    """
    Collection of built-in tools for CrewGraph AI workflows.
    
    Provides a comprehensive set of ready-to-use tools that cover
    common workflow operations including file handling, data processing,
    web operations, and utility functions.
    
    Created by: Vatsal216
    Date: 2025-07-22 12:40:39 UTC
    """
    
    @staticmethod
    @tool_decorator(
        name="text_processor",
        description="Process and transform text data with various operations",
        category=ToolCategory.DATA_PROCESSING
    )
    def text_processor(text: str, operation: str = "clean", **options) -> str:
        """
        Process text with various operations.
        
        Args:
            text: Input text to process
            operation: Operation type (clean, upper, lower, title, reverse, etc.)
            **options: Additional processing options
            
        Returns:
            Processed text
        """
        logger.info(f"Processing text with operation: {operation}")
        
        if operation == "clean":
            # Remove extra whitespace and normalize
            result = " ".join(text.split())
        elif operation == "upper":
            result = text.upper()
        elif operation == "lower":
            result = text.lower()
        elif operation == "title":
            result = text.title()
        elif operation == "reverse":
            result = text[::-1]
        elif operation == "count_words":
            result = str(len(text.split()))
        elif operation == "count_chars":
            result = str(len(text))
        elif operation == "extract_numbers":
            import re
            numbers = re.findall(r'\d+', text)
            result = " ".join(numbers)
        else:
            result = text
        
        return result
    
    @staticmethod
    @tool_decorator(
        name="file_handler",
        description="Handle file operations including read, write, and management",
        category=ToolCategory.FILE_OPERATIONS
    )
    def file_handler(filepath: str, operation: str = "read", content: str = "", **options) -> str:
        """
        Handle various file operations.
        
        Args:
            filepath: Path to the file
            operation: Operation type (read, write, append, exists, delete, etc.)
            content: Content to write (for write/append operations)
            **options: Additional options
            
        Returns:
            Operation result or file content
        """
        logger.info(f"File operation: {operation} on {filepath}")
        
        try:
            if operation == "read":
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif operation == "write":
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Successfully wrote {len(content)} characters to {filepath}"
            
            elif operation == "append":
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write(content)
                return f"Successfully appended {len(content)} characters to {filepath}"
            
            elif operation == "exists":
                return str(os.path.exists(filepath))
            
            elif operation == "delete":
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return f"Successfully deleted {filepath}"
                else:
                    return f"File {filepath} does not exist"
            
            elif operation == "info":
                if os.path.exists(filepath):
                    stat = os.stat(filepath)
                    return json.dumps({
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "created": stat.st_ctime,
                        "is_file": os.path.isfile(filepath),
                        "is_dir": os.path.isdir(filepath)
                    }, indent=2)
                else:
                    return f"File {filepath} does not exist"
            
            else:
                return f"Unknown operation: {operation}"
        
        except Exception as e:
            return f"File operation failed: {str(e)}"
    
    @staticmethod  
    @tool_decorator(
        name="data_converter",
        description="Convert data between different formats (JSON, CSV, etc.)",
        category=ToolCategory.DATA_PROCESSING
    )
    def data_converter(data: str, input_format: str = "json", output_format: str = "json") -> str:
        """
        Convert data between different formats.
        
        Args:
            data: Input data string
            input_format: Input format (json, csv, yaml, etc.)
            output_format: Output format (json, csv, yaml, etc.)
            
        Returns:
            Converted data string
        """
        logger.info(f"Converting data from {input_format} to {output_format}")
        
        try:
            # Parse input data
            parsed_data = None
            
            if input_format.lower() == "json":
                parsed_data = json.loads(data)
            elif input_format.lower() == "csv":
                # Simple CSV parsing
                lines = data.strip().split('\n')
                if len(lines) > 1:
                    headers = [h.strip() for h in lines[0].split(',')]
                    rows = []
                    for line in lines[1:]:
                        values = [v.strip() for v in line.split(',')]
                        row_dict = dict(zip(headers, values))
                        rows.append(row_dict)
                    parsed_data = rows
                else:
                    parsed_data = []
            else:
                # Treat as plain text
                parsed_data = {"text": data}
            
            # Convert to output format
            if output_format.lower() == "json":
                return json.dumps(parsed_data, indent=2)
            elif output_format.lower() == "csv":
                if isinstance(parsed_data, list) and parsed_data:
                    # Convert list of dicts to CSV
                    if isinstance(parsed_data[0], dict):
                        headers = list(parsed_data[0].keys())
                        csv_lines = [','.join(headers)]
                        for item in parsed_data:
                            row = [str(item.get(h, '')) for h in headers]
                            csv_lines.append(','.join(row))
                        return '\n'.join(csv_lines)
                    else:
                        return ','.join(str(item) for item in parsed_data)
                else:
                    return str(parsed_data)
            else:
                return str(parsed_data)
        
        except Exception as e:
            return f"Data conversion failed: {str(e)}"
    
    @staticmethod
    @tool_decorator(
        name="hash_generator", 
        description="Generate various hash values for data integrity and security",
        category=ToolCategory.SECURITY
    )
    def hash_generator(data: str, algorithm: str = "md5") -> str:
        """
        Generate hash values using various algorithms.
        
        Args:
            data: Input data to hash
            algorithm: Hash algorithm (md5, sha1, sha256, sha512)
            
        Returns:
            Generated hash value
        """
        logger.info(f"Generating {algorithm} hash")
        
        try:
            data_bytes = data.encode('utf-8')
            
            if algorithm.lower() == "md5":
                return hashlib.md5(data_bytes).hexdigest()
            elif algorithm.lower() == "sha1":
                return hashlib.sha1(data_bytes).hexdigest()
            elif algorithm.lower() == "sha256":
                return hashlib.sha256(data_bytes).hexdigest()
            elif algorithm.lower() == "sha512":
                return hashlib.sha512(data_bytes).hexdigest()
            else:
                return f"Unsupported algorithm: {algorithm}"
        
        except Exception as e:
            return f"Hash generation failed: {str(e)}"
    
    @staticmethod
    @tool_decorator(
        name="timestamp_utility",
        description="Work with timestamps and date/time formatting",
        category=ToolCategory.GENERAL
    )
    def timestamp_utility(operation: str = "current", timestamp: Optional[float] = None, 
                         format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Utility for working with timestamps.
        
        Args:
            operation: Operation type (current, format, parse, add, subtract)
            timestamp: Unix timestamp (for format operations)
            format_string: Format string for datetime formatting
            
        Returns:
            Formatted timestamp or operation result
        """
        import datetime
        
        logger.info(f"Timestamp operation: {operation}")
        
        try:
            if operation == "current":
                return str(time.time())
            
            elif operation == "current_formatted":
                return datetime.datetime.now().strftime(format_string)
            
            elif operation == "format":
                if timestamp is not None:
                    dt = datetime.datetime.fromtimestamp(timestamp)
                    return dt.strftime(format_string)
                else:
                    return "Timestamp required for format operation"
            
            elif operation == "parse":
                # Parse formatted string back to timestamp
                if timestamp:  # Using timestamp parameter as string input
                    dt = datetime.datetime.strptime(str(timestamp), format_string)
                    return str(dt.timestamp())
                else:
                    return "String timestamp required for parse operation"
            
            else:
                return f"Unknown operation: {operation}"
        
        except Exception as e:
            return f"Timestamp operation failed: {str(e)}"
    
    @staticmethod
    @tool_decorator(
        name="math_calculator",
        description="Perform mathematical calculations and operations",
        category=ToolCategory.ANALYSIS
    )
    def math_calculator(expression: str, operation: str = "evaluate") -> str:
        """
        Perform mathematical calculations.
        
        Args:
            expression: Mathematical expression to evaluate
            operation: Operation type (evaluate, validate, etc.)
            
        Returns:
            Calculation result or operation output
        """
        logger.info(f"Math operation: {operation} on '{expression}'")
        
        try:
            if operation == "evaluate":
                # Safe evaluation of mathematical expressions
                allowed_chars = "0123456789+-*/().e "
                if not all(c in allowed_chars for c in expression):
                    return "Error: Expression contains invalid characters"
                
                # Use eval safely (only with verified safe expressions)
                result = eval(expression)
                return str(result)
            
            elif operation == "validate":
                allowed_chars = "0123456789+-*/().e "
                is_valid = all(c in allowed_chars for c in expression)
                return str(is_valid)
            
            else:
                return f"Unknown operation: {operation}"
        
        except Exception as e:
            return f"Math calculation failed: {str(e)}"
    
    @staticmethod
    @tool_decorator(
        name="list_processor",
        description="Process and manipulate list data structures",
        category=ToolCategory.DATA_PROCESSING
    )
    def list_processor(data: str, operation: str = "parse", separator: str = ",") -> str:
        """
        Process list data with various operations.
        
        Args:
            data: Input data (JSON list or separated values)
            operation: Operation type (parse, sort, unique, filter, etc.)
            separator: Separator for parsing string data
            
        Returns:
            Processed list data
        """
        logger.info(f"List operation: {operation}")
        
        try:
            # Parse input data
            if data.strip().startswith('['):
                # JSON list
                items = json.loads(data)
            else:
                # Separated values
                items = [item.strip() for item in data.split(separator)]
            
            # Perform operation
            if operation == "parse":
                result = items
            elif operation == "sort":
                result = sorted(items)
            elif operation == "reverse":
                result = list(reversed(items))
            elif operation == "unique":
                result = list(set(items))
            elif operation == "count":
                return str(len(items))
            elif operation == "filter_empty":
                result = [item for item in items if item.strip()]
            elif operation == "to_json":
                return json.dumps(items, indent=2)
            else:
                result = items
            
            # Return result
            if operation == "to_json":
                return json.dumps(result, indent=2)
            else:
                return separator.join(str(item) for item in result)
        
        except Exception as e:
            return f"List processing failed: {str(e)}"
    
    @classmethod
    def get_all_tools(cls) -> Dict[str, ToolWrapper]:
        """Get all built-in tools as ToolWrapper instances"""
        tools = {}
        
        # Get all methods that are decorated tools
        for method_name in dir(cls):
            method = getattr(cls, method_name)
            if isinstance(method, ToolWrapper):
                tools[method.metadata.name] = method
        
        return tools
    
    @classmethod
    def get_tool(cls, name: str) -> Optional[ToolWrapper]:
        """Get a specific built-in tool by name"""
        tools = cls.get_all_tools()
        return tools.get(name)
    
    @classmethod
    def list_tools(cls) -> List[str]:
        """List all available built-in tool names"""
        return list(cls.get_all_tools().keys())
    
    @classmethod
    def get_tools_by_category(cls, category: ToolCategory) -> Dict[str, ToolWrapper]:
        """Get tools filtered by category"""
        all_tools = cls.get_all_tools()
        return {
            name: tool for name, tool in all_tools.items()
            if tool.metadata.category == category
        }
    
    @classmethod
    def get_tool_info(cls) -> Dict[str, Any]:
        """Get comprehensive information about all built-in tools"""
        tools = cls.get_all_tools()
        
        info = {
            'total_tools': len(tools),
            'categories': {},
            'tools': {},
            'created_by': 'Vatsal216',
            'created_at': '2025-07-22 12:40:39'
        }
        
        # Categorize tools
        for name, tool in tools.items():
            category = tool.metadata.category.value
            if category not in info['categories']:
                info['categories'][category] = []
            info['categories'][category].append(name)
            
            # Add tool details
            info['tools'][name] = {
                'description': tool.metadata.description,
                'category': category,
                'status': tool.metadata.status.value,
                'input_types': tool.metadata.input_types,
                'output_types': tool.metadata.output_types
            }
        
        return info


# Convenience function to get all built-in tools
def get_builtin_tools() -> Dict[str, ToolWrapper]:
    """Get all built-in tools"""
    return BuiltinTools.get_all_tools()


# Convenience function to get a specific tool
def get_builtin_tool(name: str) -> Optional[ToolWrapper]:
    """Get specific built-in tool by name"""
    return BuiltinTools.get_tool(name)