"""
CrewGraph AI Tools Module - Complete Tool Management System
Comprehensive tool management with registry, discovery, validation, and built-in tools

Author: Vatsal216
Updated: 2025-07-22 12:40:39 UTC
Current User: Vatsal216
"""

from .discovery import DiscoveryConfig, ToolDefinition, ToolDiscovery, create_tool_discovery

# Import existing modules
from .registry import ToolRegistry, get_global_registry
from .validator import ToolValidator, ValidationConfig, ValidationResult, create_tool_validator

# Import missing modules that need to be created
try:
    from .base import BaseTool, ToolMetadata
except ImportError:
    # Will create these below
    BaseTool = None
    ToolMetadata = None

try:
    from .wrapper import ToolWrapper
except ImportError:
    # Will create this below
    ToolWrapper = None

try:
    from .builtin import BuiltinTools
except ImportError:
    # Will create this below
    BuiltinTools = None

__all__ = [
    # Base components
    "BaseTool",
    "ToolMetadata",
    "ToolWrapper",
    # Registry system
    "ToolRegistry",
    "get_global_registry",
    # Discovery system
    "ToolDiscovery",
    "ToolDefinition",
    "DiscoveryConfig",
    "create_tool_discovery",
    # Validation system
    "ToolValidator",
    "ValidationResult",
    "ValidationConfig",
    "create_tool_validator",
    # Built-in tools
    "BuiltinTools",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Vatsal216"
__updated__ = "2025-07-22 12:40:39"

print("🔧 CrewGraph AI Tools Module loaded")
print("📁 Includes: Base, Wrapper, Registry, Discovery, Validation & Built-in Tools")
print(f"👤 Updated by: {__author__}")
print(f"⏰ Time: {__updated__} UTC")

# Provide helpful information about missing components
if BaseTool is None:
    print("⚠️  BaseTool not found - will be created")
if ToolMetadata is None:
    print("⚠️  ToolMetadata not found - will be created")
if ToolWrapper is None:
    print("⚠️  ToolWrapper not found - will be created")
if BuiltinTools is None:
    print("⚠️  BuiltinTools not found - will be created")
