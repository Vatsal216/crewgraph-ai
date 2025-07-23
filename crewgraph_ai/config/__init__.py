"""
CrewGraph AI Configuration Package
Enhanced enterprise-grade configuration management
"""

from datetime import datetime
import time
import os

# Utility functions for dynamic values (replaces hardcoded values)
def get_current_user() -> str:
    """Get the current system user from configuration"""
    return os.getenv("CREWGRAPH_SYSTEM_USER", "crewgraph_system")


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()


def get_formatted_timestamp() -> str:
    """Get current timestamp in the format used throughout the codebase"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_unix_timestamp() -> float:
    """Get current Unix timestamp"""
    return time.time()

# Basic configuration class (simplified version)
class CrewGraphSettings:
    """Basic configuration for CrewGraph AI"""
    
    def __init__(self):
        self.system_user = get_current_user()
        self.environment = os.getenv("CREWGRAPH_ENVIRONMENT", "production")
    
    @classmethod
    def from_env(cls):
        return cls()

def get_settings():
    """Get basic settings"""
    return CrewGraphSettings.from_env()

def configure(settings):
    """Configure settings"""
    pass

def quick_setup():
    """Quick setup"""
    return get_settings()

__all__ = [
    "CrewGraphSettings",
    "get_settings",
    "configure",
    "quick_setup",
    "get_current_user",
    "get_current_timestamp",
    "get_formatted_timestamp",
    "get_unix_timestamp",
]