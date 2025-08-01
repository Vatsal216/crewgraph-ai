"""
CrewGraph AI Configuration Management
Provides centralized configuration for API keys, models, and providers
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime
import time


@dataclass
class ProviderConfig:
    """Configuration for an AI provider"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: List[str] = field(default_factory=list)
    default_model: Optional[str] = None
    enabled: bool = True


@dataclass
class CrewGraphSettings:
    """
    Centralized configuration for CrewGraph AI
    
    Example:
        ```python
        from crewgraph_ai.config import CrewGraphSettings
        
        # Configure with environment variables
        settings = CrewGraphSettings.from_env()
        
        # Or configure manually
        settings = CrewGraphSettings(
            openai_api_key="your-key",
            default_model="gpt-4"
        )
        
        # Apply to workflow
        workflow = CrewGraph("my_workflow")
        workflow.configure(settings)
        ```
    """
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_org_id: Optional[str] = None
    openai_base_url: Optional[str] = None
    
    # Anthropic Configuration  
    anthropic_api_key: Optional[str] = None
    
    # Default Settings
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    
    # Advanced Settings
    max_retries: int = 3
    timeout: int = 30
    temperature: float = 0.7
    
    # Logging
    log_level: str = "INFO"
    enable_debug: bool = False
    
    # Memory
    default_memory_backend: str = "dict"
    redis_url: Optional[str] = None
    
    # Security
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    
    # User/System Identification (replaces hardcoded values)
    system_user: str = "crewgraph_system"
    organization: Optional[str] = None
    environment: str = "production"
    
    @classmethod
    def from_env(cls) -> "CrewGraphSettings":
        """Create configuration from environment variables"""
        return cls(
            # OpenAI
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_org_id=os.getenv("OPENAI_ORG_ID"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            
            # Anthropic
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            
            # Defaults
            default_model=os.getenv("CREWGRAPH_DEFAULT_MODEL", "gpt-3.5-turbo"),
            default_provider=os.getenv("CREWGRAPH_DEFAULT_PROVIDER", "openai"),
            
            # Advanced
            max_retries=int(os.getenv("CREWGRAPH_MAX_RETRIES", "3")),
            timeout=int(os.getenv("CREWGRAPH_TIMEOUT", "30")),
            temperature=float(os.getenv("CREWGRAPH_TEMPERATURE", "0.7")),
            
            # Logging
            log_level=os.getenv("CREWGRAPH_LOG_LEVEL", "INFO"),
            enable_debug=os.getenv("CREWGRAPH_DEBUG", "false").lower() == "true",
            
            # Memory
            default_memory_backend=os.getenv("CREWGRAPH_MEMORY_BACKEND", "dict"),
            redis_url=os.getenv("REDIS_URL"),
            
            # Security
            enable_encryption=os.getenv("CREWGRAPH_ENCRYPTION", "false").lower() == "true",
            encryption_key=os.getenv("CREWGRAPH_ENCRYPTION_KEY"),
            
            # User/System Identification
            system_user=os.getenv("CREWGRAPH_SYSTEM_USER", "crewgraph_system"),
            organization=os.getenv("CREWGRAPH_ORGANIZATION"),
            environment=os.getenv("CREWGRAPH_ENVIRONMENT", "production"),
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> "CrewGraphSettings":
        """Load configuration from YAML or JSON file"""
        import json
        from pathlib import Path
        
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(path) as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")
        elif path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
        
        return cls(**data)
    
    def configure_litellm(self):
        """Configure LiteLLM with the current settings"""
        try:
            import litellm
            import os
            
            # Set API keys in environment for LiteLLM
            if self.openai_api_key:
                os.environ["OPENAI_API_KEY"] = self.openai_api_key
                
            if self.openai_org_id:
                os.environ["OPENAI_ORG_ID"] = self.openai_org_id
                
            if self.anthropic_api_key:
                os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
            
            # Configure LiteLLM settings
            litellm.set_verbose = self.enable_debug
            
            return True
            
        except ImportError:
            return False
    
    def get_provider_config(self, provider: str) -> ProviderConfig:
        """Get configuration for a specific provider"""
        if provider == "openai":
            return ProviderConfig(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                models=["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
                default_model=self.default_model if self.default_provider == "openai" else "gpt-3.5-turbo",
                enabled=bool(self.openai_api_key)
            )
        elif provider == "anthropic":
            return ProviderConfig(
                api_key=self.anthropic_api_key,
                models=["claude-3-sonnet", "claude-3-haiku"],
                default_model="claude-3-sonnet",
                enabled=bool(self.anthropic_api_key)
            )
        else:
            return ProviderConfig(enabled=False)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check if at least one provider is configured
        if not self.openai_api_key and not self.anthropic_api_key:
            issues.append("No AI provider API keys configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        
        # Validate model selection
        provider_config = self.get_provider_config(self.default_provider)
        if not provider_config.enabled:
            issues.append(f"Default provider '{self.default_provider}' is not configured")
        
        # Validate memory backend
        if self.default_memory_backend == "redis" and not self.redis_url:
            issues.append("Redis memory backend selected but REDIS_URL not configured")
        
        # Validate numeric ranges
        if self.max_retries < 0:
            issues.append("max_retries must be non-negative")
        
        if self.timeout <= 0:
            issues.append("timeout must be positive")
            
        if not 0.0 <= self.temperature <= 2.0:
            issues.append("temperature must be between 0.0 and 2.0")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            issues.append(f"log_level must be one of: {', '.join(valid_log_levels)}")
        
        # Validate provider-specific settings
        if self.openai_api_key:
            if not self.openai_api_key.startswith(('sk-', 'org-')):
                issues.append("OpenAI API key format appears invalid")
        
        return issues
    
    def create_env_file(self, path: str = ".env") -> None:
        """Create a .env file template with current settings"""
        env_content = f"""# CrewGraph AI Configuration
# Copy this file to .env and fill in your API keys

# OpenAI Configuration
OPENAI_API_KEY={self.openai_api_key or "your-openai-api-key-here"}
OPENAI_ORG_ID={self.openai_org_id or "your-org-id-here"}

# Anthropic Configuration  
ANTHROPIC_API_KEY={self.anthropic_api_key or "your-anthropic-api-key-here"}

# Default Settings
CREWGRAPH_DEFAULT_MODEL={self.default_model}
CREWGRAPH_DEFAULT_PROVIDER={self.default_provider}

# Advanced Settings
CREWGRAPH_MAX_RETRIES={self.max_retries}
CREWGRAPH_TIMEOUT={self.timeout}
CREWGRAPH_TEMPERATURE={self.temperature}

# Logging
CREWGRAPH_LOG_LEVEL={self.log_level}
CREWGRAPH_DEBUG={str(self.enable_debug).lower()}

# Memory Backend
CREWGRAPH_MEMORY_BACKEND={self.default_memory_backend}
REDIS_URL={self.redis_url or "redis://localhost:6379"}

# Security (optional)
CREWGRAPH_ENCRYPTION={str(self.enable_encryption).lower()}
CREWGRAPH_ENCRYPTION_KEY={self.encryption_key or "your-encryption-key-here"}
"""
        
        with open(path, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Configuration template created: {path}")
        print("📝 Please edit the file and add your API keys")


# Global configuration instance
_global_settings: Optional[CrewGraphSettings] = None


def get_settings() -> CrewGraphSettings:
    """Get the global CrewGraph settings"""
    global _global_settings
    if _global_settings is None:
        _global_settings = CrewGraphSettings.from_env()
    return _global_settings


def configure(settings: CrewGraphSettings) -> None:
    """Set the global CrewGraph settings"""
    global _global_settings
    _global_settings = settings
    
    # Apply configuration
    settings.configure_litellm()


def quick_setup() -> CrewGraphSettings:
    """
    Quick setup wizard for first-time users
    
    Returns configured settings
    """
    print("🚀 CrewGraph AI Quick Setup")
    print("=" * 40)
    
    settings = CrewGraphSettings.from_env()
    issues = settings.validate()
    
    if issues:
        print("⚠️  Configuration Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        
        # Offer to create .env file
        create_env = input("📁 Create .env file template? (y/n): ").lower() == 'y'
        if create_env:
            settings.create_env_file()
            print("\n✅ Setup complete! Edit .env file with your API keys and restart.")
            return settings
    else:
        print("✅ Configuration looks good!")
        
    configure(settings)
    return settings


def quick_setup() -> CrewGraphSettings:
    """
    Quick setup wizard for first-time users
    
    Returns configured settings
    """
    print("🚀 CrewGraph AI Quick Setup")
    print("=" * 40)
    
    settings = CrewGraphSettings.from_env()
    issues = settings.validate()
    
    if issues:
        print("⚠️  Configuration Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        
        # Offer to create .env file
        create_env = input("📁 Create .env file template? (y/n): ").lower() == 'y'
        if create_env:
            settings.create_env_file()
            print("\n✅ Setup complete! Edit .env file with your API keys and restart.")
            return settings
    else:
        print("✅ Configuration looks good!")
        
    configure(settings)
    return settings


def validate_configuration(settings: CrewGraphSettings = None) -> bool:
    """
    Validate current configuration and print detailed report
    
    Args:
        settings: Settings to validate (uses global if None)
        
    Returns:
        True if configuration is valid
    """
    if settings is None:
        settings = get_settings()
    
    print("🔍 Configuration Validation Report")
    print("=" * 50)
    
    issues = settings.validate()
    
    if not issues:
        print("✅ Configuration is valid and ready to use!")
        print("\n📋 Current Configuration:")
        print(f"  Default Provider: {settings.default_provider}")
        print(f"  Default Model: {settings.default_model}")
        print(f"  Memory Backend: {settings.default_memory_backend}")
        print(f"  Log Level: {settings.log_level}")
        print(f"  Max Retries: {settings.max_retries}")
        print(f"  Timeout: {settings.timeout}s")
        return True
    
    print("❌ Configuration Issues Found:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print(f"\n📝 Found {len(issues)} issue(s) that need attention.")
    print("💡 Run quick_setup() to fix these issues interactively.")
    
    return False


def load_config_file(file_path: str) -> CrewGraphSettings:
    """
    Load configuration from file with enhanced error handling
    
    Args:
        file_path: Path to configuration file (.yaml, .yml, or .json)
        
    Returns:
        CrewGraphSettings instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
        ValidationError: If loaded config is invalid
    """
    try:
        settings = CrewGraphSettings.from_file(file_path)
        
        # Validate loaded configuration
        issues = settings.validate()
        if issues:
            error_msg = f"Configuration validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues)
            raise ValueError(error_msg)
        
        print(f"✅ Configuration loaded successfully from {file_path}")
        return settings
        
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {file_path}")
        raise
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        raise


if __name__ == "__main__":
    # Run quick setup when script is executed directly
    quick_setup()


# Utility functions for dynamic values (replaces hardcoded values)
def get_current_user() -> str:
    """Get the current system user from configuration"""
    settings = get_settings()
    return settings.system_user


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()


def get_formatted_timestamp() -> str:
    """Get current timestamp in the format used throughout the codebase"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_unix_timestamp() -> float:
    """Get current Unix timestamp"""
    return time.time()