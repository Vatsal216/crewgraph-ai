"""
Enterprise Configuration Management for CrewGraph AI
Provides centralized, production-ready configuration management with validation,
environment-based loading, and enterprise security features.

Author: Vatsal216
Created: 2025-07-23
"""

import os
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from enum import Enum

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError, ValidationError

logger = get_logger(__name__)


class ConfigEnvironment(Enum):
    """Configuration environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider"""
    provider: LLMProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    organization_id: Optional[str] = None
    project_id: Optional[str] = None
    models: List[str] = field(default_factory=list)
    default_model: Optional[str] = None
    rate_limit_rpm: int = 60  # Requests per minute
    rate_limit_tpm: int = 1000  # Tokens per minute
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    enabled: bool = True
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate provider configuration"""
        issues = []
        
        if self.enabled and not self.api_key and self.provider != LLMProvider.LOCAL:
            issues.append(f"API key required for {self.provider.value} provider")
        
        if self.rate_limit_rpm <= 0:
            issues.append("Rate limit RPM must be positive")
            
        if self.rate_limit_tpm <= 0:
            issues.append("Rate limit TPM must be positive")
            
        if self.max_retries < 0:
            issues.append("Max retries cannot be negative")
            
        if self.timeout <= 0:
            issues.append("Timeout must be positive")
            
        return issues


@dataclass
class SecurityConfig:
    """Security configuration"""
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_expiration: int = 3600  # 1 hour
    api_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    audit_logging: bool = True
    secure_headers: bool = True
    cors_enabled: bool = False
    cors_origins: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate security configuration"""
        issues = []
        
        if self.encryption_enabled and not self.encryption_key:
            issues.append("Encryption key required when encryption is enabled")
            
        if self.jwt_expiration <= 0:
            issues.append("JWT expiration must be positive")
            
        if self.max_requests_per_minute <= 0:
            issues.append("Max requests per minute must be positive")
            
        return issues


@dataclass
class ScalingConfig:
    """Scaling and performance configuration"""
    max_concurrent_workflows: int = 10
    max_concurrent_agents: int = 50
    max_concurrent_tasks: int = 100
    auto_scaling_enabled: bool = False
    auto_scaling_target_cpu: float = 70.0  # Percentage
    auto_scaling_min_instances: int = 1
    auto_scaling_max_instances: int = 10
    memory_limit_mb: Optional[int] = None
    cpu_limit_cores: Optional[float] = None
    
    def validate(self) -> List[str]:
        """Validate scaling configuration"""
        issues = []
        
        if self.max_concurrent_workflows <= 0:
            issues.append("Max concurrent workflows must be positive")
            
        if self.max_concurrent_agents <= 0:
            issues.append("Max concurrent agents must be positive")
            
        if self.max_concurrent_tasks <= 0:
            issues.append("Max concurrent tasks must be positive")
            
        if not 0 < self.auto_scaling_target_cpu <= 100:
            issues.append("Auto scaling target CPU must be between 0 and 100")
            
        if self.auto_scaling_min_instances <= 0:
            issues.append("Auto scaling min instances must be positive")
            
        if self.auto_scaling_max_instances < self.auto_scaling_min_instances:
            issues.append("Auto scaling max instances must be >= min instances")
            
        return issues


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    metrics_enabled: bool = True
    metrics_port: int = 8080
    health_check_enabled: bool = True
    health_check_port: int = 8081
    logging_level: str = "INFO"
    structured_logging: bool = True
    trace_enabled: bool = False
    trace_sample_rate: float = 0.1
    performance_monitoring: bool = True
    error_reporting: bool = True
    alerting_enabled: bool = False
    alert_webhooks: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate monitoring configuration"""
        issues = []
        
        if not 1024 <= self.metrics_port <= 65535:
            issues.append("Metrics port must be between 1024 and 65535")
            
        if not 1024 <= self.health_check_port <= 65535:
            issues.append("Health check port must be between 1024 and 65535")
            
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging_level.upper() not in valid_log_levels:
            issues.append(f"Logging level must be one of: {', '.join(valid_log_levels)}")
            
        if not 0.0 <= self.trace_sample_rate <= 1.0:
            issues.append("Trace sample rate must be between 0.0 and 1.0")
            
        return issues


@dataclass
class EnterpriseConfig:
    """
    Enterprise-grade configuration for CrewGraph AI
    
    Provides comprehensive configuration management with:
    - Multiple LLM provider support
    - Environment-based configuration
    - Security and compliance features
    - Auto-scaling capabilities
    - Monitoring and observability
    - Validation and error checking
    """
    
    # Environment
    environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT
    
    # LLM Providers
    llm_providers: Dict[str, LLMProviderConfig] = field(default_factory=dict)
    default_provider: str = "openai"
    
    # Core settings
    workflow_timeout: float = 300.0  # 5 minutes
    task_timeout: float = 60.0      # 1 minute
    agent_timeout: float = 30.0     # 30 seconds
    
    # Memory configuration
    memory_backend: str = "dict"
    redis_url: Optional[str] = None
    redis_cluster_nodes: List[str] = field(default_factory=list)
    faiss_index_path: Optional[str] = None
    
    # Security
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Scaling
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    
    # Monitoring
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Internal state
    _validation_callbacks: List[Callable] = field(default_factory=list, init=False)
    _last_validated: Optional[float] = field(default=None, init=False)
    _config_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    
    def __post_init__(self):
        """Initialize default configurations"""
        if not self.llm_providers:
            self._setup_default_providers()
    
    def _setup_default_providers(self):
        """Setup default LLM provider configurations"""
        # OpenAI
        openai_config = LLMProviderConfig(
            provider=LLMProvider.OPENAI,
            api_key=os.getenv("OPENAI_API_KEY"),
            models=["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
            default_model="gpt-3.5-turbo",
            rate_limit_rpm=60,
            rate_limit_tpm=1000
        )
        if openai_config.api_key:
            self.llm_providers["openai"] = openai_config
        
        # Anthropic
        anthropic_config = LLMProviderConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            models=["claude-3-sonnet", "claude-3-haiku", "claude-3-opus"],
            default_model="claude-3-sonnet",
            rate_limit_rpm=50,
            rate_limit_tpm=800
        )
        if anthropic_config.api_key:
            self.llm_providers["anthropic"] = anthropic_config
        
        # Azure OpenAI
        azure_config = LLMProviderConfig(
            provider=LLMProvider.AZURE,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            models=["gpt-4", "gpt-35-turbo"],
            default_model="gpt-35-turbo",
            rate_limit_rpm=60,
            rate_limit_tpm=1000
        )
        if azure_config.api_key and azure_config.base_url:
            self.llm_providers["azure"] = azure_config
    
    @classmethod
    def from_env(cls, environment: Optional[ConfigEnvironment] = None) -> "EnterpriseConfig":
        """Create configuration from environment variables"""
        if environment is None:
            env_str = os.getenv("CREWGRAPH_ENVIRONMENT", "development").lower()
            environment = ConfigEnvironment(env_str)
        
        config = cls(environment=environment)
        
        # Load environment-specific settings
        config._load_from_environment()
        
        return config
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Core settings
        self.workflow_timeout = float(os.getenv("CREWGRAPH_WORKFLOW_TIMEOUT", "300"))
        self.task_timeout = float(os.getenv("CREWGRAPH_TASK_TIMEOUT", "60"))
        self.agent_timeout = float(os.getenv("CREWGRAPH_AGENT_TIMEOUT", "30"))
        
        # Memory settings
        self.memory_backend = os.getenv("CREWGRAPH_MEMORY_BACKEND", "dict")
        self.redis_url = os.getenv("REDIS_URL")
        
        if redis_cluster := os.getenv("REDIS_CLUSTER_NODES"):
            self.redis_cluster_nodes = redis_cluster.split(",")
        
        self.faiss_index_path = os.getenv("FAISS_INDEX_PATH")
        
        # Security settings
        self.security.encryption_enabled = os.getenv("CREWGRAPH_ENCRYPTION", "false").lower() == "true"
        self.security.encryption_key = os.getenv("CREWGRAPH_ENCRYPTION_KEY")
        self.security.jwt_secret = os.getenv("JWT_SECRET")
        self.security.api_rate_limiting = os.getenv("API_RATE_LIMITING", "true").lower() == "true"
        
        # Scaling settings
        self.scaling.max_concurrent_workflows = int(os.getenv("MAX_CONCURRENT_WORKFLOWS", "10"))
        self.scaling.max_concurrent_agents = int(os.getenv("MAX_CONCURRENT_AGENTS", "50"))
        self.scaling.max_concurrent_tasks = int(os.getenv("MAX_CONCURRENT_TASKS", "100"))
        self.scaling.auto_scaling_enabled = os.getenv("AUTO_SCALING_ENABLED", "false").lower() == "true"
        
        # Monitoring settings
        self.monitoring.metrics_enabled = os.getenv("METRICS_ENABLED", "true").lower() == "true"
        self.monitoring.logging_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.monitoring.trace_enabled = os.getenv("TRACE_ENABLED", "false").lower() == "true"
    
    @classmethod
    def from_file(cls, config_path: str) -> "EnterpriseConfig":
        """Load configuration from YAML or JSON file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required for YAML configuration files")
                data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError("Configuration file must be .yaml, .yml, or .json")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnterpriseConfig":
        """Create configuration from dictionary"""
        # Handle nested configurations
        if 'security' in data:
            data['security'] = SecurityConfig(**data['security'])
        
        if 'scaling' in data:
            data['scaling'] = ScalingConfig(**data['scaling'])
        
        if 'monitoring' in data:
            data['monitoring'] = MonitoringConfig(**data['monitoring'])
        
        if 'llm_providers' in data:
            providers = {}
            for name, provider_data in data['llm_providers'].items():
                if 'provider' in provider_data:
                    provider_data['provider'] = LLMProvider(provider_data['provider'])
                providers[name] = LLMProviderConfig(**provider_data)
            data['llm_providers'] = providers
        
        if 'environment' in data:
            data['environment'] = ConfigEnvironment(data['environment'])
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
                
            if isinstance(value, Enum):
                result[key] = value.value
            elif hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif isinstance(value, dict):
                if key == 'llm_providers':
                    result[key] = {
                        name: {
                            'provider': config.provider.value,
                            'api_key': config.api_key,
                            'base_url': config.base_url,
                            'api_version': config.api_version,
                            'organization_id': config.organization_id,
                            'project_id': config.project_id,
                            'models': config.models,
                            'default_model': config.default_model,
                            'rate_limit_rpm': config.rate_limit_rpm,
                            'rate_limit_tpm': config.rate_limit_tpm,
                            'max_retries': config.max_retries,
                            'retry_delay': config.retry_delay,
                            'timeout': config.timeout,
                            'enabled': config.enabled,
                            'custom_headers': config.custom_headers
                        }
                        for name, config in value.items()
                    }
                else:
                    result[key] = value
            elif hasattr(value, '__dict__'):
                # Handle dataclass instances
                result[key] = {
                    k: v.value if isinstance(v, Enum) else v
                    for k, v in value.__dict__.items()
                    if not k.startswith('_')
                }
            else:
                result[key] = value
        
        return result
    
    def save_to_file(self, config_path: str, format: str = "yaml"):
        """Save configuration to file"""
        path = Path(config_path)
        data = self.to_dict()
        
        with open(path, 'w') as f:
            if format.lower() == "yaml":
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required for YAML format")
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError("Format must be 'yaml' or 'json'")
        
        logger.info(f"Configuration saved to {config_path}")
    
    def validate(self) -> List[str]:
        """Validate entire configuration"""
        with self._config_lock:
            issues = []
            
            # Validate LLM providers
            if not self.llm_providers:
                issues.append("No LLM providers configured")
            else:
                for name, provider in self.llm_providers.items():
                    provider_issues = provider.validate()
                    issues.extend([f"Provider {name}: {issue}" for issue in provider_issues])
            
            # Validate default provider
            if self.default_provider not in self.llm_providers:
                issues.append(f"Default provider '{self.default_provider}' not configured")
            
            # Validate timeouts
            if self.workflow_timeout <= 0:
                issues.append("Workflow timeout must be positive")
            
            if self.task_timeout <= 0:
                issues.append("Task timeout must be positive")
            
            if self.agent_timeout <= 0:
                issues.append("Agent timeout must be positive")
            
            # Validate memory backend
            valid_backends = ["dict", "redis", "faiss", "sql"]
            if self.memory_backend not in valid_backends:
                issues.append(f"Invalid memory backend. Must be one of: {valid_backends}")
            
            if self.memory_backend == "redis" and not self.redis_url and not self.redis_cluster_nodes:
                issues.append("Redis URL or cluster nodes required for Redis memory backend")
            
            if self.memory_backend == "faiss" and not self.faiss_index_path:
                issues.append("FAISS index path required for FAISS memory backend")
            
            # Validate nested configurations
            issues.extend(self.security.validate())
            issues.extend(self.scaling.validate())
            issues.extend(self.monitoring.validate())
            
            # Run custom validation callbacks
            for callback in self._validation_callbacks:
                try:
                    custom_issues = callback(self)
                    if custom_issues:
                        issues.extend(custom_issues)
                except Exception as e:
                    issues.append(f"Validation callback error: {e}")
            
            self._last_validated = time.time()
            
            return issues
    
    def add_validation_callback(self, callback: Callable):
        """Add custom validation callback"""
        self._validation_callbacks.append(callback)
    
    def get_provider_config(self, provider_name: Optional[str] = None) -> Optional[LLMProviderConfig]:
        """Get configuration for a specific provider"""
        if provider_name is None:
            provider_name = self.default_provider
        
        return self.llm_providers.get(provider_name)
    
    def add_provider(self, name: str, config: LLMProviderConfig):
        """Add or update LLM provider configuration"""
        with self._config_lock:
            self.llm_providers[name] = config
            logger.info(f"Added LLM provider: {name}")
    
    def remove_provider(self, name: str):
        """Remove LLM provider configuration"""
        with self._config_lock:
            if name in self.llm_providers:
                del self.llm_providers[name]
                logger.info(f"Removed LLM provider: {name}")
                
                # Update default provider if necessary
                if self.default_provider == name and self.llm_providers:
                    self.default_provider = next(iter(self.llm_providers.keys()))
                    logger.info(f"Updated default provider to: {self.default_provider}")
    
    def create_env_template(self, path: str = ".env.template"):
        """Create environment variable template"""
        template = f"""# CrewGraph AI Enterprise Configuration
# Environment: {self.environment.value}

# Core Settings
CREWGRAPH_ENVIRONMENT={self.environment.value}
CREWGRAPH_WORKFLOW_TIMEOUT={self.workflow_timeout}
CREWGRAPH_TASK_TIMEOUT={self.task_timeout}
CREWGRAPH_AGENT_TIMEOUT={self.agent_timeout}

# LLM Provider API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here
AZURE_OPENAI_ENDPOINT=your-azure-endpoint-here
AZURE_OPENAI_API_VERSION=2023-05-15

# Memory Backend
CREWGRAPH_MEMORY_BACKEND={self.memory_backend}
REDIS_URL=redis://localhost:6379
REDIS_CLUSTER_NODES=node1:6379,node2:6379,node3:6379
FAISS_INDEX_PATH=/path/to/faiss/index

# Security
CREWGRAPH_ENCRYPTION={str(self.security.encryption_enabled).lower()}
CREWGRAPH_ENCRYPTION_KEY=your-encryption-key-here
JWT_SECRET=your-jwt-secret-here
API_RATE_LIMITING={str(self.security.api_rate_limiting).lower()}

# Scaling
MAX_CONCURRENT_WORKFLOWS={self.scaling.max_concurrent_workflows}
MAX_CONCURRENT_AGENTS={self.scaling.max_concurrent_agents}
MAX_CONCURRENT_TASKS={self.scaling.max_concurrent_tasks}
AUTO_SCALING_ENABLED={str(self.scaling.auto_scaling_enabled).lower()}

# Monitoring
METRICS_ENABLED={str(self.monitoring.metrics_enabled).lower()}
LOG_LEVEL={self.monitoring.logging_level}
TRACE_ENABLED={str(self.monitoring.trace_enabled).lower()}
"""
        
        with open(path, 'w') as f:
            f.write(template)
        
        logger.info(f"Environment template created: {path}")


# Global configuration instance
_global_enterprise_config: Optional[EnterpriseConfig] = None
_config_lock = threading.Lock()


def get_enterprise_config() -> EnterpriseConfig:
    """Get the global enterprise configuration"""
    global _global_enterprise_config
    
    with _config_lock:
        if _global_enterprise_config is None:
            _global_enterprise_config = EnterpriseConfig.from_env()
        
        return _global_enterprise_config


def configure_enterprise(config: EnterpriseConfig):
    """Set the global enterprise configuration"""
    global _global_enterprise_config
    
    with _config_lock:
        _global_enterprise_config = config
        logger.info(f"Enterprise configuration set for environment: {config.environment.value}")


def validate_enterprise_config(config: Optional[EnterpriseConfig] = None) -> bool:
    """Validate enterprise configuration and return True if valid"""
    if config is None:
        config = get_enterprise_config()
    
    issues = config.validate()
    
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("Configuration validation successful")
    return True


def enterprise_setup_wizard() -> EnterpriseConfig:
    """Interactive setup wizard for enterprise configuration"""
    print("🏢 CrewGraph AI Enterprise Setup Wizard")
    print("=" * 50)
    
    # Environment selection
    print("\n1. Select Environment:")
    environments = list(ConfigEnvironment)
    for i, env in enumerate(environments, 1):
        print(f"   {i}. {env.value.title()}")
    
    while True:
        try:
            choice = int(input("Choose environment (1-4): "))
            if 1 <= choice <= len(environments):
                environment = environments[choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    config = EnterpriseConfig.from_env(environment)
    
    # Validate current configuration
    issues = config.validate()
    if issues:
        print(f"\n⚠️  Found {len(issues)} configuration issues:")
        for issue in issues:
            print(f"   - {issue}")
        
        if input("\n📝 Create environment template? (y/n): ").lower() == 'y':
            config.create_env_template()
            print("✅ Template created! Please update with your settings and restart.")
    else:
        print("\n✅ Configuration is valid!")
    
    return config


if __name__ == "__main__":
    # Run setup wizard when executed directly
    enterprise_setup_wizard()