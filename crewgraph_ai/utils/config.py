"""
Configuration management for CrewGraph AI
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .exceptions import ConfigurationError
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""

    host: str = "localhost"
    port: int = 5432
    database: str = "crewgraph"
    username: str = "crewgraph"
    password: str = ""
    ssl_mode: str = "prefer"
    connection_pool_size: int = 10
    connection_timeout: int = 30


@dataclass
class RedisConfig:
    """Redis configuration"""

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    database: int = 0
    connection_pool_size: int = 10
    socket_timeout: int = 30
    ssl: bool = False


@dataclass
class SecurityConfig:
    """Security configuration"""

    enable_encryption: bool = True
    encryption_key: str = ""
    enable_authentication: bool = True
    jwt_secret: str = ""
    jwt_expiration: int = 3600
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600
    allowed_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""

    enable_metrics: bool = True
    metrics_port: int = 8080
    enable_health_checks: bool = True
    health_check_interval: int = 30
    enable_tracing: bool = False
    tracing_endpoint: str = ""
    prometheus_endpoint: str = "/metrics"


@dataclass
class APIConfig:
    """API configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 300
    enable_cors: bool = True
    enable_swagger: bool = True
    api_prefix: str = "/api/v1"


@dataclass
class WorkflowConfig:
    """Workflow execution configuration"""

    max_concurrent_workflows: int = 50
    max_concurrent_tasks: int = 5
    default_task_timeout: float = 300.0
    enable_planning: bool = True
    planning_strategy: str = "optimal"
    enable_auto_retry: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration"""

    level: str = "INFO"
    format: str = "json"
    enable_file_logging: bool = True
    log_file: str = "crewgraph_ai.log"
    log_dir: str = "/var/log/crewgraph"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    backup_count: int = 5
    enable_structured_logging: bool = True


@dataclass
class MemoryConfig:
    """Memory backend configuration"""

    backend: str = "dict"  # dict, redis, faiss, sql
    ttl: int = 3600
    max_size: int = 10000
    compression: bool = False
    encryption: bool = False


@dataclass
class CrewGraphConfig:
    """Complete CrewGraph AI configuration"""

    environment: str = "development"
    debug: bool = False
    version: str = "1.0.0"

    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    api: APIConfig = field(default_factory=APIConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation and environment variable loading"""
        self._load_from_environment()
        self._validate_config()

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Environment
        self.environment = os.getenv("CREWGRAPH_ENVIRONMENT", self.environment)
        self.debug = os.getenv("CREWGRAPH_DEBUG", "false").lower() == "true"

        # Database
        self.database.host = os.getenv("DATABASE_HOST", self.database.host)
        self.database.port = int(os.getenv("DATABASE_PORT", str(self.database.port)))
        self.database.database = os.getenv("DATABASE_NAME", self.database.database)
        self.database.username = os.getenv("DATABASE_USER", self.database.username)
        self.database.password = os.getenv("DATABASE_PASSWORD", self.database.password)

        # Redis
        self.redis.host = os.getenv("REDIS_HOST", self.redis.host)
        self.redis.port = int(os.getenv("REDIS_PORT", str(self.redis.port)))
        self.redis.password = os.getenv("REDIS_PASSWORD", self.redis.password)
        self.redis.database = int(os.getenv("REDIS_DB", str(self.redis.database)))

        # Security
        self.security.encryption_key = os.getenv(
            "CREWGRAPH_ENCRYPTION_KEY", self.security.encryption_key
        )
        self.security.jwt_secret = os.getenv("JWT_SECRET", self.security.jwt_secret)

        # API
        self.api.host = os.getenv("API_HOST", self.api.host)
        self.api.port = int(os.getenv("API_PORT", str(self.api.port)))
        self.api.workers = int(os.getenv("API_WORKERS", str(self.api.workers)))

        # Workflow
        self.workflow.max_concurrent_workflows = int(
            os.getenv("MAX_CONCURRENT_WORKFLOWS", str(self.workflow.max_concurrent_workflows))
        )
        self.workflow.max_concurrent_tasks = int(
            os.getenv("MAX_CONCURRENT_TASKS", str(self.workflow.max_concurrent_tasks))
        )

        # Logging
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.log_dir = os.getenv("LOG_DIR", self.logging.log_dir)

        logger.info(f"Configuration loaded from environment for {self.environment}")

    def _validate_config(self):
        """Validate configuration values"""
        errors = []

        # Validate ports
        if not (1 <= self.api.port <= 65535):
            errors.append(f"Invalid API port: {self.api.port}")

        if not (1 <= self.monitoring.metrics_port <= 65535):
            errors.append(f"Invalid metrics port: {self.monitoring.metrics_port}")

        # Validate security settings
        if self.security.enable_encryption and not self.security.encryption_key:
            errors.append("Encryption enabled but no encryption key provided")

        if self.security.enable_authentication and not self.security.jwt_secret:
            errors.append("Authentication enabled but no JWT secret provided")

        # Validate workflow settings
        if self.workflow.max_concurrent_workflows <= 0:
            errors.append("Max concurrent workflows must be positive")

        if self.workflow.max_concurrent_tasks <= 0:
            errors.append("Max concurrent tasks must be positive")

        # Validate memory backend
        valid_backends = ["dict", "redis", "faiss", "sql"]
        if self.memory.backend not in valid_backends:
            errors.append(
                f"Invalid memory backend: {self.memory.backend}. Must be one of {valid_backends}"
            )

        if errors:
            raise ConfigurationError("Configuration validation failed", details={"errors": errors})

        logger.info("Configuration validation passed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)

    def get_database_url(self) -> str:
        """Get database connection URL"""
        if self.database.password:
            auth = f"{self.database.username}:{self.database.password}"
        else:
            auth = self.database.username

        return (
            f"postgresql://{auth}@{self.database.host}:{self.database.port}/"
            f"{self.database.database}?sslmode={self.database.ssl_mode}"
        )

    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.redis.password:
            auth = f":{self.redis.password}@"
        else:
            auth = ""

        protocol = "rediss" if self.redis.ssl else "redis"
        return f"{protocol}://{auth}{self.redis.host}:{self.redis.port}/{self.redis.database}"

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"


def load_config(config_path: Optional[Union[str, Path]] = None) -> CrewGraphConfig:
    """
    Load configuration from file or environment.

    Args:
        config_path: Path to configuration file

    Returns:
        CrewGraphConfig instance
    """
    if config_path is None:
        # Try common configuration file locations
        search_paths = [
            "config.yaml",
            "config.yml",
            "crewgraph.yaml",
            "crewgraph.yml",
            "/etc/crewgraph/config.yaml",
            "/app/config/config.yaml",
            os.path.expanduser("~/.crewgraph/config.yaml"),
        ]

        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break

    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")

        try:
            with open(config_path, "r") as f:
                if str(config_path).endswith(".json"):
                    config_data = json.load(f)
                else:
                    config_data = yaml.safe_load(f)

            # Convert nested dictionaries to dataclass instances
            return _dict_to_config(config_data)

        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")

    else:
        logger.info(
            "No configuration file found, using default configuration with environment variables"
        )
        return CrewGraphConfig()


def save_config(config: CrewGraphConfig, config_path: Union[str, Path]) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        config_path: Path to save configuration
    """
    try:
        config_dict = config.to_dict()

        # Ensure directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            if str(config_path).endswith(".json"):
                json.dump(config_dict, f, indent=2)
            else:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {config_path}")

    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise ConfigurationError(f"Failed to save configuration: {e}")


def _dict_to_config(config_data: Dict[str, Any]) -> CrewGraphConfig:
    """Convert dictionary to CrewGraphConfig instance"""

    # Extract nested configurations
    database_config = DatabaseConfig(**config_data.get("database", {}))
    redis_config = RedisConfig(**config_data.get("redis", {}))
    security_config = SecurityConfig(**config_data.get("security", {}))
    monitoring_config = MonitoringConfig(**config_data.get("monitoring", {}))
    api_config = APIConfig(**config_data.get("api", {}))
    workflow_config = WorkflowConfig(**config_data.get("workflow", {}))
    logging_config = LoggingConfig(**config_data.get("logging", {}))
    memory_config = MemoryConfig(**config_data.get("memory", {}))

    # Create main configuration
    main_config = {
        k: v
        for k, v in config_data.items()
        if k
        not in [
            "database",
            "redis",
            "security",
            "monitoring",
            "api",
            "workflow",
            "logging",
            "memory",
        ]
    }

    return CrewGraphConfig(
        database=database_config,
        redis=redis_config,
        security=security_config,
        monitoring=monitoring_config,
        api=api_config,
        workflow=workflow_config,
        logging=logging_config,
        memory=memory_config,
        **main_config,
    )


# Global configuration instance
_global_config: Optional[CrewGraphConfig] = None


def get_config() -> CrewGraphConfig:
    """Get global configuration instance"""
    global _global_config

    if _global_config is None:
        _global_config = load_config()

    return _global_config


def set_config(config: CrewGraphConfig) -> None:
    """Set global configuration instance"""
    global _global_config
    _global_config = config
