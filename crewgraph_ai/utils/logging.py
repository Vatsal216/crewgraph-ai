"""
Production-ready logging configuration and utilities
"""

import json
import logging
import logging.handlers
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import structlog


class LogLevel(Enum):
    """Log level enumeration"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Log format types"""

    JSON = "json"
    STRUCTURED = "structured"
    SIMPLE = "simple"
    DETAILED = "detailed"


@dataclass
class LoggerConfig:
    """Logger configuration"""

    level: LogLevel = LogLevel.INFO
    format_type: LogFormat = LogFormat.STRUCTURED
    enable_file_logging: bool = True
    log_file: str = "crewgraph_ai.log"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    backup_count: int = 5
    enable_console_logging: bool = True
    enable_structured_logging: bool = True
    log_dir: str = "logs"
    include_caller: bool = True
    include_timestamp: bool = True
    include_process_info: bool = True
    custom_fields: Dict[str, Any] = field(default_factory=dict)


class CrewGraphFormatter(logging.Formatter):
    """Custom formatter for CrewGraph AI logs"""

    def __init__(self, config: LoggerConfig):
        self.config = config
        self.hostname = os.uname().nodename if hasattr(os, "uname") else "unknown"

        if config.format_type == LogFormat.JSON:
            super().__init__()
        elif config.format_type == LogFormat.DETAILED:
            fmt = (
                "%(asctime)s | %(levelname)-8s | %(name)s | "
                "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
            )
            super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        else:
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        if self.config.format_type == LogFormat.JSON:
            return self._format_json(record)
        else:
            return super().format(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.config.include_caller:
            log_entry.update(
                {
                    "file": record.filename,
                    "line": record.lineno,
                    "function": record.funcName,
                }
            )

        if self.config.include_process_info:
            log_entry.update(
                {
                    "process_id": record.process,
                    "thread_id": record.thread,
                    "hostname": self.hostname,
                }
            )

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add custom fields
        if self.config.custom_fields:
            log_entry.update(self.config.custom_fields)

        # Add extra fields from log record
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
            ):
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


def setup_logging(config: Optional[LoggerConfig] = None) -> None:
    """
    Setup production-ready logging for CrewGraph AI.

    Args:
        config: Logger configuration
    """
    if config is None:
        config = LoggerConfig()

    # Create log directory
    log_dir = Path(config.log_dir)
    log_dir.mkdir(exist_ok=True)

    # Configure structlog if enabled
    if config.enable_structured_logging:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                (
                    structlog.processors.JSONRenderer()
                    if config.format_type == LogFormat.JSON
                    else structlog.dev.ConsoleRenderer()
                ),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.value))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = CrewGraphFormatter(config)

    # Console handler
    if config.enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level.value))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if config.enable_file_logging:
        log_file_path = log_dir / config.log_file
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, config.level.value))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Error file handler (separate file for errors)
    error_log_path = log_dir / f"error_{config.log_file}"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_path,
        maxBytes=config.max_file_size,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # Set level for specific loggers to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Log setup completion
    logger = get_logger(__name__)
    logger.info(
        "CrewGraph AI logging configured",
        level=config.level.value,
        format=config.format_type.value,
        log_dir=str(log_dir),
        user=os.getenv("USER", "unknown"),
        timestamp="2025-07-22 10:13:13",
    )


def get_logger(name: str) -> Union[logging.Logger, structlog.BoundLogger]:
    """
    Get logger instance for the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    # Try to get structlog logger first
    try:
        return structlog.get_logger(name)
    except:
        # Fallback to standard logger
        return logging.getLogger(name)


class LogContext:
    """Context manager for adding context to logs"""

    def __init__(self, logger: Union[logging.Logger, structlog.BoundLogger], **context):
        self.logger = logger
        self.context = context
        self.bound_logger = None

    def __enter__(self):
        if hasattr(self.logger, "bind"):
            self.bound_logger = self.logger.bind(**self.context)
            return self.bound_logger
        else:
            # For standard logger, add context to extra
            return LoggerAdapter(self.logger, self.context)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class LoggerAdapter:
    """Adapter to add context to standard logger"""

    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context

    def debug(self, msg, *args, **kwargs):
        kwargs.setdefault("extra", {}).update(self.context)
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        kwargs.setdefault("extra", {}).update(self.context)
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        kwargs.setdefault("extra", {}).update(self.context)
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        kwargs.setdefault("extra", {}).update(self.context)
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        kwargs.setdefault("extra", {}).update(self.context)
        self.logger.critical(msg, *args, **kwargs)


def log_execution_time(func):
    """Decorator to log function execution time"""

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.utcnow()

        try:
            result = func(*args, **kwargs)
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()

            logger.info(
                f"Function {func.__name__} completed",
                execution_time=execution_time,
                function=func.__name__,
                module=func.__module__,
            )

            return result

        except Exception as e:
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()

            logger.error(
                f"Function {func.__name__} failed",
                execution_time=execution_time,
                function=func.__name__,
                module=func.__module__,
                error=str(e),
            )
            raise

    return wrapper
