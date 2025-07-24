"""Dynamic configuration management with validation."""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from datetime import datetime

@dataclass
class ConfigValidator:
    """Configuration validation rules."""
    required_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, type] = field(default_factory=dict)
    value_ranges: Dict[str, tuple] = field(default_factory=dict)
    custom_validators: Dict[str, Callable] = field(default_factory=dict)

class DynamicConfigManager:
    """Dynamic configuration management with hot reloading."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv('CREWGRAPH_CONFIG_PATH', 'config.yaml')
        self._config: Dict[str, Any] = {}
        self._validators: Dict[str, ConfigValidator] = {}
        self._lock = threading.RLock()
        self._last_modified = 0
        self._watchers: List[Callable] = []
        
        # Load initial configuration
        self.reload_config()
        
        # Start file watcher if config file exists
        if os.path.exists(self.config_path):
            self._start_file_watcher()
    
    def get(self, key: str, default: Any = None, config_section: Optional[str] = None) -> Any:
        """Get configuration value with optional section."""
        with self._lock:
            if config_section:
                section = self._config.get(config_section, {})
                return section.get(key, default)
            return self._config.get(key, default)
    
    def set(self, key: str, value: Any, config_section: Optional[str] = None, 
            persist: bool = False) -> None:
        """Set configuration value."""
        with self._lock:
            if config_section:
                if config_section not in self._config:
                    self._config[config_section] = {}
                self._config[config_section][key] = value
            else:
                self._config[key] = value
            
            if persist:
                self._persist_config()
            
            # Notify watchers
            self._notify_watchers(key, value)
    
    def reload_config(self) -> bool:
        """Reload configuration from file."""
        if not os.path.exists(self.config_path):
            # Create default config if it doesn't exist
            self._create_default_config()
            return True
        
        try:
            with self._lock:
                stat = os.stat(self.config_path)
                if stat.st_mtime <= self._last_modified:
                    return False  # No changes
                
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.json'):
                        new_config = json.load(f)
                    else:
                        new_config = yaml.safe_load(f) or {}
                
                # Validate new configuration
                validation_errors = self._validate_config(new_config)
                if validation_errors:
                    raise ConfigurationError(
                        f"Configuration validation failed: {validation_errors}"
                    )
                
                self._config = new_config
                self._last_modified = stat.st_mtime
                
                # Notify all watchers of config reload
                for watcher in self._watchers:
                    try:
                        watcher('__config_reloaded__', self._config)
                    except Exception as e:
                        logger.error(f"Config watcher failed: {e}")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to reload config from {self.config_path}: {e}")
            return False
    
    def _create_default_config(self) -> None:
        """Create default configuration file."""
        default_config = {
            'created_by': 'Vatsal216',
            'created_at': datetime.utcnow().isoformat(),
            'environment': os.getenv('CREWGRAPH_ENVIRONMENT', 'development'),
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'memory': {
                'backend': 'dict',
                'redis': {
                    'host': os.getenv('REDIS_HOST', 'localhost'),
                    'port': int(os.getenv('REDIS_PORT', 6379)),
                    'db': int(os.getenv('REDIS_DB', 0))
                }
            },
            'llm_providers': {
                'default': 'openai',
                'openai': {
                    'api_key': os.getenv('OPENAI_API_KEY', ''),
                    'model': 'gpt-4',
                    'max_tokens': 2048
                }
            },
            'performance': {
                'max_workers': int(os.getenv('CREWGRAPH_MAX_WORKERS', 10)),
                'timeout': int(os.getenv('CREWGRAPH_TIMEOUT', 300)),
                'enable_optimization': True
            }
        }
        
        self._persist_config(default_config)
        self._config = default_config
    
    def add_validator(self, section: str, validator: ConfigValidator) -> None:
        """Add configuration validator for a section."""
        self._validators[section] = validator
    
    def _validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against registered validators."""
        errors = []
        
        for section, validator in self._validators.items():
            section_config = config.get(section, {})
            
            # Check required fields
            for field in validator.required_fields:
                if field not in section_config:
                    errors.append(f"Missing required field: {section}.{field}")
            
            # Check field types
            for field, expected_type in validator.field_types.items():
                if field in section_config:
                    value = section_config[field]
                    if not isinstance(value, expected_type):
                        errors.append(f"Invalid type for {section}.{field}: expected {expected_type.__name__}, got {type(value).__name__}")
            
            # Check value ranges
            for field, (min_val, max_val) in validator.value_ranges.items():
                if field in section_config:
                    value = section_config[field]
                    if isinstance(value, (int, float)) and not (min_val <= value <= max_val):
                        errors.append(f"Value out of range for {section}.{field}: {value} not in [{min_val}, {max_val}]")
            
            # Run custom validators
            for field, validator_func in validator.custom_validators.items():
                if field in section_config:
                    try:
                        if not validator_func(section_config[field]):
                            errors.append(f"Custom validation failed for {section}.{field}")
                    except Exception as e:
                        errors.append(f"Validator error for {section}.{field}: {e}")
        
        return errors
    
    def watch(self, callback: Callable[[str, Any], None]) -> None:
        """Register a callback to watch for configuration changes."""
        self._watchers.append(callback)
    
    def _notify_watchers(self, key: str, value: Any) -> None:
        """Notify all watchers of configuration changes."""
        for watcher in self._watchers:
            try:
                watcher(key, value)
            except Exception as e:
                logger.error(f"Config watcher failed for key {key}: {e}")
    
    def _start_file_watcher(self) -> None:
        """Start background thread to watch config file changes."""
        def watch_file():
            while True:
                try:
                    self.reload_config()
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    logger.error(f"Config file watcher error: {e}")
                    time.sleep(10)  # Wait longer on error
        
        watcher_thread = threading.Thread(target=watch_file, daemon=True)
        watcher_thread.start()
    
    def _persist_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Persist configuration to file."""
        config_to_save = config or self._config
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.json'):
                    json.dump(config_to_save, f, indent=2)
                else:
                    yaml.safe_dump(config_to_save, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to persist config to {self.config_path}: {e}")

# Global configuration manager
_config_manager: Optional[DynamicConfigManager] = None

def get_config_manager() -> DynamicConfigManager:
    """Get global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = DynamicConfigManager()
    return _config_manager

def get_config(key: str, default: Any = None, section: Optional[str] = None) -> Any:
    """Get configuration value."""
    return get_config_manager().get(key, default, section)

def set_config(key: str, value: Any, section: Optional[str] = None, persist: bool = False) -> None:
    """Set configuration value."""
    get_config_manager().set(key, value, section, persist)