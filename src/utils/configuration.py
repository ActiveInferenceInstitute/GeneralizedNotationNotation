#!/usr/bin/env python3
"""
GNN Pipeline Unified Configuration System

This module provides a centralized configuration management system for the GNN processing
pipeline with environment variable support, schema validation, and hot reloading capabilities.
"""

import os
import json
import yaml
import time
import hashlib
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic
from contextlib import contextmanager
import threading
import logging
from functools import lru_cache


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"


class ConfigSource(Enum):
    """Configuration source priorities."""
    ENVIRONMENT = 1
    FILE = 2
    DEFAULT = 3


@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigValue:
    """Configuration value with metadata."""
    value: Any
    source: ConfigSource
    timestamp: float = field(default_factory=time.time)
    environment_key: Optional[str] = None
    file_path: Optional[Path] = None


class ConfigurationError(Exception):
    """Configuration related errors."""
    pass


class ConfigurationManager:
    """Centralized configuration management system."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config: Dict[str, ConfigValue] = {}
        self.schemas: Dict[str, ConfigSchema] = {}
        self.config_files: List[Path] = []
        self.environment_prefix = "GNN_"
        self.hot_reload_enabled = False
        self.reload_interval = 30  # seconds
        self.file_hashes: Dict[Path, str] = {}
        self._reload_thread: Optional[threading.Thread] = None
        self._stop_reload = threading.Event()

    def register_schema(self, schema: ConfigSchema):
        """Register a configuration schema."""
        self.schemas[schema.name] = schema
        self.logger.debug(f"Registered configuration schema: {schema.name}")

    def load_from_file(self, config_path: Union[str, Path], format: Optional[ConfigFormat] = None) -> bool:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            self.logger.warning(f"Configuration file not found: {config_path}")
            return False

        if format is None:
            format = self._detect_format(config_path)

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if format == ConfigFormat.JSON:
                    data = json.load(f)
                elif format == ConfigFormat.YAML:
                    try:
                        import yaml
                        data = yaml.safe_load(f)
                    except ImportError:
                        raise ConfigurationError("PyYAML required for YAML configuration files")
                elif format == ConfigFormat.TOML:
                    try:
                        import tomllib
                        data = tomllib.load(f)
                    except ImportError:
                        try:
                            import tomli
                            data = tomli.load(f)
                        except ImportError:
                            raise ConfigurationError("tomllib or tomli required for TOML configuration files")
                else:
                    raise ConfigurationError(f"Unsupported configuration format: {format}")

            # Flatten nested configuration and set values
            flat_config = self._flatten_config(data)
            for key, value in flat_config.items():
                self.set(key, value, ConfigSource.FILE, file_path=config_path)

            self.config_files.append(config_path)
            self.file_hashes[config_path] = self._calculate_file_hash(config_path)

            self.logger.info(f"Loaded configuration from {config_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            return False

    def load_from_environment(self, prefix: Optional[str] = None):
        """Load configuration from environment variables."""
        prefix = prefix or self.environment_prefix

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace('_', '.')
                self.set(config_key, self._parse_env_value(value), ConfigSource.ENVIRONMENT, environment_key=key)

        self.logger.info(f"Loaded environment configuration with prefix: {prefix}")

    def set(self, key: str, value: Any, source: ConfigSource, **metadata):
        """Set a configuration value."""
        config_value = ConfigValue(
            value=value,
            source=source,
            **metadata
        )
        self.config[key] = config_value

    def get(self, key: str, default: Any = None, validate: bool = True) -> Any:
        """Get a configuration value."""
        if key in self.config:
            value = self.config[key].value
            if validate:
                self._validate_value(key, value)
            return value
        return default

    def get_with_metadata(self, key: str) -> Optional[ConfigValue]:
        """Get a configuration value with metadata."""
        return self.config.get(key)

    def validate_config(self, schema_name: Optional[str] = None) -> List[str]:
        """Validate configuration against registered schemas."""
        errors = []

        schemas_to_check = [self.schemas[schema_name]] if schema_name else self.schemas.values()

        for schema in schemas_to_check:
            errors.extend(self._validate_schema(schema))

        return errors

    def _validate_schema(self, schema: ConfigSchema) -> List[str]:
        """Validate configuration against a specific schema."""
        errors = []

        # Check required properties
        for required_key in schema.required:
            if required_key not in self.config:
                if required_key not in schema.defaults:
                    errors.append(f"Required configuration property missing: {required_key}")

        # Validate property types and constraints
        for key, constraints in schema.properties.items():
            if key in self.config:
                value = self.config[key].value
                errors.extend(self._validate_property(key, value, constraints))

        return errors

    def _validate_property(self, key: str, value: Any, constraints: Dict[str, Any]) -> List[str]:
        """Validate a single property against constraints."""
        errors = []

        # Type checking
        expected_type = constraints.get('type')
        if expected_type:
            if not self._check_type(value, expected_type):
                errors.append(f"Property {key}: expected type {expected_type}, got {type(value).__name__}")

        # Range checking for numbers
        if isinstance(value, (int, float)):
            min_val = constraints.get('minimum')
            max_val = constraints.get('maximum')
            if min_val is not None and value < min_val:
                errors.append(f"Property {key}: value {value} is below minimum {min_val}")
            if max_val is not None and value > max_val:
                errors.append(f"Property {key}: value {value} is above maximum {max_val}")

        # Enum checking
        enum_values = constraints.get('enum')
        if enum_values and value not in enum_values:
            errors.append(f"Property {key}: value {value} not in allowed values {enum_values}")

        # Pattern checking for strings
        pattern = constraints.get('pattern')
        if pattern and isinstance(value, str):
            import re
            if not re.match(pattern, value):
                errors.append(f"Property {key}: value {value} does not match pattern {pattern}")

        return errors

    def enable_hot_reload(self, interval: int = 30):
        """Enable hot reloading of configuration files."""
        if self.hot_reload_enabled:
            return

        self.hot_reload_enabled = True
        self.reload_interval = interval
        self._stop_reload.clear()

        self._reload_thread = threading.Thread(target=self._hot_reload_worker, daemon=True)
        self._reload_thread.start()
        self.logger.info(f"Hot reload enabled with {interval}s interval")

    def disable_hot_reload(self):
        """Disable hot reloading."""
        if not self.hot_reload_enabled:
            return

        self.hot_reload_enabled = False
        self._stop_reload.set()

        if self._reload_thread:
            self._reload_thread.join(timeout=5)
        self.logger.info("Hot reload disabled")

    def _hot_reload_worker(self):
        """Worker thread for hot reloading configuration files."""
        while not self._stop_reload.wait(self.reload_interval):
            self._check_for_config_changes()

    def _check_for_config_changes(self):
        """Check for changes in configuration files and reload if needed."""
        for config_file in self.config_files:
            if not config_file.exists():
                continue

            current_hash = self._calculate_file_hash(config_file)
            if self.file_hashes.get(config_file) != current_hash:
                self.logger.info(f"Configuration file changed: {config_file}")
                self.load_from_file(config_file)
                self.file_hashes[config_file] = current_hash

    def export_config(self, output_path: Union[str, Path], format: ConfigFormat = ConfigFormat.JSON):
        """Export current configuration to file."""
        output_path = Path(output_path)
        config_data = {}

        # Group configuration by source
        for key, config_value in self.config.items():
            parts = key.split('.')
            current = config_data
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = config_value.value

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format == ConfigFormat.JSON:
                    json.dump(config_data, f, indent=2)
                elif format == ConfigFormat.YAML:
                    import yaml
                    yaml.dump(config_data, f, default_flow_style=False)
                elif format == ConfigFormat.TOML:
                    try:
                        import tomllib
                        # For TOML export, we'd need a TOML library with write support
                        raise ConfigurationError("TOML export not yet supported")
                    except ImportError:
                        raise ConfigurationError("TOML export requires additional dependencies")

            self.logger.info(f"Configuration exported to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            raise

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        summary = {
            "total_properties": len(self.config),
            "sources": {},
            "schemas": list(self.schemas.keys()),
            "config_files": [str(f) for f in self.config_files],
            "hot_reload_enabled": self.hot_reload_enabled,
            "environment_prefix": self.environment_prefix
        }

        # Count by source
        for config_value in self.config.values():
            source_name = config_value.source.name
            summary["sources"][source_name] = summary["sources"].get(source_name, 0) + 1

        return summary

    def _detect_format(self, config_path: Path) -> ConfigFormat:
        """Detect configuration file format from extension."""
        suffix = config_path.suffix.lower()
        if suffix in ['.json']:
            return ConfigFormat.JSON
        elif suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        elif suffix in ['.toml']:
            return ConfigFormat.TOML
        else:
            raise ConfigurationError(f"Cannot detect format for file: {config_path}")

    def _flatten_config(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested configuration dictionary."""
        flat_config = {}

        for key, value in data.items():
            flat_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flat_config.update(self._flatten_config(value, flat_key))
            else:
                flat_config[flat_key] = value

        return flat_config

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to convert to boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'

        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            if isinstance(expected_python_type, tuple):
                return isinstance(value, expected_python_type)
            return isinstance(value, expected_python_type)

        return True  # Unknown type, assume valid

    def _validate_value(self, key: str, value: Any):
        """Validate a configuration value (lightweight validation)."""
        # Basic type validation
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            raise ConfigurationError(f"Invalid configuration value type for {key}: {type(value)}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


# Global configuration manager instance
_config_manager = ConfigurationManager()


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    return _config_manager


def init_config(config_files: Optional[List[Union[str, Path]]] = None,
                environment_prefix: str = "GNN_",
                enable_hot_reload: bool = False) -> ConfigurationManager:
    """Initialize the configuration system."""
    global _config_manager

    _config_manager = ConfigurationManager()
    _config_manager.environment_prefix = environment_prefix

    # Load default configuration
    _load_default_config()

    # Load from environment
    _config_manager.load_from_environment()

    # Load from configuration files
    if config_files:
        for config_file in config_files:
            _config_manager.load_from_file(config_file)

    # Enable hot reload if requested
    if enable_hot_reload:
        _config_manager.enable_hot_reload()

    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value."""
    return _config_manager.get(key, default)


def set_config(key: str, value: Any, source: ConfigSource = ConfigSource.DEFAULT):
    """Set configuration value."""
    _config_manager.set(key, value, source)


def validate_config(schema_name: Optional[str] = None) -> List[str]:
    """Validate configuration."""
    return _config_manager.validate_config(schema_name)


@lru_cache(maxsize=1)
def _load_default_config():
    """Load default configuration values."""
    # Pipeline defaults
    defaults = {
        "pipeline.max_concurrent_steps": 1,
        "pipeline.default_timeout_seconds": 300,
        "pipeline.retry_attempts": 3,
        "pipeline.log_level": "INFO",
        "pipeline.output_format": "json",

        # Step-specific defaults
        "gnn.discovery_patterns": ["*.gnn", "*.json", "*.yaml"],
        "gnn.validation_enabled": True,
        "gnn.export_formats": ["json", "xml", "yaml"],

        # Logging defaults
        "logging.structured_enabled": True,
        "logging.correlation_id_enabled": True,
        "logging.performance_tracking": True,

        # Test defaults
        "test.coverage_target": 80.0,
        "test.timeout_seconds": 60,
        "test.parallel_execution": False,

        # Export defaults
        "export.compress_output": False,
        "export.include_metadata": True,
    }

    for key, value in defaults.items():
        _config_manager.set(key, value, ConfigSource.DEFAULT)


# Convenience functions for common configuration patterns
def get_pipeline_config() -> Dict[str, Any]:
    """Get pipeline-specific configuration."""
    return {
        key.split('.', 1)[1]: value
        for key, value in _config_manager.config.items()
        if key.startswith('pipeline.')
    }


def get_step_config(step_name: str) -> Dict[str, Any]:
    """Get step-specific configuration."""
    return {
        key.split('.', 1)[1]: value
        for key, value in _config_manager.config.items()
        if key.startswith(f'{step_name}.')
    }


def get_logging_config() -> Dict[str, Any]:
    """Get logging-specific configuration."""
    return {
        key.split('.', 1)[1]: value
        for key, value in _config_manager.config.items()
        if key.startswith('logging.')
    }


def get_test_config() -> Dict[str, Any]:
    """Get test-specific configuration."""
    return {
        key.split('.', 1)[1]: value
        for key, value in _config_manager.config.items()
        if key.startswith('test.')
    }

