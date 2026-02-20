"""
MCP Data Models and Status Classes.

Contains MCPTool, MCPResource, MCPModuleInfo, MCPPerformanceMetrics,
and MCPSDKStatus dataclasses/classes used throughout the MCP module.

Extracted from mcp.py for maintainability.
"""

import hashlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional

logger = logging.getLogger("mcp")


@dataclass
class MCPTool:
    """Enhanced MCP tool representation with better validation and utilities."""
    name: str
    func: Callable
    schema: Dict[str, Any]
    description: str
    module: str = ""
    category: str = ""
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    deprecated: bool = False
    experimental: bool = False
    timeout: Optional[float] = None
    max_concurrent: int = 1
    requires_auth: bool = False
    rate_limit: Optional[float] = None
    cache_ttl: Optional[float] = None
    input_validation: bool = True
    output_validation: bool = True
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    use_count: int = 0

    def __post_init__(self):
        """Validate tool configuration after initialization."""
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not callable(self.func):
            raise ValueError("Tool function must be callable")
        if not isinstance(self.schema, dict):
            raise ValueError("Tool schema must be a dictionary")
        if not isinstance(self.description, str):
            raise ValueError("Tool description must be a string")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Timeout must be positive if specified")
        if self.max_concurrent < 1:
            raise ValueError("Max concurrent must be at least 1")
        if self.rate_limit is not None and self.rate_limit <= 0:
            raise ValueError("Rate limit must be positive if specified")
        if self.cache_ttl is not None and self.cache_ttl <= 0:
            raise ValueError("Cache TTL must be positive if specified")

    def get_signature(self) -> str:
        """Get a unique signature for this tool."""
        return hashlib.md5(
            f"{self.name}:{self.module}:{self.version}:{self.schema}".encode()
        ).hexdigest()

    def mark_used(self) -> None:
        """Mark the tool as used and update statistics."""
        self.last_used = time.time()
        self.use_count += 1

    def is_rate_limited(self) -> bool:
        """Check if the tool is currently rate limited."""
        if not self.rate_limit:
            return False

        current_time = time.time()
        if self.last_used is not None:
             elapsed = current_time - self.last_used
             if elapsed < (1.0 / self.rate_limit):
                 return True
        return False

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of tool usage statistics."""
        return {
            "name": self.name,
            "use_count": self.use_count,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "avg_usage_interval": None if self.use_count < 2 else
                (self.last_used - self.created_at) / (self.use_count - 1) if self.last_used else None
        }

    def validate_schema(self) -> List[str]:
        """Validate the tool schema and return any issues."""
        issues = []

        if not isinstance(self.schema, dict):
            issues.append("Schema must be a dictionary")
            return issues

        schema = self.schema
        if "type" not in schema:
            issues.append("Schema missing 'type' field")
        elif schema["type"] != "object":
            issues.append("Schema type must be 'object' for MCP tools")

        if "properties" not in schema:
            issues.append("Schema missing 'properties' field")

        if "required" in schema and not isinstance(schema["required"], list):
            issues.append("Schema 'required' field must be a list")

        return issues


@dataclass
class MCPResource:
    """Enhanced MCP resource representation with better validation and utilities."""
    uri_template: str
    retriever: Callable
    description: str
    module: str = ""
    category: str = ""
    version: str = "1.0.0"
    mime_type: str = "application/json"
    cacheable: bool = True
    tags: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    requires_auth: bool = False
    rate_limit: Optional[float] = None
    cache_ttl: Optional[float] = None
    compression: bool = False
    encryption: bool = False
    created_at: float = field(default_factory=time.time)
    last_accessed: Optional[float] = None
    access_count: int = 0

    def __post_init__(self):
        """Validate resource configuration after initialization."""
        if not self.uri_template:
            raise ValueError("Resource URI template cannot be empty")
        if not callable(self.retriever):
            raise ValueError("Resource retriever must be callable")
        if not isinstance(self.description, str):
            raise ValueError("Resource description must be a string")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Timeout must be positive if specified")
        if self.rate_limit is not None and self.rate_limit <= 0:
            raise ValueError("Rate limit must be positive if specified")
        if self.cache_ttl is not None and self.cache_ttl <= 0:
            raise ValueError("Cache TTL must be positive if specified")

    def mark_accessed(self) -> None:
        """Mark the resource as accessed and update statistics."""
        self.last_accessed = time.time()
        self.access_count += 1

    def get_access_summary(self) -> Dict[str, Any]:
        """Get a summary of resource access statistics."""
        return {
            "uri_template": self.uri_template,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "avg_access_interval": None if self.access_count < 2 else
                (self.last_accessed - self.created_at) / (self.access_count - 1) if self.last_accessed else None
        }

    def validate_uri_template(self, uri: str) -> bool:
        """Validate if a URI matches this resource template."""
        return uri.startswith(self.uri_template.split('{')[0]) if '{' in self.uri_template else uri == self.uri_template


@dataclass
class MCPModuleInfo:
    """Information about a discovered MCP module."""
    name: str
    path: Path
    tools_count: int = 0
    resources_count: int = 0
    status: str = "loaded"
    error_message: Optional[str] = None
    load_time: float = 0.0
    version: str = "1.0.0"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    file_size: int = 0
    file_hash: str = ""
    import_time: float = 0.0
    register_time: float = 0.0
    memory_usage: Optional[int] = None

    def __post_init__(self):
        """Calculate file size and hash if path exists."""
        if self.path.exists():
            try:
                self.file_size = self.path.stat().st_size
                self.file_hash = hashlib.md5(self.path.read_bytes()).hexdigest()
            except Exception as e:
                logger.warning(f"Could not calculate file info for {self.path}: {e}")


@dataclass
class MCPPerformanceMetrics:
    """Enhanced performance metrics for MCP operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    request_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    tool_usage_stats: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    module_load_times: Dict[str, float] = field(default_factory=dict)
    cache_hit_ratio: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage: Optional[int] = None
    concurrent_requests: int = 0
    max_concurrent_requests: int = 0
    request_queue_size: int = 0
    max_request_queue_size: int = 0

    def update_execution_time(self, execution_time: float):
        """Update execution time statistics."""
        self.total_execution_time += execution_time
        self.max_execution_time = max(self.max_execution_time, execution_time)
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.average_execution_time = self.total_execution_time / self.total_requests

    def update_cache_stats(self, hit: bool):
        """Update cache statistics."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            self.cache_hit_ratio = self.cache_hits / total_cache_requests


class MCPSDKStatus:
    """Enhanced MCP SDK status tracking."""

    def __init__(self):
        self._sdk_found = False
        self._sdk_version = None
        self._sdk_path = None
        self._last_check = 0.0
        self._check_interval = 300.0  # 5 minutes
        self._sdk_capabilities: Dict[str, Any] = {}
        self._sdk_health = "unknown"
        self._sdk_last_health_check = 0.0

    def check_status(self) -> bool:
        """Check if MCP SDK is available with enhanced detection."""
        current_time = time.time()

        if current_time - self._last_check < self._check_interval:
            return self._sdk_found

        self._last_check = current_time

        try:
            sdk_paths = [
                Path.home() / ".mcp" / "sdk",
                Path("/usr/local/mcp/sdk"),
                Path("/opt/mcp/sdk"),
                Path.cwd() / "mcp_sdk",
                Path(__file__).parent / "sdk"
            ]

            for sdk_path in sdk_paths:
                if sdk_path.exists() and sdk_path.is_dir():
                    self._sdk_found = True
                    self._sdk_path = sdk_path

                    version_file = sdk_path / "version.txt"
                    if version_file.exists():
                        self._sdk_version = version_file.read_text().strip()

                    self._check_sdk_health(sdk_path)

                    logger.info(f"MCP SDK found at {sdk_path} (version: {self._sdk_version})")
                    return True

            if os.environ.get("MCP_SDK_PATH"):
                sdk_path = Path(os.environ["MCP_SDK_PATH"])
                if sdk_path.exists() and sdk_path.is_dir():
                    self._sdk_found = True
                    self._sdk_path = sdk_path
                    self._check_sdk_health(sdk_path)
                    logger.info(f"MCP SDK found via environment at {sdk_path}")
                    return True

            self._sdk_found = False
            self._sdk_path = None
            self._sdk_version = None
            self._sdk_health = "not_found"
            return False

        except Exception as e:
            logger.warning(f"Error checking MCP SDK status: {e}")
            self._sdk_found = False
            self._sdk_health = "error"
            return False

    def _check_sdk_health(self, sdk_path: Path):
        """Check the health of the SDK installation."""
        try:
            essential_files = ["mcp.py", "server.py", "client.py"]
            missing_files = [f for f in essential_files if not (sdk_path / f).exists()]

            if missing_files:
                self._sdk_health = "incomplete"
                logger.warning(f"MCP SDK missing files: {missing_files}")
            else:
                self._sdk_health = "healthy"

            capabilities_file = sdk_path / "capabilities.json"
            if capabilities_file.exists():
                try:
                    self._sdk_capabilities = json.loads(capabilities_file.read_text())
                except Exception as e:
                    logger.warning(f"Could not parse SDK capabilities: {e}")

        except Exception as e:
            logger.warning(f"Error checking SDK health: {e}")
            self._sdk_health = "error"

    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary."""
        return {
            "sdk_found": self._sdk_found,
            "sdk_version": self._sdk_version,
            "sdk_path": str(self._sdk_path) if self._sdk_path else None,
            "sdk_health": self._sdk_health,
            "sdk_capabilities": self._sdk_capabilities,
            "last_check": self._last_check
        }
