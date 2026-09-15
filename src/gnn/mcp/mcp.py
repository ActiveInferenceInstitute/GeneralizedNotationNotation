#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Core Implementation for GNN

Core MCP server for the GNN project: discovers modules, registers tools and
resources, executes them with thread-safe caching and rate limiting, and
exposes them to MCP-compatible clients via stdio or HTTP transport.

Mechanical split facade (MAJ-04 sibling-mixin pattern): the five
responsibilities live in sibling mixin modules (``discovery``, ``registry``,
``execution``, ``introspection``, ``metrics``); ``MCP`` inherits them and
this module keeps the class assembly, the module facade functions, and
every previously importable name — verified by
``tests/mcp/test_mcp_facade_contract.py``.
"""

import copy as copy
import hashlib as hashlib
import importlib as importlib
import json as json
import logging
import sys as sys
import threading
import time
from collections import defaultdict
from concurrent.futures import (
    ThreadPoolExecutor,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,  # noqa: F401
)
from contextlib import contextmanager as contextmanager
from pathlib import Path as Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)
from typing import (
    Callable as Callable,
)
from typing import (
    Union as Union,
)

from .discovery import MCPDiscoveryMixin as _MCPDiscoveryMixin

# --- Import MCP Exceptions from dedicated module ---
from .exceptions import (
    MCPInvalidParamsError as MCPInvalidParamsError,
)
from .exceptions import (
    MCPRateLimitError as MCPRateLimitError,
)
from .exceptions import (
    MCPResourceNotFoundError as MCPResourceNotFoundError,
)
from .exceptions import (
    MCPSDKNotFoundError,
)
from .exceptions import (
    MCPToolExecutionError as MCPToolExecutionError,
)
from .exceptions import (
    MCPToolNotFoundError as MCPToolNotFoundError,
)
from .exceptions import (
    MCPToolTimeoutError as MCPToolTimeoutError,
)
from .exceptions import (
    MCPValidationError as MCPValidationError,
)
from .execution import _TOOL_TIMEOUT_POOL_SIZE
from .execution import MCPExecutionMixin as _MCPExecutionMixin
from .introspection import MCPIntrospectionMixin as _MCPIntrospectionMixin
from .jsonrpc import tag_non_json_values as tag_non_json_values
from .metrics import _MCP_SDK_STATUS
from .metrics import MCPMetricsMixin as _MCPMetricsMixin
from .models import (
    MCPModuleInfo,
    MCPPerformanceMetrics,
    MCPResource,
    MCPTool,
)
from .models import (
    MCPSDKStatus as MCPSDKStatus,
)
from .registry import MCPRegistryMixin as _MCPRegistryMixin
from .validation import _validate_params as _validate_params

# Configure logging
logger = logging.getLogger("mcp")


# --- Enhanced Main MCP Class ---
class MCP(
    _MCPDiscoveryMixin,
    _MCPRegistryMixin,
    _MCPExecutionMixin,
    _MCPIntrospectionMixin,
    _MCPMetricsMixin,
):
    """
    Enhanced Model Context Protocol implementation.

    This class provides the core functionality for:
    - Discovering and loading MCP modules with caching
    - Registering tools and resources with enhanced metadata
    - Executing tools and retrieving resources with performance tracking
    - Managing server capabilities and status
    - Performance monitoring and metrics collection
    - Thread-safe operations with proper locking
    - Enhanced error handling and validation
    """

    def __init__(
        self,
        enable_caching: bool = True,
        enable_rate_limiting: bool = True,
        strict_validation: bool = False,
        max_workers: int = 4,
    ) -> None:
        """Initialize the enhanced MCP server with configurable features."""
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.modules: Dict[str, MCPModuleInfo] = {}
        self._modules_discovered = False
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        self._lock = threading.RLock()
        self._registration_context = threading.local()

        self._performance_metrics = MCPPerformanceMetrics()
        self._tool_execution_times: Dict[str, List[float]] = defaultdict(list)
        self._last_activity = time.time()

        self._discovery_cache: Dict[str, Any] = {}
        self._cache_timestamp = 0.0
        self._cache_ttl = 300.0  # 5 minutes
        self._discovery_cache_lock = threading.Lock()
        self._registration_lock = threading.RLock()

        self._active_executions: Dict[str, int] = defaultdict(int)
        self._execution_lock = threading.Lock()

        self._rate_limit_timestamps: Dict[str, List[float]] = defaultdict(list)
        self._rate_limit_lock = threading.Lock()

        self._result_cache: Dict[str, Tuple[Any, float]] = {}
        self._result_cache_lock = threading.Lock()

        # Tools-list cache: invalidated on register_tool/unregister. The
        # list is rebuilt only when the registry changes, not on every call.
        self._tools_list_cache: List[Dict[str, Any]] | None = None
        self._tools_names_cache: List[str] | None = None

        self._executor: Optional[ThreadPoolExecutor]
        try:
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="MCP"
            )
        except Exception as e:
            logger.warning(f"Failed to create thread pool executor: {e}")
            self._executor = None

        self._tool_timeout_executor: Optional[ThreadPoolExecutor]
        try:
            self._tool_timeout_executor = ThreadPoolExecutor(
                max_workers=_TOOL_TIMEOUT_POOL_SIZE,
                thread_name_prefix="MCP-ToolTimeout",
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Failed to create tool timeout executor: {e}")
            self._tool_timeout_executor = None

        self._enable_caching = enable_caching
        self._enable_rate_limiting = enable_rate_limiting
        self._strict_validation = strict_validation

        logger.info(
            f"Enhanced MCP server initialized (caching={enable_caching}, "
            f"rate_limiting={enable_rate_limiting}, strict_validation={strict_validation})"
        )

    @property
    def uptime(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self._start_time

    @property
    def request_count(self) -> int:
        """Get total number of requests processed."""
        return self._request_count

    @property
    def error_count(self) -> int:
        """Get total number of errors encountered."""
        return self._error_count

    @property
    def performance_metrics(self) -> MCPPerformanceMetrics:
        """Get performance metrics."""
        return self._performance_metrics

    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "enable_caching": self._enable_caching,
            "enable_rate_limiting": self._enable_rate_limiting,
            "strict_validation": self._strict_validation,
            "cache_ttl": self._cache_ttl,
            "max_workers": self._executor._max_workers if self._executor else 0,
            "modules_discovered": self._modules_discovered,
        }

    @staticmethod
    def _strip_legacy_schema_keys(value: Any) -> Any:
        """Remove non-JSON-schema compatibility keys while preserving nested structure."""
        if isinstance(value, dict):
            return {
                key: MCP._strip_legacy_schema_keys(item)
                for key, item in value.items()
                if key != "optional"
            }
        if isinstance(value, list):
            return [MCP._strip_legacy_schema_keys(item) for item in value]
        return value

    @classmethod
    def _normalize_tool_schema(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize compatibility MCP schemas into JSON-schema object form."""
        if not schema:
            return {"type": "object", "properties": {}}

        if schema.get("type") == "object" or "properties" in schema:
            normalized = cls._strip_legacy_schema_keys(dict(schema))
            normalized["type"] = "object"
            existing_properties = normalized.get("properties")
            if not isinstance(existing_properties, dict):
                normalized["properties"] = {}
            if "required" in normalized and not isinstance(
                normalized["required"], list
            ):
                normalized["required"] = []
            return cast("Dict[str, Any]", normalized)

        properties: Dict[str, Any] = {}
        required: List[str] = []
        for field_name, field_schema in schema.items():
            if not isinstance(field_schema, dict):
                properties[field_name] = {"description": str(field_schema)}
                required.append(field_name)
                continue

            is_optional = bool(field_schema.get("optional", False)) or (
                "default" in field_schema and "optional" not in field_schema
            )
            properties[field_name] = cast(
                "Dict[str, Any]", cls._strip_legacy_schema_keys(field_schema)
            )
            if not is_optional:
                required.append(field_name)

        normalized_schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            normalized_schema["required"] = required
        return normalized_schema


# --- Global MCP Instance (lazy) ---
# Deferred to first access so importing this module does not allocate a
# ThreadPoolExecutor or other resources before the caller is ready.
_mcp_instance: Optional["MCP"] = None


class _LazyMCP:
    """Proxy that creates the real MCP singleton on first attribute access.

    Forwards both reads and writes to the underlying :class:`MCP` instance so
    call sites can safely do e.g. ``mcp_instance._enable_caching = False``
    without silently shadowing the attribute on the proxy itself.
    """

    _PROXY_ONLY: Any = frozenset()  # reserved for internal proxy state

    def _target(self) -> "MCP":
        """Handle target for internal callers."""
        global _mcp_instance
        if _mcp_instance is None:
            _mcp_instance = MCP()
        return _mcp_instance

    def __getattr__(self, name: str) -> Any:
        """Handle getattr for internal callers."""
        return getattr(self._target(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle setattr for internal callers."""
        if name in self._PROXY_ONLY:
            object.__setattr__(self, name, value)
            return
        setattr(self._target(), name, value)


mcp_instance: Any = _LazyMCP()


# --- Initialization Function ---
def initialize(
    halt_on_missing_sdk: bool = True,
    force_proceed_flag: bool = False,
    performance_mode: str = "low",
    modules_allowlist: Optional[List[str]] = None,
    per_module_timeout: float = 30.0,
    overall_timeout: float = 120.0,
    enable_caching: Optional[bool] = None,
    enable_rate_limiting: Optional[bool] = None,
    strict_validation: Optional[bool] = None,
    cache_ttl: Optional[float] = None,
    force_refresh: bool = False,
) -> Tuple[MCP, bool, bool]:
    """
    Initialize the MCP by discovering modules and checking SDK status.

    Args:
        halt_on_missing_sdk: If True, raises MCPSDKNotFoundError if SDK is missing.
        force_proceed_flag: If True, proceeds even if SDK is missing.
        performance_mode: ``"low"``, ``"medium"`` (unused; treated as low), or
            ``"high"``. Applied via :meth:`MCP.set_performance_mode` before any
            fine-grained overrides below.
        modules_allowlist: If set, only load these package names under ``src/``.
        per_module_timeout: Max seconds to wait per module during parallel
            discovery (see :meth:`MCP.discover_modules`).
        overall_timeout: Wall-clock budget for the parallel wait loop.
        enable_caching: Optional override for result-cache enablement. When
            None the value chosen by ``performance_mode`` is preserved.
        enable_rate_limiting: Optional override for rate-limiting enablement.
        strict_validation: Optional override for strict schema validation.
        cache_ttl: Optional override for result-cache TTL (seconds).
        force_refresh: If True, force re-discovery even if the singleton has
            already loaded modules in this process.

    Returns:
        Tuple of (mcp_instance, sdk_found, all_modules_loaded)

    Raises:
        MCPSDKNotFoundError: If SDK is missing and halt_on_missing_sdk is True
    """
    sdk_found = _MCP_SDK_STATUS.check_status()

    if not sdk_found:
        if halt_on_missing_sdk and not force_proceed_flag:
            error_message = (
                "MCP SDK is critical for full functionality and was not found or failed to load. "
                "Pipeline is configured to halt. To proceed with limited MCP capabilities, "
                "use a flag like --proceed-without-mcp-sdk or adjust pipeline configuration."
            )
            logger.error(error_message)
            raise MCPSDKNotFoundError(error_message)
        else:
            logger.debug(
                "MCP SDK optional dependency not available - proceeding with core functionality"
            )

    try:
        mcp_instance.set_performance_mode(performance_mode)
    except (AttributeError, TypeError):
        logger.debug("Performance mode setting not supported on this MCP instance")

    # Fine-grained overrides applied after performance_mode so callers can
    # opt into, e.g., high performance with strict_validation disabled.
    if enable_caching is not None:
        mcp_instance._enable_caching = bool(enable_caching)
    if enable_rate_limiting is not None:
        mcp_instance._enable_rate_limiting = bool(enable_rate_limiting)
    if strict_validation is not None:
        mcp_instance._strict_validation = bool(strict_validation)
    if cache_ttl is not None:
        mcp_instance._cache_ttl = float(cache_ttl)

    all_modules_loaded = mcp_instance.discover_modules(
        force_refresh=force_refresh,
        modules_allowlist=modules_allowlist,
        per_module_timeout=per_module_timeout,
        overall_timeout=overall_timeout,
    )

    if all_modules_loaded:
        logger.info("MCP initialization completed successfully")
    else:
        logger.warning("MCP initialization completed with some module loading failures")

    return mcp_instance, sdk_found, all_modules_loaded


def get_mcp_instance() -> MCP:
    """Get the global MCP instance, creating it on first call."""
    global _mcp_instance
    if _mcp_instance is None:
        _mcp_instance = MCP()
    return _mcp_instance


def list_available_tools() -> List[Dict[str, Any]]:
    """List all available tools with metadata."""
    return cast(List[Dict[str, Any]], mcp_instance.list_available_tools())


def list_available_resources() -> List[Dict[str, Any]]:
    """List all available resources with metadata."""
    return cast(List[Dict[str, Any]], mcp_instance.list_available_resources())


def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific tool."""
    return cast(Optional[Dict[str, Any]], mcp_instance.get_tool_info(tool_name))


def get_resource_info(uri_template: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific resource."""
    if uri_template in mcp_instance.resources:
        resource = mcp_instance.resources[uri_template]
        return {
            "uri_template": resource.uri_template,
            "description": resource.description,
            "module": resource.module,
            "category": resource.category,
            "version": resource.version,
            "mime_type": resource.mime_type,
            "cacheable": resource.cacheable,
            "tags": resource.tags,
        }
    return None


def register_tools(server: Any) -> Any:
    """Register core MCP introspection tools."""

    def list_core_tools() -> List[Dict[str, Any]]:
        """Provide list core tools behavior."""
        return list_available_tools()

    def list_core_resources() -> List[Dict[str, Any]]:
        """Provide list core resources behavior."""
        return list_available_resources()

    server.register_tool(
        name="mcp.list_available_tools",
        func=list_core_tools,
        schema={},
        description="List all tools currently registered with the MCP registry.",
    )
    server.register_tool(
        name="mcp.list_available_resources",
        func=list_core_resources,
        schema={},
        description="List all resources currently registered with the MCP registry.",
    )
