#!/usr/bin/env python3
"""Facade-contract probe for the MAJ-04 MCP mixin split.

The five ``MCP`` responsibilities moved verbatim into sibling mixin modules
(``discovery``/``registry``/``execution``/``introspection``/``metrics``);
``gnn.mcp.mcp`` keeps the class assembly and the module facade functions.
This test pins the split's only observable contract: every public name that
was importable from ``gnn.mcp.mcp`` before the split is still importable,
the module gained no unplanned public names, and responsibility ownership
(what lives on the facade vs. the mixins) is as documented.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mcp

# Captured from ``dir(gnn.mcp.mcp)`` (non-underscore names) immediately
# before the split. The module defines no ``__all__``, so this list IS the
# facade contract.
MCP_FACADE_PUBLIC_NAMES: frozenset[str] = frozenset(
    {
        "Any",
        "Callable",
        "Dict",
        "FuturesTimeoutError",
        "List",
        "MCP",
        "MCPInvalidParamsError",
        "MCPModuleInfo",
        "MCPPerformanceMetrics",
        "MCPRateLimitError",
        "MCPResource",
        "MCPResourceNotFoundError",
        "MCPSDKNotFoundError",
        "MCPSDKStatus",
        "MCPTool",
        "MCPToolExecutionError",
        "MCPToolNotFoundError",
        "MCPToolTimeoutError",
        "MCPValidationError",
        "Optional",
        "Path",
        "ThreadPoolExecutor",
        "Tuple",
        "Union",
        "cast",
        "contextmanager",
        "copy",
        "defaultdict",
        "get_mcp_instance",
        "get_resource_info",
        "get_tool_info",
        "hashlib",
        "importlib",
        "initialize",
        "json",
        "list_available_resources",
        "list_available_tools",
        "logger",
        "logging",
        "mcp_instance",
        "register_tools",
        "sys",
        "tag_non_json_values",
        "threading",
        "time",
    }
)


def test_every_pinned_public_name_is_still_importable() -> None:
    import gnn.mcp.mcp as mcp_module

    missing = sorted(
        name for name in MCP_FACADE_PUBLIC_NAMES if not hasattr(mcp_module, name)
    )
    assert missing == [], f"gnn.mcp.mcp lost public names: {missing}"


def test_facade_gained_no_unplanned_public_names() -> None:
    import gnn.mcp.mcp as mcp_module

    gained = {
        name for name in vars(mcp_module) if not name.startswith("_")
    } - MCP_FACADE_PUBLIC_NAMES
    assert gained == set(), f"gnn.mcp.mcp gained public names: {sorted(gained)}"


def test_mcp_inherits_all_five_responsibility_mixins() -> None:
    from gnn.mcp.discovery import MCPDiscoveryMixin
    from gnn.mcp.execution import MCPExecutionMixin
    from gnn.mcp.introspection import MCPIntrospectionMixin
    from gnn.mcp.mcp import MCP
    from gnn.mcp.metrics import MCPMetricsMixin
    from gnn.mcp.registry import MCPRegistryMixin

    for mixin in (
        MCPDiscoveryMixin,
        MCPRegistryMixin,
        MCPExecutionMixin,
        MCPIntrospectionMixin,
        MCPMetricsMixin,
    ):
        assert issubclass(MCP, mixin), f"MCP is not a subclass of {mixin.__name__}"


def test_responsibility_ownership_matches_the_split_plan() -> None:
    from gnn.mcp.discovery import MCPDiscoveryMixin
    from gnn.mcp.execution import MCPExecutionMixin
    from gnn.mcp.introspection import MCPIntrospectionMixin
    from gnn.mcp.mcp import MCP
    from gnn.mcp.metrics import MCPMetricsMixin
    from gnn.mcp.registry import MCPRegistryMixin

    moved_to: dict[str, type] = {
        "discover_modules": MCPDiscoveryMixin,
        "_load_module": MCPDiscoveryMixin,
        "_configure_local_imports": MCPDiscoveryMixin,
        "_tool_registration_context": MCPDiscoveryMixin,
        "_default_tool_metadata": MCPDiscoveryMixin,
        "list_available_tools": MCPRegistryMixin,
        "list_available_resources": MCPRegistryMixin,
        "register_tool": MCPRegistryMixin,
        "register_resource": MCPRegistryMixin,
        "execute_tool": MCPExecutionMixin,
        "_result_cache_key": MCPExecutionMixin,
        "_execute_with_timeout": MCPExecutionMixin,
        "_cache_get": MCPExecutionMixin,
        "_check_rate_limit": MCPExecutionMixin,
        "_track_performance": MCPExecutionMixin,
        "get_resource": MCPIntrospectionMixin,
        "get_capabilities": MCPIntrospectionMixin,
        "get_server_status": MCPIntrospectionMixin,
        "get_module_info": MCPIntrospectionMixin,
        "get_tool_info": MCPIntrospectionMixin,
        "_validate_params": MCPIntrospectionMixin,
        "_match_uri_template": MCPIntrospectionMixin,
        "_validate_output": MCPIntrospectionMixin,
        "get_enhanced_server_status": MCPMetricsMixin,
        "clear_cache": MCPMetricsMixin,
        "get_tool_performance_stats": MCPMetricsMixin,
        "shutdown": MCPMetricsMixin,
        "set_performance_mode": MCPMetricsMixin,
    }
    for name, mixin in moved_to.items():
        assert name not in MCP.__dict__, f"{name} should not stay on the facade"
        assert name in mixin.__dict__, f"{name} missing from {mixin.__name__}"

    # The assembly plus the members that cannot move mechanically (the
    # static schema helper recurses through the concrete class name).
    for name in (
        "__init__",
        "uptime",
        "request_count",
        "error_count",
        "performance_metrics",
        "config",
        "_strip_legacy_schema_keys",
        "_normalize_tool_schema",
    ):
        assert name in MCP.__dict__, f"{name} should stay on the facade"


def test_late_bound_clock_resolves_through_the_facade_namespace() -> None:
    """``execution.time`` honors a swap of ``gnn.mcp.mcp.time``.

    The moved bodies read ``time`` via a module reference instead of a plain
    import precisely so the fake-clock patch seam
    (``monkeypatch.setattr(mcp_module, "time", ...)``) keeps working.
    """
    import gnn.mcp.execution as execution_module
    import gnn.mcp.mcp as mcp_module

    assert execution_module.time.time is mcp_module.time.time

    class _FakeClock:
        @staticmethod
        def time() -> float:
            return 12345.0

    original = mcp_module.time
    try:
        mcp_module.time = _FakeClock  # type: ignore[assignment]
        assert execution_module.time.time() == 12345.0
    finally:
        mcp_module.time = original
