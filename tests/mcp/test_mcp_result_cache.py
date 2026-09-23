"""Result-cache contract tests (tier-2 row 13, MCP phase 2).

Pins the shipped-tool opt-in semantics: a tool is served from the result
cache only when it registered ``cacheable=True`` AND the server has caching
enabled; the entry TTL defaults to the server-wide ``_cache_ttl`` knob
(``initialize(cache_ttl=...)``) and a tool's own ``cache_ttl`` overrides
that default. Expiry uses the same controlled clock as
``test_registry_internals`` (execution.py's late-bound ``time``).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mcp

import gnn.mcp.mcp as mcp_module
from gnn.mcp.mcp import MCP
from tests.helpers import FakeMCPTime


class TestCacheableOptIn:
    """Caching requires the per-tool cacheable flag, not just a cache_ttl."""

    @pytest.mark.unit
    def test_cacheable_tool_executes_once_then_serves_from_cache(self) -> None:
        registry = MCP(enable_caching=True, enable_rate_limiting=False)
        calls: list[int] = []
        registry.register_tool(
            name="report_tool",
            func=lambda: calls.append(1) or {"n": len(calls)},
            schema={},
            description="cacheable computation",
            cacheable=True,
        )
        first = registry.execute_tool("report_tool", {})
        second = registry.execute_tool("report_tool", {})

        assert first == {"n": 1}
        assert second == {"n": 1}
        assert len(calls) == 1
        assert len(registry._result_cache) == 1
        assert registry._performance_metrics.cache_hits == 1
        assert registry._performance_metrics.cache_misses == 1

    @pytest.mark.unit
    def test_non_cacheable_tool_never_caches_even_when_enabled(self) -> None:
        registry = MCP(enable_caching=True, enable_rate_limiting=False)
        calls: list[int] = []
        registry.register_tool(
            name="mutating_tool",
            func=lambda: calls.append(1) or {"n": len(calls)},
            schema={},
            description="mutating side effects, never cached",
            cache_ttl=60.0,
        )
        registry.execute_tool("mutating_tool", {})
        registry.execute_tool("mutating_tool", {})

        assert len(calls) == 2
        assert registry._result_cache == {}
        assert registry._performance_metrics.cache_hits == 0
        assert registry._performance_metrics.cache_misses == 0


class TestServerDefaultTTL:
    """The server-wide _cache_ttl knob is honored as the default entry TTL."""

    @pytest.mark.unit
    def test_entry_expires_after_server_ttl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_time = FakeMCPTime()
        monkeypatch.setattr(mcp_module, "time", fake_time)
        registry = MCP(enable_caching=True, enable_rate_limiting=False)
        registry._cache_ttl = 50.0
        calls: list[int] = []
        registry.register_tool(
            name="server_ttl_tool",
            func=lambda: calls.append(1) or {"n": len(calls)},
            schema={},
            description="uses the server default TTL",
            cacheable=True,
        )
        registry.execute_tool("server_ttl_tool", {})
        fake_time.advance(49.0)
        registry.execute_tool("server_ttl_tool", {})
        assert len(calls) == 1  # still inside the default TTL window

        fake_time.advance(2.0)  # entry is now past the server default TTL
        result = registry.execute_tool("server_ttl_tool", {})
        assert result == {"n": 2}
        assert len(calls) == 2  # re-executed after expiry

    @pytest.mark.unit
    def test_per_tool_cache_ttl_overrides_server_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_time = FakeMCPTime()
        monkeypatch.setattr(mcp_module, "time", fake_time)
        registry = MCP(enable_caching=True, enable_rate_limiting=False)
        registry._cache_ttl = 50.0
        calls: list[int] = []
        registry.register_tool(
            name="short_ttl_tool",
            func=lambda: calls.append(1) or {"n": len(calls)},
            schema={},
            description="registers its own shorter TTL",
            cacheable=True,
            cache_ttl=5.0,
        )
        registry.execute_tool("short_ttl_tool", {})
        fake_time.advance(10.0)  # past the tool TTL, inside the server default
        result = registry.execute_tool("short_ttl_tool", {})
        assert result == {"n": 2}
        assert len(calls) == 2


class TestInitializeCacheTTLKnob:
    """initialize(cache_ttl=...) sets the server-wide default TTL."""

    @pytest.mark.unit
    def test_initialize_cache_ttl_propagates(self) -> None:
        from gnn.mcp import initialize

        mcp = initialize(
            halt_on_missing_sdk=False,
            force_proceed_flag=True,
            performance_mode="low",
            modules_allowlist=["mcp"],
            per_module_timeout=5.0,
            overall_timeout=20.0,
            cache_ttl=123.5,
        )[0]
        try:
            assert mcp._cache_ttl == 123.5
            assert mcp.config["cache_ttl"] == 123.5
        finally:
            mcp._cache_ttl = 300.0
