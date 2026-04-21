"""
Validate the MCP server's configurable surface end-to-end.

Covers contracts introduced after 2026-04-16:
  * ``register_tools(mcp)`` actually populates the registry (was a no-op stub).
  * ``JSONRPCServer.register_tool`` returns a truthful bool.
  * ``MCPServer`` remains an alias of :class:`mcp.MCP`.
  * Fine-grained overrides (``strict_validation``, ``cache_ttl``,
    ``modules_allowlist``, …) propagate from ``process_mcp`` → ``initialize`` →
    the lazy singleton attribute storage.
  * The lazy :class:`mcp.mcp._LazyMCP` proxy forwards writes to the underlying
    instance rather than shadowing the attribute on itself.
  * Every symbol declared in :data:`mcp.__all__` is actually importable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def fresh_mcp(monkeypatch):
    """Provide a pristine MCP instance without touching the shared singleton."""
    from mcp.mcp import MCP

    instance = MCP()
    yield instance
    try:
        instance.shutdown()
    except Exception:
        pass


class TestMCPExportsContract:
    def test_all_symbols_importable(self) -> None:
        import mcp as _mcp

        missing = [s for s in _mcp.__all__ if not hasattr(_mcp, s)]
        assert missing == [], f"mcp.__all__ lists symbols not on the module: {missing}"

    def test_mcp_server_is_core_alias(self) -> None:
        from mcp import MCP, MCPServer

        assert MCPServer is MCP

    def test_jsonrpc_server_is_request_handler(self) -> None:
        from mcp import JSONRPCServer

        server = JSONRPCServer()
        assert hasattr(server, "handle_request"), "JSONRPCServer must expose handle_request"
        assert hasattr(server, "register_tool")


class TestJSONRPCServerRegisterTool:
    def test_register_tool_returns_true_on_success(self) -> None:
        from mcp import JSONRPCServer

        server = JSONRPCServer()
        ok = server.register_tool(
            "__configurability_test_ok__",
            lambda **_: {"success": True},
            {"type": "object"},
            "test tool",
        )
        assert ok is True
        assert "__configurability_test_ok__" in server.mcp.tools

    def test_register_tool_returns_false_on_invalid_inputs(self) -> None:
        from mcp import JSONRPCServer

        server = JSONRPCServer()
        # Empty name is rejected by the core MCP.register_tool
        assert server.register_tool("", lambda **_: None, {}, "broken") is False


class TestRegisterToolsHelper:
    def test_register_tools_populates_registry(self, fresh_mcp) -> None:
        from mcp.server_core import register_tools

        # Pristine instance starts with no tools
        assert len(fresh_mcp.tools) == 0
        ok = register_tools(fresh_mcp)
        # Discovery should succeed and populate at least the core modules
        assert ok in (True, False)  # bool, regardless of module-level failures
        assert len(fresh_mcp.tools) > 0, "register_tools must actually register tools"
        assert len(fresh_mcp.modules) > 0


class TestLazyMCPProxyWrites:
    def test_attribute_writes_forward_to_underlying_instance(self) -> None:
        """The proxy must not silently shadow attributes on itself."""
        from mcp.mcp import get_mcp_instance, mcp_instance

        real = get_mcp_instance()
        original = real._cache_ttl

        try:
            mcp_instance._cache_ttl = 777.0
            assert real._cache_ttl == 777.0
            assert mcp_instance._cache_ttl == 777.0
        finally:
            real._cache_ttl = original


class TestInitializeOverrides:
    def test_fine_grained_overrides_propagate(self) -> None:
        from mcp import initialize
        from mcp.mcp import get_mcp_instance

        mcp = get_mcp_instance()
        initialize(
            halt_on_missing_sdk=False,
            force_proceed_flag=True,
            performance_mode="high",
            strict_validation=False,
            enable_caching=False,
            enable_rate_limiting=False,
            cache_ttl=42.5,
            modules_allowlist=["gnn"],
            per_module_timeout=5.0,
            overall_timeout=20.0,
            force_refresh=True,
        )
        cfg = mcp.config
        assert cfg["strict_validation"] is False
        assert cfg["enable_caching"] is False
        assert cfg["enable_rate_limiting"] is False
        assert cfg["cache_ttl"] == 42.5
        assert "gnn" in mcp.modules


class TestProcessMCPConfigWiring:
    def test_cli_style_kwargs_reach_singleton(self, tmp_path: Path) -> None:
        from mcp import process_mcp
        from mcp.mcp import get_mcp_instance

        target = tmp_path / "in"
        target.mkdir()
        out = tmp_path / "out"
        out.mkdir()

        ok = process_mcp(
            target,
            out,
            verbose=False,
            mcp_strict_validation=True,
            mcp_cache_ttl=60.0,
            mcp_modules_allowlist="gnn",
            mcp_per_module_timeout=5.0,
            mcp_overall_timeout=20.0,
            force_refresh=True,
        )
        assert ok is True

        cfg = get_mcp_instance().config
        assert cfg["cache_ttl"] == 60.0
        assert cfg["strict_validation"] is True

        # Allowlist restricts discovery to "gnn" (plus the always-on special
        # modules: "mcp" self + "sympy_mcp" inside mcp/).
        discovered = set(get_mcp_instance().modules.keys())
        assert "gnn" in discovered
        # The allowlist filter must exclude at least one normally-loaded module
        assert "audio" not in discovered
