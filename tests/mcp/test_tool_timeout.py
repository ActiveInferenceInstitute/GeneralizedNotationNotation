"""Tool-timeout enforcement contract tests (wave-2 MAJ-09).

``MCPTool.timeout`` is registered, advertised in capabilities, and
documented — these tests pin that it is also ENFORCED: the caller stops
waiting at ``tool.timeout`` and receives ``MCPToolTimeoutError`` (wire code
-32008, in the JSON-RPC server-defined range) on every transport, while
un-timed tools keep the inline (unbounded) execution path.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

pytestmark = pytest.mark.mcp

from gnn.mcp.exceptions import MCPToolTimeoutError
from gnn.mcp.mcp import MCP
from gnn.mcp.server_core import MCPServer


def _registry() -> MCP:
    return MCP(enable_caching=False, enable_rate_limiting=False)


class TestToolTimeoutEnforcement:
    """execute_tool must bound the caller's wait at tool.timeout."""

    @pytest.mark.unit
    def test_hung_tool_raises_timeout_quickly(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="hung_tool",
            func=lambda: time.sleep(1.5),
            schema={},
            description="sleeps past its timeout",
            timeout=0.2,
        )
        start = time.monotonic()
        with pytest.raises(MCPToolTimeoutError) as excinfo:
            registry.execute_tool("hung_tool", {})
        elapsed = time.monotonic() - start
        assert excinfo.value.code == -32008
        assert excinfo.value.data["tool_name"] == "hung_tool"
        assert excinfo.value.data["timeout"] == 0.2
        assert elapsed < 1.0  # bounded wait, not the full 1.5s sleep

    @pytest.mark.unit
    def test_timed_tool_completing_within_timeout_returns_result(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="quick_tool",
            func=lambda: {"ok": True},
            schema={},
            description="fast",
            timeout=5.0,
        )
        assert registry.execute_tool("quick_tool", {}) == {"ok": True}

    @pytest.mark.unit
    def test_untimed_tool_keeps_unbounded_inline_path(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="slow_untimed_tool",
            func=lambda: {"finished": True},
            schema={},
            description="no timeout registered",
        )
        assert registry.execute_tool("slow_untimed_tool", {}) == {"finished": True}

    @pytest.mark.unit
    def test_timeout_counts_as_failed_request(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="hung_tool",
            func=lambda: time.sleep(1.5),
            schema={},
            description="sleeps past its timeout",
            timeout=0.1,
        )
        with pytest.raises(MCPToolTimeoutError):
            registry.execute_tool("hung_tool", {})
        assert registry._performance_metrics.failed_requests == 1
        assert registry._performance_metrics.error_counts["hung_tool"] == 1
        assert registry._error_count == 1

    @pytest.mark.unit
    def test_timeout_not_double_wrapped_as_execution_error(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="hung_tool",
            func=lambda: time.sleep(1.5),
            schema={},
            description="sleeps past its timeout",
            timeout=0.1,
        )
        with pytest.raises(MCPToolTimeoutError) as excinfo:
            registry.execute_tool("hung_tool", {})
        # -32008 (tool timeout), not -32603 (generic execution failure).
        assert excinfo.value.code == -32008


class TestTimeoutWireContract:
    """The reserved code must reach JSON-RPC clients, not just the registry."""

    @pytest.mark.unit
    def test_server_core_maps_timeout_to_reserved_code(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="hung_tool",
            func=lambda: time.sleep(1.5),
            schema={},
            description="sleeps past its timeout",
            timeout=0.1,
        )
        server = MCPServer(mcp_instance=registry)
        response: dict[str, Any] | None = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "t-1",
                "method": "tools/call",
                "params": {"name": "hung_tool", "arguments": {}},
            }
        )
        assert response is not None
        assert response["id"] == "t-1"
        assert response["error"]["code"] == -32008
        assert "timed out" in response["error"]["message"]

    @pytest.mark.unit
    def test_stdio_transport_maps_timeout_to_reserved_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import queue as _queue

        from gnn.mcp.server_stdio import StdioServer

        registry = _registry()
        registry.register_tool(
            name="hung_tool",
            func=lambda: time.sleep(1.5),
            schema={},
            description="sleeps past its timeout",
            timeout=0.1,
        )
        monkeypatch.setattr("gnn.mcp.server_stdio.mcp_instance", registry)
        server = StdioServer()
        server._process_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": "t-2",
                "method": "mcp.tool.execute",
                "params": {"name": "hung_tool", "params": {}},
            }
        )
        messages: list[dict[str, Any]] = []
        while True:
            try:
                messages.append(server.response_queue.get_nowait())
            except _queue.Empty:
                break
        assert messages, "stdio transport must emit an error response"
        error = messages[-1]
        assert error["id"] == "t-2"
        assert error["error"]["code"] == -32008
