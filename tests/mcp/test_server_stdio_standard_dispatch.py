"""Standard MCP (2024-11-05) dispatch contract tests for the stdio transport.

Pins that the stdio transport routes the standard surface through
``server_core.MCPServer`` — over the real in-process thread pipeline
(reader → processor → writer) — while the direct dialect keeps its raw,
unwrapped result shapes. The standard surface and the direct dialect must
stay wire-distinguishable: ``tools/call`` returns a content-wrapped text
item, direct calls return the raw tool result.
"""

from __future__ import annotations

import io
import json
import queue
import sys
import threading
import time
from typing import Any

import pytest

pytestmark = pytest.mark.mcp

from gnn.mcp import server_stdio
from gnn.mcp.mcp import MCP


def _registry() -> MCP:
    registry = MCP(enable_caching=False, enable_rate_limiting=False)
    registry.register_tool(
        "echo",
        lambda value: value,
        {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        "Echo a value",
    )
    return registry


def _drain(server: server_stdio.StdioServer) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    while True:
        try:
            messages.append(server.response_queue.get_nowait())
        except queue.Empty:
            return messages


class TestStdioStandardRoundTrip:
    """Full pipeline: real reader/processor/writer threads over fed stdin."""

    def _drive(
        self, monkeypatch: pytest.MonkeyPatch, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        registry = _registry()
        monkeypatch.setattr(server_stdio, "mcp_instance", registry)
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)

        lines = [json.dumps(m).encode("utf-8") + b"\n" for m in messages]
        expected_responses = sum(1 for m in messages if "id" in m)
        cursor = {"i": 0}
        eof_gate = threading.Event()

        def fake_read(fd: int, size: int) -> bytes:
            i = cursor["i"]
            if i < len(lines):
                cursor["i"] = i + 1
                return lines[i]
            eof_gate.wait(timeout=5.0)
            return b""

        monkeypatch.setattr(server_stdio.os, "read", fake_read)

        server = server_stdio.StdioServer()
        server.running = True
        threads = [
            threading.Thread(target=server._reader_thread, daemon=True),
            threading.Thread(target=server._processor_thread, daemon=True),
            threading.Thread(target=server._writer_thread, daemon=True),
        ]
        for thread in threads:
            thread.start()

        # Every request produces exactly one stdout line, in processing
        # order; the notification produces none. Once all expected lines
        # are written, every queued message (FIFO) has been processed.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if out.getvalue().count("\n") >= expected_responses:
                break
            time.sleep(0.01)

        server.running = False
        eof_gate.set()
        for thread in threads:
            thread.join(timeout=5.0)
            assert thread.is_alive() is False

        return [json.loads(line) for line in out.getvalue().splitlines()]

    def test_full_round_trip_over_real_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "contract-test", "version": "0.0.0"},
                },
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"value": 42}},
            },
            {"jsonrpc": "2.0", "id": 4, "method": "ping"},
        ]

        responses = self._drive(monkeypatch, messages)

        # notifications/initialized produced no response line.
        assert len(responses) == 4
        assert [r["id"] for r in responses] == [1, 2, 3, 4]

        initialize = responses[0]["result"]
        assert initialize["protocolVersion"] == "2024-11-05"
        assert initialize["serverInfo"] == {
            "name": "GNN MCP Server",
            "version": "1.0.0",
        }
        assert "capabilities" in initialize

        tools_list = responses[1]["result"]
        tool_names = [t["name"] for t in tools_list["tools"]]
        assert tool_names == ["echo"]

        tool_call = responses[2]
        assert tool_call["result"] == {"content": [{"type": "text", "text": "42"}]}

        assert responses[3]["result"] == {}


class TestStdioDirectDialectRegression:
    """Direct-dialect entries keep their raw result shapes, byte for byte."""

    def _server(self, monkeypatch: pytest.MonkeyPatch) -> server_stdio.StdioServer:
        monkeypatch.setattr(server_stdio, "mcp_instance", _registry())
        return server_stdio.StdioServer()

    @pytest.mark.unit
    def test_mcp_capabilities_returns_full_caps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(monkeypatch)
        server._process_jsonrpc(
            {"jsonrpc": "2.0", "id": 1, "method": "mcp.capabilities"}
        )
        (response,) = _drain(server)
        assert response["result"] == server_stdio.mcp_instance.get_capabilities()
        assert response["result"]["tools"][0]["name"] == "echo"

    @pytest.mark.unit
    def test_get_mcp_server_capabilities_returns_full_caps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(monkeypatch)
        server._process_jsonrpc(
            {"jsonrpc": "2.0", "id": 2, "method": "get_mcp_server_capabilities"}
        )
        (response,) = _drain(server)
        assert response["result"] == server_stdio.mcp_instance.get_capabilities()

    @pytest.mark.unit
    def test_mcp_tool_execute_returns_raw_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(monkeypatch)
        server._process_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "mcp.tool.execute",
                "params": {"name": "echo", "params": {"value": 7}},
            }
        )
        (response,) = _drain(server)
        # RAW result, NOT the standard tools/call content envelope.
        assert response["result"] == 7

    @pytest.mark.unit
    def test_direct_tool_name_call_returns_raw_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(monkeypatch)
        server._process_jsonrpc(
            {"jsonrpc": "2.0", "id": 4, "method": "echo", "params": {"value": 8}}
        )
        (response,) = _drain(server)
        assert response["result"] == 8

    @pytest.mark.unit
    def test_mcp_resource_get_returns_raw_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = _registry()
        registry.register_resource(
            "test://greeting",
            lambda uri: {"greeting": "hello"},
            "A greeting resource",
        )
        monkeypatch.setattr(server_stdio, "mcp_instance", registry)
        server = server_stdio.StdioServer()
        server._process_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "mcp.resource.get",
                "params": {"uri": "test://greeting"},
            }
        )
        (response,) = _drain(server)
        result = response["result"]
        assert result["content"] == {"greeting": "hello"}
        assert result["uri"] == "test://greeting"
        # RAW resource dict, NOT the standard resources/read contents envelope.
        assert "contents" not in result


class TestStdioStandardErrorContract:
    """Error shapes the standard core guarantees through the stdio wiring."""

    def _server(self, monkeypatch: pytest.MonkeyPatch) -> server_stdio.StdioServer:
        monkeypatch.setattr(server_stdio, "mcp_instance", _registry())
        return server_stdio.StdioServer()

    @pytest.mark.unit
    def test_unknown_method_routes_to_standard_32601(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(monkeypatch)
        server._process_jsonrpc(
            {"jsonrpc": "2.0", "id": "u-1", "method": "foo/bar", "params": {}}
        )
        (response,) = _drain(server)
        assert response["id"] == "u-1"
        error = response["error"]
        assert error["code"] == -32601
        assert error["data"] == "Method 'foo/bar' not found"
        # The old direct-dialect "Method not found: <method>" message is gone.
        assert error["message"] != "Method not found: foo/bar"

    @pytest.mark.unit
    def test_tools_call_missing_name_is_32602(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(monkeypatch)
        server._process_jsonrpc(
            {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {}}
        )
        (response,) = _drain(server)
        assert response["id"] == 6
        assert response["error"]["code"] == -32602

    @pytest.mark.unit
    def test_tools_call_non_string_name_is_32602(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(monkeypatch)
        server._process_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": 3, "arguments": {}},
            }
        )
        (response,) = _drain(server)
        assert response["id"] == 7
        assert response["error"]["code"] == -32602

    @pytest.mark.unit
    def test_tools_call_mcp_error_code_passthrough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A timed-out tool raises MCPToolTimeoutError (an MCPError with its
        # own reserved code); server_core passes e.code straight to the wire.
        registry = _registry()
        registry.register_tool(
            "hung_tool",
            lambda: time.sleep(1.5),
            {"type": "object", "properties": {}},
            "sleeps past its timeout",
            timeout=0.1,
        )
        monkeypatch.setattr(server_stdio, "mcp_instance", registry)
        server = server_stdio.StdioServer()
        server._process_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {"name": "hung_tool", "arguments": {}},
            }
        )
        (response,) = _drain(server)
        assert response["id"] == 8
        assert response["error"]["code"] == -32008

    @pytest.mark.unit
    def test_tools_call_unexpected_exception_is_32603(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = _registry()
        registry.register_tool(
            "boomer", _boom, {"type": "object", "properties": {}}, "always raises"
        )
        monkeypatch.setattr(server_stdio, "mcp_instance", registry)
        server = server_stdio.StdioServer()
        server._process_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "boomer", "arguments": {}},
            }
        )
        (response,) = _drain(server)
        assert response["id"] == 9
        assert response["error"]["code"] == -32603


def _boom() -> None:
    raise ValueError("boom")


class TestStdioNotificationSilence:
    """A no-id notification executes and enqueues nothing."""

    @pytest.mark.unit
    def test_initialized_notification_enqueues_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server_stdio, "mcp_instance", _registry())
        server = server_stdio.StdioServer()
        server._process_jsonrpc(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        assert _drain(server) == []

    @pytest.mark.unit
    def test_unknown_method_notification_enqueues_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server_stdio, "mcp_instance", _registry())
        server = server_stdio.StdioServer()
        server._process_jsonrpc({"jsonrpc": "2.0", "method": "foo/bar"})
        assert _drain(server) == []


class TestMCPErrorSurfacesDirectly:
    """MCPError raised outside tool execution keeps its code on the wire."""

    @pytest.mark.unit
    def test_unknown_tool_name_is_32601_mcp_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # tools/call of an unregistered name raises MCPToolNotFoundError
        # (code -32601) from the registry; server_core passes e.code through.
        monkeypatch.setattr(server_stdio, "mcp_instance", _registry())
        server = server_stdio.StdioServer()
        server._process_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "no_such_tool", "arguments": {}},
            }
        )
        (response,) = _drain(server)
        assert response["id"] == 10
        assert response["error"]["code"] == -32601
