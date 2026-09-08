"""Wire-serialization contract tests for the MCP transports (wave-2 MAJ-08).

Tool results are produced by arbitrary callables, so no transport may assume
a payload is JSON-serializable. These tests pin the shared
``gnn.mcp.jsonrpc.serialize_response`` choke point:

- sets/bytes/datetimes/NaN results sanitize into wire-valid JSON instead of
  crashing the stdio writer, aborting the HTTP response, or emitting bare
  ``NaN``/``Infinity`` tokens (invalid per RFC 8259 §6);
- a cyclic (deeply unserializable) result degrades to a protocol-valid -32603
  error envelope with the request id preserved on every transport;
- result-cache keys type-tag non-JSON-native params so ``{1}`` and ``"{1}"``
  can never alias to one key.
"""

import datetime
import io
import json
import sys
import threading
import time
from http.server import HTTPServer
from typing import Any, cast

import pytest

pytestmark = pytest.mark.mcp

from gnn.mcp.jsonrpc import (
    INTERNAL_ERROR,
    serialize_response,
    tag_non_json_values,
)
from gnn.mcp.mcp import MCP
from gnn.mcp.server_core import MCPServer
from gnn.mcp.server_http import (
    _RATE_LIMIT_STATE,
    MCPHTTPHandler,
    initialize,
)
from gnn.mcp.server_stdio import StdioServer


class TestSerializeResponse:
    """sanitize_json_value must make any tool result wire-valid."""

    @pytest.mark.unit
    def test_non_finite_floats_become_canonical_tokens(self) -> None:
        payload = {"nan": float("nan"), "inf": float("inf"), "neg": float("-inf")}
        text = serialize_response(payload)
        assert json.loads(text) == {"nan": "NaN", "inf": "Infinity", "neg": "-Infinity"}

    @pytest.mark.unit
    def test_non_json_native_types_sanitize(self) -> None:
        payload = {
            "set": {3, 1, 2},
            "frozenset": frozenset({"b", "a"}),
            "bytes": b"raw\xff",
            "when": datetime.datetime(2026, 9, 8, 12, 0, 0),
            "tuple": (1, 2),
        }
        parsed = json.loads(serialize_response(payload))
        assert parsed["set"] == [1, 2, 3]  # deterministic order, valid JSON
        assert parsed["frozenset"] == ["a", "b"]
        assert parsed["bytes"] == "raw\\xff"
        assert parsed["when"] == "2026-09-08T12:00:00"
        assert parsed["tuple"] == [1, 2]

    @pytest.mark.unit
    def test_set_serialization_is_deterministic(self) -> None:
        text = serialize_response({"s": {"x", "y", "z", "w", "v"}})
        for _ in range(10):
            assert serialize_response({"s": {"x", "y", "z", "w", "v"}}) == text

    @pytest.mark.unit
    def test_cyclic_structure_raises_for_fallback(self) -> None:
        cyclic: dict[str, Any] = {"ok": True}
        cyclic["self"] = cyclic
        with pytest.raises((RecursionError, ValueError)):
            serialize_response(cyclic)


class TestResultCacheKeyAliasing:
    """Cache keys must distinguish structurally distinct params."""

    @pytest.mark.unit
    def test_set_param_never_aliases_string_param(self) -> None:
        key_set = MCP._result_cache_key("t", {"x": {1}})
        key_str = MCP._result_cache_key("t", {"x": "{1}"})
        assert key_set != key_str
        assert key_set != ""

    @pytest.mark.unit
    def test_non_finite_param_does_not_alias_its_string(self) -> None:
        key_nan = MCP._result_cache_key("t", {"x": float("nan")})
        key_str = MCP._result_cache_key("t", {"x": "nan"})
        assert key_nan != key_str


class TestServerCoreWireSafety:
    """In-process transport: sanitized results, -32603 for cycles."""

    @pytest.fixture
    def server(self) -> MCPServer:
        registry = MCP(enable_caching=False, enable_rate_limiting=False)
        return MCPServer(mcp_instance=registry)

    def _register(self, server: MCPServer, name: str, result: Any) -> None:
        server.mcp.register_tool(
            name=name,
            func=lambda: result,
            schema={},
            description="wire test tool",
        )

    @pytest.mark.unit
    def test_tool_returning_set_is_wire_valid(self, server: MCPServer) -> None:
        self._register(server, "bad_set_tool", {"s": {1, 2}})
        response = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "bad_set_tool", "arguments": {}},
            }
        )
        assert response is not None
        # The whole envelope serializes; the embedded text is valid JSON.
        envelope_text = json.dumps(response)
        assert json.loads(envelope_text)["id"] == 1
        text = response["result"]["content"][0]["text"]
        assert json.loads(text) == {"s": [1, 2]}

    @pytest.mark.unit
    def test_tool_returning_nan_is_wire_valid(self, server: MCPServer) -> None:
        self._register(server, "bad_nan_tool", {"p": float("nan")})
        response = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "bad_nan_tool", "arguments": {}},
            }
        )
        assert response is not None
        text = response["result"]["content"][0]["text"]
        assert "NaN" not in json.dumps(json.loads(text)) or '"NaN"' in text
        assert json.loads(text) == {"p": "NaN"}

    @pytest.mark.unit
    def test_cyclic_result_yields_32603_envelope(self, server: MCPServer) -> None:
        cyclic: dict[str, Any] = {}
        cyclic["self"] = cyclic
        self._register(server, "cyclic_tool", cyclic)
        response = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "cyclic_tool", "arguments": {}},
            }
        )
        assert response is not None
        assert response["id"] == 3
        assert response["error"]["code"] == INTERNAL_ERROR
        json.dumps(response)  # the error envelope itself must be wire-valid


class _StreamSys:
    """Namespace exposing ``stdout`` as the capture stream; rest delegates."""

    def __init__(self, real: Any, stream: io.StringIO) -> None:
        self._real = real
        self.stdout = stream

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class TestStdioWriterWireSafety:
    """The writer thread must never hang or emit invalid JSON."""

    def _run_writer(self, monkeypatch: pytest.MonkeyPatch, message: Any) -> str:
        import gnn.mcp.server_stdio as server_stdio_module

        server = StdioServer()
        stream = io.StringIO()
        server.running = True
        server.response_queue.put(message)
        monkeypatch.setattr(server_stdio_module, "sys", _StreamSys(sys, stream))
        writer = threading.Thread(target=server._writer_thread, daemon=True)
        writer.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not stream.getvalue():
            time.sleep(0.02)
        server.running = False
        writer.join(timeout=5)
        return stream.getvalue()

    @pytest.mark.unit
    def test_unserializable_result_still_writes_valid_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = self._run_writer(
            monkeypatch, {"jsonrpc": "2.0", "id": 7, "result": {"s": {1}}}
        )
        assert output.strip(), "writer must emit a line for a bad result"
        parsed = json.loads(output.strip().splitlines()[-1])
        assert parsed == {"jsonrpc": "2.0", "id": 7, "result": {"s": [1]}}

    @pytest.mark.unit
    def test_cyclic_result_writes_32603_with_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cyclic: dict[str, Any] = {}
        cyclic["self"] = cyclic
        output = self._run_writer(
            monkeypatch, {"jsonrpc": "2.0", "id": 8, "result": cyclic}
        )
        assert output.strip(), "writer must emit a line for a cyclic result"
        parsed = json.loads(output.strip().splitlines()[-1])
        assert parsed["id"] == 8
        assert parsed["error"]["code"] == INTERNAL_ERROR


class TestHTTPWireSafety:
    """The HTTP transport must answer, never abort, on bad tool results."""

    BAD_TOOL = "wire_serialization_bad_tool"

    def _start(self) -> tuple[HTTPServer, threading.Thread]:
        server = HTTPServer(("127.0.0.1", 0), MCPHTTPHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _post(self, port: int, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        import http.client
        import os

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer wire-test-token",
        }
        try:
            conn.request("POST", "/", body=json.dumps(payload), headers=headers)
            response = conn.getresponse()
            body = response.read().decode("utf-8")
            return response.status, cast(dict[str, Any], json.loads(body))
        finally:
            conn.close()
            assert os.environ.get("GNN_MCP_TOKEN") is not None

    @pytest.mark.unit
    def test_unserializable_tool_result_returns_valid_envelope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GNN_MCP_TOKEN", "wire-test-token")
        monkeypatch.setenv("GNN_MCP_SAFE_TOOLS", self.BAD_TOOL)
        _RATE_LIMIT_STATE.clear()
        registry = initialize(force_refresh=True)[0]
        try:
            registry.register_tool(
                name=self.BAD_TOOL,
                func=lambda: {"cyclic": None, "s": {1, 2}},
                schema={},
                description="wire test tool",
            )
            tool = registry.tools[self.BAD_TOOL]
            tool.func = lambda: _cyclic()  # type: ignore[method-assign]

            server, thread = self._start()
            try:
                status, payload = self._post(
                    server.server_port,
                    {
                        "jsonrpc": "2.0",
                        "id": "wire-1",
                        "method": "mcp.tool.execute",
                        "params": {"name": self.BAD_TOOL, "params": {}},
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            assert status == 200
            assert payload["id"] == "wire-1"
            assert payload["error"]["code"] == INTERNAL_ERROR
        finally:
            registry.tools.pop(self.BAD_TOOL, None)

    @pytest.mark.unit
    def test_sanitizable_tool_result_returns_result_envelope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GNN_MCP_TOKEN", "wire-test-token")
        monkeypatch.setenv("GNN_MCP_SAFE_TOOLS", self.BAD_TOOL)
        _RATE_LIMIT_STATE.clear()
        registry = initialize(force_refresh=True)[0]
        try:
            registry.register_tool(
                name=self.BAD_TOOL,
                func=lambda: {"s": {1, 2}, "p": float("nan")},
                schema={},
                description="wire test tool",
            )
            server, thread = self._start()
            try:
                status, payload = self._post(
                    server.server_port,
                    {
                        "jsonrpc": "2.0",
                        "id": "wire-2",
                        "method": "mcp.tool.execute",
                        "params": {"name": self.BAD_TOOL, "params": {}},
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            assert status == 200
            assert payload["id"] == "wire-2"
            assert payload["result"] == {"s": [1, 2], "p": "NaN"}
        finally:
            registry.tools.pop(self.BAD_TOOL, None)


def _cyclic() -> dict[str, Any]:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    return cyclic


def test_tag_non_json_values_shapes() -> None:
    assert tag_non_json_values({"x": {1}}) == {
        "x": {"__json_type__": "set", "items": ["1"]}
    }
    assert tag_non_json_values({"x": "{1}"}) == {"x": "{1}"}
    assert tag_non_json_values(1) == 1
    assert tag_non_json_values(True) is True
    assert tag_non_json_values(float("nan")) == {
        "__json_type__": "float",
        "repr": "nan",
    }
