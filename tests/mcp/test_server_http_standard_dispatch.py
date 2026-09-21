"""Standard MCP (2024-11-05) dispatch tests for the HTTP transport.

Pins that ``MCPHTTPHandler`` routes the standard method surface through
``server_core.MCPServer`` (per-call, with the HTTP-filtered capabilities
getter), that the HTTP exposure allowlists apply to both surfaces, and
that the direct dialect keeps its raw-result contract on the same wire.
"""

from __future__ import annotations

import http.client
import json
import threading
from http.server import HTTPServer
from typing import Any, cast

import pytest

pytestmark = pytest.mark.mcp

import gnn.mcp.server_http as server_http
from gnn.mcp.mcp import MCP
from gnn.mcp.server_http import (
    _RATE_LIMIT_STATE,
    DEFAULT_SAFE_HTTP_TOOL_NAMES,
    MCPHTTPHandler,
)

DISPATCH_TOKEN = "dispatch-test-token"
ECHO_TOOL = "dispatch_echo_tool"
SHADOW_TOOL = "dispatch_shadow_tool"
ECHO_RESOURCE = "gnn://dispatch/echo"


def _register_dispatch_registry() -> MCP:
    registry = MCP(enable_caching=False, enable_rate_limiting=False)
    registry.register_tool(
        name=ECHO_TOOL,
        func=lambda **kwargs: {"echo": kwargs},
        schema={},
        description="Echoes its arguments",
    )
    registry.register_tool(
        name=SHADOW_TOOL,
        func=lambda: {"shadow": True},
        schema={},
        description="Registered but not HTTP-exposed",
    )
    registry.register_resource(
        uri_template=ECHO_RESOURCE,
        retriever=lambda uri: {"payload": uri},
        description="Dispatch test resource",
    )
    return registry


def _dispatch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNN_MCP_TOKEN", DISPATCH_TOKEN)
    monkeypatch.setenv("GNN_MCP_SAFE_TOOLS", ECHO_TOOL)
    monkeypatch.setenv("GNN_MCP_SAFE_RESOURCES", ECHO_RESOURCE)
    monkeypatch.delenv("GNN_MCP_RATE_LIMIT_PER_MINUTE", raising=False)
    monkeypatch.delenv("GNN_MCP_ALLOW_UNSAFE_TOOLS", raising=False)


def _start_server() -> tuple[HTTPServer, threading.Thread]:
    server = HTTPServer(("127.0.0.1", 0), MCPHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _post_jsonrpc(
    port: int, payload: dict[str, Any], *, token: str | None = DISPATCH_TOKEN
) -> tuple[int, str, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("POST", "/", body=json.dumps(payload), headers=headers)
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        parsed = json.loads(body) if body else {}
        return response.status, body, cast(dict[str, Any], parsed)
    finally:
        conn.close()


class TestStandardRoundTripOverHTTP:
    """Standard methods answer over the real HTTP wire through server_core."""

    def _boot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[HTTPServer, threading.Thread]:
        _dispatch_env(monkeypatch)
        _RATE_LIMIT_STATE.clear()
        monkeypatch.setattr(server_http, "mcp_instance", _register_dispatch_registry())
        return _start_server()

    @pytest.mark.unit
    def test_full_standard_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, thread = self._boot(monkeypatch)
        try:
            port = server.server_port

            # initialize: filtered capabilities prove the getter seam.
            status, _, payload = _post_jsonrpc(
                port,
                {
                    "jsonrpc": "2.0",
                    "id": "init-1",
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                },
            )
            assert status == 200
            assert payload["id"] == "init-1"
            assert payload["result"]["protocolVersion"] == "2024-11-05"
            http_access = payload["result"]["capabilities"]["server"]["http_access"]
            assert http_access["safe_tools_only"] is True
            assert http_access["safe_tool_count"] == 1
            assert payload["result"]["serverInfo"]["name"] == "GNN MCP Server"

            # notifications/initialized: notification → 204 empty body.
            status, raw, _ = _post_jsonrpc(
                port,
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            assert status == 204
            assert raw == ""

            # tools/list: only the allowlisted tool is visible.
            status, _, payload = _post_jsonrpc(
                port, {"jsonrpc": "2.0", "id": "list-1", "method": "tools/list"}
            )
            assert status == 200
            tool_names = {tool["name"] for tool in payload["result"]["tools"]}
            assert tool_names == {ECHO_TOOL}

            # tools/call: content-wrapped text result.
            status, _, payload = _post_jsonrpc(
                port,
                {
                    "jsonrpc": "2.0",
                    "id": "call-1",
                    "method": "tools/call",
                    "params": {
                        "name": ECHO_TOOL,
                        "arguments": {"value": "hello"},
                    },
                },
            )
            assert status == 200
            assert "error" not in payload
            content = payload["result"]["content"]
            assert content[0]["type"] == "text"
            assert json.loads(content[0]["text"]) == {"echo": {"value": "hello"}}

            # resources/read: allowlisted URI returns standard contents.
            status, _, payload = _post_jsonrpc(
                port,
                {
                    "jsonrpc": "2.0",
                    "id": "res-1",
                    "method": "resources/read",
                    "params": {"uri": ECHO_RESOURCE},
                },
            )
            assert status == 200
            contents = payload["result"]["contents"]
            assert contents[0]["uri"] == ECHO_RESOURCE
            assert json.loads(contents[0]["text"]) == {"payload": ECHO_RESOURCE}
            # ping: empty result object.
            status, _, payload = _post_jsonrpc(
                port, {"jsonrpc": "2.0", "id": "ping-1", "method": "ping"}
            )
            assert status == 200
            assert payload["result"] == {}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            _RATE_LIMIT_STATE.clear()


class TestStandardExposurePolicy:
    """The HTTP allowlists gate the standard surface before server_core runs."""

    def _boot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[HTTPServer, threading.Thread]:
        _dispatch_env(monkeypatch)
        _RATE_LIMIT_STATE.clear()
        monkeypatch.setattr(server_http, "mcp_instance", _register_dispatch_registry())
        return _start_server()

    @pytest.mark.unit
    def test_unexposed_tool_call_is_rejected_32001(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server, thread = self._boot(monkeypatch)
        try:
            status, _, payload = _post_jsonrpc(
                server.server_port,
                {
                    "jsonrpc": "2.0",
                    "id": "shadow-1",
                    "method": "tools/call",
                    "params": {"name": SHADOW_TOOL, "arguments": {}},
                },
            )
            assert status == 200
            assert payload["id"] == "shadow-1"
            assert payload["error"]["code"] == -32001
            assert SHADOW_TOOL in payload["error"]["message"]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            _RATE_LIMIT_STATE.clear()

    @pytest.mark.unit
    def test_unexposed_resource_read_is_rejected_32002(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server, thread = self._boot(monkeypatch)
        try:
            status, _, payload = _post_jsonrpc(
                server.server_port,
                {
                    "jsonrpc": "2.0",
                    "id": "res-x",
                    "method": "resources/read",
                    "params": {"uri": "gnn://dispatch/other"},
                },
            )
            assert status == 200
            assert payload["error"]["code"] == -32002
            assert "gnn://dispatch/other" in payload["error"]["message"]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            _RATE_LIMIT_STATE.clear()


class TestDirectDialectRegression:
    """The direct dialect keeps raw results and filtered capabilities."""

    def _boot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[HTTPServer, threading.Thread]:
        _dispatch_env(monkeypatch)
        _RATE_LIMIT_STATE.clear()
        monkeypatch.setattr(server_http, "mcp_instance", _register_dispatch_registry())
        return _start_server()

    @pytest.mark.unit
    def test_capabilities_both_capitalizations_are_filtered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server, thread = self._boot(monkeypatch)
        try:
            for method in ("mcp.capabilities", "get_mcp_server_capabilities"):
                status, _, payload = _post_jsonrpc(
                    server.server_port,
                    {"jsonrpc": "2.0", "id": method, "method": method},
                )
                assert status == 200
                assert payload["id"] == method
                result = payload["result"]
                tool_names = {tool["name"] for tool in result["tools"]}
                assert tool_names == {ECHO_TOOL}
                assert "http_access" in result["server"]
                assert result["server"]["http_access"]["safe_tool_count"] == 1
                resource_templates = {
                    resource["uri_template"] for resource in result["resources"]
                }
                assert resource_templates == {ECHO_RESOURCE}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            _RATE_LIMIT_STATE.clear()

    @pytest.mark.unit
    def test_tool_execute_and_direct_call_stay_raw(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server, thread = self._boot(monkeypatch)
        try:
            port = server.server_port

            # mcp.tool.execute: RAW result, never content-wrapped.
            status, _, payload = _post_jsonrpc(
                port,
                {
                    "jsonrpc": "2.0",
                    "id": "raw-1",
                    "method": "mcp.tool.execute",
                    "params": {"name": ECHO_TOOL, "params": {"value": "raw"}},
                },
            )
            assert status == 200
            assert payload["result"] == {"echo": {"value": "raw"}}
            assert "content" not in payload["result"]

            # Direct tool-name invocation: RAW result.
            status, _, payload = _post_jsonrpc(
                port,
                {
                    "jsonrpc": "2.0",
                    "id": "raw-2",
                    "method": ECHO_TOOL,
                    "params": {"value": "raw2"},
                },
            )
            assert status == 200
            assert payload["result"] == {"echo": {"value": "raw2"}}
            assert "content" not in payload["result"]

            # mcp.resource.get: RAW resource result.
            status, _, payload = _post_jsonrpc(
                port,
                {
                    "jsonrpc": "2.0",
                    "id": "raw-3",
                    "method": "mcp.resource.get",
                    "params": {"uri": ECHO_RESOURCE},
                },
            )
            assert status == 200
            assert payload["result"]["content"] == {"payload": ECHO_RESOURCE}
            assert "contents" not in payload["result"]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            _RATE_LIMIT_STATE.clear()

    @pytest.mark.unit
    def test_default_safe_tool_names_still_apply(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _dispatch_env(monkeypatch)
        assert ECHO_TOOL in server_http.get_safe_http_tool_names()  # type: ignore[union-attr]
        assert ECHO_TOOL not in DEFAULT_SAFE_HTTP_TOOL_NAMES


class TestStandardTransportSeams:
    """Bearer auth and rate limiting apply before standard dispatch."""

    @pytest.mark.unit
    def test_initialize_requires_bearer_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _dispatch_env(monkeypatch)
        _RATE_LIMIT_STATE.clear()
        monkeypatch.setattr(server_http, "mcp_instance", _register_dispatch_registry())
        server, thread = _start_server()
        try:
            request = {
                "jsonrpc": "2.0",
                "id": "init-auth",
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            }
            status, _, _ = _post_jsonrpc(server.server_port, request, token=None)
            assert status == 401
            status, _, _ = _post_jsonrpc(
                server.server_port, request, token="wrong-token"
            )
            assert status == 401
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            _RATE_LIMIT_STATE.clear()

    @pytest.mark.unit
    def test_initialize_is_rate_limited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _dispatch_env(monkeypatch)
        monkeypatch.setenv("GNN_MCP_RATE_LIMIT_PER_MINUTE", "1")
        _RATE_LIMIT_STATE.clear()
        monkeypatch.setattr(server_http, "mcp_instance", _register_dispatch_registry())
        server, thread = _start_server()
        try:
            request = {
                "jsonrpc": "2.0",
                "id": "init-rl",
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            }
            first_status, _, _ = _post_jsonrpc(server.server_port, request)
            second_status, _, second_payload = _post_jsonrpc(
                server.server_port, request
            )
            assert first_status == 200
            assert second_status == 429
            assert "rate limit" in second_payload["error"]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            _RATE_LIMIT_STATE.clear()


class TestStandardErrorContracts:
    """Unknown methods and invalid tool names produce standard envelopes."""

    def _boot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[HTTPServer, threading.Thread]:
        _dispatch_env(monkeypatch)
        _RATE_LIMIT_STATE.clear()
        monkeypatch.setattr(server_http, "mcp_instance", _register_dispatch_registry())
        return _start_server()

    @pytest.mark.unit
    def test_unknown_method_returns_32601_with_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server, thread = self._boot(monkeypatch)
        try:
            status, _, payload = _post_jsonrpc(
                server.server_port,
                {
                    "jsonrpc": "2.0",
                    "id": "unknown-9",
                    "method": "totally/unknown/method",
                },
            )
            assert status == 200
            assert payload["id"] == "unknown-9"
            assert payload["error"]["code"] == -32601
            assert payload["error"]["message"] == "Method not found"
            assert payload["error"]["data"] == (
                "Method 'totally/unknown/method' not found"
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            _RATE_LIMIT_STATE.clear()

    @pytest.mark.unit
    def test_tools_call_requires_string_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server, thread = self._boot(monkeypatch)
        try:
            port = server.server_port
            for request_id, params in (
                ("noname-1", {"arguments": {}}),
                ("badname-2", {"name": 42, "arguments": {}}),
            ):
                status, _, payload = _post_jsonrpc(
                    port,
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "tools/call",
                        "params": params,
                    },
                )
                assert status == 200
                assert payload["id"] == request_id
                assert payload["error"]["code"] == -32602
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            _RATE_LIMIT_STATE.clear()
