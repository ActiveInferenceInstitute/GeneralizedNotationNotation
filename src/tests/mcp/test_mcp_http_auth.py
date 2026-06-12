from __future__ import annotations

import http.client
import json
import threading
from http.server import HTTPServer
from typing import Any, cast

import pytest

from mcp.meta_mcp import get_mcp_auth_status
from mcp.server_http import (
    _RATE_LIMIT_STATE,
    MCPHTTPHandler,
    allow_insecure_local_http,
    get_http_capabilities,
    get_safe_http_resource_uris,
    get_safe_http_tool_names,
    initialize,
    is_authorized,
    is_rate_limited,
    is_safe_http_resource,
    is_safe_http_tool,
)


def test_mcp_http_rejects_requests_without_token_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GNN_MCP_TOKEN", raising=False)
    monkeypatch.delenv("GNN_MCP_ALLOW_INSECURE_LOCAL", raising=False)
    assert is_authorized({}) is False


def test_mcp_http_insecure_local_opt_in_allows_missing_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GNN_MCP_TOKEN", raising=False)
    monkeypatch.setenv("GNN_MCP_ALLOW_INSECURE_LOCAL", "1")
    assert allow_insecure_local_http() is True
    assert is_authorized({}, client_host="127.0.0.1") is True
    assert is_authorized({}, client_host="::1") is True
    assert is_authorized({}, client_host="192.0.2.10") is False
    assert is_authorized({}) is False


def test_mcp_http_requires_matching_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GNN_MCP_TOKEN", "secret-token")
    assert is_authorized({}) is False
    assert is_authorized({"Authorization": "Bearer wrong"}) is False
    assert is_authorized({"Authorization": "Bearer secret-token"}) is True


def test_mcp_http_rate_limit_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GNN_MCP_RATE_LIMIT_PER_MINUTE", raising=False)
    _RATE_LIMIT_STATE.clear()

    assert is_rate_limited("127.0.0.1", now=100.0) is False


def test_mcp_http_rate_limit_blocks_after_configured_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GNN_MCP_RATE_LIMIT_PER_MINUTE", "2")
    _RATE_LIMIT_STATE.clear()

    assert is_rate_limited("127.0.0.1", now=100.0) is False
    assert is_rate_limited("127.0.0.1", now=101.0) is False
    assert is_rate_limited("127.0.0.1", now=102.0) is True
    assert is_rate_limited("127.0.0.1", now=200.0) is False


def test_mcp_http_exposes_only_safe_tools_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GNN_MCP_ALLOW_UNSAFE_TOOLS", raising=False)
    monkeypatch.delenv("GNN_MCP_SAFE_TOOLS", raising=False)

    assert is_safe_http_tool("get_pipeline_status") is True
    assert is_safe_http_tool("get_environment_info") is False
    assert is_safe_http_tool("get_system_info") is False
    assert is_safe_http_tool("process_execute") is False


def test_mcp_http_resources_are_denied_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GNN_MCP_SAFE_RESOURCES", raising=False)
    assert get_safe_http_resource_uris() == set()
    assert is_safe_http_resource("gnn://pipeline/status") is False

    monkeypatch.setenv("GNN_MCP_SAFE_RESOURCES", "gnn://pipeline/status")
    assert is_safe_http_resource("gnn://pipeline/status") is True
    assert is_safe_http_resource("gnn://pipeline/config") is False


def test_mcp_http_capabilities_are_filtered_to_safe_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GNN_MCP_ALLOW_UNSAFE_TOOLS", raising=False)
    monkeypatch.delenv("GNN_MCP_SAFE_RESOURCES", raising=False)
    initialize(force_refresh=True)

    capabilities = get_http_capabilities()
    tool_names = {tool["name"] for tool in capabilities["tools"]}

    assert "get_pipeline_status" in tool_names
    assert "process_execute" not in tool_names
    assert "get_environment_info" not in tool_names
    assert capabilities["resources"] == []
    assert (
        capabilities["server"]["http_access"]["resource_allowlist_env"]
        == "GNN_MCP_SAFE_RESOURCES"
    )


def test_mcp_auth_status_documents_resource_allowlist() -> None:
    status = get_mcp_auth_status(None)
    assert status["access_level"] == "safe_http_tool_and_resource_allowlists"
    assert "GNN_MCP_SAFE_RESOURCES" in " ".join(status["recommendations"])


def test_mcp_http_safe_tools_can_be_extended_or_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GNN_MCP_SAFE_TOOLS", "process_execute")
    monkeypatch.delenv("GNN_MCP_ALLOW_UNSAFE_TOOLS", raising=False)
    assert is_safe_http_tool("process_execute") is True

    monkeypatch.setenv("GNN_MCP_ALLOW_UNSAFE_TOOLS", "true")
    assert get_safe_http_tool_names() is None
    assert is_safe_http_tool("any_registered_tool") is True


def test_mcp_http_jsonrpc_smoke_auth_safe_tool_and_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GNN_MCP_TOKEN", "local-dev-token")
    monkeypatch.delenv("GNN_MCP_RATE_LIMIT_PER_MINUTE", raising=False)
    monkeypatch.delenv("GNN_MCP_ALLOW_UNSAFE_TOOLS", raising=False)
    monkeypatch.delenv("GNN_MCP_SAFE_TOOLS", raising=False)
    _RATE_LIMIT_STATE.clear()
    initialize(force_refresh=True)
    server, thread = _start_http_server()
    try:
        assert _post_jsonrpc(server.server_port, {}, token=None)[0] == 401
        assert _post_jsonrpc(server.server_port, {}, token="wrong")[0] == 401

        unsafe_status, unsafe_payload = _post_jsonrpc(
            server.server_port,
            {
                "jsonrpc": "2.0",
                "id": "unsafe",
                "method": "mcp.tool.execute",
                "params": {"name": "process_execute", "params": {}},
            },
            token="local-dev-token",
        )
        assert unsafe_status == 200
        assert unsafe_payload["error"]["code"] == -32001

        resource_status, resource_payload = _post_jsonrpc(
            server.server_port,
            {
                "jsonrpc": "2.0",
                "id": "resource",
                "method": "mcp.resource.get",
                "params": {"uri": "gnn://pipeline/status"},
            },
            token="local-dev-token",
        )
        assert resource_status == 200
        assert resource_payload["error"]["code"] == -32002

        capabilities_status, capabilities_payload = _post_jsonrpc(
            server.server_port,
            {"jsonrpc": "2.0", "id": "caps", "method": "mcp.capabilities"},
            token="local-dev-token",
        )
        assert capabilities_status == 200
        capability_tool_names = {
            tool["name"] for tool in capabilities_payload["result"]["tools"]
        }
        assert "process_execute" not in capability_tool_names
        assert capabilities_payload["result"]["resources"] == []

        safe_status, safe_payload = _post_jsonrpc(
            server.server_port,
            {"jsonrpc": "2.0", "id": "safe", "method": "get_pipeline_status"},
            token="local-dev-token",
        )
        assert safe_status == 200
        assert safe_payload["id"] == "safe"
        assert "result" in safe_payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    monkeypatch.setenv("GNN_MCP_RATE_LIMIT_PER_MINUTE", "1")
    _RATE_LIMIT_STATE.clear()
    server, thread = _start_http_server()
    try:
        first_status, _ = _post_jsonrpc(
            server.server_port,
            {"jsonrpc": "2.0", "id": "first", "method": "mcp.capabilities"},
            token="local-dev-token",
        )
        second_status, second_payload = _post_jsonrpc(
            server.server_port,
            {"jsonrpc": "2.0", "id": "second", "method": "mcp.capabilities"},
            token="local-dev-token",
        )
        assert first_status == 200
        assert second_status == 429
        assert "rate limit" in second_payload["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_mcp_http_rate_limit_applies_before_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GNN_MCP_TOKEN", "local-dev-token")
    monkeypatch.setenv("GNN_MCP_RATE_LIMIT_PER_MINUTE", "1")
    _RATE_LIMIT_STATE.clear()

    server, thread = _start_http_server()
    try:
        first_status, _ = _post_jsonrpc(server.server_port, {}, token="wrong")
        second_status, second_payload = _post_jsonrpc(
            server.server_port, {}, token="wrong"
        )
        assert first_status == 401
        assert second_status == 429
        assert "rate limit" in second_payload["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _start_http_server() -> tuple[HTTPServer, threading.Thread]:
    server = HTTPServer(("127.0.0.1", 0), MCPHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _post_jsonrpc(
    port: int, payload: dict[str, Any], *, token: str | None
) -> tuple[int, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("POST", "/", body=json.dumps(payload), headers=headers)
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        parsed = json.loads(body) if body else {}
        return response.status, cast(dict[str, Any], parsed)
    finally:
        conn.close()
