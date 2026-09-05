"""Exercise JSON-RPC HTTP envelopes on a real local server."""

import io
import json
import socket
import socketserver
from typing import Any, cast

import pytest

from mcp import server_http
from mcp.mcp import MCP

pytestmark = pytest.mark.mcp


@pytest.fixture
def rpc_server(monkeypatch: pytest.MonkeyPatch) -> MCP:
    monkeypatch.setenv("GNN_MCP_TOKEN", "test-local-token")
    monkeypatch.setenv("GNN_MCP_RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("GNN_MCP_SAFE_TOOLS", "reliability_echo")
    registry = MCP(
        enable_caching=False, enable_rate_limiting=False, strict_validation=True
    )
    registry.register_tool(
        "reliability_echo",
        lambda value: value,
        {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        "Return an integer",
    )
    monkeypatch.setattr(server_http, "mcp_instance", registry)
    return registry


class HTTPByteStream:
    """Socket interface over actual HTTP bytes; no registry/handler simulation."""

    def __init__(self, body: bytes):
        self.input = io.BytesIO(
            b"POST / HTTP/1.0\r\nAuthorization: Bearer test-local-token\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        self.output = io.BytesIO()

    def makefile(self, mode: str, *args: object) -> io.BytesIO:
        return self.input

    def sendall(self, data: bytes) -> None:
        self.output.write(data)


def post(
    port: object, payload: object, *, raw: bool = False
) -> tuple[int, dict[str, Any] | None]:
    body = str(payload).encode() if raw else json.dumps(payload).encode()
    stream = HTTPByteStream(body)
    server_http.MCPHTTPHandler(
        cast(socket.socket, stream),
        ("127.0.0.1", 1234),
        cast(socketserver.BaseServer, None),
    )
    headers, body = stream.output.getvalue().split(b"\r\n\r\n", 1)
    return int(headers.split()[1]), json.loads(body) if body else None


@pytest.mark.parametrize(
    "payload,code",
    [
        ([], -32600),
        (None, -32600),
        ({"jsonrpc": "2.0", "method": []}, -32600),
        (
            {"jsonrpc": "2.0", "id": 7, "method": "mcp.capabilities", "params": []},
            -32602,
        ),
        (
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "mcp.tool.execute",
                "params": {"name": [], "params": {}},
            },
            -32602,
        ),
        (
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "reliability_echo",
                "params": {"value": "bad"},
            },
            -32602,
        ),
    ],
)
def test_http_invalid_requests_are_jsonrpc(
    rpc_server: int, payload: object, code: int
) -> None:
    status, result = post(rpc_server, payload)
    assert status == 200
    assert result is not None
    assert result["jsonrpc"] == "2.0"
    assert "id" in result
    assert result["error"]["code"] == code


@pytest.mark.parametrize(
    "method,params",
    [("reliability_echo", {"value": 4}), ("missing", {}), ("reliability_echo", [])],
)
def test_http_notifications_have_no_jsonrpc_response(
    rpc_server: int, method: str, params: object
) -> None:
    assert post(rpc_server, {"jsonrpc": "2.0", "method": method, "params": params}) == (
        204,
        None,
    )


def test_http_parse_error_and_explicit_null_id(rpc_server: int) -> None:
    status, result = post(rpc_server, "{", raw=True)
    assert status == 200
    assert result is not None
    assert result["error"]["code"] == -32700
    assert result["id"] is None
    assert post(
        rpc_server,
        {
            "jsonrpc": "2.0",
            "id": None,
            "method": "reliability_echo",
            "params": {"value": 4},
        },
    ) == (200, {"jsonrpc": "2.0", "id": None, "result": 4})


def test_core_and_stdio_notifications_and_null_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp import server_stdio
    from mcp.server_core import MCPServer

    registry = MCP(enable_caching=False, enable_rate_limiting=False)
    registry.register_tool("echo", lambda: {"ok": True}, {}, "Echo")
    core = MCPServer(registry)
    monkeypatch.setattr(server_stdio, "mcp_instance", registry)
    stdio = server_stdio.StdioServer()
    params: object
    request: dict[str, object]
    for params in ({}, [], None):
        request = {"jsonrpc": "2.0", "method": "echo", "params": params}
        assert core.handle_request(request) is None
        stdio._process_message(request)
        assert stdio.response_queue.empty()
    for params in ([], None):
        invalid = {"jsonrpc": "2.0", "method": "echo", "id": 1, "params": params}
        response = core.handle_request(invalid)
        assert response is not None
        assert response["error"]["code"] == -32602
        stdio._process_message(invalid)
        assert stdio.response_queue.get_nowait()["error"]["code"] == -32602
    request = {"jsonrpc": "2.0", "method": "echo", "id": None}
    expected = {"jsonrpc": "2.0", "id": None, "result": {"ok": True}}
    assert core.handle_request(request) == expected
    stdio._process_message(request)
    assert stdio.response_queue.get_nowait() == expected
