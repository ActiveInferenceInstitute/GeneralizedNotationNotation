"""Exercise JSON-RPC HTTP envelopes on a real local server."""

import io
import json
import socket
import socketserver
import threading
from typing import Any, cast

import pytest

from gnn.mcp import server_http
from gnn.mcp.mcp import MCP

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

    def __init__(self, body: bytes, content_length: str | None = None):
        header_value = content_length if content_length is not None else str(len(body))
        self.input = io.BytesIO(
            b"POST / HTTP/1.0\r\nAuthorization: Bearer test-local-token\r\n"
            + f"Content-Length: {header_value}\r\n\r\n".encode()
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
    from gnn.mcp import server_stdio
    from gnn.mcp.server_core import MCPServer

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


def post_with_length_header(
    payload: object, length_header: str
) -> tuple[int, dict[str, Any] | None]:
    """POST with an arbitrary raw Content-Length header value."""
    body = json.dumps(payload).encode()
    stream = HTTPByteStream(body, content_length=length_header)
    server_http.MCPHTTPHandler(
        cast(socket.socket, stream),
        ("127.0.0.1", 1234),
        cast(socketserver.BaseServer, None),
    )
    headers, out_body = stream.output.getvalue().split(b"\r\n\r\n", 1)
    return int(headers.split()[1]), json.loads(out_body) if out_body else None


def test_http_malformed_content_length_returns_400(rpc_server: int) -> None:
    """A garbage Content-Length must yield HTTP 400, never an abort."""
    status, body = post_with_length_header({}, length_header="garbage")
    assert status == 400
    assert body is not None
    assert "error" in body


def test_http_oversize_body_returns_413(rpc_server: int) -> None:
    """A body above MAX_REQUEST_BYTES is rejected 413 without reading it."""
    from gnn.mcp.jsonrpc import MAX_REQUEST_BYTES

    status, body = post_with_length_header(
        {}, length_header=str(MAX_REQUEST_BYTES + 1)
    )
    assert status == 413
    assert body is not None
    assert "error" in body


def test_stdio_oversize_message_rejected_with_invalid_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-cap stdio line gets a -32600 envelope; the server keeps running."""
    from gnn.mcp import server_stdio
    from gnn.mcp.jsonrpc import MAX_REQUEST_BYTES

    registry = MCP(enable_caching=False, enable_rate_limiting=False)
    registry.register_tool("echo", lambda: {"ok": True}, {}, "Echo")
    monkeypatch.setattr(server_stdio, "mcp_instance", registry)
    stdio = server_stdio.StdioServer()

    oversize = "x" * (MAX_REQUEST_BYTES + 10)
    fake_stdin = io.StringIO(
        json.dumps({"jsonrpc": "2.0", "method": "echo", "id": 1}) + "\n"
        + oversize
        + "\n"
    )
    monkeypatch.setattr("sys.stdin", fake_stdin)
    stdio.running = True
    reader = threading.Thread(target=stdio._reader_thread, daemon=True)
    reader.start()
    reader.join(timeout=10)
    assert not reader.is_alive()

    # First response is the -32600 oversize rejection.
    response = stdio.response_queue.get(timeout=1)
    assert response["error"]["code"] == -32600
    # The valid line was queued, not dropped.
    message = stdio.request_queue.get_nowait()
    assert message["method"] == "echo"


def test_tools_call_embed_truncates_oversize_results() -> None:
    """The tools/call embed applies the documented truncate policy."""
    from gnn.mcp.server_core import MCPServer
    from gnn.mcp.jsonrpc import (
        MAX_RESPONSE_EMBED_CHARS,
        TRUNCATION_NOTICE,
    )

    registry = MCP(enable_caching=False, enable_rate_limiting=False)
    registry.register_tool(
        "big_matrix",
        lambda: {"matrix": ["x" * 100] * (MAX_RESPONSE_EMBED_CHARS // 50)},
        {},
        "Return a huge matrix",
    )
    core = MCPServer(registry)
    response = core.handle_request(
        {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "big_matrix"},
            "id": 9,
        }
    )
    assert response is not None
    text = response["result"]["content"][0]["text"]
    assert len(text) <= MAX_RESPONSE_EMBED_CHARS + len(TRUNCATION_NOTICE)
    assert text.endswith(TRUNCATION_NOTICE)
