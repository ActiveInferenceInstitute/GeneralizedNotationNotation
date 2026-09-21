"""ThreadingHTTPServer concurrency contract tests (tier-2 row 13, G-6).

``MCPHTTPServer.start()`` must serve each request on its own thread
(``ThreadingHTTPServer``): a handler blocked inside a long-running tool must
not serialize the server — a second, independent request must complete while
the first is still blocked. Deterministic: the block is a pair of
``threading.Event`` objects, never a wall-clock race.
"""

from __future__ import annotations

import http.client
import json
import select
import threading
import time
from typing import Any, cast

import pytest

pytestmark = pytest.mark.mcp

import gnn.mcp.server_http as server_http
from gnn.mcp.mcp import MCP
from gnn.mcp.server_http import (
    _RATE_LIMIT_STATE,
    MCPHTTPHandler,
    MCPHTTPServer,
)

DISPATCH_TOKEN = "threading-test-token"
BLOCK_TOOL = "threading_block_tool"
FAST_TOOL = "threading_fast_tool"


def _register_threading_registry() -> tuple[MCP, threading.Event, threading.Event]:
    registry = MCP(enable_caching=False, enable_rate_limiting=False)
    started = threading.Event()
    release = threading.Event()

    def _blocked() -> dict[str, Any]:
        started.set()
        release.wait(timeout=10.0)
        return {"released": True}

    registry.register_tool(
        name=BLOCK_TOOL,
        func=_blocked,
        schema={},
        description="Signals started, then blocks until the test releases it",
    )
    registry.register_tool(
        name=FAST_TOOL,
        func=lambda: {"fast": True},
        schema={},
        description="Answers immediately",
    )
    return registry, started, release


def _threading_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNN_MCP_TOKEN", DISPATCH_TOKEN)
    monkeypatch.setenv("GNN_MCP_SAFE_TOOLS", f"{BLOCK_TOOL},{FAST_TOOL}")
    monkeypatch.delenv("GNN_MCP_SAFE_RESOURCES", raising=False)
    monkeypatch.delenv("GNN_MCP_RATE_LIMIT_PER_MINUTE", raising=False)
    monkeypatch.delenv("GNN_MCP_ALLOW_UNSAFE_TOOLS", raising=False)


def _open_post(port: int, payload: dict[str, Any]) -> http.client.HTTPConnection:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(
        "POST",
        "/",
        body=json.dumps(payload),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DISPATCH_TOKEN}",
        },
    )
    return conn


def _read_response(conn: http.client.HTTPConnection) -> dict[str, Any]:
    response = conn.getresponse()
    body = response.read().decode("utf-8")
    return cast(dict[str, Any], json.loads(body)) if body else {}


def _try_read_response(
    conn: http.client.HTTPConnection, wait_seconds: float
) -> dict[str, Any] | None:
    """Read the response only if the socket delivers within the bound."""
    sock = conn.sock
    assert sock is not None
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        ready, _, _ = select.select([sock], [], [], 0.1)
        if ready:
            return _read_response(conn)
    return None


class TestThreadingHTTPConcurrency:
    """A blocked request must not stall the request loop."""

    @pytest.mark.unit
    def test_second_request_completes_while_first_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _threading_env(monkeypatch)
        _RATE_LIMIT_STATE.clear()
        registry, started, release = _register_threading_registry()
        monkeypatch.setattr(server_http, "mcp_instance", registry)

        server = server_http.ThreadingHTTPServer(("127.0.0.1", 0), MCPHTTPHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_port

            # Request #1: blocked inside the handler's tool body.
            blocked_conn = _open_post(
                port,
                {
                    "jsonrpc": "2.0",
                    "id": "blocked-1",
                    "method": "tools/call",
                    "params": {"name": BLOCK_TOOL, "arguments": {}},
                },
            )
            try:
                # The handler must already be INSIDE the blocked tool before
                # request #2 is sent, else a serialized server could pass
                # vacuously.
                assert started.wait(timeout=5.0), "blocked tool never started"

                # Request #2: a fast tool on a separate connection must be
                # served WHILE #1 is still blocked. A single-threaded
                # HTTPServer cannot deliver this — accept() is serialized
                # behind the blocked handler.
                fast_conn = _open_post(
                    port,
                    {
                        "jsonrpc": "2.0",
                        "id": "fast-2",
                        "method": "tools/call",
                        "params": {"name": FAST_TOOL, "arguments": {}},
                    },
                )
                try:
                    fast_payload = _try_read_response(fast_conn, wait_seconds=3.0)
                    assert fast_payload is not None, (
                        "fast request never completed while the first was "
                        "blocked: server is serializing requests"
                    )
                    assert "error" not in fast_payload
                    assert fast_payload["id"] == "fast-2"
                finally:
                    fast_conn.close()

                # Release the blocked handler; request #1 must now complete.
                release.set()
                blocked_payload = _read_response(blocked_conn)
                assert "error" not in blocked_payload
                assert blocked_payload["id"] == "blocked-1"
            finally:
                release.set()
                blocked_conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            _RATE_LIMIT_STATE.clear()


class TestStartUsesThreadingHTTPServer:
    """MCPHTTPServer.start() must construct a ThreadingHTTPServer."""

    @pytest.mark.unit
    def test_start_instantiates_threading_http_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        constructed: list[Any] = []
        real_cls = server_http.ThreadingHTTPServer

        class _CapturingServer(real_cls):  # type: ignore[misc,valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                constructed.append(self)

            def serve_forever(self, *args: Any, **kwargs: Any) -> None:
                # Do not run a real accept loop; the server thread exits
                # immediately and start()'s watchdog loop returns.
                return

        monkeypatch.setattr(server_http, "ThreadingHTTPServer", _CapturingServer)
        monkeypatch.setattr(server_http, "initialize", lambda *a, **k: None)

        http_server = MCPHTTPServer("127.0.0.1", 0)
        http_server.start()  # returns once the (no-op) server thread dies

        assert len(constructed) == 1
        assert isinstance(constructed[0], server_http.ThreadingHTTPServer)
        assert constructed[0].daemon_threads is True
        assert http_server.server is not None
        http_server.server.server_close()
