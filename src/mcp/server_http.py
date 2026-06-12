#!/usr/bin/env python3
"""
MCP HTTP Server Implementation

This module provides a JSON-RPC 2.0 HTTP server for the Model Context Protocol (MCP),
exposing all registered GNN tools and resources to HTTP clients.

Key Features:
- Supports both standard MCP methods (capabilities, tool/resource execution) and direct tool invocation
- Robust error handling with custom MCP error codes and JSON-RPC compliance
- Detailed logging of all requests and responses
- Extensible for meta-tools and future MCP extensions
"""

import ipaddress
import json
import logging
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)
DEFAULT_SAFE_HTTP_TOOL_NAMES = frozenset(
    {
        "cli.health",
        "cli.preflight",
        "get_pipeline_steps",
        "get_pipeline_status",
        "validate_pipeline_dependencies",
        "get_pipeline_config_info",
        "check_execute_dependencies",
        "get_execute_module_info",
        "get_logging_info",
        "validate_dependencies",
    }
)
DEFAULT_SAFE_HTTP_RESOURCE_URIS: frozenset[str] = frozenset()
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_RATE_LIMIT_STATE: Dict[str, List[float]] = {}

# Import MCP
try:
    from . import MCPError, initialize, mcp_instance
except ImportError:
    from mcp import MCPError, initialize, mcp_instance


def get_required_bearer_token() -> Optional[str]:
    """Return the configured MCP HTTP bearer token, if auth is enabled."""
    token = os.environ.get("GNN_MCP_TOKEN")
    if token is None or token.strip() == "":
        return None
    return token


def allow_insecure_local_http() -> bool:
    """Return True only for explicit local development without bearer auth."""
    return os.environ.get("GNN_MCP_ALLOW_INSECURE_LOCAL", "").lower() in {
        "1",
        "true",
        "yes",
    }


def is_loopback_client(client_host: str | None) -> bool:
    """Return True only for loopback client addresses."""
    if not client_host:
        return False
    if client_host == "localhost":
        return True
    try:
        return ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        return False


def is_authorized(headers: Any, *, client_host: str | None = None) -> bool:
    """Validate HTTP headers against ``GNN_MCP_TOKEN``.

    HTTP transport requires bearer authentication by default. Developers may
    opt into unauthenticated loopback-only experimentation with
    ``GNN_MCP_ALLOW_INSECURE_LOCAL=1``.
    """
    token = get_required_bearer_token()
    if token is None:
        return allow_insecure_local_http() and is_loopback_client(client_host)
    auth_header = headers.get("Authorization", "")
    return bool(auth_header == f"Bearer {token}")


def get_rate_limit_per_minute() -> int:
    """Return the configured per-client HTTP rate limit, or 0 when disabled."""
    raw_value = os.environ.get("GNN_MCP_RATE_LIMIT_PER_MINUTE", "0")
    try:
        return max(0, int(raw_value))
    except ValueError:
        logger.warning("Invalid GNN_MCP_RATE_LIMIT_PER_MINUTE=%r; disabling", raw_value)
        return 0


def is_rate_limited(client_id: str, *, now: float | None = None) -> bool:
    """Return True when ``client_id`` has exceeded the configured rate limit."""
    limit = get_rate_limit_per_minute()
    if limit <= 0:
        return False
    timestamp = time.time() if now is None else now
    cutoff = timestamp - _RATE_LIMIT_WINDOW_SECONDS
    with _RATE_LIMIT_LOCK:
        recent = [
            seen_at
            for seen_at in _RATE_LIMIT_STATE.get(client_id, [])
            if seen_at >= cutoff
        ]
        if len(recent) >= limit:
            _RATE_LIMIT_STATE[client_id] = recent
            return True
        recent.append(timestamp)
        _RATE_LIMIT_STATE[client_id] = recent
    return False


def get_safe_http_tool_names() -> set[str] | None:
    """Return the default/extra safe HTTP tool names, or None if unsafe tools are allowed."""
    if os.environ.get("GNN_MCP_ALLOW_UNSAFE_TOOLS", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return None
    configured = {
        item.strip()
        for item in os.environ.get("GNN_MCP_SAFE_TOOLS", "").split(",")
        if item.strip()
    }
    return set(DEFAULT_SAFE_HTTP_TOOL_NAMES) | configured


def get_safe_http_resource_uris() -> set[str]:
    """Return resource URIs explicitly exposed over HTTP."""
    configured = {
        item.strip()
        for item in os.environ.get("GNN_MCP_SAFE_RESOURCES", "").split(",")
        if item.strip()
    }
    return set(DEFAULT_SAFE_HTTP_RESOURCE_URIS) | configured


def is_safe_http_tool(tool_name: str) -> bool:
    """Return True when a tool may be executed over HTTP by default."""
    safe_tools = get_safe_http_tool_names()
    return True if safe_tools is None else tool_name in safe_tools


def is_safe_http_resource(uri: str) -> bool:
    """Return True when a resource URI is explicitly exposed over HTTP."""
    return uri in get_safe_http_resource_uris()


def get_http_capabilities() -> Dict[str, Any]:
    """Return capabilities filtered to the HTTP-exposed allowlists."""
    capabilities = mcp_instance.get_capabilities()
    safe_tools = get_safe_http_tool_names()
    safe_resources = get_safe_http_resource_uris()
    tools = capabilities.get("tools", [])
    resources = capabilities.get("resources", [])
    if safe_tools is not None:
        tools = [
            tool
            for tool in tools
            if isinstance(tool, dict) and str(tool.get("name")) in safe_tools
        ]
    resources = [
        resource
        for resource in resources
        if isinstance(resource, dict)
        and str(resource.get("uri_template")) in safe_resources
    ]
    server = dict(capabilities.get("server", {}))
    server["http_access"] = {
        "safe_tools_only": safe_tools is not None,
        "safe_tool_count": len(tools),
        "safe_resource_count": len(resources),
        "resource_allowlist_env": "GNN_MCP_SAFE_RESOURCES",
    }
    return {"tools": tools, "resources": resources, "server": server}


class MCPHTTPHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for MCP JSON-RPC 2.0 requests.
    Supports both standard MCP methods and direct tool invocation.
    """

    def do_POST(self) -> Any:
        """Handle POST requests (JSON-RPC 2.0)."""
        urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        client_host = self.client_address[0] if self.client_address else None
        client_id = client_host or "unknown"
        if is_rate_limited(client_id):
            self._discard_request_body(content_length)
            self._send_error(429, "MCP HTTP rate limit exceeded")
            return
        if not is_authorized(self.headers, client_host=client_host):
            self._discard_request_body(content_length)
            self._send_error(401, "Missing or invalid bearer token")
            return
        if content_length <= 0:
            self._send_error(400, "Missing request body")
            return
        try:
            request_body = self.rfile.read(content_length)
            request = json.loads(request_body)
            logger.debug(f"HTTP IN: {request}")
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON request")
            return
        # Process JSON-RPC message
        if "jsonrpc" in request and request["jsonrpc"] == "2.0":
            self._handle_jsonrpc(request)
        else:
            self._send_error(
                400, "Invalid request format (missing or invalid jsonrpc field)"
            )

    def _handle_jsonrpc(self, request: Dict[str, Any]) -> Any:
        """
        Process a JSON-RPC message, supporting both standard MCP methods and direct tool invocation.
        """
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        if not method:
            self._send_jsonrpc_error(
                request_id, -32600, "Invalid Request: missing method"
            )
            return
        try:
            # Standard MCP methods
            if method in ("mcp.capabilities", "get_mcp_server_capabilities"):
                result = get_http_capabilities()
                self._send_jsonrpc_result(request_id, result)
            elif method == "mcp.tool.execute":
                if not (
                    isinstance(params, dict) and "name" in params and "params" in params
                ):
                    self._send_jsonrpc_error(
                        request_id, -32602, "Invalid params for tool execution"
                    )
                    return
                tool_name = params["name"]
                tool_params = params["params"]
                if not is_safe_http_tool(str(tool_name)):
                    self._send_jsonrpc_error(
                        request_id,
                        -32001,
                        f"Tool not exposed over MCP HTTP by default: {tool_name}",
                    )
                    return
                result = mcp_instance.execute_tool(tool_name, tool_params)
                self._send_jsonrpc_result(request_id, result)
            elif method == "mcp.resource.get":
                if not (isinstance(params, dict) and "uri" in params):
                    self._send_jsonrpc_error(
                        request_id, -32602, "Invalid params for resource retrieval"
                    )
                    return
                uri = params["uri"]
                if not is_safe_http_resource(str(uri)):
                    self._send_jsonrpc_error(
                        request_id,
                        -32002,
                        f"Resource not exposed over MCP HTTP by default: {uri}",
                    )
                    return
                result = mcp_instance.get_resource(uri)
                self._send_jsonrpc_result(request_id, result)
            # Direct tool invocation (meta-tools, registered tools, etc.)
            elif method in mcp_instance.tools:
                if not isinstance(params, dict):
                    self._send_jsonrpc_error(
                        request_id, -32602, "Params must be an object (dictionary)"
                    )
                    return
                if not is_safe_http_tool(str(method)):
                    self._send_jsonrpc_error(
                        request_id,
                        -32001,
                        f"Tool not exposed over MCP HTTP by default: {method}",
                    )
                    return
                result = mcp_instance.execute_tool(method, params)
                self._send_jsonrpc_result(request_id, result)
            else:
                self._send_jsonrpc_error(
                    request_id, -32601, f"Method not found: {method}"
                )
        except MCPError as mcpe:
            logger.error(f"MCPError in method {method}: {mcpe}")
            self._send_jsonrpc_error(
                request_id, mcpe.code, str(mcpe), data=getattr(mcpe, "data", None)
            )
        except Exception as e:
            logger.exception(f"Unhandled error in method {method}: {e}")
            self._send_jsonrpc_error(request_id, -32603, f"Internal error: {str(e)}")

    def _send_jsonrpc_result(self, request_id: Optional[str], result: Any) -> Any:
        """Send a successful JSON-RPC response."""
        response: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }
        logger.debug(f"HTTP OUT: {response}")
        self._send_json_response(200, response)

    def _send_jsonrpc_error(
        self, request_id: Optional[str], code: int, message: str, data: Any = None
    ) -> Any:
        """Send a JSON-RPC error response, including optional data."""
        error_obj: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error_obj["data"] = data
        response: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error_obj,
        }
        logger.debug(f"HTTP OUT (error): {response}")
        self._send_json_response(200, response)

    def _send_json_response(self, status_code: int, data: Any) -> Any:
        """Send a JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response_body = json.dumps(data).encode("utf-8")
        self.wfile.write(response_body)

    def _send_error(self, status_code: int, message: str) -> Any:
        """Send an HTTP error response."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        error_body = json.dumps({"error": message}).encode("utf-8")
        self.wfile.write(error_body)

    def _discard_request_body(self, content_length: int) -> None:
        """Drain a rejected request body so clients can read the error cleanly."""
        if content_length <= 0:
            return
        try:
            self.rfile.read(content_length)
        except OSError:
            logger.debug(
                "Could not drain rejected MCP HTTP request body", exc_info=True
            )

    def log_message(self, format: Any, *args: Any) -> Any:
        """Override log_message to use our logger."""
        logger.info(
            "%s - - [%s] %s"
            % (self.client_address[0], self.log_date_time_string(), format % args)
        )


class MCPHTTPServer:
    """
    HTTP server for MCP. Runs in a background thread and supports graceful shutdown.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Initialize the instance."""
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.running = False

    def start(self) -> Any:
        """Start the HTTP server."""
        # Initialize MCP
        initialize()
        self.server = HTTPServer((self.host, self.port), MCPHTTPHandler)
        self.running = True
        logger.info(f"Starting MCP HTTP server on {self.host}:{self.port}")
        self.server_thread = threading.Thread(target=self._server_thread)
        self.server_thread.daemon = True
        self.server_thread.start()
        try:
            while self.running:
                self.server_thread.join(0.1)
                if not self.server_thread.is_alive():
                    self.running = False
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping server")
            self.stop()

    def _server_thread(self) -> Any:
        """Thread function that runs the HTTP server."""
        try:
            if self.server is None:
                raise RuntimeError("HTTP server has not been initialized")
            self.server.serve_forever()
        except Exception as e:
            logger.error(f"Error in HTTP server: {str(e)}")
            self.running = False

    def stop(self) -> Any:
        """Stop the HTTP server."""
        if self.server:
            logger.info("Stopping HTTP server")
            self.server.shutdown()
            self.server.server_close()
            self.running = False


def start_http_server(host: str = "127.0.0.1", port: int = 8080) -> Any:
    """Start an MCP server using HTTP transport."""
    server = MCPHTTPServer(host, port)
    server.start()


if __name__ == "__main__":
    start_http_server()
