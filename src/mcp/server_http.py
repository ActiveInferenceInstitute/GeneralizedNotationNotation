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
import json
import logging
import traceback
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import urllib.parse

# Configure logging
logger = logging.getLogger(__name__)

# Import MCP
try:
    from . import mcp_instance, initialize, MCPError
except ImportError:
    from mcp import mcp_instance, initialize, MCPError

class MCPHTTPHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for MCP JSON-RPC 2.0 requests.
    Supports both standard MCP methods and direct tool invocation.
    """
    def do_POST(self):
        """Handle POST requests (JSON-RPC 2.0)."""
        parsed_path = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
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
            self._send_error(400, "Invalid request format (missing or invalid jsonrpc field)")

    def _handle_jsonrpc(self, request: Dict[str, Any]):
        """
        Process a JSON-RPC message, supporting both standard MCP methods and direct tool invocation.
        """
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        if not method:
            self._send_jsonrpc_error(request_id, -32600, "Invalid Request: missing method")
            return
        try:
            # Standard MCP methods
            if method in ("mcp.capabilities", "get_mcp_server_capabilities"):
                result = mcp_instance.get_capabilities()
                self._send_jsonrpc_result(request_id, result)
            elif method == "mcp.tool.execute":
                if not (isinstance(params, dict) and "name" in params and "params" in params):
                    self._send_jsonrpc_error(request_id, -32602, "Invalid params for tool execution")
                    return
                tool_name = params["name"]
                tool_params = params["params"]
                result = mcp_instance.execute_tool(tool_name, tool_params)
                self._send_jsonrpc_result(request_id, result)
            elif method == "mcp.resource.get":
                if not (isinstance(params, dict) and "uri" in params):
                    self._send_jsonrpc_error(request_id, -32602, "Invalid params for resource retrieval")
                    return
                uri = params["uri"]
                result = mcp_instance.get_resource(uri)
                self._send_jsonrpc_result(request_id, result)
            # Direct tool invocation (meta-tools, registered tools, etc.)
            elif method in mcp_instance.tools:
                if not isinstance(params, dict):
                    self._send_jsonrpc_error(request_id, -32602, "Params must be an object (dictionary)")
                    return
                result = mcp_instance.execute_tool(method, params)
                self._send_jsonrpc_result(request_id, result)
            else:
                self._send_jsonrpc_error(request_id, -32601, f"Method not found: {method}")
        except MCPError as mcpe:
            logger.error(f"MCPError in method {method}: {mcpe}")
            self._send_jsonrpc_error(request_id, mcpe.code, str(mcpe), data=getattr(mcpe, 'data', None))
        except Exception as e:
            logger.error(f"Unhandled error in method {method}: {e}")
            traceback.print_exc()
            self._send_jsonrpc_error(request_id, -32603, f"Internal error: {str(e)}")

    def _send_jsonrpc_result(self, request_id: Optional[str], result: Any):
        """Send a successful JSON-RPC response."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
        logger.debug(f"HTTP OUT: {response}")
        self._send_json_response(200, response)

    def _send_jsonrpc_error(self, request_id: Optional[str], code: int, message: str, data: Any = None):
        """Send a JSON-RPC error response, including optional data."""
        error_obj = {"code": code, "message": message}
        if data is not None:
            error_obj["data"] = data
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error_obj
        }
        logger.debug(f"HTTP OUT (error): {response}")
        self._send_json_response(200, response)

    def _send_json_response(self, status_code: int, data: Any):
        """Send a JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response_body = json.dumps(data).encode('utf-8')
        self.wfile.write(response_body)

    def _send_error(self, status_code: int, message: str):
        """Send an HTTP error response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        error_body = json.dumps({"error": message}).encode('utf-8')
        self.wfile.write(error_body)

    def log_message(self, format, *args):
        """Override log_message to use our logger."""
        logger.info("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))

class MCPHTTPServer:
    """
    HTTP server for MCP. Runs in a background thread and supports graceful shutdown.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False

    def start(self):
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

    def _server_thread(self):
        """Thread function that runs the HTTP server."""
        try:
            self.server.serve_forever()
        except Exception as e:
            logger.error(f"Error in HTTP server: {str(e)}")
            self.running = False

    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            logger.info("Stopping HTTP server")
            self.server.shutdown()
            self.server.server_close()
            self.running = False

def start_http_server(host: str = "127.0.0.1", port: int = 8080):
    """Start an MCP server using HTTP transport."""
    server = MCPHTTPServer(host, port)
    server.start()

if __name__ == "__main__":
    start_http_server() 