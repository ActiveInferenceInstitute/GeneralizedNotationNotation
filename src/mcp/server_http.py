#!/usr/bin/env python3
import json
import logging
import traceback
from typing import Dict, Any, Optional, List
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import urllib.parse

# Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     filename='mcp_http_server.log',
#                     filemode='a')
logger = logging.getLogger(__name__)

# Import MCP
try:
    from . import mcp_instance, initialize
except ImportError:
    from mcp import mcp_instance, initialize

class MCPHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MCP."""
    
    def do_POST(self):
        """Handle POST requests."""
        # Parse path
        parsed_path = urllib.parse.urlparse(self.path)
        
        # Check content length
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length <= 0:
            self._send_error(400, "Missing request body")
            return
        
        # Read and parse request body
        try:
            request_body = self.rfile.read(content_length)
            request = json.loads(request_body)
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON request")
            return
        
        # Process JSON-RPC message
        if "jsonrpc" in request and request["jsonrpc"] == "2.0":
            self._handle_jsonrpc(request)
        else:
            self._send_error(400, "Invalid request format")
    
    def _handle_jsonrpc(self, request: Dict[str, Any]):
        """Process a JSON-RPC message."""
        if "method" not in request:
            self._send_jsonrpc_error(request.get("id"), -32600, "Invalid Request")
            return
        
        method = request["method"]
        params = request.get("params", {})
        
        # Handle standard methods
        try:
            if method == "mcp.capabilities":
                self._handle_capabilities(request.get("id"))
            elif method == "mcp.tool.execute":
                self._handle_execute_tool(request.get("id"), params)
            elif method == "mcp.resource.get":
                self._handle_get_resource(request.get("id"), params)
            else:
                self._send_jsonrpc_error(request.get("id"), -32601, f"Method not found: {method}")
        except Exception as e:
            logger.error(f"Error handling method {method}: {str(e)}")
            traceback.print_exc()
            self._send_jsonrpc_error(request.get("id"), -32603, f"Internal error: {str(e)}")
    
    def _handle_capabilities(self, request_id: Optional[str]):
        """Handle capabilities request."""
        capabilities = mcp_instance.get_capabilities()
        self._send_jsonrpc_result(request_id, capabilities)
    
    def _handle_execute_tool(self, request_id: Optional[str], params: Dict[str, Any]):
        """Handle tool execution request."""
        if "name" not in params or "params" not in params:
            self._send_jsonrpc_error(request_id, -32602, "Invalid params for tool execution")
            return
        
        tool_name = params["name"]
        tool_params = params["params"]
        
        try:
            result = mcp_instance.execute_tool(tool_name, tool_params)
            self._send_jsonrpc_result(request_id, result)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            self._send_jsonrpc_error(request_id, -32603, f"Internal error: {str(e)}")
    
    def _handle_get_resource(self, request_id: Optional[str], params: Dict[str, Any]):
        """Handle resource retrieval request."""
        if "uri" not in params:
            self._send_jsonrpc_error(request_id, -32602, "Invalid params for resource retrieval")
            return
        
        uri = params["uri"]
        
        try:
            result = mcp_instance.get_resource(uri)
            self._send_jsonrpc_result(request_id, result)
        except Exception as e:
            logger.error(f"Error retrieving resource {uri}: {str(e)}")
            self._send_jsonrpc_error(request_id, -32603, f"Internal error: {str(e)}")
    
    def _send_jsonrpc_result(self, request_id: Optional[str], result: Any):
        """Send a successful JSON-RPC response."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
        self._send_json_response(200, response)
    
    def _send_jsonrpc_error(self, request_id: Optional[str], code: int, message: str):
        """Send a JSON-RPC error response."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
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
        
        error_body = json.dumps({
            "error": message
        }).encode('utf-8')
        
        self.wfile.write(error_body)
    
    def log_message(self, format, *args):
        """Override log_message to use our logger."""
        logger.info("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))

class MCPHTTPServer:
    """HTTP server for MCP."""
    
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
        
        # Create and start server
        self.server = HTTPServer((self.host, self.port), MCPHTTPHandler)
        self.running = True
        
        logger.info(f"Starting MCP HTTP server on {self.host}:{self.port}")
        
        # Run server in a separate thread
        self.server_thread = threading.Thread(target=self._server_thread)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Wait for Ctrl+C
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