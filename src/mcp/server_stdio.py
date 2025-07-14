#!/usr/bin/env python3
"""
MCP Stdio Server Implementation

This module provides a JSON-RPC 2.0 stdio server for the Model Context Protocol (MCP),
exposing all registered GNN tools and resources via standard input/output streams.

Key Features:
- Supports both standard MCP methods (capabilities, tool/resource execution) and direct tool invocation
- Robust error handling with custom MCP error codes and JSON-RPC compliance
- Multi-threaded architecture for concurrent request processing
- Detailed logging of all requests and responses
- Extensible for meta-tools and future MCP extensions
"""
import json
import sys
import logging
import threading
import queue
from typing import Dict, Any, Optional
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# Import MCP
try:
    from . import mcp_instance, initialize, MCPError
except ImportError:
    from mcp import mcp_instance, initialize, MCPError

class StdioServer:
    """
    A Model Context Protocol server implementation using stdio transport.
    
    This server reads JSON-RPC 2.0 requests from stdin and writes responses to stdout,
    supporting both standard MCP methods and direct tool invocation.
    """
    
    def __init__(self):
        """Initialize the stdio server with queues and thread management."""
        self.running = False
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.next_id = 1
        self.pending_requests = {}
        
    def start(self):
        """Start the server with reader, processor, and writer threads."""
        self.running = True
        
        # Initialize MCP
        initialize()
        logger.info("MCP stdio server initialized and ready")
        
        # Start reader and writer threads
        reader_thread = threading.Thread(target=self._reader_thread)
        writer_thread = threading.Thread(target=self._writer_thread)
        processor_thread = threading.Thread(target=self._processor_thread)
        
        reader_thread.daemon = True
        writer_thread.daemon = True
        processor_thread.daemon = True
        
        reader_thread.start()
        writer_thread.start()
        processor_thread.start()
        
        # Wait for threads to exit
        try:
            while self.running:
                reader_thread.join(0.1)
                if not reader_thread.is_alive():
                    self.running = False
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping server")
            self.running = False
        
        writer_thread.join()
        processor_thread.join()
        logger.info("MCP stdio server stopped")
    
    def _reader_thread(self):
        """Thread that reads JSON-RPC messages from stdin."""
        try:
            while self.running:
                line = sys.stdin.readline()
                if not line:
                    logger.info("End of input, stopping server")
                    self.running = False
                    break
                
                try:
                    message = json.loads(line)
                    logger.debug(f"STDIO IN: {message}")
                    self.request_queue.put(message)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON message: {line}")
                    # Send JSON-RPC parse error
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": "Parse error"},
                        "id": None
                    }
                    self.response_queue.put(error_response)
        except Exception as e:
            logger.error(f"Error in reader thread: {str(e)}")
            self.running = False
    
    def _processor_thread(self):
        """Thread that processes messages from the request queue."""
        try:
            while self.running:
                try:
                    message = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                try:
                    self._process_message(message)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    traceback.print_exc()
                
                self.request_queue.task_done()
        except Exception as e:
            logger.error(f"Error in processor thread: {str(e)}")
            self.running = False
    
    def _writer_thread(self):
        """Thread that writes JSON-RPC responses to stdout."""
        try:
            while self.running:
                try:
                    message = self.response_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                try:
                    json_str = json.dumps(message)
                    logger.debug(f"STDIO OUT: {json_str}")
                    sys.stdout.write(json_str + "\n")
                    sys.stdout.flush()
                except Exception as e:
                    logger.error(f"Error writing message: {str(e)}")
                
                self.response_queue.task_done()
        except Exception as e:
            logger.error(f"Error in writer thread: {str(e)}")
            self.running = False
    
    def _process_message(self, message: Dict[str, Any]):
        """Process an incoming JSON-RPC message."""
        if not isinstance(message, dict):
            logger.error(f"Invalid message format: {message}")
            return
        
        # Check for JSON-RPC message
        if "jsonrpc" in message and message["jsonrpc"] == "2.0":
            self._process_jsonrpc(message)
        else:
            logger.error(f"Unsupported message format: {message}")
    
    def _process_jsonrpc(self, message: Dict[str, Any]):
        """
        Process a JSON-RPC message, supporting both standard MCP methods and direct tool invocation.
        """
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})
        
        if not method:
            self._send_error(request_id, -32600, "Invalid Request: missing method")
            return
        
        try:
            # Standard MCP methods
            if method in ("mcp.capabilities", "get_mcp_server_capabilities"):
                result = mcp_instance.get_capabilities()
                self._send_result(request_id, result)
            elif method == "mcp.tool.execute":
                if not (isinstance(params, dict) and "name" in params and "params" in params):
                    self._send_error(request_id, -32602, "Invalid params for tool execution")
                    return
                tool_name = params["name"]
                tool_params = params["params"]
                result = mcp_instance.execute_tool(tool_name, tool_params)
                self._send_result(request_id, result)
            elif method == "mcp.resource.get":
                if not (isinstance(params, dict) and "uri" in params):
                    self._send_error(request_id, -32602, "Invalid params for resource retrieval")
                    return
                uri = params["uri"]
                result = mcp_instance.get_resource(uri)
                self._send_result(request_id, result)
            # Direct tool invocation (meta-tools, registered tools, etc.)
            elif method in mcp_instance.tools:
                if not isinstance(params, dict):
                    self._send_error(request_id, -32602, "Params must be an object (dictionary)")
                    return
                result = mcp_instance.execute_tool(method, params)
                self._send_result(request_id, result)
            else:
                self._send_error(request_id, -32601, f"Method not found: {method}")
        except MCPError as mcpe:
            logger.error(f"MCPError in method {method}: {mcpe}")
            self._send_error(request_id, mcpe.code, str(mcpe), data=getattr(mcpe, 'data', None))
        except Exception as e:
            logger.error(f"Unhandled error in method {method}: {e}")
            traceback.print_exc()
            self._send_error(request_id, -32603, f"Internal error: {str(e)}")
    
    def _send_result(self, request_id: str, result: Any):
        """Send a successful JSON-RPC result response."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
        self.response_queue.put(response)
    
    def _send_error(self, request_id: Optional[str], code: int, message: str, data: Any = None):
        """Send a JSON-RPC error response, including optional data."""
        error_obj = {"code": code, "message": message}
        if data is not None:
            error_obj["data"] = data
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error_obj
        }
        self.response_queue.put(response)

def start_stdio_server():
    """Start an MCP server using stdio transport."""
    server = StdioServer()
    server.start()

if __name__ == "__main__":
    start_stdio_server() 