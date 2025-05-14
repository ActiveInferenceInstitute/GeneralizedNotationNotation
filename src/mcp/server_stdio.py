#!/usr/bin/env python3
import json
import sys
import logging
import threading
import queue
from typing import Dict, Any, Optional, List
import traceback

# Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     filename='mcp_server.log',
#                     filemode='a')
logger = logging.getLogger(__name__)

# Import MCP
try:
    from . import mcp_instance, initialize
except ImportError:
    from mcp import mcp_instance, initialize

class StdioServer:
    """A Model Context Protocol server implementation using stdio transport."""
    
    def __init__(self):
        self.running = False
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.next_id = 1
        self.pending_requests = {}
        
    def start(self):
        """Start the server."""
        self.running = True
        
        # Initialize MCP
        initialize()
        
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
            self.running = False
        
        writer_thread.join()
        processor_thread.join()
    
    def _reader_thread(self):
        """Thread that reads messages from stdin."""
        try:
            while self.running:
                line = sys.stdin.readline()
                if not line:
                    logger.info("End of input, stopping server")
                    self.running = False
                    break
                
                try:
                    message = json.loads(line)
                    self.request_queue.put(message)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON message: {line}")
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
        """Thread that writes messages to stdout."""
        try:
            while self.running:
                try:
                    message = self.response_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                try:
                    json_str = json.dumps(message)
                    sys.stdout.write(json_str + "\n")
                    sys.stdout.flush()
                except Exception as e:
                    logger.error(f"Error writing message: {str(e)}")
                
                self.response_queue.task_done()
        except Exception as e:
            logger.error(f"Error in writer thread: {str(e)}")
            self.running = False
    
    def _process_message(self, message: Dict[str, Any]):
        """Process an incoming message."""
        if not isinstance(message, dict):
            logger.error(f"Invalid message format: {message}")
            return
        
        # Check for JSON-RPC message
        if "jsonrpc" in message and message["jsonrpc"] == "2.0":
            self._process_jsonrpc(message)
        else:
            logger.error(f"Unsupported message format: {message}")
    
    def _process_jsonrpc(self, message: Dict[str, Any]):
        """Process a JSON-RPC message."""
        if "method" not in message:
            self._send_error(message.get("id"), -32600, "Invalid Request")
            return
        
        method = message["method"]
        params = message.get("params", {})
        
        # Handle standard methods
        if method == "mcp.capabilities":
            self._handle_capabilities(message["id"])
        elif method == "mcp.tool.execute":
            self._handle_execute_tool(message["id"], params)
        elif method == "mcp.resource.get":
            self._handle_get_resource(message["id"], params)
        else:
            self._send_error(message.get("id"), -32601, f"Method not found: {method}")
    
    def _handle_capabilities(self, request_id: str):
        """Handle capabilities request."""
        capabilities = mcp_instance.get_capabilities()
        self._send_result(request_id, capabilities)
    
    def _handle_execute_tool(self, request_id: str, params: Dict[str, Any]):
        """Handle tool execution request."""
        if "name" not in params or "params" not in params:
            self._send_error(request_id, -32602, "Invalid params for tool execution")
            return
        
        tool_name = params["name"]
        tool_params = params["params"]
        
        try:
            result = mcp_instance.execute_tool(tool_name, tool_params)
            self._send_result(request_id, result)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            self._send_error(request_id, -32603, f"Internal error: {str(e)}")
    
    def _handle_get_resource(self, request_id: str, params: Dict[str, Any]):
        """Handle resource retrieval request."""
        if "uri" not in params:
            self._send_error(request_id, -32602, "Invalid params for resource retrieval")
            return
        
        uri = params["uri"]
        
        try:
            result = mcp_instance.get_resource(uri)
            self._send_result(request_id, result)
        except Exception as e:
            logger.error(f"Error retrieving resource {uri}: {str(e)}")
            self._send_error(request_id, -32603, f"Internal error: {str(e)}")
    
    def _send_result(self, request_id: str, result: Any):
        """Send a successful result response."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
        self.response_queue.put(response)
    
    def _send_error(self, request_id: Optional[str], code: int, message: str):
        """Send an error response."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
        self.response_queue.put(response)

def start_stdio_server():
    """Start an MCP server using stdio transport."""
    server = StdioServer()
    server.start()

if __name__ == "__main__":
    start_stdio_server() 