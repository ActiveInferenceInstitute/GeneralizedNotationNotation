#!/usr/bin/env python3
"""
Enhanced MCP Stdio Server Implementation

This module provides a robust JSON-RPC 2.0 stdio server for the Model Context Protocol (MCP),
exposing all registered GNN tools and resources via standard input/output streams.

Key Features:
- Enhanced error handling with custom MCP error codes and JSON-RPC compliance
- Multi-threaded architecture for concurrent request processing
- Comprehensive logging and request/response tracking
- Graceful shutdown and resource cleanup
- Connection health monitoring and automatic recovery
- Extensible for meta-tools and future MCP extensions
- Performance monitoring and metrics collection
"""
import json
import sys
import logging
import threading
import queue
import time
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
    
    def __init__(self, max_queue_size: int = 1000, request_timeout: float = 30.0):
        """Initialize the stdio server with enhanced queue and thread management.

        Args:
            max_queue_size: Maximum size of request/response queues
            request_timeout: Timeout for request processing in seconds
        """
        self.running = False
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.response_queue = queue.Queue(maxsize=max_queue_size)
        self.next_id = 1
        self.pending_requests = {}
        self.request_timeout = request_timeout

        # Connection monitoring
        self._connection_errors = 0
        self._max_connection_errors = 10
        self._last_activity = time.time()

        # Performance tracking
        self._requests_processed = 0
        self._responses_sent = 0
        self._errors_encountered = 0
        
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

    def get_stats(self) -> Dict[str, Any]:
        """Get server performance statistics."""
        return {
            "running": self.running,
            "requests_processed": self._requests_processed,
            "responses_sent": self._responses_sent,
            "errors_encountered": self._errors_encountered,
            "connection_errors": self._connection_errors,
            "last_activity": self._last_activity,
            "uptime": time.time() - self._start_time if hasattr(self, '_start_time') else 0,
            "queue_sizes": {
                "requests": self.request_queue.qsize(),
                "responses": self.response_queue.qsize()
            }
        }
    
    def _reader_thread(self):
        """Enhanced thread that reads JSON-RPC messages from stdin with connection monitoring."""
        try:
            while self.running:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        logger.info("End of input detected, stopping server")
                        self.running = False
                        break

                    # Update activity timestamp
                    self._last_activity = time.time()

                    try:
                        message = json.loads(line.strip())
                        logger.debug(f"STDIO IN: {message}")
                        self._requests_processed += 1

                        # Check for connection health
                        if self._connection_errors > self._max_connection_errors:
                            logger.error("Too many connection errors, stopping server")
                            self.running = False
                            break

                        self.request_queue.put(message, timeout=1.0)

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON message: {line.strip()} - {e}")
                        self._connection_errors += 1

                        # Send JSON-RPC parse error
                        error_response = {
                            "jsonrpc": "2.0",
                            "error": {"code": -32700, "message": "Parse error"},
                            "id": None
                        }
                        try:
                            self.response_queue.put(error_response, timeout=1.0)
                        except queue.Full:
                            logger.warning("Response queue full, dropping error response")

                    except queue.Full:
                        logger.warning("Request queue full, dropping message")
                        self._connection_errors += 1

                except (IOError, OSError) as e:
                    logger.error(f"IO error in reader thread: {e}")
                    self._connection_errors += 1
                    if self._connection_errors > self._max_connection_errors:
                        logger.error("Too many IO errors, stopping server")
                        self.running = False
                        break

                    # Brief pause before retrying
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Unexpected error in reader thread: {str(e)}")
            self.running = False
    
    def _processor_thread(self):
        """Enhanced thread that processes messages from the request queue with better error handling."""
        try:
            while self.running:
                try:
                    message = self.request_queue.get(timeout=0.1)
                    self._process_message(message)
                    self.request_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in processor thread: {str(e)}")
                    self._errors_encountered += 1
                    # Continue processing other messages
                    try:
                        self.request_queue.task_done()
                    except ValueError:
                        pass  # Task already done or never queued
        except Exception as e:
            logger.error(f"Fatal error in processor thread: {str(e)}")
            self.running = False
    
    def _writer_thread(self):
        """Enhanced thread that writes JSON-RPC responses to stdout with error recovery."""
        try:
            while self.running:
                try:
                    message = self.response_queue.get(timeout=0.1)
                    self._responses_sent += 1

                    try:
                        json_str = json.dumps(message, separators=(',', ':'), ensure_ascii=False)
                        logger.debug(f"STDIO OUT: {json_str}")
                        sys.stdout.write(json_str + "\n")
                        sys.stdout.flush()
                    except (BrokenPipeError, IOError) as e:
                        logger.error(f"IO error writing response: {e}")
                        self._connection_errors += 1
                        if self._connection_errors > self._max_connection_errors:
                            logger.error("Too many write errors, stopping server")
                            self.running = False
                            break
                    except Exception as e:
                        logger.error(f"Error serializing/writing message: {str(e)}")
                        self._errors_encountered += 1

                    self.response_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error in writer thread: {str(e)}")
                    self._errors_encountered += 1

        except Exception as e:
            logger.error(f"Fatal error in writer thread: {str(e)}")
            self.running = False
    
    def _process_message(self, message: Dict[str, Any]):
        """Process an incoming JSON-RPC message with enhanced validation."""
        try:
            if not isinstance(message, dict):
                logger.error(f"Invalid message format: expected dict, got {type(message)}")
                return

            # Validate JSON-RPC structure
            if "jsonrpc" not in message:
                logger.error("Missing 'jsonrpc' field in message")
                self._send_error(None, -32600, "Invalid Request: missing jsonrpc field")
                return

            if message["jsonrpc"] != "2.0":
                logger.error(f"Unsupported JSON-RPC version: {message['jsonrpc']}")
                self._send_error(None, -32600, f"Invalid Request: unsupported jsonrpc version '{message['jsonrpc']}'")
                return

            if "method" not in message:
                logger.error("Missing 'method' field in message")
                self._send_error(None, -32600, "Invalid Request: missing method field")
                return

            # Check for JSON-RPC message
            self._process_jsonrpc(message)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            self._errors_encountered += 1
            try:
                self._send_error(None, -32603, f"Internal error processing message: {str(e)}")
            except Exception:
                logger.error("Failed to send error response")
    
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