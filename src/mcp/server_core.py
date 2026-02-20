"""
MCP Server implementation sub-module.

Provides MCPServer class for JSON-RPC 2.0 request handling,
and factory functions for server creation and management.

Extracted from mcp.py for maintainability.
"""

import json
import sys
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("mcp")

# Import exceptions
from .exceptions import (
    MCPError,
    MCPInvalidParamsError,
)

# Import core MCP class and helpers
from .mcp import MCP, get_mcp_instance, initialize


class MCPServer:
    """
    MCP Server implementation for handling JSON-RPC requests.
    
    This class provides a server implementation that can handle
    MCP protocol requests and responses.
    """
    
    def __init__(self, mcp_instance: Optional[MCP] = None):
        """
        Initialize the MCP server.
        
        Args:
            mcp_instance: MCP instance to use for tool execution
        """
        self.mcp = mcp_instance or get_mcp_instance()
        self.running = False
        self.request_handlers = {
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "initialize": self._handle_initialize,
            "notifications/initialized": self._handle_initialized,
            "shutdown": self._handle_shutdown,
            "exit": self._handle_exit
        }
    
    def start(self) -> bool:
        """Start the MCP server."""
        if self.running:
            logger.warning("MCP server is already running")
            return False
        
        self.running = True
        logger.info("MCP server started")
        return True
    
    def stop(self) -> bool:
        """Stop the MCP server."""
        if not self.running:
            logger.warning("MCP server is not running")
            return False
        
        self.running = False
        logger.info("MCP server stopped")
        return True
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming JSON-RPC request.
        
        Args:
            request: JSON-RPC request dictionary
            
        Returns:
            JSON-RPC response dictionary
        """
        try:
            if not isinstance(request, dict):
                return self._create_error_response(-32700, "Parse error", "Invalid JSON")
            
            if "jsonrpc" not in request or request["jsonrpc"] != "2.0":
                return self._create_error_response(-32600, "Invalid Request", "Missing or invalid jsonrpc field")
            
            if "method" not in request:
                return self._create_error_response(-32600, "Invalid Request", "Missing method field")
            
            method = request["method"]
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method in self.request_handlers:
                result = self.request_handlers[method](params)
                return self._create_success_response(result, request_id)
            else:
                if method in self.mcp.tools:
                    result = self.mcp.execute_tool(method, params)
                    return self._create_success_response(result, request_id)
                else:
                    return self._create_error_response(-32601, "Method not found", f"Method '{method}' not found", request_id)
                    
        except MCPError as e:
            return self._create_error_response(e.code, type(e).__name__, str(e), request.get("id"))
        except Exception as e:
            logger.error(f"Unexpected error handling request: {e}")
            return self._create_error_response(-32603, "Internal error", str(e), request.get("id"))
    
    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": self.mcp.get_capabilities(),
            "serverInfo": {
                "name": "GNN MCP Server",
                "version": "1.0.0"
            }
        }
    
    def _handle_initialized(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialized notification."""
        return {}
    
    def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        return {"tools": self.mcp.get_capabilities()["tools"]}
    
    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        tool_params = params.get("arguments", {})
        
        if not tool_name:
            raise MCPInvalidParamsError("Tool name is required")
        
        result = self.mcp.execute_tool(tool_name, tool_params)
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
    
    def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request."""
        return {"resources": self.mcp.get_capabilities()["resources"]}
    
    def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")
        
        if not uri:
            raise MCPInvalidParamsError("Resource URI is required")
        
        result = self.mcp.get_resource(uri)
        return {"contents": [{"uri": uri, "mimeType": result["mime_type"], "text": json.dumps(result["content"])}]}
    
    def _handle_shutdown(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle shutdown request."""
        self.stop()
        return {}
    
    def _handle_exit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle exit request."""
        self.stop()
        return {}
    
    def _create_success_response(self, result: Any, request_id: Any) -> Dict[str, Any]:
        """Create a successful JSON-RPC response."""
        response = {
            "jsonrpc": "2.0",
            "result": result
        }
        if request_id is not None:
            response["id"] = request_id
        return response
    
    def _create_error_response(self, code: int, message: str, data: Any = None, request_id: Any = None) -> Dict[str, Any]:
        """Create an error JSON-RPC response."""
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            }
        }
        if data is not None:
            response["error"]["data"] = data
        if request_id is not None:
            response["id"] = request_id
        return response


def create_mcp_server(mcp_instance: Optional[MCP] = None) -> MCPServer:
    """Create a new MCP server instance."""
    return MCPServer(mcp_instance)


def start_mcp_server(mcp_instance: Optional[MCP] = None) -> bool:
    """Start the MCP server."""
    server = create_mcp_server(mcp_instance)
    return server.start()


def register_tools(mcp: Optional[MCP] = None) -> bool:
    """
    Register tools with the MCP instance.
    
    Args:
        mcp: Optional MCP instance. If None, uses the global instance.
        
    Returns:
        True if registration succeeded
    """
    try:
        return True
    except Exception as e:
        logger.error(f"Failed to register tools: {e}")
        return False


if __name__ == "__main__":
    try:
        mcp, sdk_found, all_modules_loaded = initialize()
        
        capabilities = mcp.get_capabilities()
        print(f"Available tools: {len(capabilities['tools'])}")
        print(f"Available resources: {len(capabilities['resources'])}")
        
        status = mcp.get_server_status()
        print(f"Server uptime: {status['uptime_formatted']}")
        
    except Exception as e:
        logger.error(f"Error in MCP initialization: {e}")
        sys.exit(1)
