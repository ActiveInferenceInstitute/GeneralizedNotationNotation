"""MCP (Model Context Protocol) integration for the GNN pipeline: tool discovery, registration, and execution.

See ``src/mcp/AGENTS.md`` for the public API, step-21 wiring, and tool registration patterns.
"""

from __future__ import annotations

from typing import Any

# -- Core API: the five classes and the instance factory most callers need ----------
from .exceptions import (
    MCPError,
    MCPToolExecutionError,
    MCPToolNotFoundError,
    MCPValidationError,
)
from .mcp import MCP, get_mcp_instance, initialize, mcp_instance
from .models import MCPResource, MCPTool
from .processor import (
    generate_mcp_report,
    get_available_tools,
    handle_mcp_request,
    process_mcp,
    register_module_tools,
)
from .server_core import create_mcp_server, register_tools, start_mcp_server

# -- Aliases for backward compatibility (don't add new code depending on these) -----
MCPRegistry = MCP
from .server import MCPServer as _JSONRPCServer  # noqa: E402

MCPServer = _JSONRPCServer
JSONRPCServer = _JSONRPCServer

# list_available_tools is an alias for get_available_tools
list_available_tools = get_available_tools

# -- Module metadata -----------------------------------------------------------------
__version__ = "1.6.0"
__author__ = "Active Inference Institute"
__description__ = "Enhanced Model Context Protocol implementation for GNN"

FEATURES: dict[str, Any] = {
    "tool_registration": True,
    "resource_access": True,
    "module_discovery": True,
    "json_rpc": True,
    "server_implementation": True,
    "mcp_integration": True,
    "performance_monitoring": True,
    "caching": True,
    "rate_limiting": True,
    "concurrent_control": True,
}


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "mcp",
        "version": __version__,
        "description": "Model Context Protocol tool registration and discovery",
        "features": FEATURES,
    }


__all__: list[str] = [
    # Core MCP classes
    "mcp_instance",
    "initialize",
    "MCP",
    "MCPRegistry",
    "MCPServer",
    "MCPTool",
    "MCPResource",
    "get_mcp_instance",
    "create_mcp_server",
    "start_mcp_server",
    "register_tools",
    # Processor functions
    "register_module_tools",
    "generate_mcp_report",
    "handle_mcp_request",
    "process_mcp",
    "get_available_tools",
    # Exception classes (most common)
    "MCPError",
    "MCPToolExecutionError",
    "MCPToolNotFoundError",
    "MCPValidationError",
    # Aliases
    "list_available_tools",
    # Metadata
    "FEATURES",
    "__version__",
    "get_module_info",
]