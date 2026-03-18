"""
MCP SDK shim: delegates to the parent mcp module implementation.
Present so MCPSDKStatus health check finds a complete SDK under src/mcp/sdk/.
"""
from __future__ import annotations

# Re-export core MCP API from parent package
import sys
from pathlib import Path

# Ensure parent mcp package is importable when this file is run or imported in isolation
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent.parent))

# Delegate to real implementation
from mcp.mcp import (
    MCP,
    get_mcp_instance,
    initialize,
    list_available_tools,
    list_available_resources,
    get_tool_info,
    get_resource_info,
)
from mcp import __version__, FEATURES, process_mcp

__all__ = [
    "MCP",
    "get_mcp_instance",
    "initialize",
    "list_available_tools",
    "list_available_resources",
    "get_tool_info",
    "get_resource_info",
    "__version__",
    "FEATURES",
    "process_mcp",
]
