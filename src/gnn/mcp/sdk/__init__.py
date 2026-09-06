"""MCP SDK — facades delegating to the parent mcp module.

The files under ``src/mcp/sdk/`` exist so that ``MCPSDKStatus`` health checks
find a complete SDK directory.  Every public name is re-exported from the
parent ``mcp`` package; import from ``mcp`` or ``mcp.sdk`` as needed.
"""

from .client import (
    get_mcp_instance,
    list_available_resources,
    list_available_tools,
)
from .mcp import FEATURES, __version__
from .server import MCPServer

__all__ = [
    "get_mcp_instance",
    "list_available_tools",
    "list_available_resources",
    "MCPServer",
    "FEATURES",
    "__version__",
]
