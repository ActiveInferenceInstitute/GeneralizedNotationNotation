"""
MCP SDK client shim: minimal client interface delegating to the parent mcp module.
Present so MCPSDKStatus health check finds a complete SDK under src/mcp/sdk/.
"""
from __future__ import annotations

import sys
from pathlib import Path

_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent.parent))

from mcp.mcp import get_mcp_instance, list_available_resources, list_available_tools

__all__ = ["get_mcp_instance", "list_available_tools", "list_available_resources"]
