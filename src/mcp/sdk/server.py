"""
MCP SDK server shim: delegates to the parent mcp.server implementation.
Present so MCPSDKStatus health check finds a complete SDK under src/mcp/sdk/.
"""
from __future__ import annotations

import sys
from pathlib import Path

_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent.parent))

from mcp.server import MCPServer

__all__ = ["MCPServer"]
