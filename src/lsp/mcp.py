"""
MCP Tools for Language Server Protocol (LSP) Integration.

LSP has no MCP-facing tools because the LSP itself is invoked out-of-band
(over stdio) by IDE clients, not through the MCP server. ``register_tools``
exists to satisfy the MCP auto-discovery contract.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_tools(mcp_instance: Any) -> None:
    """Register LSP tools with the MCP server (no-op).

    LSP operates out-of-band over stdio and has no MCP-callable tools.
    """
    logger.info("lsp.mcp: no tools to register (LSP operates out-of-band via stdio)")
