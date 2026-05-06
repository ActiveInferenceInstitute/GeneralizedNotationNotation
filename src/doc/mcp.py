"""
MCP Tools for Architecture Documentation Integration.

The ``src/doc/`` tree is static Markdown documentation and exposes no
callable tools. ``register_tools`` exists to satisfy the MCP auto-discovery
contract (every top-level ``src/<module>/`` is expected to expose an
``mcp.py`` entry point).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_tools(mcp_instance: Any) -> None:
    """Register doc-module tools with the MCP server (no-op).

    The doc module is static Markdown and has nothing to register.
    """
    logger.info("doc.mcp: no tools to register (static Markdown tree)")
