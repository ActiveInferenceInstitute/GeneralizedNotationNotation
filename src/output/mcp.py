"""
MCP Tools for Output Integration.

This module is an intentional structural stub — the ``src/output/`` directory
holds pipeline artifacts at runtime and has no callable tools to expose.
``register_tools`` exists to satisfy the MCP auto-discovery contract (every
top-level ``src/<module>/`` is expected to expose an ``mcp.py`` entry point).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_tools(mcp_instance: Any) -> None:
    """Register output-module tools with the MCP server (no-op).

    The output module has no tools to register — it's a pure artifact
    directory. We log the no-op registration so the MCP audit log can
    confirm every module was discovered.
    """
    logger.info("output.mcp: no tools to register (structural stub)")
