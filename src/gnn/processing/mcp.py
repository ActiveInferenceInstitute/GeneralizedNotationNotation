"""
MCP integration for the processing module.

Exposes the lightweight GNN processing surface (``gnn.processing``) through
MCP: single-file parsing, structural checking, and directory discovery from
``gnn.processing.processor``.

The processing imports happen lazily inside the tool bodies: the
``gnn.processing`` package ``__init__`` also pulls in the five-phase
``GNNProcessor`` orchestration engine, which is too heavy for module
discovery (which loads every ``mcp.py`` under a 30s timeout).
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

from gnn.utils.mcp.dispatch import run_tool_envelope


def parse_gnn_file_mcp(file_path: str) -> Dict[str, Any]:
    """
    Parse a GNN file and extract basic structural information.

    Args:
        file_path: Path to the GNN specification file to parse.

    Returns:
        Dictionary with ``success`` and ``file_path``. On success, the
        lightweight parse result: ``sections``, ``variables``,
        ``structure_info`` (variable/section/line/character counts), and
        ``file_name``. On failure, ``success`` is ``False`` with the
        processor's ``error`` detail. Exceptions are converted to
        ``{"success": False, "error": str(e)}`` by the envelope.
    """

    def _build() -> Dict[str, Any]:
        from gnn.processing import parse_gnn_file

        result = parse_gnn_file(file_path)
        return {"success": True, **result}

    return run_tool_envelope(_build, wrapper_name="parse_gnn_file_mcp", logger=logger)


def check_gnn_file_structure_mcp(file_path: str) -> Dict[str, Any]:
    """
    Validate the structure of a GNN file (lightweight checks).

    Args:
        file_path: Path to the GNN specification file to check.

    Returns:
        Dictionary with ``success`` and ``file_path``. On success, the
        structural verdict: ``valid``, ``errors``, and ``warnings`` from
        the lightweight structure checker. Exceptions are converted to
        ``{"success": False, "error": str(e)}`` by the envelope.
    """

    def _build() -> Dict[str, Any]:
        from gnn.processing import check_gnn_file_structure

        result = check_gnn_file_structure(file_path)
        return {"success": True, **result}

    return run_tool_envelope(
        _build, wrapper_name="check_gnn_file_structure_mcp", logger=logger
    )


def discover_gnn_files_mcp(directory: str, recursive: bool = True) -> Dict[str, Any]:
    """
    Discover GNN model files in a directory.

    Args:
        directory: Directory to scan for GNN specification files.
        recursive: Whether to scan subdirectories. Defaults to true.

    Returns:
        Dictionary with ``success`` and ``directory``. On success,
        ``files`` (sorted path strings) and ``count``. Exceptions are
        converted to ``{"success": False, "error": str(e)}`` by the
        envelope.
    """

    def _build() -> Dict[str, Any]:
        from gnn.processing import discover_gnn_files

        files = discover_gnn_files(directory, recursive=recursive)
        return {
            "success": True,
            "directory": directory,
            "files": sorted(str(f) for f in files),
            "count": len(files),
        }

    return run_tool_envelope(
        _build, wrapper_name="discover_gnn_files_mcp", logger=logger
    )


# ── MCP Registration ────────────────────────────────────────────────────────


def register_tools(mcp_instance: Any) -> None:
    """Register processing tools with the MCP server."""

    mcp_instance.register_tool(
        name="processing.parse_gnn_file",
        func=parse_gnn_file_mcp,
        schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the GNN specification file to parse",
                },
            },
            "required": ["file_path"],
        },
        description=(
            "Parse a GNN file and extract basic structural information "
            "(sections, variables, counts)."
        ),
        module=__package__,
        category="processing",
    )

    mcp_instance.register_tool(
        name="processing.check_gnn_file_structure",
        func=check_gnn_file_structure_mcp,
        schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the GNN specification file to check",
                },
            },
            "required": ["file_path"],
        },
        description=(
            "Validate the structure of a GNN file with lightweight checks "
            "(empty/short content, missing sections, unmatched brackets)."
        ),
        module=__package__,
        category="processing",
    )

    mcp_instance.register_tool(
        name="processing.discover_gnn_files",
        func=discover_gnn_files_mcp,
        schema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to scan for GNN specification files",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to scan subdirectories",
                    "default": True,
                },
            },
            "required": ["directory"],
        },
        description=(
            "Discover GNN model files in a directory and return their paths."
        ),
        module=__package__,
        category="processing",
    )

    logger.info("processing module MCP tools registered (3 tools).")
