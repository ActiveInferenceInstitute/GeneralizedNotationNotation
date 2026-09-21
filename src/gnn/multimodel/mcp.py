"""
MCP integration for the multimodel module.

Exposes inter-model dependency-graph rendering (the ``gnn graph`` CLI
parity surface) through the Model Context Protocol: parse a — possibly
multi-model — GNN file, build the model dependency graph, and render it
as a Mermaid flowchart or a plain-text adjacency list.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

from gnn.utils.mcp.dispatch import run_tool_envelope

# ── Domain tools ─────────────────────────────────────────────────────────────


def generate_dependency_graph_mcp(
    file_path: str,
    output_format: str = "mermaid",
) -> Dict[str, Any]:
    """
    Render the inter-model dependency graph of a GNN file.

    Parses ``file_path`` (a single- or multi-model GNN file), builds the
    dependency graph over the models it contains, and renders it in the
    requested format.

    Args:
        file_path: Path to the GNN file to analyze. Must exist; a missing
            or non-file path yields ``{"success": False, "error": ...}``.
        output_format: Render format. ``"mermaid"`` produces a Mermaid
            ``graph TD`` flowchart; ``"text"`` produces a plain-text
            adjacency list. Anything else yields a typed error.

    Returns:
        Success dict: ``{"success": True, "file_path": ..., "format": ...,
        "graph": <rendered string>}``. Failure dict:
        ``{"success": False, "error": <message>}``.
    """
    if output_format not in ("mermaid", "text"):
        return {"success": False, "error": "output_format must be 'mermaid' or 'text'"}

    def _build() -> Dict[str, Any]:
        if not Path(file_path).is_file():
            return {"success": False, "error": f"GNN file not found: {file_path}"}
        from gnn.multimodel import render_graph_from_file

        graph_str = render_graph_from_file(file_path, output_format=output_format)
        return {
            "success": True,
            "file_path": file_path,
            "format": output_format,
            "graph": graph_str,
        }

    return run_tool_envelope(_build, wrapper_name="generate_dependency_graph_mcp", logger=logger)


# ── MCP Registration ──────────────────────────────────────────────────────────


def register_tools(mcp_instance: Any) -> None:
    """Register multimodel domain tools with the MCP server."""

    mcp_instance.register_tool(
        "generate_dependency_graph",
        generate_dependency_graph_mcp,
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the (possibly multi-model) GNN file to analyze",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["mermaid", "text"],
                    "default": "mermaid",
                    "description": "Render format for the dependency graph",
                },
            },
            "required": ["file_path"],
        },
        "Render the inter-model dependency graph of a GNN file as a Mermaid diagram or text adjacency list.",
        module=__package__,
        category="multimodel",
    )

    logger.info("multimodel module MCP tools registered (1 tool).")
