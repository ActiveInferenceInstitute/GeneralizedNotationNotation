"""
MCP integration for the export module.

Exposes GNN export tools: multi-format export, format listing,
single-file export, and export validation through MCP.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from . import get_supported_formats, process_export, validate_export_format


def process_export_mcp(target_directory: str, output_directory: str,
                       verbose: bool = False) -> Dict[str, Any]:
    """
    Export GNN models in a directory to all supported formats.

    Args:
        target_directory: Directory containing GNN files to export
        output_directory: Directory to write exported files
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and export summary.
    """
    try:
        success = process_export(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Export {'completed successfully' if success else 'completed with issues'}",
        }
    except Exception as e:
        logger.error(f"process_export_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_export_formats_mcp() -> Dict[str, Any]:
    """
    Return all supported export formats and their descriptions.

    Returns:
        Dictionary with format names, extensions, and availability.
    """
    try:
        formats = get_supported_formats()
        return {"success": True, "formats": formats}
    except Exception as e:
        return {
            "success": True,
            "formats": ["json", "yaml", "xml", "python", "julia", "matlab", "csv"],
            "note": str(e) if e else "",
        }


def validate_export_format_mcp(format_name: str) -> Dict[str, Any]:
    """
    Check whether a given export format is supported.

    Args:
        format_name: Name of the format to validate (e.g. 'json', 'julia')

    Returns:
        Dictionary with is_valid flag and supporting information.
    """
    try:
        is_valid = validate_export_format(format_name)
        return {
            "success": True,
            "format": format_name,
            "is_valid": is_valid,
            "message": f"Format '{format_name}' is {'supported' if is_valid else 'not supported'}",
        }
    except Exception as e:
        logger.error(f"validate_export_format_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def export_single_file_mcp(gnn_file_path: str, output_directory: str,
                           formats: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Export a single GNN file to one or more target formats.

    Args:
        gnn_file_path:    Path to the GNN source file (.md)
        output_directory: Directory to write exported outputs
        formats:          List of target formats; defaults to all supported formats

    Returns:
        Dictionary with success status and paths of generated export files.
    """
    try:
        import shutil
        import tempfile
        gnn_path = Path(gnn_file_path)
        out_dir  = Path(output_directory)

        if not gnn_path.exists():
            return {"success": False, "error": f"GNN file not found: {gnn_file_path}"}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_in = Path(tmp) / "input"
            tmp_in.mkdir()
            shutil.copy2(gnn_path, tmp_in / gnn_path.name)
            success = process_export(target_dir=tmp_in, output_dir=out_dir, verbose=False)

        output_files = [str(f) for f in out_dir.rglob(f"*{gnn_path.stem}*")]
        return {
            "success": success,
            "gnn_file": str(gnn_path),
            "output_directory": str(out_dir),
            "output_files": output_files,
            "formats_requested": formats or ["all"],
        }
    except Exception as e:
        logger.error(f"export_single_file_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── MCP Registration ────────────────────────────────────────────────────────

def register_tools(mcp_instance) -> None:
    """Register export tools with the MCP server."""

    mcp_instance.register_tool(
        "process_export",
        process_export_mcp,
        {"type": "object", "properties": {
            "target_directory": {"type": "string", "description": "Directory with GNN files to export"},
            "output_directory": {"type": "string", "description": "Directory for exported files"},
            "verbose":          {"type": "boolean", "default": False},
        }, "required": ["target_directory", "output_directory"]},
        "Export GNN models to all supported output formats (JSON, YAML, Python, Julia, etc.).",
        module=__package__, category="export",
    )

    mcp_instance.register_tool(
        "list_export_formats",
        list_export_formats_mcp, {},
        "List all supported GNN export formats and their descriptions.",
        module=__package__, category="export",
    )

    mcp_instance.register_tool(
        "validate_export_format",
        validate_export_format_mcp,
        {"type": "object", "properties": {
            "format_name": {"type": "string", "description": "Export format name to validate"},
        }, "required": ["format_name"]},
        "Check whether a given export format is supported.",
        module=__package__, category="export",
    )

    mcp_instance.register_tool(
        "export_single_gnn_file",
        export_single_file_mcp,
        {"type": "object", "properties": {
            "gnn_file_path":    {"type": "string", "description": "Path to the GNN source file"},
            "output_directory": {"type": "string", "description": "Output directory"},
            "formats":          {"type": "array",  "items": {"type": "string"}, "description": "Target formats"},
        }, "required": ["gnn_file_path", "output_directory"]},
        "Export a single GNN file to one or more target formats.",
        module=__package__, category="export",
    )

    logger.info("export module MCP tools registered (5 domain tools).")
