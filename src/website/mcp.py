"""
MCP (Model Context Protocol) integration for the website module.

Exposes website generation and inspection tools through MCP with
full typed schemas and domain-specific functionality.
"""

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

from .generator import generate_website as _generate_website
from .renderer import process_website


def process_website_mcp(target_directory: str, output_directory: str,
                        verbose: bool = False) -> Dict[str, Any]:
    """
    Generate a static website from GNN pipeline artifacts.

    Args:
        target_directory: Directory containing GNN files / pipeline input
        output_directory: Directory to write the website HTML pages
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status, pages_created count, errors and warnings.
    """
    try:
        result = process_website(Path(target_directory), Path(output_directory), verbose=verbose)
        # process_website returns bool or dict
        if isinstance(result, dict):
            return {
                "success": bool(result.get("success", False)),
                "pages_created": result.get("pages_created", 0),
                "target_directory": target_directory,
                "output_directory": output_directory,
                "errors": result.get("errors", []),
                "warnings": result.get("warnings", []),
            }
        return {
            "success": bool(result),
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Website processing {'completed successfully' if result else 'failed'}",
        }
    except Exception as e:
        logger.error(f"process_website_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def build_from_pipeline_output_mcp(pipeline_output_directory: str,
                                   website_output_directory: str,
                                   verbose: bool = False) -> Dict[str, Any]:
    """
    Build the full premium website by scanning a pipeline output root directory.

    Automatically discovers analysis results, visualizations, GNN files, and
    JSON reports from numbered pipeline output folders (e.g. 03_gnn_output/,
    08_visualization_output/, 16_analysis_output/, etc.).

    Args:
        pipeline_output_directory: Root of the numbered pipeline output directories
        website_output_directory: Destination for the generated website
        verbose: Enable verbose logging

    Returns:
        Dictionary with pages_created, discovered artifacts, errors.
    """
    try:
        import logging as _log
        _logger = _log.getLogger("website.mcp.build")
        if verbose:
            _logger.setLevel(_log.DEBUG)

        p_root = Path(pipeline_output_directory)
        out    = Path(website_output_directory)

        result = _generate_website(_logger, p_root, out, pipeline_output_root=p_root)
        return {
            "success":             result.get("success", False),
            "pages_created":       result.get("pages_created", 0),
            "pipeline_output_dir": str(p_root),
            "website_output_dir":  str(out),
            "errors":              result.get("errors", []),
            "warnings":            result.get("warnings", []),
        }
    except Exception as e:
        logger.error(f"build_from_pipeline_output_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_website_status_mcp(website_directory: str) -> Dict[str, Any]:
    """
    Inspect an existing generated website directory.

    Returns the list of HTML pages found, total size, and whether key pages
    (index.html, pipeline.html, mcp.html) are present.

    Args:
        website_directory: Path to a previously generated website directory

    Returns:
        Dictionary with page inventory, sizes, and completeness flags.
    """
    try:
        wdir = Path(website_directory)
        if not wdir.exists():
            return {"success": False, "error": f"Directory not found: {website_directory}"}

        pages = sorted(wdir.glob("*.html"))
        assets = list((wdir / "assets").glob("*")) if (wdir / "assets").exists() else []
        key_pages = ["index.html", "pipeline.html", "gnn_files.html",
                     "analysis.html", "visualization.html", "reports.html", "mcp.html"]
        completeness = {p: (wdir / p).exists() for p in key_pages}
        total_size = sum(f.stat().st_size for f in pages if f.exists())

        return {
            "success":    True,
            "directory":  str(wdir),
            "pages":      [p.name for p in pages],
            "pages_count": len(pages),
            "assets_count": len(assets),
            "total_size_bytes": total_size,
            "completeness": completeness,
            "all_key_pages_present": all(completeness.values()),
        }
    except Exception as e:
        logger.error(f"get_website_status_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_generated_pages_mcp(website_directory: str) -> Dict[str, Any]:
    """
    List all HTML pages in a generated website directory with metadata.

    Args:
        website_directory: Path to a previously generated website directory

    Returns:
        Dictionary with page list, sizes, and last-modified timestamps.
    """
    try:
        wdir = Path(website_directory)
        if not wdir.exists():
            return {"success": False, "error": f"Directory not found: {website_directory}"}

        from datetime import datetime as _dt
        pages = []
        for html_file in sorted(wdir.glob("*.html")):
            stat = html_file.stat()
            pages.append({
                "name":     html_file.name,
                "size_bytes": stat.st_size,
                "modified": _dt.fromtimestamp(stat.st_mtime).isoformat(),
            })

        return {
            "success": True,
            "directory": str(wdir),
            "pages": pages,
            "total_pages": len(pages),
        }
    except Exception as e:
        logger.error(f"list_generated_pages_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_website_module_info_mcp() -> Dict[str, Any]:
    """
    Return metadata about the website module capabilities.

    Returns:
        Dictionary with version, features, supported formats, and MCP tool list.
    """
    try:
        from . import FEATURES, SUPPORTED_FILE_TYPES, __version__
        return {
            "success": True,
            "version": __version__,
            "features": FEATURES,
            "supported_file_types": SUPPORTED_FILE_TYPES,
            "pages": ["index", "pipeline", "gnn_files", "analysis",
                      "visualization", "reports", "mcp"],
            "mcp_tools": [
                "process_website",
                "build_from_pipeline_output",
                "get_website_status",
                "list_generated_pages",
                "get_website_module_info",
            ],
        }
    except Exception as e:
        logger.error(f"get_website_module_info_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── MCP Registration ────────────────────────────────────────────────────────

def register_tools(mcp_instance) -> None:
    """Register all website module tools with the MCP server."""

    # Generic introspection tools (namespaced)
    # Domain-specific tools
    mcp_instance.register_tool(
        "process_website",
        process_website_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {"type": "string", "description": "Directory containing GNN pipeline input files"},
                "output_directory": {"type": "string", "description": "Directory to write the generated website"},
                "verbose":          {"type": "boolean", "description": "Enable verbose logging", "default": False},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Generate a premium 7-page static HTML website from GNN pipeline artifacts.",
        module=__package__, category="website",
    )

    mcp_instance.register_tool(
        "build_website_from_pipeline_output",
        build_from_pipeline_output_mcp,
        {
            "type": "object",
            "properties": {
                "pipeline_output_directory": {"type": "string", "description": "Root of numbered pipeline output dirs (e.g. output/)"},
                "website_output_directory":  {"type": "string", "description": "Destination for the generated website"},
                "verbose":                   {"type": "boolean", "description": "Enable verbose logging", "default": False},
            },
            "required": ["pipeline_output_directory", "website_output_directory"],
        },
        "Build the full GNN website by auto-discovering all pipeline artifacts from numbered output directories.",
        module=__package__, category="website",
    )

    mcp_instance.register_tool(
        "get_website_status",
        get_website_status_mcp,
        {
            "type": "object",
            "properties": {
                "website_directory": {"type": "string", "description": "Path to a previously generated website directory"},
            },
            "required": ["website_directory"],
        },
        "Inspect an existing generated website: list pages, sizes, and check completeness of key pages.",
        module=__package__, category="website",
    )

    mcp_instance.register_tool(
        "list_generated_website_pages",
        list_generated_pages_mcp,
        {
            "type": "object",
            "properties": {
                "website_directory": {"type": "string", "description": "Path to a previously generated website directory"},
            },
            "required": ["website_directory"],
        },
        "List all HTML pages in a generated website directory with sizes and timestamps.",
        module=__package__, category="website",
    )

    mcp_instance.register_tool(
        "get_website_module_info",
        get_website_module_info_mcp,
        {},
        "Return metadata about the website module: version, supported file types, and available MCP tools.",
        module=__package__, category="website",
    )

    logger.info("website module MCP tools registered (6 tools).")
