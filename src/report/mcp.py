"""
MCP integration for the report module.

Exposes GNN report generation tools: comprehensive report creation,
format listing, validation, and report inspection through MCP.
"""

from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

from .generator import generate_comprehensive_report
from . import process_report, get_supported_formats, get_module_info as _get_mod_info






def generate_report_mcp(target_directory: str, output_directory: str,
                        verbose: bool = False) -> Dict[str, Any]:
    """
    Generate a comprehensive pipeline execution report.

    Collects data from all pipeline step outputs, computes aggregate statistics,
    and writes a detailed JSON + summary report to the output directory.

    Args:
        target_directory: Directory containing GNN files (pipeline input)
        output_directory: Directory to save generated reports
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and report summary.
    """
    try:
        import logging as _log
        _logger = _log.getLogger("report.mcp")
        if verbose:
            _logger.setLevel(_log.DEBUG)
        success = generate_comprehensive_report(Path(target_directory), Path(output_directory), _logger)
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Report generation {'completed successfully' if success else 'completed with issues'}",
        }
    except Exception as e:
        logger.error(f"generate_report_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def process_report_mcp(target_directory: str, output_directory: str,
                       verbose: bool = False) -> Dict[str, Any]:
    """
    Run the full report processing pipeline step.

    Args:
        target_directory: Pipeline input directory
        output_directory: Report output directory
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status.
    """
    try:
        success = process_report(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
        }
    except Exception as e:
        logger.error(f"process_report_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_report_formats_mcp() -> Dict[str, Any]:
    """
    Return all supported report output formats.

    Returns:
        Dictionary with format names, extensions, and descriptions.
    """
    try:
        formats = get_supported_formats()
        return {"success": True, "formats": formats}
    except Exception as e:
        return {
            "success": True,
            "formats": ["json", "html", "txt", "markdown"],
            "note": str(e) if e else "",
        }


def read_report_mcp(report_file_path: str) -> Dict[str, Any]:
    """
    Read and return the contents of a generated report file.

    Supports JSON, text, and markdown report formats.

    Args:
        report_file_path: Path to the report file to read

    Returns:
        Dictionary with report content and metadata.
    """
    try:
        import json as _json
        rpath = Path(report_file_path)
        if not rpath.exists():
            return {"success": False, "error": f"Report file not found: {report_file_path}"}

        content = rpath.read_text(encoding="utf-8", errors="replace")
        parsed = None
        if rpath.suffix == ".json":
            try:
                parsed = _json.loads(content)
            except Exception:
                pass

        return {
            "success":   True,
            "file":      str(rpath),
            "format":    rpath.suffix.lstrip("."),
            "size_bytes": rpath.stat().st_size,
            "content":   content[:5000],
            "data":      parsed,
        }
    except Exception as e:
        logger.error(f"read_report_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_report_module_info_mcp() -> Dict[str, Any]:
    """Return metadata about the report module."""
    try:
        info = _get_mod_info()
        return {"success": True, **info}
    except Exception as e:
        logger.error(f"get_report_module_info_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── MCP Registration ────────────────────────────────────────────────────────

def register_tools(mcp_instance) -> None:
    """Register report tools with the MCP server."""

    mcp_instance.register_tool(
        "generate_report",
        generate_report_mcp,
        {"type": "object", "properties": {
            "target_directory": {"type": "string", "description": "Pipeline input directory"},
            "output_directory": {"type": "string", "description": "Report output directory"},
            "verbose":          {"type": "boolean", "default": False},
        }, "required": ["target_directory", "output_directory"]},
        "Generate a comprehensive pipeline execution report with statistics and summaries.",
        module=__package__, category="report",
    )

    mcp_instance.register_tool(
        "process_report",
        process_report_mcp,
        {"type": "object", "properties": {
            "target_directory": {"type": "string"},
            "output_directory": {"type": "string"},
            "verbose": {"type": "boolean", "default": False},
        }, "required": ["target_directory", "output_directory"]},
        "Run the full report pipeline step.",
        module=__package__, category="report",
    )

    mcp_instance.register_tool(
        "list_report_formats",
        list_report_formats_mcp, {},
        "Return all supported report output formats (JSON, HTML, Markdown, etc.).",
        module=__package__, category="report",
    )

    mcp_instance.register_tool(
        "read_report",
        read_report_mcp,
        {"type": "object", "properties": {
            "report_file_path": {"type": "string", "description": "Path to the report file"},
        }, "required": ["report_file_path"]},
        "Read and return the contents of a generated pipeline report file.",
        module=__package__, category="report",
    )

    mcp_instance.register_tool(
        "get_report_module_info",
        get_report_module_info_mcp, {},
        "Return metadata about the report module (version, supported formats).",
        module=__package__, category="report",
    )

    logger.info("report module MCP tools registered (5 domain tools).")
