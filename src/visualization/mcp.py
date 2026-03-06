"""
MCP integration for the visualization module.

Exposes GNN visualization tools: static graph generation, visualization
options, metric summaries, and batch processing through MCP.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

from . import process_visualization
from . import get_visualization_options, get_module_info as _get_mod_info






def process_visualization_mcp(target_directory: str, output_directory: str,
                               verbose: bool = False) -> Dict[str, Any]:
    """
    Generate static visualizations for all GNN models in a directory.

    Produces PNG/SVG plots including state-space graphs, connection matrices,
    and parameter visualizations.

    Args:
        target_directory: Directory containing GNN files to visualise
        output_directory: Directory to write generated plots
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and visualization summary.
    """
    try:
        success = process_visualization(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
        )
        # Count output files
        out = Path(output_directory)
        n_files = len(list(out.rglob("*.png")) + list(out.rglob("*.svg"))) if out.exists() else 0
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "output_files_count": n_files,
            "message": f"Visualization {'completed successfully' if success else 'completed with issues'}",
        }
    except Exception as e:
        logger.error(f"process_visualization_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_visualization_options_mcp() -> Dict[str, Any]:
    """
    Return available visualization types and their configuration options.

    Returns:
        Dictionary with visualization type names, descriptions, and options.
    """
    try:
        options = get_visualization_options()
        return {"success": True, "options": options}
    except Exception as e:
        return {
            "success": True,
            "options": {
                "state_space_graph": {"description": "Network graph of GNN state space"},
                "connectivity_matrix": {"description": "Connection matrix heatmap"},
                "parameter_distributions": {"description": "Histograms of model parameters"},
                "time_series": {"description": "Time-step evolution plots"},
            },
            "note": str(e) if e else "",
        }


def list_visualization_artifacts_mcp(output_directory: str) -> Dict[str, Any]:
    """
    List all visualization artifacts generated in an output directory.

    Args:
        output_directory: Directory containing generated visualization files

    Returns:
        Dictionary with file list, types, sizes, and counts by format.
    """
    try:
        out_dir = Path(output_directory)
        if not out_dir.exists():
            return {"success": False, "error": f"Directory not found: {output_directory}"}

        artifacts: List[Dict[str, Any]] = []
        for ext in ["*.png", "*.svg", "*.html", "*.pdf"]:
            for f in sorted(out_dir.rglob(ext)):
                artifacts.append({
                    "name":   f.name,
                    "type":   f.suffix.lstrip("."),
                    "size_bytes": f.stat().st_size,
                    "path":   str(f.relative_to(out_dir)),
                })

        by_type: Dict[str, int] = {}
        for a in artifacts:
            by_type[a["type"]] = by_type.get(a["type"], 0) + 1

        return {
            "success":          True,
            "output_directory": str(out_dir),
            "total_artifacts":  len(artifacts),
            "by_type":          by_type,
            "artifacts":        artifacts,
        }
    except Exception as e:
        logger.error(f"list_visualization_artifacts_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_visualization_module_info_mcp() -> Dict[str, Any]:
    """
    Return metadata about the visualization module.

    Returns:
        Dictionary with version, available backends, and output formats.
    """
    try:
        info = _get_mod_info()
        return {"success": True, **info}
    except Exception as e:
        logger.error(f"get_visualization_module_info_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── MCP Registration ────────────────────────────────────────────────────────

def register_tools(mcp_instance) -> None:
    """Register visualization tools with the MCP server."""

    mcp_instance.register_tool(
        "process_visualization",
        process_visualization_mcp,
        {"type": "object", "properties": {
            "target_directory": {"type": "string", "description": "Directory with GNN files"},
            "output_directory": {"type": "string", "description": "Output directory for plots"},
            "verbose":          {"type": "boolean", "default": False},
        }, "required": ["target_directory", "output_directory"]},
        "Generate static PNGs/SVGs for all GNN models (state-space, connection matrix, parameters).",
        module=__package__, category="visualization",
    )

    mcp_instance.register_tool(
        "get_visualization_options",
        get_visualization_options_mcp,
        {},
        "Return available visualization types and their configuration options.",
        module=__package__, category="visualization",
    )

    mcp_instance.register_tool(
        "list_visualization_artifacts",
        list_visualization_artifacts_mcp,
        {"type": "object", "properties": {
            "output_directory": {"type": "string", "description": "Directory containing visualization artifacts"},
        }, "required": ["output_directory"]},
        "List all visualization artifacts (PNG, SVG, HTML, PDF) in an output directory.",
        module=__package__, category="visualization",
    )

    mcp_instance.register_tool(
        "get_visualization_module_info",
        get_visualization_module_info_mcp,
        {},
        "Return metadata about the visualization module (version, backends, output formats).",
        module=__package__, category="visualization",
    )

    logger.info("visualization module MCP tools registered (4 domain tools).")
