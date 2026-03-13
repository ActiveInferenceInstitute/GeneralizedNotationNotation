"""
Analysis MCP Integration

Exposes GNN analysis processing, complexity metrics, and result inspection
through the Model Context Protocol with full typed schemas.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

from . import process_analysis


def process_analysis_mcp(target_directory: str, output_directory: str,
                         verbose: bool = False) -> Dict[str, Any]:
    """
    Run GNN statistical and complexity analysis on a directory of GNN files.

    Args:
        target_directory: Directory containing GNN files to analyse
        output_directory: Directory to save analysis results (JSON)
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and summary of analysis performed.
    """
    try:
        success = process_analysis(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Analysis {'completed successfully' if success else 'completed with issues'}",
        }
    except Exception as e:
        logger.error(f"process_analysis_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_analysis_results_mcp(output_directory: str,
                              model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Read and return saved analysis results from a previous analysis run.

    Searches the output directory for analysis_results.json or per-model JSON
    files and returns their contents as structured data.

    Args:
        output_directory: Directory where analysis results were saved
        model_name: Optional model name filter (returns only that model's results)

    Returns:
        Dictionary containing analysis results data.
    """
    try:
        out_dir = Path(output_directory)
        if not out_dir.exists():
            return {"success": False, "error": f"Output directory not found: {output_directory}"}

        results: List[Dict[str, Any]] = []
        for json_file in sorted(out_dir.rglob("*.json")):
            try:
                data = json.loads(json_file.read_text())
                if model_name and model_name not in json_file.name:
                    continue
                results.append({"file": json_file.name, "data": data})
            except Exception as e:
                logger.debug(f"Skipping {json_file.name}: {e}")

        return {
            "success": True,
            "output_directory": str(out_dir),
            "results_count": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"get_analysis_results_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def compute_complexity_metrics_mcp(gnn_content: str,
                                   model_name: str = "model") -> Dict[str, Any]:
    """
    Compute complexity metrics for a GNN model provided as string content.

    Parses the GNN file and computes:
    - Number of state variables
    - Number of connections / edges
    - Parameter count
    - Estimated cyclomatic complexity
    - Section count

    Args:
        gnn_content: GNN model content as a string
        model_name: Human-readable name for this model

    Returns:
        Dictionary with computed complexity metrics.
    """
    try:
        lines = gnn_content.splitlines()
        sections    = [l.strip("# ").strip() for l in lines if l.startswith("## ")]
        connections = [l for l in lines if "->" in l or "<->" in l or "→" in l]
        variables   = [l for l in lines if "[" in l and "]" in l
                       and not l.strip().startswith("#")]
        parameters  = [l for l in lines if "=" in l and not l.strip().startswith("#")
                       and "->" not in l]

        n_vars   = len(variables)
        n_conns  = len(connections)
        n_params = len(parameters)
        n_secs   = len(sections)
        cyclomatic = max(1, n_conns - n_vars + 2)

        return {
            "success": True,
            "model_name": model_name,
            "sections":           n_secs,
            "state_variables":    n_vars,
            "connections":        n_conns,
            "parameters":         n_params,
            "total_lines":        len(lines),
            "cyclomatic_complexity": cyclomatic,
            "complexity_rating": (
                "low" if cyclomatic < 5 else
                "medium" if cyclomatic < 15 else
                "high"
            ),
        }
    except Exception as e:
        logger.error(f"compute_complexity_metrics_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_analysis_tools_mcp() -> Dict[str, Any]:
    """
    Return information about available analysis tools and their capabilities.

    Returns:
        Dictionary with tool names, descriptions, and module metadata.
    """
    try:
        from . import check_analysis_tools
        tools_info = check_analysis_tools()
        return {"success": True, "tools": tools_info}
    except Exception as e:
        return {
            "success": True,
            "tools": {
                "statistical_analysis": {"available": True, "description": "Statistical measures on GNN model structure"},
                "complexity_metrics":   {"available": True, "description": "Cyclomatic and cognitive complexity"},
                "network_analysis":     {"available": True, "description": "Graph-theoretic analysis of connections"},
            },
            "note": str(e) if e else "",
        }


# ── MCP Registration ────────────────────────────────────────────────────────

def register_tools(mcp_instance) -> None:
    """Register analysis tools with the MCP server."""

    mcp_instance.register_tool(
        "process_analysis",
        process_analysis_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {"type": "string", "description": "Directory containing GNN files to analyse"},
                "output_directory": {"type": "string", "description": "Directory to save analysis results"},
                "verbose":          {"type": "boolean", "description": "Enable verbose logging", "default": False},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Run statistical and complexity analysis on GNN files in a directory.",
        module=__package__, category="analysis",
    )

    mcp_instance.register_tool(
        "get_analysis_results",
        get_analysis_results_mcp,
        {
            "type": "object",
            "properties": {
                "output_directory": {"type": "string", "description": "Directory containing saved analysis JSON results"},
                "model_name":       {"type": "string", "description": "Optional model name to filter results"},
            },
            "required": ["output_directory"],
        },
        "Read and return saved analysis results from a previous analysis run.",
        module=__package__, category="analysis",
    )

    mcp_instance.register_tool(
        "compute_complexity_metrics",
        compute_complexity_metrics_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_content": {"type": "string", "description": "GNN model content as a string"},
                "model_name":  {"type": "string", "description": "Human-readable model name", "default": "model"},
            },
            "required": ["gnn_content"],
        },
        "Compute complexity metrics (variables, connections, cyclomatic complexity) for GNN content.",
        module=__package__, category="analysis",
    )

    mcp_instance.register_tool(
        "list_analysis_tools",
        list_analysis_tools_mcp,
        {},
        "Return information about available GNN analysis tools and capabilities.",
        module=__package__, category="analysis",
    )

    logger.info("analysis module MCP tools registered (4 tools).")
