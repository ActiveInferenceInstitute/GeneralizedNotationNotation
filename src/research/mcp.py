"""
MCP integration for the research module.

Exposes GNN-related research tools: pipeline research processing,
literature topic listing, experiment results reading, and research
metadata retrieval through MCP.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

from . import process_research


def process_research_mcp(target_directory: str, output_directory: str,
                         verbose: bool = False) -> Dict[str, Any]:
    """
    Run GNN research processing on a directory of GNN files.

    Generates research-oriented metadata, cross-references, and structured
    experiment summaries compatible with academic and reproducibility workflows.

    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save research outputs
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and research summary.
    """
    try:
        success = process_research(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Research processing {'completed successfully' if success else 'completed with issues'}",
        }
    except Exception as e:
        logger.error(f"process_research_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_research_topics_mcp() -> Dict[str, Any]:
    """
    Return a list of Active Inference and GNN research topics and domains.

    Returns:
        Dictionary with topic names, descriptions, and related GNN constructs.
    """
    topics = {
        "active_inference":  "Free Energy Principle / Active Inference agents",
        "pomdp":             "Partially Observable Markov Decision Processes",
        "bayesian_mechanics":"Bayesian mechanics and path integral formulation",
        "predictive_coding": "Predictive coding and hierarchical generative models",
        "markov_blankets":   "Markov blankets and statistical self-organisation",
        "variational_bayes": "Variational Bayes and message passing algorithms",
        "epistemic_value":   "Epistemic value, information gain and exploration",
        "policy_selection":  "Policy selection via Expected Free Energy",
    }
    return {"success": True, "topics": topics, "count": len(topics)}


def read_research_results_mcp(output_directory: str,
                               file_pattern: str = "*.json") -> Dict[str, Any]:
    """
    Read and return research results from a previous research processing run.

    Args:
        output_directory: Directory containing saved research output files
        file_pattern:     Glob pattern to match files (default '*.json')

    Returns:
        Dictionary with file contents, counts, and metadata.
    """
    try:
        out_dir = Path(output_directory)
        if not out_dir.exists():
            return {"success": False, "error": f"Directory not found: {output_directory}"}

        results: List[Dict[str, Any]] = []
        for f in sorted(out_dir.rglob(file_pattern))[:20]:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                try:
                    data = json.loads(content)
                except Exception:
                    data = content[:1000]
                results.append({"file": f.name, "content": data})
            except Exception:
                pass

        return {
            "success": True,
            "output_directory": str(out_dir),
            "file_pattern": file_pattern,
            "results_found": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"read_research_results_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_research_module_info_mcp() -> Dict[str, Any]:
    """Return metadata about the research module."""
    return {
        "success": True,
        "module": __package__,
        "capabilities": ["experiment_metadata", "cross_reference", "literature_mapping"],
        "supported_output_formats": ["json", "markdown"],
    }


# ── MCP Registration ────────────────────────────────────────────────────────

def register_tools(mcp_instance) -> None:
    """Register research tools with the MCP server."""

    mcp_instance.register_tool(
        "process_research",
        process_research_mcp,
        {"type": "object", "properties": {
            "target_directory": {"type": "string", "description": "Directory with GNN files"},
            "output_directory": {"type": "string", "description": "Research output directory"},
            "verbose":          {"type": "boolean", "default": False},
        }, "required": ["target_directory", "output_directory"]},
        "Run GNN research processing: generate experiment metadata and cross-references.",
        module=__package__, category="research",
    )

    mcp_instance.register_tool(
        "list_research_topics",
        list_research_topics_mcp,
        {},
        "Return Active Inference and GNN research topic taxonomy.",
        module=__package__, category="research",
    )

    mcp_instance.register_tool(
        "read_research_results",
        read_research_results_mcp,
        {"type": "object", "properties": {
            "output_directory": {"type": "string", "description": "Directory with research outputs"},
            "file_pattern":     {"type": "string", "default": "*.json",
                                 "description": "Glob pattern for output files"},
        }, "required": ["output_directory"]},
        "Read and return research output files from a previous research processing run.",
        module=__package__, category="research",
    )

    mcp_instance.register_tool(
        "get_research_module_info",
        get_research_module_info_mcp,
        {},
        "Return metadata about the research module capabilities.",
        module=__package__, category="research",
    )

    logger.info("research module MCP tools registered (4 tools).")
