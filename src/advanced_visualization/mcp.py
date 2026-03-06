"""
Advanced Visualization MCP Integration

This module exposes advanced visualization processing tools via MCP.
"""

from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Import utilities from the advanced_visualization module
from . import process_advanced_viz_standardized_impl, D2_AVAILABLE

# MCP Tools for Advanced Visualization Module


def process_advanced_visualization_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
    generate_d2: bool = True
) -> Dict[str, Any]:
    """
    Process advanced visualization for GNN files. Exposed via MCP.

    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save visualization results
        verbose: Enable verbose output
        generate_d2: Whether to generate D2 diagrams (if available)

    Returns:
        Dictionary with operation status and results.
    """
    try:
        success = process_advanced_viz_standardized_impl(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "d2_available": D2_AVAILABLE,
            "message": f"Advanced visualization processing {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in process_advanced_visualization_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def check_visualization_capabilities_mcp() -> Dict[str, Any]:
    """
    Check available advanced visualization capabilities.

    Probes D2 availability, dashboard generation support, and network
    visualization backends. Returns full feature flag dictionary.

    Returns:
        Dictionary with D2 availability, feature flags, and backend status.
    """
    from . import FEATURES
    return {
        "success":      True,
        "d2_available": D2_AVAILABLE,
        "features":     FEATURES,
    }


def list_d2_visualization_types_mcp() -> Dict[str, Any]:
    """
    Return all D2 diagram types supported for GNN model visualization.

    Includes: state-space diagrams, factor graphs, generative model diagrams,
    belief propagation graphs, and policy trees.

    Returns:
        Dictionary with visualization type names and descriptions.
    """
    types = {
        "state_space":      "Visualizes the hidden state space (X) and observation space (Y)",
        "factor_graph":     "Factor graph representation of the generative model",
        "generative_model": "Full A/B/C/D matrix generative model diagram",
        "belief_prop":      "Belief propagation / message-passing graph",
        "policy_tree":      "Policy tree (action sequences) for planning agents",
        "network":          "General network/graph diagram of GNN model connections",
        "dashboard":        "Multi-panel dashboard with all visualization types combined",
    }
    return {
        "success":           True,
        "types":             types,
        "count":             len(types),
        "d2_required":       ["state_space", "factor_graph", "generative_model", "belief_prop", "policy_tree"],
        "no_d2_required":    ["network", "dashboard"],
        "d2_available":      D2_AVAILABLE,
    }


def get_advanced_visualization_module_info_mcp() -> Dict[str, Any]:
    """
    Return version, feature flags, and API surface of the advanced visualization module.

    Returns:
        Dictionary with module metadata, D2 status, and tool inventory.
    """
    try:
        import importlib
        mod = importlib.import_module(__package__)
        version  = getattr(mod, "__version__", "unknown")
        features = getattr(mod, "FEATURES", {})
        return {
            "success":      True,
            "module":       __package__,
            "version":      version,
            "d2_available": D2_AVAILABLE,
            "features":     features,
            "tools": [
                "process_advanced_visualization",
                "check_visualization_capabilities",
                "list_d2_visualization_types",
                "get_advanced_visualization_module_info",
            ],
        }
    except Exception as e:
        logger.error(f"get_advanced_visualization_module_info_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}



# MCP Registration Function
def register_tools(mcp_instance):
    """Register advanced visualization domain tools with the MCP."""

    mcp_instance.register_tool(
        "process_advanced_visualization",
        process_advanced_visualization_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {"type": "string", "description": "Directory containing GNN files"},
                "output_directory": {"type": "string", "description": "Directory to save visualization results"},
                "verbose":          {"type": "boolean", "default": False},
                "generate_d2":      {"type": "boolean", "default": True, "description": "Generate D2 diagrams if available"},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Process advanced visualization for GNN files: D2 diagrams, dashboards, network visualizations.",
        module=__package__, category="advanced_visualization",
    )

    mcp_instance.register_tool(
        "check_visualization_capabilities",
        check_visualization_capabilities_mcp,
        {},
        "Check available advanced visualization capabilities (D2, dashboard, network backends).",
        module=__package__, category="advanced_visualization",
    )

    mcp_instance.register_tool(
        "list_d2_visualization_types",
        list_d2_visualization_types_mcp,
        {},
        "Return all D2 diagram types supported for GNN model visualization.",
        module=__package__, category="advanced_visualization",
    )

    mcp_instance.register_tool(
        "get_advanced_visualization_module_info",
        get_advanced_visualization_module_info_mcp,
        {},
        "Return version, feature flags, and tool inventory of the advanced visualization module.",
        module=__package__, category="advanced_visualization",
    )

    logger.info("advanced_visualization module MCP tools registered (4 domain tools).")

