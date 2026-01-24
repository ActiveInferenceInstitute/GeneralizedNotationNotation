"""
Advanced Visualization MCP Integration

This module exposes advanced visualization processing tools via MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
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
    Check available visualization capabilities. Exposed via MCP.

    Returns:
        Dictionary with available visualization features.
    """
    from . import FEATURES
    return {
        "success": True,
        "features": FEATURES,
        "d2_available": D2_AVAILABLE
    }


# MCP Registration Function
def register_tools(mcp_instance):
    """Register advanced visualization utility tools with the MCP."""

    mcp_instance.register_tool(
        "process_advanced_visualization",
        process_advanced_visualization_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to process."},
            "output_directory": {"type": "string", "description": "Directory to save visualization results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True},
            "generate_d2": {"type": "boolean", "description": "Generate D2 diagrams if available. Defaults to true.", "optional": True}
        },
        "Process advanced visualization for GNN files including D2 diagrams, dashboards, and network visualizations."
    )

    mcp_instance.register_tool(
        "check_visualization_capabilities",
        check_visualization_capabilities_mcp,
        {},
        "Check available visualization capabilities and features."
    )

    logger.info("Advanced visualization module MCP tools registered.")
