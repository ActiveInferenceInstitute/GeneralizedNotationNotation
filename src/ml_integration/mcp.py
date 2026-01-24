"""
ML Integration MCP Integration

This module exposes ML integration processing tools via MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import utilities from the ml_integration module
from . import process_ml_integration, check_ml_frameworks, FEATURES

# MCP Tools for ML Integration Module


def process_ml_integration_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Process ML integration for GNN files. Exposed via MCP.

    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save ML integration results
        verbose: Enable verbose output

    Returns:
        Dictionary with operation status and results.
    """
    try:
        success = process_ml_integration(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"ML integration processing {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in process_ml_integration_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def check_ml_frameworks_mcp() -> Dict[str, Any]:
    """
    Check available ML frameworks. Exposed via MCP.

    Returns:
        Dictionary with available ML frameworks and their status.
    """
    try:
        frameworks = check_ml_frameworks()
        return {
            "success": True,
            "frameworks": frameworks,
            "features": FEATURES
        }
    except Exception as e:
        logger.error(f"Error checking ML frameworks: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


# MCP Registration Function
def register_tools(mcp_instance):
    """Register ML integration utility tools with the MCP."""

    mcp_instance.register_tool(
        "process_ml_integration",
        process_ml_integration_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to process."},
            "output_directory": {"type": "string", "description": "Directory to save ML integration results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        "Process ML integration for GNN files including model training and inference setup."
    )

    mcp_instance.register_tool(
        "check_ml_frameworks",
        check_ml_frameworks_mcp,
        {},
        "Check available ML frameworks (PyTorch, TensorFlow, JAX, scikit-learn) and their versions."
    )

    logger.info("ML integration module MCP tools registered.")
