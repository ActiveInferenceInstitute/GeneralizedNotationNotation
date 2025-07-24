"""
MCP (Model Context Protocol) integration for visualization utilities.

This module exposes utility functions from the visualization module through MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import utilities from the visualization module
from . import process_visualization

# MCP Tools for Visualization Utilities Module

def process_visualization_mcp(target_directory: str, output_directory: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Process visualization for GNN files. Exposed via MCP.
    
    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save results
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and results.
    """
    try:
        success = process_visualization(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Visualization processing {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in process_visualization_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance):
    """Register visualization utility tools with the MCP."""
    
    mcp_instance.register_tool(
        "process_visualization",
        process_visualization_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to process."},
            "output_directory": {"type": "string", "description": "Directory to save visualization results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        "Process visualization for GNN files in the specified directory."
    )
    
    logger.info("Visualization module MCP tools registered.")
