"""
MCP (Model Context Protocol) integration for render utilities.

This module exposes utility functions from the render module through MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import utilities from the render module
from . import process_render

# MCP Tools for Render Utilities Module

def process_render_mcp(target_directory: str, output_directory: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Process render for GNN files. Exposed via MCP.
    
    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save results
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and results.
    """
    try:
        success = process_render(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"render processing {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in process_render_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance):
    """Register render utility tools with the MCP."""
    
    mcp_instance.register_tool(
        "process_render",
        process_render_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to process."},
            "output_directory": {"type": "string", "description": "Directory to save render results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        f"Process render for GNN files in the specified directory."
    )
    
    logger.info("render module MCP tools registered.")
