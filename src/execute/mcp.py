"""
MCP (Model Context Protocol) integration for execute utilities.

This module exposes utility functions from the execute module through MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import utilities from the execute module
from . import process_execute

# MCP Tools for Execute Utilities Module

def process_execute_mcp(target_directory: str, output_directory: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Process execute for GNN files. Exposed via MCP.
    
    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save results
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and results.
    """
    try:
        success = process_execute(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"execute processing {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in process_execute_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance):
    """Register execute utility tools with the MCP."""
    
    mcp_instance.register_tool(
        "process_execute",
        process_execute_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to process."},
            "output_directory": {"type": "string", "description": "Directory to save execute results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        f"Process execute for GNN files in the specified directory."
    )
    
    logger.info("execute module MCP tools registered.")
