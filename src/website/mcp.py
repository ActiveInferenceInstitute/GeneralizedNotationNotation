"""
MCP (Model Context Protocol) integration for website utilities.

This module exposes website rendering functions through MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

from .renderer import process_website

# MCP Tools for $module Utilities Module

def process_website_mcp(target_directory: str, output_directory: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Process website generation for pipeline artifacts. Exposed via MCP.
    
    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save results
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and results.
    """
    try:
        success = process_website(Path(target_directory), Path(output_directory), verbose=verbose)
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"website processing {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in process_$module_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance):
    """Register website utility tools with the MCP."""
    
    mcp_instance.register_tool(
        "process_website",
        process_website_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to process."},
            "output_directory": {"type": "string", "description": "Directory to save website results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        "Generate website from pipeline artifacts in the specified directory."
    )
    
    logger.info("website module MCP tools registered.")
