"""
MCP (Model Context Protocol) integration for ontology utilities.

This module exposes utility functions from the ontology module through MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

from . import process_ontology

# MCP Tools for $module Utilities Module

def process_ontology_mcp(target_directory: str, output_directory: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Process ontology for GNN files. Exposed via MCP.
    
    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save results
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and results.
    """
    try:
        success = process_ontology(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Ontology processing {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in process_ontology_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance):
    """Register ontology utility tools with the MCP."""
    
    mcp_instance.register_tool(
        "process_ontology",
        process_ontology_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to process."},
            "output_directory": {"type": "string", "description": "Directory to save ontology results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        f"Process ontology for GNN files in the specified directory."
    )
    
    logger.info("Ontology module MCP tools registered.")
