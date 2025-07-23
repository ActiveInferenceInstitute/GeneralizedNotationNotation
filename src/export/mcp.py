"""
MCP (Model Context Protocol) integration for export utilities.

This module exposes utility functions from the export module through MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import utilities from the export module
from . import generate_exports, export_single_gnn_file

# MCP Tools for Export Utilities Module

def generate_exports_mcp(target_directory: str, output_directory: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Generate exports in multiple formats for GNN files. Exposed via MCP.
    
    Args:
        target_directory: Directory containing GNN files to export
        output_directory: Directory to save exports
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and export results.
    """
    try:
        success = generate_exports(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Export generation {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in generate_exports_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def export_single_gnn_file_mcp(gnn_file_path: str, output_directory: str) -> Dict[str, Any]:
    """
    Export a single GNN file to multiple formats. Exposed via MCP.
    
    Args:
        gnn_file_path: Path to the GNN file to export
        output_directory: Directory to save exports
        
    Returns:
        Dictionary with export results.
    """
    try:
        result = export_single_gnn_file(
            gnn_file=Path(gnn_file_path),
            exports_dir=Path(output_directory)
        )
        return {
            "success": result["success"],
            "file": gnn_file_path,
            "exports": result["exports"],
            "message": f"File export {'completed successfully' if result['success'] else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in export_single_gnn_file_mcp for {gnn_file_path}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance):
    """Register export utility tools with the MCP."""
    
    mcp_instance.register_tool(
        "generate_exports",
        generate_exports_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to export."},
            "output_directory": {"type": "string", "description": "Directory to save exports."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        "Generate exports in multiple formats for GNN files."
    )
    
    mcp_instance.register_tool(
        "export_single_gnn_file",
        export_single_gnn_file_mcp,
        {
            "gnn_file_path": {"type": "string", "description": "Path to the GNN file to export."},
            "output_directory": {"type": "string", "description": "Directory to save exports."}
        },
        "Export a single GNN file to multiple formats."
    )
    
    logger.info("Export module MCP tools registered.")
