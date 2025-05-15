"""
MCP (Model Context Protocol) integration for setup utilities.

This module exposes utility functions from the setup module through MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import utilities from the setup module
# Assuming the file is src/setup/mcp.py, utils.py is in the same directory
from .utils import ensure_directory, find_gnn_files, get_output_paths

# MCP Tools for Setup Utilities Module

def ensure_directory_exists_mcp(directory_path: str) -> Dict[str, Any]:
    """
    Ensure a directory exists, creating it if necessary. Exposed via MCP.
    
    Args:
        directory_path: Directory path to ensure existence of.
        
    Returns:
        Dictionary with operation status and path.
    """
    try:
        path_obj = ensure_directory(directory_path)
        return {
            "success": True,
            "path": str(path_obj),
            "created": not Path(directory_path).exists() # Check if it was created now or existed before
        }
    except Exception as e:
        logger.error(f"Error in ensure_directory_exists_mcp for {directory_path}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def find_project_gnn_files_mcp(search_directory: str, recursive: bool = False) -> Dict[str, Any]:
    """
    Find all GNN (.md) files in a directory. Exposed via MCP.
    
    Args:
        search_directory: Directory to search.
        recursive: Whether to search recursively (default: False).
        
    Returns:
        Dictionary with list of found file paths or an error.
    """
    try:
        files = find_gnn_files(search_directory, recursive)
        return {
            "success": True,
            "files": [str(f) for f in files],
            "count": len(files)
        }
    except Exception as e:
        logger.error(f"Error in find_project_gnn_files_mcp for {search_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def get_standard_output_paths_mcp(base_output_directory: str) -> Dict[str, Any]:
    """
    Get standard output paths for the pipeline. Exposed via MCP.
    
    Args:
        base_output_directory: Base output directory.
        
    Returns:
        Dictionary of named output paths or an error.
    """
    try:
        paths = get_output_paths(base_output_directory)
        return {
            "success": True,
            "paths": {name: str(p) for name, p in paths.items()}
        }
    except Exception as e:
        logger.error(f"Error in get_standard_output_paths_mcp for {base_output_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance): # Changed 'mcp' to 'mcp_instance' for clarity
    """Register setup utility tools with the MCP."""
    
    mcp_instance.register_tool(
        "ensure_directory_exists",
        ensure_directory_exists_mcp,
        {
            "directory_path": {"type": "string", "description": "Path of the directory to create if it doesn\'t exist."}
        },
        "Ensures a directory exists, creating it if necessary. Returns the absolute path."
    )
    
    mcp_instance.register_tool(
        "find_project_gnn_files",
        find_project_gnn_files_mcp,
        {
            "search_directory": {"type": "string", "description": "The directory to search for GNN (.md) files."},
            "recursive": {"type": "boolean", "description": "Set to true to search recursively. Defaults to false.", "optional": True}
        },
        "Finds all GNN (.md) files in a specified directory within the project."
    )
    
    mcp_instance.register_tool(
        "get_standard_output_paths",
        get_standard_output_paths_mcp,
        {
            "base_output_directory": {"type": "string", "description": "The base directory where output subdirectories will be managed."}
        },
        "Gets a dictionary of standard output directory paths (e.g., for type_check, visualization), creating them if needed."
    )
    
    logger.info("Setup module MCP tools registered.") 