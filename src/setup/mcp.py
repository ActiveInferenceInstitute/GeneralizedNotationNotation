"""
MCP (Model Context Protocol) integration for UV-based setup utilities.

This module exposes utility functions from the setup module through MCP,
with support for UV-based environment management and modern Python packaging.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import utilities from the setup module
from .utils import (
    ensure_directory, 
    find_gnn_files, 
    get_output_paths,
    check_uv_project_status,
    get_uv_environment_info,
    setup_uv_project_structure
)

# MCP Tools for UV-based Setup Utilities Module

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

def check_uv_project_status_mcp(project_directory: str) -> Dict[str, Any]:
    """
    Check the status of a UV project. Exposed via MCP.
    
    Args:
        project_directory: Path to the project directory.
        
    Returns:
        Dictionary with UV project status information.
    """
    try:
        project_root = Path(project_directory)
        status = check_uv_project_status(project_root)
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"Error in check_uv_project_status_mcp for {project_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def get_uv_environment_info_mcp() -> Dict[str, Any]:
    """
    Get information about the current UV environment. Exposed via MCP.
    
    Returns:
        Dictionary with UV environment information.
    """
    try:
        env_info = get_uv_environment_info()
        return {
            "success": True,
            "environment_info": env_info
        }
    except Exception as e:
        logger.error(f"Error in get_uv_environment_info_mcp: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def setup_uv_project_structure_mcp(project_directory: str) -> Dict[str, Any]:
    """
    Set up a new UV project structure. Exposed via MCP.
    
    Args:
        project_directory: Path to the project directory.
        
    Returns:
        Dictionary with setup status.
    """
    try:
        project_root = Path(project_directory)
        success = setup_uv_project_structure(project_root, logger)
        return {
            "success": success,
            "project_directory": str(project_root),
            "message": "UV project structure setup completed" if success else "UV project structure setup failed"
        }
    except Exception as e:
        logger.error(f"Error in setup_uv_project_structure_mcp for {project_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def install_uv_dependency_mcp(package_name: str, extras: Optional[str] = None) -> Dict[str, Any]:
    """
    Install a dependency using UV. Exposed via MCP.
    
    Args:
        package_name: Name of the package to install.
        extras: Optional extras to install (e.g., "dev", "ml-ai").
        
    Returns:
        Dictionary with installation status.
    """
    try:
        import subprocess
        
        cmd = ["uv", "add", package_name]
        if extras:
            cmd.extend(["--extras", extras])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            return {
                "success": True,
                "package": package_name,
                "extras": extras,
                "message": f"Successfully installed {package_name}"
            }
        else:
            return {
                "success": False,
                "package": package_name,
                "extras": extras,
                "error": result.stderr,
                "message": f"Failed to install {package_name}"
            }
            
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "package": package_name,
            "extras": extras,
            "error": "Installation timed out",
            "message": f"Installation of {package_name} timed out"
        }
    except Exception as e:
        logger.error(f"Error in install_uv_dependency_mcp for {package_name}: {e}", exc_info=True)
        return {
            "success": False,
            "package": package_name,
            "extras": extras,
            "error": str(e),
            "message": f"Error installing {package_name}"
        }

def sync_uv_dependencies_mcp(project_directory: str) -> Dict[str, Any]:
    """
    Sync dependencies using UV. Exposed via MCP.
    
    Args:
        project_directory: Path to the project directory.
        
    Returns:
        Dictionary with sync status.
    """
    try:
        import subprocess
        
        project_root = Path(project_directory)
        result = subprocess.run(
            ["uv", "sync"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode == 0:
            return {
                "success": True,
                "project_directory": str(project_root),
                "message": "Dependencies synced successfully"
            }
        else:
            return {
                "success": False,
                "project_directory": str(project_root),
                "error": result.stderr,
                "message": "Failed to sync dependencies"
            }
            
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "project_directory": str(project_directory),
            "error": "Sync timed out",
            "message": "Dependency sync timed out"
        }
    except Exception as e:
        logger.error(f"Error in sync_uv_dependencies_mcp for {project_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "project_directory": str(project_directory),
            "error": str(e),
            "message": "Error syncing dependencies"
        }

# MCP Registration Function
def register_tools(mcp_instance):
    """Register UV-based setup utility tools with the MCP."""
    
    mcp_instance.register_tool(
        "ensure_directory_exists",
        ensure_directory_exists_mcp,
        {
            "directory_path": {"type": "string", "description": "Path of the directory to create if it doesn't exist."}
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
    
    mcp_instance.register_tool(
        "check_uv_project_status",
        check_uv_project_status_mcp,
        {
            "project_directory": {"type": "string", "description": "Path to the project directory to check."}
        },
        "Checks the status of a UV project including pyproject.toml, uv.lock, and virtual environment."
    )
    
    mcp_instance.register_tool(
        "get_uv_environment_info",
        get_uv_environment_info_mcp,
        {},
        "Gets information about the current UV environment including paths and status."
    )
    
    mcp_instance.register_tool(
        "setup_uv_project_structure",
        setup_uv_project_structure_mcp,
        {
            "project_directory": {"type": "string", "description": "Path to the project directory to set up."}
        },
        "Sets up a new UV project structure with standard directories and configuration."
    )
    
    mcp_instance.register_tool(
        "install_uv_dependency",
        install_uv_dependency_mcp,
        {
            "package_name": {"type": "string", "description": "Name of the package to install."},
            "extras": {"type": "string", "description": "Optional extras to install (e.g., 'dev', 'ml-ai').", "optional": True}
        },
        "Installs a dependency using UV with optional extras support."
    )
    
    mcp_instance.register_tool(
        "sync_uv_dependencies",
        sync_uv_dependencies_mcp,
        {
            "project_directory": {"type": "string", "description": "Path to the project directory to sync dependencies for."}
        },
        "Syncs dependencies using UV from pyproject.toml and updates the lock file."
    )
    
    logger.info("UV-based setup module MCP tools registered.") 