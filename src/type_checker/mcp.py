"""
MCP (Model Context Protocol) integration for type_checker utilities.

This module exposes utility functions from the type_checker module through MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import utilities from the type_checker module via class API exposed in __init__
from . import GNNTypeChecker

# MCP Tools for Type Checker Utilities Module

def validate_gnn_files_mcp(target_directory: str, output_directory: str, strict: bool = False, estimate_resources: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """
    Validate GNN files for syntax and type correctness. Exposed via MCP.
    
    Args:
        target_directory: Directory containing GNN files to validate
        output_directory: Directory to save validation results
        strict: Enable strict type checking mode
        estimate_resources: Estimate computational resources
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and validation results.
    """
    try:
        checker = GNNTypeChecker()
        success = checker.validate_gnn_files(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "strict_mode": strict,
            "resource_estimation": estimate_resources,
            "message": f"GNN validation {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in validate_gnn_files_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def validate_single_gnn_file_mcp(gnn_file_path: str, strict: bool = False, estimate_resources: bool = False) -> Dict[str, Any]:
    """
    Validate a single GNN file. Exposed via MCP.
    
    Args:
        gnn_file_path: Path to the GNN file to validate
        strict: Enable strict validation mode
        estimate_resources: Estimate computational resources
        
    Returns:
        Dictionary with validation results.
    """
    try:
        checker = GNNTypeChecker()
        result = checker.validate_single_gnn_file(
            file_path=Path(gnn_file_path),
        )
        return {
            "success": result["valid"],
            "file": gnn_file_path,
            "validation_result": result,
            "message": f"File validation {'passed' if result['valid'] else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in validate_single_gnn_file_mcp for {gnn_file_path}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance):
    """Register type_checker utility tools with the MCP."""
    
    mcp_instance.register_tool(
        "validate_gnn_files",
        validate_gnn_files_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to validate."},
            "output_directory": {"type": "string", "description": "Directory to save validation results."},
            "strict": {"type": "boolean", "description": "Enable strict type checking mode. Defaults to false.", "optional": True},
            "estimate_resources": {"type": "boolean", "description": "Estimate computational resources. Defaults to false.", "optional": True},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        "Validate GNN files for syntax and type correctness."
    )
    
    mcp_instance.register_tool(
        "validate_single_gnn_file",
        validate_single_gnn_file_mcp,
        {
            "gnn_file_path": {"type": "string", "description": "Path to the GNN file to validate."},
            "strict": {"type": "boolean", "description": "Enable strict validation mode. Defaults to false.", "optional": True},
            "estimate_resources": {"type": "boolean", "description": "Estimate computational resources. Defaults to false.", "optional": True}
        },
        "Validate a single GNN file for syntax and type correctness."
    )
    
    logger.info("Type checker module MCP tools registered.")
