"""
MCP (Model Context Protocol) integration for tests utilities.

This module exposes utility functions from the tests module through MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import utilities from the tests module
from . import run_all_tests, run_unit_tests, run_integration_tests

# MCP Tools for Tests Utilities Module

def run_all_tests_mcp(target_directory: str, output_directory: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Run comprehensive test suite for the GNN pipeline. Exposed via MCP.
    
    Args:
        target_directory: Directory containing GNN files to test
        output_directory: Directory to save test results
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and test results.
    """
    try:
        success = run_all_tests(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Test suite {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in run_all_tests_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def run_unit_tests_mcp(output_directory: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Run unit test suite for the GNN pipeline. Exposed via MCP.
    
    Args:
        output_directory: Directory to save test results
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and test results.
    """
    try:
        success = run_unit_tests(
            test_results_dir=Path(output_directory),
            verbose=verbose
        )
        return {
            "success": success,
            "output_directory": output_directory,
            "message": f"Unit tests {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in run_unit_tests_mcp for {output_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def run_integration_tests_mcp(target_directory: str, output_directory: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Run integration test suite for the GNN pipeline. Exposed via MCP.
    
    Args:
        target_directory: Directory containing GNN files to test
        output_directory: Directory to save test results
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and test results.
    """
    try:
        success = run_integration_tests(
            target_dir=Path(target_directory),
            test_results_dir=Path(output_directory),
            verbose=verbose
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Integration tests {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in run_integration_tests_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance):
    """Register tests utility tools with the MCP."""
    
    mcp_instance.register_tool(
        "run_all_tests",
        run_all_tests_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to test."},
            "output_directory": {"type": "string", "description": "Directory to save test results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        "Run comprehensive test suite for the GNN pipeline."
    )
    
    mcp_instance.register_tool(
        "run_unit_tests",
        run_unit_tests_mcp,
        {
            "output_directory": {"type": "string", "description": "Directory to save test results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        "Run unit test suite for the GNN pipeline."
    )
    
    mcp_instance.register_tool(
        "run_integration_tests",
        run_integration_tests_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to test."},
            "output_directory": {"type": "string", "description": "Directory to save test results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        "Run integration test suite for the GNN pipeline."
    )
    
    logger.info("Tests module MCP tools registered.")
