"""
MCP (Model Context Protocol) integration for GNN Type Checker module.

This module exposes GNN type checking and resource estimation 
functionality through the Model Context Protocol.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Add parent directory to Python path if running this module directly for testing
# and to ensure imports from sibling directories (like visualization for parser) work.
# This might be needed if GNNResourceEstimator or GNNTypeChecker try to import
# GNNParser from the visualization module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from .checker import GNNTypeChecker
except ImportError:
    # Fallback if checker.py is not found or GNNTypeChecker is not directly in it
    # This is a placeholder; ideally, the import should be direct.
    logger.warning("GNNTypeChecker not found in gnn_type_checker.checker. Using a mock.")
    class GNNTypeChecker:
        def check_file(self, file_path: str) -> tuple[bool, list, list]:
            logger.debug(f"Mock GNNTypeChecker.check_file called for {file_path}")
            return True, [], []
        def check_directory(self, dir_path: str, recursive: bool = False) -> dict:
            logger.debug(f"Mock GNNTypeChecker.check_directory called for {dir_path}")
            return {dir_path: {"is_valid": True, "errors": [], "warnings": []}}
        def generate_report(self, results: dict, output_dir_base: Path, report_md_filename: str = "type_check_report.md") -> str:
            logger.debug(f"Mock GNNTypeChecker.generate_report called.")
            return "Mock report content"

try:
    from .resource_estimator import GNNResourceEstimator
except ImportError:
    logger.warning("GNNResourceEstimator not found. Resource estimation tools will be unavailable.")
    class GNNResourceEstimator:
        def estimate_from_file(self, file_path: str) -> dict:
            return {"error": "GNNResourceEstimator not available"}
        def estimate_from_directory(self, dir_path: str, recursive: bool = False) -> dict:
            return {"error": "GNNResourceEstimator not available"}

# MCP Tools for GNN Type Checker Module

# --- Type Checker Tools ---
def type_check_gnn_file_mcp(file_path: str) -> Dict[str, Any]:
    """
    Run the GNN type checker on a single GNN file. Exposed via MCP.
    
    Args:
        file_path: Path to the GNN file to check.
        
    Returns:
        Dictionary containing type checker results (is_valid, errors, warnings).
    """
    try:
        checker = GNNTypeChecker()
        is_valid, errors, warnings = checker.check_file(file_path)
        return {
            "success": True,
            "file_path": file_path,
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings
        }
    except Exception as e:
        logger.error(f"Error in type_check_gnn_file_mcp for {file_path}: {e}", exc_info=True)
        return {
            "success": False,
            "file_path": file_path,
            "error": str(e)
        }

def type_check_gnn_directory_mcp(dir_path: str, recursive: bool = False, output_dir_base: Optional[str] = None, report_md_filename: Optional[str] = "type_check_report.md") -> Dict[str, Any]:
    """
    Run the GNN type checker on all GNN files in a directory. Exposed via MCP.
    
    Args:
        dir_path: Path to the directory containing GNN files.
        recursive: Whether to search recursively (default: False).
        output_dir_base: Optional base directory path to save the generated report.
        report_md_filename: Optional filename for the markdown report. Defaults to type_check_report.md.

    Returns:
        Dictionary containing aggregated type checker results and report path if generated.
    """
    try:
        checker = GNNTypeChecker()
        results = checker.check_directory(dir_path, recursive=recursive)

        report_generated_path = None
        if output_dir_base and report_md_filename:
            output_path = Path(output_dir_base)
            # The checker's generate_report now handles creating subdirs and all its files (HTML, JSON)
            # and returns the main markdown report content.
            # It also prints where files are saved.
            checker.generate_report(results, output_path, report_md_filename=report_md_filename)
            report_generated_path = str(output_path / report_md_filename)
            
        return {
            "success": True,
            "directory_path": dir_path,
            "results_summary": {
                "total_files": len(results),
                "valid_count": sum(1 for r in results.values() if r.get("is_valid", False)), # Ensure key exists
                "invalid_count": sum(1 for r in results.values() if not r.get("is_valid", True)), # Ensure key exists
            },
            "results_detail": results,
            "report_generated_at": report_generated_path
        }
    except Exception as e:
        logger.error(f"Error in type_check_gnn_directory_mcp for {dir_path}: {e}", exc_info=True)
        return {
            "success": False,
            "directory_path": dir_path,
            "error": str(e)
        }

# --- Resource Estimator Tools ---
def estimate_resources_for_gnn_file_mcp(file_path: str) -> Dict[str, Any]:
    """
    Estimate computational resources for a single GNN file. Exposed via MCP.
    
    Args:
        file_path: Path to the GNN file.
        
    Returns:
        Dictionary containing resource estimates.
    """
    try:
        estimator = GNNResourceEstimator()
        estimates = estimator.estimate_from_file(file_path)
        return {
            "success": True,
            "file_path": file_path,
            "estimates": estimates
        }
    except Exception as e:
        logger.error(f"Error in estimate_resources_for_gnn_file_mcp for {file_path}: {e}", exc_info=True)
        return {
            "success": False,
            "file_path": file_path,
            "error": str(e)
        }

def estimate_resources_for_gnn_directory_mcp(dir_path: str, recursive: bool = False) -> Dict[str, Any]:
    """
    Estimate resources for all GNN files in a directory. Exposed via MCP.
    
    Args:
        dir_path: Path to the directory containing GNN files.
        recursive: Whether to search recursively (default: False).
        
    Returns:
        Dictionary mapping file paths to resource estimates.
    """
    try:
        estimator = GNNResourceEstimator()
        all_estimates = estimator.estimate_from_directory(dir_path, recursive=recursive)
        return {
            "success": True,
            "directory_path": dir_path,
            "all_estimates": all_estimates
        }
    except Exception as e:
        logger.error(f"Error in estimate_resources_for_gnn_directory_mcp for {dir_path}: {e}", exc_info=True)
        return {
            "success": False,
            "directory_path": dir_path,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance): # Changed 'mcp' to 'mcp_instance' for clarity
    """Register GNN type checker and resource estimator tools with the MCP."""
    
    mcp_instance.register_tool(
        "type_check_gnn_file",
        type_check_gnn_file_mcp,
        {
            "file_path": {"type": "string", "description": "Path to the GNN file to be type-checked."}
        },
        "Runs the GNN type checker on a specified GNN model file."
    )
    
    mcp_instance.register_tool(
        "type_check_gnn_directory",
        type_check_gnn_directory_mcp,
        {
            "dir_path": {"type": "string", "description": "Path to the directory containing GNN files to be type-checked."},
            "recursive": {"type": "boolean", "description": "Search directory recursively. Defaults to False.", "optional": True},
            "output_dir_base": {"type": "string", "description": "Optional base directory to save the report and other artifacts (HTML, JSON).", "optional": True},
            "report_md_filename": {"type": "string", "description": "Optional filename for the markdown report (e.g., 'my_report.md'). Defaults to 'type_check_report.md'.", "optional": True}
        },
        "Runs the GNN type checker on all GNN files in a specified directory. If output_dir_base is provided, reports are generated."
    )
    
    if 'GNNResourceEstimator' in globals() and hasattr(GNNResourceEstimator, 'estimate_from_file'): # More robust check
        # Check if the class is the actual one, not the mock, by checking a method
        try:
            # Attempt to instantiate and check a method to ensure it's not the mock
            # This is a bit heuristic; a more robust way would be to avoid mocks in production if possible
            # or have a clearer way to distinguish them.
            estimator_instance = GNNResourceEstimator()
            # Check if estimate_from_file method exists and its docstring does not contain "mock"
            if hasattr(estimator_instance, 'estimate_from_file') and \
               (estimator_instance.estimate_from_file.__doc__ is None or \
                "mock" not in estimator_instance.estimate_from_file.__doc__.lower()):
                 is_real_estimator = True
            else:
                 is_real_estimator = False
        except: # pylint: disable=bare-except
            is_real_estimator = False # Fallback if instantiation fails

        if is_real_estimator:
            mcp_instance.register_tool(
                "estimate_resources_for_gnn_file",
                estimate_resources_for_gnn_file_mcp,
                {
                    "file_path": {"type": "string", "description": "Path to the GNN file for resource estimation."}
                },
                "Estimates computational resources (memory, inference, storage) for a GNN model file."
            )
            
            mcp_instance.register_tool(
                "estimate_resources_for_gnn_directory",
                estimate_resources_for_gnn_directory_mcp,
                {
                    "dir_path": {"type": "string", "description": "Path to the directory for GNN resource estimation."},
                    "recursive": {"type": "boolean", "description": "Search directory recursively. Defaults to False.", "optional": True}
                },
                "Estimates computational resources for all GNN files in a specified directory."
            )
        else:
            logger.warning("GNNResourceEstimator appears to be a mock or unavailable. Resource estimation MCP tools will not be registered.")
            
    logger.info("GNN Type Checker module MCP tools registered.")
    # No specific resources to register for type_checker beyond the files it might create (handled by report_file param) 