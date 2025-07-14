"""
MCP (Model Context Protocol) integration for GNN Tests module.

This module exposes testing functionality through the Model Context Protocol.
"""

import os
import sys
import json
import unittest
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import logging

logger = logging.getLogger(__name__)

# Add parent directory to Python path to import modules
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from type_checker import GNNTypeChecker

# MCP Tools for Tests Module

def run_type_checker_on_file(file_path: str) -> Dict[str, Any]:
    """
    Run the GNN type checker on a file.
    
    Args:
        file_path: Path to the GNN file to check
        
    Returns:
        Dictionary containing type checker results
    """
    try:
        checker = GNNTypeChecker()
        is_valid, errors, warnings = checker.check_file(file_path)
        
        return {
            "file_path": file_path,
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings
        }
    except Exception as e:
        logger.error(f"Error in run_type_checker_on_file for {file_path}: {e}", exc_info=True)
        return {
            "file_path": file_path,
            "success": False,
            "error": str(e)
        }

def run_type_checker_on_directory(dir_path: str, report_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the GNN type checker on a directory of files.
    
    Args:
        dir_path: Path to directory containing GNN files
        report_file: Optional path to save the report
        
    Returns:
        Dictionary containing type checker results
    """
    try:
        checker = GNNTypeChecker()
        results = checker.check_directory(dir_path)
        
        # Generate report if requested
        report = None
        if report_file:
            report = checker.generate_report(results, output_file=report_file)
        
        # Format results for output
        formatted_results = {}
        for file_path, result in results.items():
            formatted_results[file_path] = {
                "is_valid": result["is_valid"],
                "error_count": len(result["errors"]),
                "warning_count": len(result["warnings"]),
                "errors": result["errors"],
                "warnings": result["warnings"]
            }
        
        return {
            "directory_path": dir_path,
            "report_file": report_file if report_file else None,
            "results": formatted_results,
            "summary": {
                "total_files": len(results),
                "valid_count": sum(1 for r in results.values() if r["is_valid"]),
                "invalid_count": sum(1 for r in results.values() if not r["is_valid"]),
                "total_errors": sum(len(r["errors"]) for r in results.values()),
                "total_warnings": sum(len(r["warnings"]) for r in results.values())
            }
        }
    except Exception as e:
        logger.error(f"Error in run_type_checker_on_directory for {dir_path}: {e}", exc_info=True)
        return {
            "directory_path": dir_path,
            "success": False,
            "error": str(e)
        }

def run_unit_tests() -> Dict[str, Any]:
    """
    Run the GNN unit tests.
    
    Returns:
        Dictionary containing test results
    """
    try:
        # Create a test loader and runner
        loader = unittest.TestLoader()
        suite = loader.discover(os.path.dirname(__file__), pattern="test_*.py")
        
        # Use a temporary file to capture test output
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            # Create a text test runner that writes to the temp file
            runner = unittest.TextTestRunner(stream=temp_file, verbosity=2)
            result = runner.run(suite)
            
            # Rewind and read the output
            temp_file.seek(0)
            test_output = temp_file.read()
        
        # Clean up the temp file
        os.unlink(temp_file.name)
        
        return {
            "success": True,
            "ran": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "was_successful": result.wasSuccessful(),
            "failures_detail": [{"test": t[0].id(), "message": t[1]} for t in result.failures],
            "errors_detail": [{"test": t[0].id(), "message": t[1]} for t in result.errors],
            "output": test_output
        }
    except Exception as e:
        logger.error(f"Error in run_unit_tests: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# Resource retrievers

def get_test_report(uri: str) -> Dict[str, Any]:
    """
    Retrieve a test report by URI.
    
    Args:
        uri: URI of the test report. Format: test-report://{report_file}
        
    Returns:
        Dictionary containing the test report
    """
    # Extract file path from URI
    if not uri.startswith("test-report://"):
        error_msg = f"Invalid URI format: {uri}"
        logger.error(f"get_test_report: {error_msg}")
        raise ValueError(error_msg)
    
    file_path_str = uri[14:]  # Remove 'test-report://' prefix
    file_path = Path(file_path_str)
    
    if not file_path.exists() or not file_path.is_file():
        error_msg = f"Report file does not exist: {file_path}"
        logger.error(f"get_test_report: {error_msg}")
        raise ValueError(error_msg)
    
    # Read the report content
    report_content = file_path.read_text()
    
    return {
        "file_path": str(file_path),
        "content": report_content
    }

# MCP Registration Function

def register_tools(mcp):
    """Register test tools with the MCP."""
    
    # Register test tools
    mcp.register_tool(
        "run_type_checker",
        run_type_checker_on_file,
        {
            "file_path": {"type": "string", "description": "Path to the GNN file to check"}
        },
        "Run the GNN type checker on a specific file (via test module)."
    )
    
    mcp.register_tool(
        "run_type_checker_on_directory",
        run_type_checker_on_directory,
        {
            "dir_path": {"type": "string", "description": "Path to directory containing GNN files"},
            "report_file": {"type": "string", "description": "Optional path to save the report"}
        },
        "Run the GNN type checker on all GNN files in a directory (via test module)."
    )
    
    mcp.register_tool(
        "run_gnn_unit_tests",
        run_unit_tests,
        {},
        "Run the GNN unit tests and return results."
    )
    
    # Register test resources
    mcp.register_resource(
        "test-report://{report_file}",
        get_test_report,
        "Retrieve a test report by file path"
    )
    
    logger.info("Tests module MCP tools and resources registered.") 