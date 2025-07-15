#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 7: MCP (Model Context Protocol)

This script handles MCP operations and tool registrations, providing comprehensive
status reporting and validation of the MCP implementation.

Usage:
    python 7_mcp.py [options]
    (Typically called by main.py)
"""

import sys
import logging
from pathlib import Path
import json
import time
from typing import Dict, Any, List, Optional

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    UTILS_AVAILABLE
)

from pipeline import (
    STEP_METADATA,
    get_output_dir_for_script
)

from mcp.processor import process_mcp_operations

# Initialize logger for this step
logger = setup_step_logging("7_mcp", verbose=False)

# Define all expected module directories that should have MCP integration
EXPECTED_MCP_MODULE_DIRS = [
    "export",
    "gnn", 
    "type_checker",
    "ontology",
    "setup",
    "tests",
    "visualization",
    "llm",
    "render",
    "execute",
    "website",
    "sapf",
    "pipeline",
    "utils",
    "src"
]

def check_module_mcp_integration() -> Dict[str, Any]:
    """Check MCP integration status for all modules."""
    src_dir = Path(__file__).parent
    module_status = {}
    
    for module_dir in EXPECTED_MCP_MODULE_DIRS:
        module_path = src_dir / module_dir
        mcp_file_path = module_path / "mcp.py"
        
        status = {
            "exists": module_path.exists(),
            "has_mcp_file": mcp_file_path.exists(),
            "has_init": (module_path / "__init__.py").exists(),
            "path": str(module_path)
        }
        
        # Check if module has register_tools function
        if mcp_file_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(f"mcp_{module_dir}", mcp_file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                status["has_register_tools"] = hasattr(module, 'register_tools')
                status["register_tools_type"] = type(getattr(module, 'register_tools', None)).__name__
                
            except Exception as e:
                status["import_error"] = str(e)
                status["has_register_tools"] = False
        
        module_status[module_dir] = status
    
    return module_status

def test_mcp_functionality() -> Dict[str, Any]:
    """Test basic MCP functionality."""
    # This function is no longer needed as process_mcp_operations handles testing
    # Keeping it for now in case it's called directly, but it will return an error
    return {"error": "MCP functionality testing is now handled by mcp.processor.process_mcp_operations"}

def process_mcp_operations_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized MCP operations processing function.
    
    Args:
        target_dir: Directory containing files to process with MCP
        output_dir: Output directory for MCP results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # --- Robust Path Handling ---
        # Determine project root (parent of src/)
        project_root = Path(__file__).resolve().parent.parent
        cwd = Path.cwd()
        
        # Defensive conversion and resolution of paths
        if not isinstance(target_dir, Path):
            target_dir = Path(target_dir)
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        if not target_dir.is_absolute():
            target_dir = (project_root / target_dir).resolve()
        if not output_dir.is_absolute():
            output_dir = (project_root / output_dir).resolve()
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log all argument values and their types
        logger.info("--- MCP Step Argument Debugging ---")
        logger.info(f"Working directory: {cwd}")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Resolved target_dir: {target_dir} (Type: {type(target_dir).__name__})")
        logger.info(f"Resolved output_dir: {output_dir} (Type: {type(output_dir).__name__})")
        logger.info(f"Verbose: {verbose}")
        logger.info("-------------------------------")
        
        # Call the existing process_mcp_operations function
        success = process_mcp_operations(
            target_dir=target_dir,
            output_dir=output_dir,
            logger=logger,
            recursive=recursive
        )
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"MCP operations failed: {e}")
        return False

def main(parsed_args):
    """Main function for MCP operations."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("7_mcp.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Model Context Protocol operations')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run MCP operations
    success = process_mcp_operations_standardized(
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        logger=logger,
        recursive=getattr(parsed_args, 'recursive', False),
        verbose=getattr(parsed_args, 'verbose', False)
    )
    
    if success:
        log_step_success(logger, "MCP operations completed successfully")
        return 0
    else:
        log_step_error(logger, "MCP operations failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("7_mcp")
    else:
        # Fallback argument parsing
        import argparse
        parser = argparse.ArgumentParser(description="Model Context Protocol operations")
        parser.add_argument("--target-dir", type=Path, required=True,
                          help="Target directory containing GNN files")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 