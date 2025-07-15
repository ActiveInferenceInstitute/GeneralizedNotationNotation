#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 1: GNN

This script discovers and parses GNN files.

Usage:
    python 1_gnn.py [options]
    (Typically called by main.py)
"""

import sys
import logging
from pathlib import Path
from typing import TypedDict, List, Dict, Any
import argparse

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

from gnn.processors import process_gnn_folder
from utils.path_utils import get_relative_path_if_possible

# Initialize logger for this step
logger = setup_step_logging("1_gnn", verbose=False)

# Define TypedDict for file_summary structure
class FileSummaryType(TypedDict):
    file_name: str
    path: str
    model_name: str
    sections_found: List[str]
    model_parameters: Dict[str, Any]
    errors: List[str]

def process_gnn_files(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized GNN file processing function.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Determine project root for path relativization
        project_root = None
        try:
            current_script_path = Path(__file__).resolve()
            project_root = current_script_path.parent.parent
        except Exception:
            pass
        
        # Update logger verbosity if needed
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Call the existing process_gnn_folder function
        success = process_gnn_folder(
            target_dir=target_dir,
            output_dir=output_dir,
            project_root=project_root,
            recursive=recursive,
            verbose=verbose,
            logger=logger
        )
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"GNN file processing failed: {e}")
        return False

def main(parsed_args):
    """Main function for GNN file discovery and processing."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("1_gnn.py", {})
    log_step_start(logger, f"{step_info.get('description', 'GNN file discovery and basic parsing')}")
    
    logger.info(f"GNN Step 1: Target directory: {parsed_args.target_dir}")
    logger.info(f"GNN Step 1: Output directory: {parsed_args.output_dir}")
    logger.info(f"GNN Step 1: Recursive: {parsed_args.recursive}")
    logger.info(f"GNN Step 1: Verbose: {parsed_args.verbose}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Process GNN files
    success = process_gnn_files(
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        logger=logger,
        recursive=parsed_args.recursive,
        verbose=parsed_args.verbose
    )
    
    if success:
        logger.info("Step 1_gnn completed successfully.")
        log_step_success(logger, "GNN file discovery and basic parsing completed successfully")
        return 0
    else:
        log_step_error(logger, "GNN file discovery and processing failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("1_gnn.py")
    else:
        # Fallback argument parsing
        import argparse
        parser = argparse.ArgumentParser(description="GNN file discovery and basic parsing")
        parser.add_argument("--target-dir", type=Path, required=True,
                          help="Target directory containing GNN files")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--recursive", action="store_true",
                          help="Search recursively in subdirectories")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 