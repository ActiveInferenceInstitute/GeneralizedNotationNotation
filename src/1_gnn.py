#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 1: GNN File Discovery and Basic Parsing

This script handles initial GNN-specific operations, such as:
- Discovering .md GNN files in the target directory.
- Performing basic parsing for key GNN sections (ModelName, StateSpaceBlock, Connections).
- Generating a summary report of findings.

Usage:
    python 1_gnn.py [options]
    (Typically run as part of main.py pipeline)
    
Options:
    Same as main.py (verbose, target-dir, output-dir, recursive)
"""

import os
import sys
import logging
from pathlib import Path
import re
import argparse
from typing import TypedDict, List, Dict, Any

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

def main(parsed_args: argparse.Namespace):
    """Main function for GNN file discovery and processing."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("1_gnn.py", {})
    log_step_start(logger, f"{step_info.get('description', 'GNN file discovery and basic parsing')}")
    
    # Determine project root for path relativization
    project_root = None
    try:
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent
    except Exception:
        pass
    
    logger.info(f"GNN Step 1: Target directory: {parsed_args.target_dir}")
    logger.info(f"GNN Step 1: Output directory: {parsed_args.output_dir}")
    logger.info(f"GNN Step 1: Recursive: {parsed_args.recursive}")
    logger.info(f"GNN Step 1: Verbose: {parsed_args.verbose}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Process GNN files
    success = process_gnn_folder(
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        project_root=project_root,
        recursive=parsed_args.recursive,
        verbose=parsed_args.verbose,
        logger=logger
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