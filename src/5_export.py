#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 5: Export

This script exports GNN models to various formats (JSON, XML, GraphML, etc.).

Usage:
    python 5_export.py [options]
    (Typically called by main.py)
"""

import sys
import logging
from pathlib import Path
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

from export.core import export_gnn_files

# Initialize logger for this step
logger = setup_step_logging("5_export", verbose=False)

def process_export_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized export processing function with consistent signature.
    
    Args:
        target_dir: Directory containing GNN files to export
        output_dir: Output directory for exported files
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Update logger verbosity if needed
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Call the existing export_gnn_files function with updated signature
        success = export_gnn_files(
            logger=logger,
            target_dir=target_dir,
            output_dir=output_dir,
            recursive=recursive,
            verbose=verbose
        )
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Export processing failed: {e}")
        return False

def main(parsed_args):
    """Main function for GNN export operations."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("5_export.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Multi-format export generation')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run export with standardized function
    success = process_export_standardized(
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        logger=logger,
        recursive=getattr(parsed_args, 'recursive', False),
        verbose=getattr(parsed_args, 'verbose', False)
    )
    
    if success:
        log_step_success(logger, "GNN export completed successfully")
        return 0
    else:
        log_step_error(logger, "GNN export failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("5_export")
    else:
        # Fallback argument parsing
        parser = argparse.ArgumentParser(description="Multi-format export generation")
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