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
from utils.pipeline_template import create_standardized_pipeline_script

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

run_script = create_standardized_pipeline_script(
    "5_export.py",
    process_export_standardized,
    "Multi-format export generation"
)

if __name__ == '__main__':
    sys.exit(run_script()) 