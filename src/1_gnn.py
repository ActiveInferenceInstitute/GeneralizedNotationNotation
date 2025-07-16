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
from utils.pipeline_template import create_standardized_pipeline_script

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

run_script = create_standardized_pipeline_script(
    "1_gnn.py",
    process_gnn_files,
    "GNN file discovery and basic parsing"
)

if __name__ == '__main__':
    sys.exit(run_script()) 