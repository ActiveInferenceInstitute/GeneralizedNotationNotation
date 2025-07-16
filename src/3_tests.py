#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 3: Tests

This script runs the test suite for the GNN processing pipeline.

Usage:
    python 3_tests.py [options]
    (Typically called by main.py)
"""

import sys
import logging
from pathlib import Path

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

from tests.runner import run_tests
from utils.pipeline_template import create_standardized_pipeline_script

# Initialize logger for this step
logger = setup_step_logging("3_tests", verbose=False)

def run_tests_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    include_slow: bool = False,
    fast_only: bool = False,  # Changed default to False to run all tests
    **kwargs
) -> bool:
    """
    Standardized test execution function.
    
    Args:
        target_dir: Directory containing files to test
        output_dir: Output directory for test results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        include_slow: Whether to include slow tests
        fast_only: Whether to run only fast tests
        **kwargs: Additional processing options
        
    Returns:
        True if tests passed, False otherwise
    """
    try:
        # Call the existing run_tests function
        success = run_tests(logger, output_dir, verbose, include_slow, fast_only)
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Test execution failed: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "3_tests.py",
    run_tests_standardized,
    "Test suite execution"
)

if __name__ == '__main__':
    sys.exit(run_script()) 
