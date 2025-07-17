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
    generate_coverage: bool = True,  # Add flag to generate coverage reports (default True for completeness)
    **kwargs
) -> bool:
    """
    Standardized test execution function.
    
    Args:
        target_dir: Directory containing files to test (not typically used for tests, but included for consistency)
        output_dir: Output directory for test results
        logger: Logger instance for this step
        recursive: Whether to process files recursively (unused for tests)
        verbose: Whether to enable verbose logging
        include_slow: Whether to include slow tests (runs all tests including slow ones)
        fast_only: Whether to run only fast tests (overrides include_slow)
        generate_coverage: Whether to generate coverage reports
        **kwargs: Additional processing options
        
    Returns:
        True if tests passed (or were executed successfully), False otherwise
    """
    try:
        # Add: Log test configuration for better documentation and debugging
        logger.info(f"Test configuration: fast_only={fast_only}, include_slow={include_slow}, verbose={verbose}, generate_coverage={generate_coverage}")
        
        # Call the existing run_tests function with enhanced parameters
        success = run_tests(logger, output_dir, verbose, include_slow, fast_only, generate_coverage=generate_coverage)
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Test execution failed: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "3_tests.py",
    run_tests_standardized,
    "Test suite execution"
)

# Add custom argument parsing before calling run_script, to include new flags
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run test suite')
    parser.add_argument('--include-slow', action='store_true', help='Include slow tests')
    parser.add_argument('--fast-only', action='store_true', help='Run only fast tests')
    parser.add_argument('--generate-coverage', action='store_true', default=True, help='Generate coverage reports')
    # Parse known args to add these, then pass to run_script
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown  # Remove parsed args from sys.argv for run_script
    sys.exit(run_script()) 
