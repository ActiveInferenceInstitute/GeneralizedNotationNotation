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

def main(parsed_args):
    """Main function for test execution."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("3_tests.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Test suite execution')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run tests
    success = run_tests_standardized(
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        logger=logger,
        recursive=getattr(parsed_args, 'recursive', False),
        verbose=getattr(parsed_args, 'verbose', False),
        include_slow=getattr(parsed_args, 'include_slow', False),
        fast_only=getattr(parsed_args, 'fast_only', False)
    )
    
    if success:
        log_step_success(logger, "Test execution completed successfully")
        return 0
    else:
        log_step_error(logger, "Test execution failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("3_tests")
    else:
        # Fallback argument parsing
        import argparse
        parser = argparse.ArgumentParser(description="Test suite execution")
        parser.add_argument("--target-dir", type=Path, required=True,
                          help="Target directory containing GNN files")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parser.add_argument("--include-slow", action="store_true",
                          help="Include slow tests in the run")
        parser.add_argument("--fast-only", action="store_true",
                          help="Run only fast tests")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 
