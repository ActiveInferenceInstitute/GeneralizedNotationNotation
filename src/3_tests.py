#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 3: Tests

This script runs the test suite for the GNN processing pipeline.

Usage:
    python 3_tests.py [options]
    (Typically called by main.py)
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess
import json
import time
import logging

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

def main(parsed_args: argparse.Namespace):
    """Main function for test execution."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("3_tests.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Test suite execution')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run tests
    success = run_tests(logger, Path(parsed_args.output_dir), parsed_args.verbose)
    
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
        parser = argparse.ArgumentParser(description="Test suite execution")
        parser.add_argument("--target-dir", type=Path, required=True,
                          help="Target directory containing GNN files")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 
