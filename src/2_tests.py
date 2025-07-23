#!/usr/bin/env python3
"""
Step 2: Test Suite Execution

This step runs comprehensive tests for the GNN pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

def main():
    """Main test execution function."""
    args = EnhancedArgumentParser.parse_step_arguments("2_tests.py")
    
    # Setup logging
    logger = setup_step_logging("tests", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("2_tests.py", config.base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import and run tests from tests module
        from tests import run_all_tests
        
        log_step_start(logger, "Running comprehensive test suite")
        
        success = run_all_tests(
            target_dir=args.target_dir,
            output_dir=output_dir,
            verbose=args.verbose
        )
        
        if success:
            log_step_success(logger, "All tests passed successfully")
            return 0
        else:
            log_step_error(logger, "Some tests failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 