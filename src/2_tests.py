#!/usr/bin/env python3
"""
Step 2: Test Suite Execution

This step runs comprehensive tests for the GNN pipeline.
"""

import sys
from pathlib import Path
import subprocess
import json
import os

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
        output_dir = get_output_dir_for_script("2_tests.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Running comprehensive test suite")
        
        # Run pytest
        test_command = [
            sys.executable, "-m", "pytest",
            "src/tests/",
            "--cov=src/",
            "--cov-report=json:coverage.json",
            "-v" if args.verbose else ""
        ]
        
        # Set PYTHONPATH to include src/
        test_env = os.environ.copy()
        src_path = str(Path(__file__).parent)
        test_env['PYTHONPATH'] = src_path + os.pathsep + test_env.get('PYTHONPATH', '')
        
        result = subprocess.run(test_command, capture_output=True, text=True, cwd=Path(__file__).parent.parent, env=test_env)
        
        # Save results
        results_file = output_dir / "test_results.json"
        test_results = {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        if result.returncode == 0:
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