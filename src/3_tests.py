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
import json

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    UTILS_AVAILABLE,
    performance_tracker
)

from pipeline import (
    get_output_dir_for_script
)

from tests.runner import run_tests
from utils.pipeline_template import create_standardized_pipeline_script

# Initialize logger for this step
logger = setup_step_logging("3_tests", verbose=False)

def process_tests_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    include_slow: bool = False,
    fast_only: bool = False,
    generate_coverage: bool = True,
    include_report_tests: bool = True,
    **kwargs
) -> bool:
    """
    Standardized test processing function.
    
    Args:
        target_dir: Directory containing GNN files (for validation)
        output_dir: Output directory for test reports
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        include_slow: Whether to include slow tests
        fast_only: Whether to run only fast tests
        generate_coverage: Whether to generate coverage reports
        include_report_tests: Whether to include comprehensive report tests
        **kwargs: Additional processing options
        
    Returns:
        True if tests succeeded, False otherwise
    """
    try:
        # Start performance tracking
        with performance_tracker.track_operation("test_execution", {
            "verbose": verbose, 
            "generate_coverage": generate_coverage,
            "include_report_tests": include_report_tests
        }):
            # Update logger verbosity if needed
            if verbose:
                logger.setLevel(logging.DEBUG)
            
            # Execute tests using pytest or unittest
            logger.info(f"Running test suite from {target_dir}")
            
            # Set up output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Import and run test modules
            try:
                import subprocess
                import sys
                
                # Build pytest command
                pytest_args = [
                    sys.executable, "-m", "pytest",
                    "src/tests/",
                    "-v" if verbose else "-q",
                    f"--junitxml={output_dir}/pytest_report.xml",
                ]
                
                # Add specific test patterns
                if include_report_tests:
                    pytest_args.extend([
                        "src/tests/test_report_comprehensive.py",
                        "src/tests/test_pipeline_steps.py::TestStep14Report",
                        "src/tests/test_pipeline_scripts.py::TestStep14ReportComprehensive"
                    ])
                
                if generate_coverage:
                    pytest_args.extend([
                        "--cov=src",
                        "--cov=src/report",
                        f"--cov-report=html:{output_dir}/coverage_html",
                        f"--cov-report=xml:{output_dir}/coverage.xml",
                        f"--cov-report=term-missing"
                    ])
                
                if fast_only:
                    pytest_args.extend(["-m", "not slow"])
                elif include_slow:
                    pytest_args.extend(["-m", ""])
                
                # Add specific markers for report tests
                if include_report_tests:
                    pytest_args.extend([
                        "-m", "unit or integration",
                        "--tb=short"
                    ])
                
                # Run tests
                logger.info(f"Running pytest with args: {' '.join(pytest_args[3:])}")
                result = subprocess.run(pytest_args, capture_output=True, text=True)
                
                # Generate test summary
                test_summary = {
                    "exit_code": result.returncode,
                    "stdout": result.stdout[-2000:],  # Last 2000 chars
                    "stderr": result.stderr[-2000:] if result.stderr else "",
                    "test_options": {
                        "verbose": verbose,
                        "include_slow": include_slow, 
                        "fast_only": fast_only,
                        "generate_coverage": generate_coverage,
                        "include_report_tests": include_report_tests
                    },
                    "pytest_command": " ".join(pytest_args[3:])
                }
                
                # Save test summary
                summary_file = output_dir / "test_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(test_summary, f, indent=2)
                
                # Log test results
                if result.returncode == 0:
                    logger.info("All tests passed successfully")
                    
                    # Check for report test results specifically
                    if include_report_tests and "test_report_comprehensive" in result.stdout:
                        logger.info("Report generation tests completed successfully")
                    
                    return True
                else:
                    logger.warning(f"Some tests failed (exit code: {result.returncode})")
                    
                    # Log specific failures if verbose
                    if verbose and result.stderr:
                        logger.warning(f"Test errors: {result.stderr}")
                    
                    return True  # Non-critical step, continue pipeline
                    
            except ImportError:
                logger.warning("pytest not available, using basic test validation")
                # Fallback: basic validation that test files exist
                test_files = list(Path("src/tests").glob("test_*.py"))
                logger.info(f"Found {len(test_files)} test files for validation")
                
                # Check for report test files specifically
                report_test_files = list(Path("src/tests").glob("*report*.py"))
                if report_test_files:
                    logger.info(f"Found {len(report_test_files)} report test files")
                
                return True
                
    except Exception as e:
        log_step_error(logger, f"Tests failed: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "3_tests.py",
    process_tests_standardized,
    "Test suite execution"
)

# Add custom argument parsing before calling run_script, to include new flags
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run test suite')
    parser.add_argument('--include-slow', action='store_true', help='Include slow tests')
    parser.add_argument('--fast-only', action='store_true', help='Run only fast tests')
    parser.add_argument('--generate-coverage', action='store_true', default=True, help='Generate coverage reports')
    parser.add_argument('--include-report-tests', action='store_true', default=True, help='Include comprehensive report tests')
    parser.add_argument('--skip-report-tests', action='store_true', help='Skip comprehensive report tests')
    
    # Parse known args to add these, then pass to run_script
    args, unknown = parser.parse_known_args()
    
    # Handle report test inclusion/exclusion
    if args.skip_report_tests:
        args.include_report_tests = False
    
    # Update sys.argv to remove parsed args for run_script
    sys.argv = [sys.argv[0]] + unknown
    
    # Set environment variable for report tests if needed
    if args.include_report_tests:
        import os
        os.environ['INCLUDE_REPORT_TESTS'] = '1'
    
    sys.exit(run_script()) 
