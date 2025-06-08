#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 3: Test Execution

This script runs the test suite for the GNN processing pipeline.
It discovers and executes tests in the tests/ directory using pytest,
capturing output and generating a detailed test report.

Usage:
    python 3_tests.py [options]
    (Typically called by main.py)
    
Options:
    Same as main.py (target-dir, output-dir, verbose)
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import datetime
import threading
import time

# Import centralized utilities
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    UTILS_AVAILABLE
)

# Initialize logger for this step
logger = setup_step_logging("3_tests", verbose=False)  # Will be updated based on args

def run_tests(tests_dir: Path, output_dir: Path, verbose: bool = False) -> int:
    """Run tests using pytest, stream output, and generate a JUnit XML report.
    Returns:
        0 if tests pass or no tests found (or tests_dir missing - considered warning).
        1 if tests fail or pytest cannot be run.
        2 if tests_dir is missing (treated as a warning, not a failure of the test runner itself)
    """
    log_step_start(logger, f"Running tests from directory: {tests_dir}")
        
    if not tests_dir.is_dir():
        log_step_warning(logger, f"Tests directory '{tests_dir}' not found or not a directory. Skipping tests.")
        return 2 # Non-fatal issue, treat as warning for the step

    # Ensure test reports directory exists
    test_reports_dir = output_dir / "test_reports"
    try:
        test_reports_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured test reports directory exists: {test_reports_dir}")
    except OSError as e:
        log_step_error(logger, f"Failed to create test reports directory {test_reports_dir}: {e}")
        return 1 # Cannot save reports, treat as failure
    
    report_xml_path = test_reports_dir / "pytest_report.xml"

    # sys.executable should be the python from the .venv when run via main.py
    pytest_command = [sys.executable, "-m", "pytest", str(tests_dir), f"--junitxml={str(report_xml_path)}"]
    
    if verbose:
        pytest_command.append("-v") # Add pytest verbose flag
        pytest_command.append("-rA") # Show summary of all tests (passed, failed, skipped etc.)
    
    logger.debug(f"Executing test command: {' '.join(pytest_command)}")

    try:
        # Run tests from the project's src/ directory (parent of this script's location)
        # This is typically where pytest is configured to run from to correctly discover modules.
        run_cwd = Path(__file__).resolve().parent.parent # Should be project_root / src
        logger.debug(f"Running tests with CWD: {run_cwd}")

        env = os.environ.copy()

        # Use Popen to stream output in real-time, preventing deadlocks from full pipe buffers.
        process = subprocess.Popen(
            pytest_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=run_cwd,
            env=env,
            errors='replace'
        )

        def log_stream(stream, log_func):
            if stream:
                for line in iter(stream.readline, ''):
                    log_func(f"[pytest] {line.strip()}")
                stream.close()

        # Use threads to log stdout and stderr concurrently
        stdout_thread = threading.Thread(target=log_stream, args=(process.stdout, logger.info))
        stderr_thread = threading.Thread(target=log_stream, args=(process.stderr, logger.error))
        
        stdout_thread.start()
        stderr_thread.start()

        stdout_thread.join()
        stderr_thread.join()

        process.wait()
        returncode = process.returncode

        # Always log where the report is, even if tests fail, as it might contain info
        logger.info(f"JUnit XML test report is at: {report_xml_path}")
        
        # Pytest exit codes:
        # 0: all tests passed
        # 1: tests were collected and run but some failed
        # 2: test execution was interrupted by the user
        # 3: internal error encountered while running tests
        # 4: pytest command line usage error
        # 5: no tests were collected

        if returncode == 0:
            log_step_success(logger, "All tests passed")
            return 0
        elif returncode == 5:
            log_step_warning(logger, "No tests were collected by pytest. Ensure tests exist and are discoverable.")
            return 2 # Treat as warning, not a hard failure of the test runner
        else:
            log_step_error(logger, f"Some tests failed or pytest reported an error. Pytest exit code: {returncode}")
            logger.warning("Continuing with pipeline despite test failures.")
            return 1 # Test failures or pytest error
            
    except FileNotFoundError:
        # This means "python -m pytest" could not be started. Highly problematic.
        log_step_error(logger, f"Failed to execute '{sys.executable} -m pytest'.")
        logger.error("Please ensure pytest is installed in the correct Python environment.")
        logger.error("Continuing with pipeline despite missing pytest.")
        return 1 # Critical failure for this step but continue pipeline
    except Exception as e:
        log_step_error(logger, f"Unexpected error occurred while trying to run tests: {e}")
        if verbose:
            logger.exception("Full traceback:")
        logger.warning("Continuing with pipeline despite test execution error.")
        return 1 # Critical failure for this step but continue pipeline

def main(parsed_args: argparse.Namespace):
    """Main function for the testing step (Step 3).

    Invokes the test runner.

    Args:
        parsed_args (argparse.Namespace): Pre-parsed command-line arguments.
            Expected attributes include: output_dir, verbose, target_dir (optional).
    """
    
    # Update logger if verbose mode enabled
    if hasattr(parsed_args, 'verbose') and parsed_args.verbose and UTILS_AVAILABLE:
        global logger
        logger = setup_step_logging("3_tests", verbose=True)
    
    log_step_start(logger, "Starting Step 3: Tests")
    logger.debug(f"Parsed arguments for tests: {parsed_args}")

    script_dir = Path(__file__).parent
    # Tests are expected to be in src/tests/
    tests_dir_to_run = script_dir / "tests" 

    # output_dir is the main pipeline output dir. Test reports go into a subdir.
    output_dir_path = Path(parsed_args.output_dir).resolve()

    result_code = run_tests(tests_dir_to_run, output_dir_path, parsed_args.verbose)
    
    if result_code == 0:
        log_step_success(logger, "Step 3: Tests completed successfully")
        sys.exit(0)
    elif result_code == 2:
        log_step_warning(logger, "Step 3: Tests completed with warnings (e.g., no tests found or tests_dir missing)")
        # Continue pipeline with warning
        sys.exit(0)
    else:
        log_step_error(logger, "Step 3: Tests failed or encountered critical errors")
        sys.exit(1)

if __name__ == "__main__":
    script_file_path = Path(__file__).resolve()
    # project_root_for_defaults should be the actual project root (parent of src/)
    project_root_for_defaults = script_file_path.parent.parent 
    # Output dir for test reports defaults to PROJECT_ROOT/output/
    default_output_dir = project_root_for_defaults / "output"

    parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 3: Tests (Standalone).")
    parser.add_argument(
        "--target-dir",
        type=Path,
        help="Target directory containing GNN files (ignored for tests, but included for compatibility with main.py)"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=default_output_dir,
        help=f"Main output directory where a 'test_reports' subdirectory will be created (default: {default_output_dir.relative_to(project_root_for_defaults) if default_output_dir.is_relative_to(project_root_for_defaults) else default_output_dir})"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        default=False, # Default to False for standalone
        help="Enable verbose (DEBUG level) logging and detailed pytest output."
    )
    cli_args = parser.parse_args()

    main(cli_args) 
