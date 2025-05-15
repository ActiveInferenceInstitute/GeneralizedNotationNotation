#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 3: Tests

This script handles the execution of tests, such as:
- Running unit tests
- Running integration tests

Usage:
    python 3_tests.py [options]
    
Options:
    Same as main.py (though many may not be relevant for testing)
"""

import os
import sys
import subprocess
from pathlib import Path
import logging # Import logging
import argparse # Added

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

def run_tests(tests_dir: Path, output_dir: Path, verbose: bool = False) -> int:
    """Run tests using pytest and generate a JUnit XML report.
    Returns:
        0 if tests pass or no tests found (or tests_dir missing - considered warning).
        1 if tests fail or pytest cannot be run.
        2 if tests_dir is missing (treated as a warning, not a failure of the test runner itself)
    """
    logger.info(f"ℹ️ Running tests from directory: {tests_dir}")
        
    if not tests_dir.is_dir():
        logger.warning(f"⚠️ Tests directory '{tests_dir}' not found or not a directory. Skipping tests.")
        return 2 # Non-fatal issue, treat as warning for the step

    # Ensure test reports directory exists
    test_reports_dir = output_dir / "test_reports"
    try:
        test_reports_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured test reports directory exists: {test_reports_dir}")
    except OSError as e:
        logger.error(f"❌ Failed to create test reports directory {test_reports_dir}: {e}")
        return 1 # Cannot save reports, treat as failure
    
    report_xml_path = test_reports_dir / "pytest_report.xml"

    # sys.executable should be the python from the .venv when run via main.py
    pytest_command = [sys.executable, "-m", "pytest", str(tests_dir), f"--junitxml={str(report_xml_path)}"]
    
    if verbose:
        pytest_command.append("-v") # Add pytest verbose flag
        pytest_command.append("-rA") # Show summary of all tests (passed, failed, skipped etc.)
    
    logger.debug(f"  🐍 Executing test command: {' '.join(pytest_command)}")

    try:
        # Run tests from the project's src/ directory (parent of this script's location)
        # This is typically where pytest is configured to run from to correctly discover modules.
        run_cwd = Path(__file__).resolve().parent.parent # Should be project_root / src
        logger.debug(f"  📂 Running tests with CWD: {run_cwd}")

        # PYTHONPATH should be set by main.py for this script's environment.
        # sys.executable -m pytest will run within that environment.
        env = os.environ.copy() # Inherit environment from main.py

        result = subprocess.run(pytest_command, capture_output=True, text=True, check=False, cwd=run_cwd, env=env, errors='replace')
        
        # Always log where the report is, even if tests fail, as it might contain info
        logger.info(f"ℹ️ JUnit XML test report will be at: {report_xml_path}")

        if result.returncode != 0:
            logger.error("--- Test Output (stdout) ---")
            logger.error(result.stdout.strip() if result.stdout else "<No stdout>")
            logger.error("--- Test Errors (stderr) ---")
            logger.error(result.stderr.strip() if result.stderr else "<No stderr>")
            logger.error("-------------------------")
        elif verbose:
            logger.debug("--- Test Output (stdout) ---")
            logger.debug(result.stdout.strip() if result.stdout else "<No stdout>")
            if result.stderr.strip(): # Only log stderr if it has content
                logger.debug("--- Test Errors (stderr) ---")
                logger.debug(result.stderr.strip())
            logger.debug("-------------------------")

        # Pytest exit codes:
        # 0: all tests passed
        # 1: tests were collected and run but some failed
        # 2: test execution was interrupted by the user
        # 3: internal error encountered while running tests
        # 4: pytest command line usage error
        # 5: no tests were collected

        if result.returncode == 0:
            logger.info("✅ All tests passed.")
            return 0
        elif result.returncode == 5:
            logger.warning("⚠️ No tests were collected by pytest. Ensure tests exist and are discoverable.")
            return 2 # Treat as warning, not a hard failure of the test runner
        else:
            logger.error(f"❌ Some tests failed or pytest reported an error. Pytest exit code: {result.returncode}")
            return 1 # Test failures or pytest error
            
    except FileNotFoundError:
        # This means "python -m pytest" could not be started. Highly problematic.
        logger.error(f"❌ Error: Failed to execute '{sys.executable} -m pytest'.")
        logger.error("   Please ensure pytest is installed in the correct Python environment ('{sys.executable}').")
        logger.error("   This step (3_tests.py) cannot proceed without pytest.")
        return 1 # Critical failure for this step
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred while trying to run tests: {e}", exc_info=verbose)
        return 1 # Critical failure for this step

def main(cmd_args=None): # Renamed 'args' to 'cmd_args'
    """Main function for the testing step (Step 3).

    Handles argument parsing if run standalone and then invokes the test runner.

    Args:
        cmd_args (argparse.Namespace | list | None):
            - If None, parses arguments from sys.argv for standalone execution.
            - If a list, assumes it's a list of string arguments to be parsed.
            - If an argparse.Namespace, uses it directly.
            Expected attributes on the Namespace object include:
            output_dir, verbose.
    """
    if cmd_args is None:
        script_file_path = Path(__file__).resolve()
        project_root_for_defaults = script_file_path.parent.parent # src/
        # Output dir for test reports defaults to output/test_reports inside project root
        default_output_dir = project_root_for_defaults.parent / "output" 

        parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 3: Tests.")
        parser.add_argument(
            "--output-dir", 
            type=Path, 
            default=default_output_dir,
            help=f"Main output directory where a 'test_reports' subdirectory will be created (default: {default_output_dir})"
        )
        parser.add_argument(
            "--verbose", 
            action="store_true", 
            help="Enable verbose (DEBUG level) logging and detailed pytest output."
        )
        parsed_args = parser.parse_args()
    elif isinstance(cmd_args, list):
        script_file_path = Path(__file__).resolve()
        project_root_for_defaults = script_file_path.parent.parent
        default_output_dir = project_root_for_defaults.parent / "output"
        parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 3: Tests.")
        parser.add_argument("--output-dir", type=Path, default=default_output_dir)
        parser.add_argument("--verbose", action="store_true")
        parsed_args = parser.parse_args(cmd_args)
    elif hasattr(cmd_args, 'output_dir'):
        parsed_args = cmd_args
    else:
        print("ERROR: Invalid 'cmd_args' type passed to 3_tests.py main().", file=sys.stderr)
        sys.exit(1)

    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stdout)
    else:
        logger.setLevel(log_level)

    logger.info("▶️ Starting Step 3: Tests")
    logger.debug(f"  Parsed arguments for tests: {parsed_args}")

    script_dir = Path(__file__).parent
    # Tests are expected to be in src/tests/
    tests_dir_to_run = script_dir / "tests" 

    # output_dir is the main pipeline output dir. Test reports go into a subdir.
    output_dir_path = Path(parsed_args.output_dir).resolve()

    result_code = run_tests(tests_dir_to_run, output_dir_path, parsed_args.verbose)
    
    if result_code == 0:
        logger.info("✅ Step 3: Tests completed successfully.")
    elif result_code == 2:
        logger.warning("⚠️ Step 3: Tests completed with warnings (e.g., no tests found or tests_dir missing).")
    else: # result_code == 1
        logger.error("❌ Step 3: Tests failed or encountered a critical error.")
        
    sys.exit(result_code)

if __name__ == "__main__":
    main() 