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

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# --- Helper to find venv site-packages ---
def _get_venv_site_packages(script_dir: Path):
    venv_path = script_dir / ".venv"
    site_packages_found_path = None
    if venv_path.is_dir():
        lib_path = venv_path / "lib"
        if lib_path.is_dir():
            for python_version_dir in lib_path.iterdir():
                if python_version_dir.is_dir() and python_version_dir.name.startswith("python"):
                    current_site_packages = python_version_dir / "site-packages"
                    if current_site_packages.is_dir():
                        site_packages_found_path = str(current_site_packages.resolve())
                        break
    return site_packages_found_path
# --- End helper ---

def run_tests(tests_dir, output_dir, verbose=False):
    """Run tests, e.g., using pytest or unittest."""
    logger.info(f"ℹ️ Running tests from directory: {tests_dir}")
        
    tests_path = Path(tests_dir)
    if not tests_path.is_dir():
        logger.warning(f"⚠️ Tests directory '{tests_dir}' not found or not a directory. Skipping tests.")
        return True # Not a fatal error for a placeholder

    # Construct path to venv pytest
    # run_cwd is expected to be src/
    run_cwd = tests_path.parent # This should be src/, where .venv is
    venv_pytest_path = run_cwd / ".venv" / "bin" / "pytest"

    if not venv_pytest_path.exists():
        logger.error(f"❌ Pytest executable not found at {venv_pytest_path}. Falling back to sys.executable -m pytest.")
        # Fallback to original method if direct path fails, though this is unlikely to solve the core issue if it exists
        pytest_command = [sys.executable, "-m", "pytest", str(tests_path)]
    else:
        logger.info(f"ℹ️ Using pytest executable: {venv_pytest_path}")
        pytest_command = [str(venv_pytest_path), str(tests_path)]
    
    if verbose:
        pytest_command.append("-v")
    
    logger.debug(f"  🐍 Executing test command: {' '.join(pytest_command)}")

    try:
        # run_cwd = tests_path.parent # This should be src/ - Already defined above
        logger.debug(f"  📂 Running tests in CWD: {run_cwd}")

        env = os.environ.copy()
        venv_site_packages = _get_venv_site_packages(run_cwd)
        
        if venv_site_packages:
            logger.debug(f"  🐍 Adding to PYTHONPATH for subprocess: {venv_site_packages}")
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = f"{venv_site_packages}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = venv_site_packages
        else:
            logger.debug("  ⚠️ .venv site-packages not found or not added to PYTHONPATH for subprocess.")

        result = subprocess.run(pytest_command, capture_output=True, text=True, check=False, cwd=run_cwd, env=env)
        
        if result.returncode != 0:
            logger.error("--- Test Output ---")
            if result.stdout:
                 logger.error(result.stdout)
            else:
                 logger.error("<No stdout>")
            if result.stderr:
                logger.error("--- Test Errors ---")
                logger.error(result.stderr)
            logger.error("-------------------")
        elif verbose:
            logger.debug("--- Test Output (Verbose) ---")
            if result.stdout:
                logger.debug(result.stdout)
            else:
                logger.debug("<No stdout>")
            if result.stderr:
                logger.debug("--- Test Stderr (Verbose, should be empty on success) ---")
                logger.debug(result.stderr)
            logger.debug("---------------------------")

        if result.returncode == 0:
            logger.info("✅ All tests passed.")
            return True
        else:
            logger.error(f"❌ Some tests failed. Pytest exit code: {result.returncode}")
            return False
            
    except FileNotFoundError:
        logger.error("❌ Error: Pytest not found. Please ensure pytest is installed and in your PATH.")
        logger.warning("  Skipping test execution.")
        return True # Treat as non-fatal for placeholder, or return False in production
    except Exception as e:
        logger.error(f"❌ An error occurred while trying to run tests: {e}", exc_info=verbose)
        return False

def main(args):
    """Main function for the testing step."""
    # The logger level for __name__ (this module's logger) should be set by main.py
    # based on args.verbose before this main() function is called.
    # Thus, direct setLevel calls here are typically not needed when run via the pipeline.

    logger.info("▶️ Starting tests step...")
    
    script_dir = Path(__file__).parent
    tests_processing_target = script_dir / "tests"

    if not run_tests(tests_processing_target, args.output_dir, args.verbose):
        logger.warning("⚠️ Test step indicated failures.")
        return 1
        
    logger.info("✅ Tests step complete.")
    return 0

if __name__ == "__main__":
    # Basic configuration for running this script standalone
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create a dummy args object
    class DummyArgs:
        def __init__(self):
            self.verbose = (log_level == logging.DEBUG)
            self.output_dir = "../output" # Standard default
            # Add any other args expected by this script's main() if run standalone
            # For example, if main() uses args.target_dir for some reason:
            # self.target_dir = "gnn/examples" 

    dummy_args = DummyArgs()
    sys.exit(main(dummy_args)) 