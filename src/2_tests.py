#!/usr/bin/env python3
"""
Step 2: Test Suite Execution (Thin Orchestrator)

This script orchestrates comprehensive tests for the GNN pipeline in staged execution.
It is a thin orchestrator that delegates core functionality to the ModularTestRunner.

How to run:
  python src/2_tests.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - All test logs and reports in the specified output directory (default: output/)
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that pytest and required plugins are installed (pip install pytest pytest-cov pytest-xdist pytest-json-report)
  - Check that src/tests/ contains test files
  - Check that the output directory is writable
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

# Import the comprehensive test runner
from tests.test_runner_comprehensive import create_test_runner

# --- Fix for Python 3.13 pathlib recursion issue ---
def apply_pathlib_patch():
    """Apply patch for Python 3.13 pathlib recursion issue."""
    try:
        import pathlib
        from pathlib import _local
        
        # Check if we're on Python 3.13+ and the issue exists
        if hasattr(_local, 'PurePosixPath'):
            # Apply a more robust patch
            original_tail = _local.PurePosixPath._tail
            
            def _patched_tail(self):
                try:
                    if not hasattr(self, '_tail_cached'):
                        # Use a safer approach to avoid recursion
                        try:
                            self._tail_cached = self._parse_path(self._raw_path)[2]
                        except (AttributeError, RecursionError):
                            # Fallback for recursion issues
                            self._tail_cached = ""
                    return self._tail_cached
                except (AttributeError, RecursionError):
                    # Ultimate fallback
                    return ""
            
            # Apply the patch
            _local.PurePosixPath._tail = property(_patched_tail)
            
            logging.getLogger(__name__).info("Applied pathlib recursion fix for Python 3.13")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not apply pathlib patch: {e}")

# Apply the patch early
apply_pathlib_patch()

# --- Robust Path and Argument Logging ---
def log_resolved_paths_and_args(args, logger):
    logger.info("\n===== GNN Test Step: Resolved Arguments and Paths =====")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script location: {Path(__file__).resolve()}")
    logger.info(f"target_dir: {getattr(args, 'target_dir', None)}")
    logger.info(f"output_dir: {getattr(args, 'output_dir', None)}")
    logger.info(f"verbose: {getattr(args, 'verbose', None)}")
    logger.info(f"fast_only: {getattr(args, 'fast_only', None)}")
    logger.info(f"include_slow: {getattr(args, 'include_slow', None)}")
    logger.info(f"include_performance: {getattr(args, 'include_performance', None)}")
    logger.info("=======================================================\n")

# --- Robust Output Directory Creation ---
def ensure_output_dir(output_dir: Path, logger):
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ensured: {output_dir.resolve()}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        logger.error("Action: Check that the output directory is writable and not locked.")
        sys.exit(1)

# --- Robust Test Directory Check ---
def ensure_test_dir_exists(logger):
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "src" / "tests"
    if not test_dir.exists() or not any(test_dir.glob("test_*.py")):
        logger.error(f"Test directory not found or contains no test_*.py files: {test_dir}")
        logger.error("Action: Ensure that src/tests/ exists and contains test files.")
        sys.exit(1)
    logger.info(f"Test directory found: {test_dir.resolve()}")

# --- Robust Dependency Check ---
def ensure_dependencies(logger):
    """Check dependencies using the correct Python environment."""
    # Get the project root and virtual environment paths
    project_root = Path(__file__).parent.parent
    venv_python = project_root / ".venv" / "bin" / "python"
    
    # Use virtual environment python if available, otherwise system python
    python_executable = str(venv_python) if venv_python.exists() else sys.executable
    
    logger.info(f"Checking dependencies using Python: {python_executable}")
    
    # Check pytest availability using the correct Python
    try:
        import subprocess
        result = subprocess.run(
            [python_executable, "-c", "import pytest; print(f'pytest {pytest.__version__}')"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            logger.info(f"✅ pytest available: {result.stdout.strip()}")
        else:
            logger.error(f"❌ pytest not available: {result.stderr}")
            logger.error("Action: Install pytest with: pip install pytest pytest-cov pytest-xdist pytest-json-report")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to check pytest: {e}")
        logger.error("Action: Install pytest with: pip install pytest pytest-cov pytest-xdist pytest-json-report")
        sys.exit(1)

# --- Enhanced Argument Parsing ---
def parse_enhanced_arguments():
    """Parse arguments with enhanced error handling and validation."""
    # Use fallback argument parsing for test-specific arguments
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive test suite execution for GNN pipeline")
    
    # Core arguments
    parser.add_argument(
        "--target-dir", 
        type=Path,
        default=Path("input/gnn_files"),
        help="Directory containing GNN files to test against"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for test results and reports"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Test execution options
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="Run only fast tests (core and pipeline categories)"
    )
    
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow test categories"
    )
    
    parser.add_argument(
        "--include-performance",
        action="store_true",
        help="Include performance test categories"
    )
    
    args = parser.parse_args()
    
    return args

# --- Main Execution Function ---
def main():
    """Main execution function with comprehensive error handling."""
    # Parse arguments
    args = parse_enhanced_arguments()
    
    # Setup logging
    logger = setup_step_logging("2_tests", args.verbose)
    
    # Log resolved paths and arguments
    log_resolved_paths_and_args(args, logger)
    
    # Ensure output directory exists
    ensure_output_dir(args.output_dir, logger)
    
    # Ensure test directory exists
    ensure_test_dir_exists(logger)
    
    # Check dependencies
    ensure_dependencies(logger)
    
    # Start step
    log_step_start(logger, "Comprehensive test suite execution")
    
    try:
        # Create test runner using the comprehensive module
        runner = create_test_runner(args, logger)
        
        # Run all test categories
        success = runner.run_all_categories()
        
        if success:
            log_step_success(logger, "All test categories completed successfully")
            return 0
        else:
            log_step_error(logger, "Some test categories failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"Unexpected error during test execution: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 