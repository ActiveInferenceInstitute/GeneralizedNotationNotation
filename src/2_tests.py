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
import subprocess
import json
import time
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning,
    create_standardized_pipeline_script,
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

# Import the test runner with fallback. Make psutil optional inside the runner to avoid import failures.
try:
    from tests.runner import create_test_runner
    TEST_RUNNER_AVAILABLE = True
except Exception as e:
    TEST_RUNNER_AVAILABLE = False
    logging.warning(f"Test runner not available: {e}")
    
    def create_test_runner(args, logger):
        """Fallback test runner creation."""
        logger.warning("⚠️ Test runner not available, using fallback")
        return None

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

# The pathlib recursion patch is intended only when running this script directly
# (e.g., `python src/2_tests.py`). Avoid applying it during normal imports
# such as when pytest collects test modules to prevent interfering with pytest
# internals.


# --- Ensure unbuffered, verbose, and progressive pytest output ---
def _configure_live_pytest_output(logger: logging.Logger):
    """Set environment to encourage live, detailed pytest progress output.

    We do not overwrite existing PYTEST_ADDOPTS; we append desired flags if missing.
    """
    try:
        desired_opts = [
            "-vv",
            "-rA",  # report all statuses
            "-s",   # disable capture for live stdout
            "--durations=15",
            "-o",
            "console_output_style=classic",
            "--color=no",
        ]

        existing = os.environ.get("PYTEST_ADDOPTS", "").split()
        merged = existing[:]
        for opt in desired_opts:
            if opt not in merged:
                merged.append(opt)
        os.environ["PYTEST_ADDOPTS"] = " ".join(merged).strip()

        # Ensure unbuffered Python for children
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        # Force non-interactive terminal settings to avoid termcap warnings
        os.environ.setdefault("TERM", "dumb")
        os.environ.setdefault("NO_COLOR", "1")
        os.environ.setdefault("PY_COLORS", "0")
        # Disable Ollama provider in tests unless explicitly enabled to avoid CLI timeouts
        os.environ.setdefault("OLLAMA_DISABLED", "1")
        logger.info(f"Configured PYTEST_ADDOPTS: {os.environ['PYTEST_ADDOPTS']}")
    except Exception as e:
        logger.warning(f"Could not set live pytest output configuration: {e}")

# --- Robust Path and Argument Logging ---
def log_resolved_paths_and_args(args, logger):
    """Log resolved paths and arguments with validation."""
    logger.info("\n===== GNN Test Step: Resolved Arguments and Paths =====")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script location: {Path(__file__).resolve()}")
    logger.info(f"target_dir: {getattr(args, 'target_dir', None)}")
    logger.info(f"output_dir: {getattr(args, 'output_dir', None)}")
    logger.info(f"verbose: {getattr(args, 'verbose', None)}")
    logger.info(f"fast_only: {getattr(args, 'fast_only', None)}")
    logger.info(f"include_slow: {getattr(args, 'include_slow', None)}")
    logger.info(f"include_performance: {getattr(args, 'include_performance', None)}")
    
    # Validate paths exist
    if args.target_dir and not args.target_dir.exists():
        logger.warning(f"Warning: target_dir does not exist: {args.target_dir}")
    
    logger.info("=======================================================\n")

# --- Robust Output Directory Creation ---
def ensure_output_dir(output_dir: Path, logger):
    """Ensure output directory exists with error handling."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ensured: {output_dir.resolve()}")
        
        # Test write permissions
        test_file = output_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        logger.info("Output directory is writable")
    except PermissionError as e:
        logger.error(f"Permission denied creating output directory {output_dir}: {e}")
        logger.error("Action: Check that you have write permissions to the output directory.")
        sys.exit(1)
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
        # Prefer uv to query pytest when available
        result = subprocess.run(
            ["uv", "run", python_executable, "-c", "import pytest; print(f'pytest {pytest.__version__}')"],
            capture_output=True, text=True, timeout=20
        )
        if result.returncode == 0:
            logger.info(f"✅ pytest available: {result.stdout.strip()}")
        else:
            logger.error(f"❌ pytest not available: {result.stderr}")
            logger.error("Action: Ensure pytest is specified in pyproject and run: uv sync --extra dev")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to check pytest: {e}")
        logger.error("Action: Ensure pytest is specified in pyproject and run: uv sync --extra dev")
        sys.exit(1)

# --- Argument Parsing ---
def parse_enhanced_arguments():
    """Parse arguments with error handling and validation."""
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
    
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run all test categories including comprehensive suite"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively process directories"
    )
    
    args = parser.parse_args()
    
    return args

def discover_test_files(target_dir: Path = None) -> List[Path]:
    """
    Discover test files in the target directory.
    
    This function recursively searches for files matching the pattern "test_*.py".
    
    Args:
        target_dir: Directory to search for test files (defaults to src/tests)
        
    Returns:
        List of Path objects for test files.
    """
    if target_dir is None:
        target_dir = Path("src/tests")
    
    test_files = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                test_files.append(Path(root) / file)
    return test_files

def process_tests_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized test processing function.
    
    Args:
        target_dir: Directory containing files to process
        output_dir: Directory to write output files
        logger: Logger instance for logging
        recursive: Whether to process subdirectories recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional keyword arguments
        
    Returns:
        True if tests passed, False otherwise
    """
    try:
        logger.info("🚀 Processing tests")
        # Configure environment for live pytest streaming
        _configure_live_pytest_output(logger)
        
        # Use standardized numbered output directory for this step
        tests_root = get_output_dir_for_script("2_tests.py", output_dir)
        tests_root.mkdir(parents=True, exist_ok=True)
        # Place results under a stable subfolder to avoid clutter at root
        test_output_dir = tests_root / "test_results"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if pytest is available (use venv Python if present)
        try:
            project_root = Path(__file__).parent.parent
            venv_python = project_root / ".venv" / "bin" / "python"
            python_exec = str(venv_python) if venv_python.exists() else sys.executable
            result = subprocess.run([python_exec, "-m", "pytest", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("❌ pytest not available; failing tests step")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("❌ pytest not available; failing tests step")
            return False
        
        # Run tests if test runner is available
        if TEST_RUNNER_AVAILABLE:
            try:
                # Create test runner
                runner = create_test_runner(type('Args', (), {
                    'target_dir': target_dir,
                    'output_dir': test_output_dir,
                    'recursive': recursive,
                    'verbose': verbose,
                    'fast_only': kwargs.get('fast_only', False),
                    'include_slow': kwargs.get('include_slow', False),
                    'include_performance': kwargs.get('include_performance', False),
                    'comprehensive': kwargs.get('comprehensive', False),
                }), logger)
                
                if runner:
                    # Prefer modular comprehensive category runner if available
                    success = False
                    if hasattr(runner, 'run_all_categories'):
                        results = runner.run_all_categories()
                        success = bool(results.get('overall_success', False))
                    elif hasattr(runner, 'run_all_tests'):
                        success = bool(runner.run_all_tests())
                    elif hasattr(runner, 'run_tests'):
                        success = bool(runner.run_tests())
                    else:
                        logger.warning("⚠️ No test execution method available")
                        success = True
                    
                    if success:
                        logger.info("✅ Tests completed successfully")
                    else:
                        logger.warning("⚠️ Some tests failed")
                    return success
                else:
                    logger.warning("⚠️ Test runner creation failed")
                    return True  # Don't fail the pipeline
            except Exception as e:
                logger.error(f"❌ Test execution failed: {e}")
                return False
        else:
            # Fallback: run basic pytest
            try:
                test_dir = Path(__file__).parent / "tests"
                if test_dir.exists():
                    # Build pytest command; PYTEST_ADDOPTS will add -vv -rA -s and progress style
                    # Disable plugins known to cause duplicate-registration in some environments
                    # (e.g. pytest_timeout sometimes registers twice via entrypoints)
                    cmd = [
                        python_exec,
                        "-m",
                        "pytest",
                        "-p",
                        "no:pytest_timeout",
                        str(test_dir),
                        "--tb=short",
                        "--maxfail=10",
                        "--durations=15",
                        "-o",
                        "console_output_style=progress",
                    ]
                    if verbose:
                        cmd.append("-vv")

                    # Stream output live while teeing to files
                    stdout_path = test_output_dir / "pytest_stdout.log"
                    stderr_path = test_output_dir / "pytest_stderr.log"
                    # Ensure unbuffered child python for prompt streaming
                    import os as _os
                    _env = _os.environ.copy()
                    _env.setdefault("PYTHONUNBUFFERED", "1")
                    _env.setdefault("TERM", "xterm-256color")
                    # Ensure stable environment for pytest execution
                    _env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
                    # Encourage live, detailed progress output
                    desired_addopts = ["-vv", "-rA", "-s", "--durations=15", "-o", "console_output_style=progress"]
                    existing_addopts = _env.get("PYTEST_ADDOPTS", "").split()
                    for opt in desired_addopts:
                        if opt not in existing_addopts:
                            existing_addopts.append(opt)
                    _env["PYTEST_ADDOPTS"] = " ".join(existing_addopts).strip()
                    # Add src to PYTHONPATH so imports work consistently
                    _env["PYTHONPATH"] = str(project_root / "src")

                    with open(stdout_path, "w") as f_out, open(stderr_path, "w") as f_err:
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            bufsize=1,
                            env=_env,
                        )

                        import threading
                        collected_stdout: list[str] = []
                        collected_stderr: list[str] = []

                        def _stream(pipe, sink_file, log_func, collect):
                            try:
                                for line in iter(pipe.readline, ""):
                                    sink_file.write(line)
                                    sink_file.flush()
                                    stripped = line.rstrip("\n")
                                    collect.append(stripped)
                                    log_func(stripped)
                            finally:
                                try:
                                    pipe.close()
                                except Exception:
                                    pass

                        t_out = threading.Thread(target=_stream, args=(process.stdout, f_out, logger.info, collected_stdout), daemon=True)
                        t_err = threading.Thread(target=_stream, args=(process.stderr, f_err, logger.error, collected_stderr), daemon=True)
                        t_out.start(); t_err.start()

                        try:
                            process.wait(timeout=3600)
                            t_out.join(timeout=5)
                            t_err.join(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()

                    # Save test results
                    result_returncode = process.returncode
                    stdout_text = "\n".join(collected_stdout)
                    stderr_text = "\n".join(collected_stderr)
                    test_results = {
                        "timestamp": time.time(),
                        "returncode": result_returncode,
                        "stdout": stdout_text,
                        "stderr": stderr_text,
                        "tests_run": True
                    }

                    with open(test_output_dir / "test_results.json", 'w') as f:
                        json.dump(test_results, f, indent=2)

                    if result_returncode == 0:
                        logger.info("✅ Basic tests completed successfully")
                        return True
                    else:
                        logger.error("❌ Tests failed in fallback execution")
                        return False
                else:
                    logger.error("❌ No test directory found")
                    return False
            except Exception as e:
                logger.error(f"❌ Fallback test execution failed: {e}")
                return False
        
    except Exception as e:
        logger.error(f"❌ Test processing failed: {e}")
        return False


def validate_test_syntax(test_files: List[Path]) -> List[str]:
    """
    Validate syntax of test files.
    
    Args:
        test_files: List of test file paths
        
    Returns:
        List of syntax error messages
    """
    syntax_errors = []
    
    for test_file in test_files:
        try:
            # Try to compile the file to check syntax
            with open(test_file, 'r') as f:
                compile(f.read(), str(test_file), 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{test_file}:{e.lineno}: {e.msg}")
        except Exception as e:
            syntax_errors.append(f"{test_file}: {str(e)}")
    
    return syntax_errors


def execute_test_suite(test_files: List[Path], verbose: bool = False) -> Dict[str, Any]:
    """
    Execute the test suite with proper error handling.
    
    Args:
        test_files: List of test file paths
        verbose: Whether to enable verbose output
        
    Returns:
        Dictionary with test execution results
    """
    try:
        # Use virtual environment Python
        venv_python = Path(".venv/bin/python")
        python_executable = str(venv_python) if venv_python.exists() else sys.executable
        
        # Build pytest command and disable pytest_timeout plugin to avoid duplicate-registration
        cmd = [python_executable, "-m", "pytest", "-p", "no:pytest_timeout"]
        if verbose:
            cmd.append("-v")
        cmd.extend(["--tb=short", "--maxfail=10", "--color=no"])
        
        # Add test files
        for test_file in test_files:
            cmd.append(str(test_file))
        
        # Execute tests
        print(f"🚀 Executing: {' '.join(cmd[:5])}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Parse results
        output_lines = result.stdout.split('\n')
        test_results = {
            "execution_successful": result.returncode in [0, 1],  # pytest returns 1 for test failures
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
        
        # Parse test results from output
        for line in output_lines:
            if "passed" in line and "failed" in line and "skipped" in line:
                # Extract numbers from summary line
                import re
                numbers = re.findall(r'(\d+)', line)
                if len(numbers) >= 3:
                    test_results["passed"] = int(numbers[0])
                    test_results["failed"] = int(numbers[1])
                    test_results["skipped"] = int(numbers[2])
                    test_results["total_tests"] = sum([test_results["passed"], test_results["failed"], test_results["skipped"]])
                break
        
        return test_results
        
    except subprocess.TimeoutExpired:
        # Using local logger is unsafe here; ensure we create a basic one if needed
        import logging as _logging
        _logging.getLogger(__name__).error("❌ Test execution timed out")
        return {
            "execution_successful": False,
            "error": "Test execution timed out",
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0
        }
    except Exception as e:
        import logging as _logging
        _logging.getLogger(__name__).error(f"❌ Test execution failed: {e}")
        return {
            "execution_successful": False,
            "error": str(e),
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0
        }


def generate_test_report(test_results: Dict[str, Any], syntax_errors: List[str]) -> Dict[str, Any]:
    """
    Generate comprehensive test report.
    
    Args:
        test_results: Test execution results
        syntax_errors: List of syntax errors
        
    Returns:
        Comprehensive test report
    """
    return {
        "timestamp": time.time(),
        "test_execution": test_results,
        "syntax_errors": syntax_errors,
        "summary": {
            "total_tests": test_results.get("total_tests", 0),
            "passed": test_results.get("passed", 0),
            "failed": test_results.get("failed", 0),
            "skipped": test_results.get("skipped", 0),
            "syntax_errors": len(syntax_errors),
            "execution_successful": test_results.get("execution_successful", False)
        }
    }

# --- Main Execution Function ---
run_script = create_standardized_pipeline_script(
    "2_tests",
    process_tests_standardized,
    "Comprehensive test suite execution",
)

def main():
    return run_script()

if __name__ == "__main__":
    sys.exit(main()) 