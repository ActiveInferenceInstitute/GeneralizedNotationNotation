import sys
import subprocess
import json
import time
from pathlib import Path
import logging
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error

def run_tests(logger: logging.Logger, output_dir: Path, verbose: bool = False, include_slow: bool = False, fast_only: bool = False):
    """Run the test suite and save results."""
    log_step_start(logger, "Running test suite")
    
    # Use centralized output directory configuration
    test_output_dir = get_output_dir_for_script("3_tests.py", output_dir)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define test report paths
    xml_report_path = test_output_dir / "pytest_report.xml"
    json_report_path = test_output_dir / "test_results.json"
    
    # Prepare pytest command with optimized settings
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        "--verbose" if verbose else "--quiet",
        "--tb=short",
        f"--junitxml={xml_report_path}",
        "--maxfail=10",  # Stop after 10 failures to prevent hanging
        "--durations=10",  # Show 10 slowest tests
        "--disable-warnings",  # Reduce noise
    ]
    
    # Add coverage only if explicitly requested or in verbose mode
    if verbose:
        pytest_cmd.extend([
            "--cov=src",
            f"--cov-report=html:{output_dir}/coverage",
            f"--cov-report=json:{output_dir}/test_coverage.json",
            "--cov-report=term-missing",
        ])
    
    # Test selection logic - Improved to run all tests by default
    if fast_only:
        # Run only fast tests when explicitly requested
        pytest_cmd.extend(["-m", "fast"])
        pytest_cmd.append("src/tests/test_fast_suite.py")
        logger.info("Running fast test suite only")
    elif not include_slow:
        # Run all tests except slow ones (default behavior)
        pytest_cmd.extend(["-m", "not slow"])
        pytest_cmd.append("src/tests/")
        logger.info("Running all tests except slow tests (use --include-slow to include them)")
    else:
        # Run all tests including slow ones
        pytest_cmd.append("src/tests/")
        logger.info("Running all tests including slow tests")
    
    # Try to add JSON reporting if the plugin is available
    try:
        # Test if pytest-json-report is available
        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if "--json-report" in test_result.stdout:
            pytest_cmd.extend([
                "--json-report",
                f"--json-report-file={json_report_path}"
            ])
            logger.debug("JSON reporting enabled")
        else:
            logger.debug("JSON reporting not available (pytest-json-report not installed)")
    except Exception as e:
        logger.debug(f"Could not check for JSON reporting support: {e}")
    
    logger.info(f"Running command: {' '.join(pytest_cmd)}")
    
    try:
        # Run pytest with appropriate timeout
        if fast_only:
            timeout_seconds = 60  # 1 minute for fast tests
        elif include_slow:
            timeout_seconds = 300  # 5 minutes for slow tests
        else:
            timeout_seconds = 120  # 2 minutes for regular tests
            
        result = subprocess.run(
            pytest_cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Project root
            timeout=timeout_seconds
        )
        
        # Log results
        if result.returncode == 0:
            log_step_success(logger, "All tests passed")
        else:
            log_step_warning(logger, f"Some tests failed (exit code: {result.returncode})")
        
        # Log output
        if verbose and result.stdout:
            logger.info("Test output:")
            for line in result.stdout.splitlines():
                logger.info(f"  {line}")
        
        if result.stderr:
            logger.warning("Test stderr:")
            for line in result.stderr.splitlines():
                logger.warning(f"  {line}")
        
        # Save detailed results
        coverage_dir = output_dir / "coverage"
        coverage_json = output_dir / "test_coverage.json"
        
        results_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_code": result.returncode,
            "success": result.returncode == 0,
            "fast_only": fast_only,
            "slow_tests_included": include_slow,
            "timeout_seconds": timeout_seconds,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "xml_report": str(xml_report_path) if xml_report_path.exists() else None,
            "json_report": str(json_report_path) if json_report_path.exists() else None,
            "coverage": {
                "html_dir": str(coverage_dir) if coverage_dir.exists() else None,
                "json_file": str(coverage_json) if coverage_json.exists() else None
            }
        }
        
        summary_file = test_output_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.debug(f"Test summary saved to: {summary_file}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        log_step_error(logger, f"Test execution timed out after {timeout_seconds} seconds")
        return False
    except Exception as e:
        log_step_error(logger, f"Error running tests: {e}")
        return False 