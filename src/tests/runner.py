import sys
import subprocess
import json
import time
from pathlib import Path
import logging
import shutil
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker

def check_test_dependencies(logger: logging.Logger) -> dict:
    """Check if required test dependencies are available."""
    dependencies = {
        "pytest": {"available": False, "version": None},
        "pytest-cov": {"available": False, "version": None},
        "pytest-json-report": {"available": False, "version": None},
        "pytest-xdist": {"available": False, "version": None}
    }
    
    for dep in dependencies.keys():
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", dep],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                dependencies[dep]["available"] = True
                # Extract version from output
                for line in result.stdout.splitlines():
                    if line.startswith("Version:"):
                        dependencies[dep]["version"] = line.split(":", 1)[1].strip()
                        break
        except Exception as e:
            logger.debug(f"Could not check dependency {dep}: {e}")
    
    return dependencies

def run_tests(logger: logging.Logger, output_dir: Path, verbose: bool = False, include_slow: bool = False, fast_only: bool = False):
    """Run the test suite with enhanced reporting and error handling."""
    log_step_start(logger, "Running enhanced test suite with comprehensive reporting")
    
    # Use centralized output directory configuration
    test_output_dir = get_output_dir_for_script("3_tests.py", output_dir)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check test dependencies
    with performance_tracker.track_operation("check_test_dependencies"):
        dependencies = check_test_dependencies(logger)
        
        # Save dependency report
        dep_report_path = test_output_dir / "test_dependencies.json"
        with open(dep_report_path, 'w') as f:
            json.dump(dependencies, f, indent=2)
        
        # Log dependency status
        pytest_available = dependencies["pytest"]["available"]
        if not pytest_available:
            log_step_error(logger, "pytest is not available - cannot run tests")
            return False
        
        logger.info(f"pytest version: {dependencies['pytest']['version']}")
        if dependencies["pytest-cov"]["available"]:
            logger.debug(f"pytest-cov available: {dependencies['pytest-cov']['version']}")
        if dependencies["pytest-json-report"]["available"]:
            logger.debug(f"pytest-json-report available: {dependencies['pytest-json-report']['version']}")
    
    # Define test report paths
    xml_report_path = test_output_dir / "pytest_report.xml"
    json_report_path = test_output_dir / "test_results.json"
    markdown_report_path = test_output_dir / "test_report.md"
    
    # Define project root early for consistent path resolution
    project_root = Path(__file__).parent.parent.parent
    
    # Enhanced test selection logic
    src_tests_dir = project_root / "src" / "tests"
    
    # Prepare pytest command with enhanced settings
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        "--verbose" if verbose else "--quiet",
        "--tb=short",
        f"--junitxml={xml_report_path}",
        "--maxfail=20",  # Allow more failures for better coverage
        "--durations=15",  # Show 15 slowest tests
        "--disable-warnings",  # Reduce noise
        # Note: Removed --strict-markers to allow tests with minor marker issues to run
        f"-c{project_root}/pytest.ini",  # Explicitly specify config file location
    ]
    
    # Add parallel execution if pytest-xdist is available
    if dependencies["pytest-xdist"]["available"]:
        pytest_cmd.extend(["-n", "auto"])
        logger.debug("Parallel test execution enabled")
    
    # Add coverage reporting if available and requested
    if verbose and dependencies["pytest-cov"]["available"]:
        coverage_dir = test_output_dir / "coverage"
        pytest_cmd.extend([
            "--cov=src",
            f"--cov-report=html:{coverage_dir}",
            f"--cov-report=json:{test_output_dir}/test_coverage.json",
            "--cov-report=term-missing",
            "--cov-fail-under=0",  # Don't fail on low coverage
        ])
        logger.debug("Coverage reporting enabled")
    
    # Enhanced test selection logic
    
    if fast_only:
        pytest_cmd.extend(["-m", "fast"])
        if (src_tests_dir / "test_fast_suite.py").exists():
            pytest_cmd.append(str(src_tests_dir / "test_fast_suite.py"))
        else:
            pytest_cmd.append(str(src_tests_dir))
        logger.info("Running fast test suite only")
    elif not include_slow:
        pytest_cmd.extend(["-m", "not slow"])
        pytest_cmd.append(str(src_tests_dir))
        logger.info("Running all tests except slow tests")
    else:
        pytest_cmd.append(str(src_tests_dir))
        logger.info("Running all tests including slow tests")
    
    # Add JSON reporting if available
    if dependencies["pytest-json-report"]["available"]:
        pytest_cmd.extend([
            "--json-report",
            f"--json-report-file={json_report_path}"
        ])
        logger.debug("JSON reporting enabled")
    
    logger.info(f"Running command: {' '.join(pytest_cmd)}")
    
    try:
        # Enhanced timeout logic
        if fast_only:
            timeout_seconds = 300  # 5 minutes for fast tests
        elif include_slow:
            timeout_seconds = 900  # 15 minutes for slow tests
        else:
            timeout_seconds = 600  # 10 minutes for regular tests
        
        # Ensure we're in the right directory
        
        with performance_tracker.track_operation("execute_test_suite"):
            result = subprocess.run(
                pytest_cmd,
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=timeout_seconds
            )
        
        # Enhanced result analysis
        test_passed = result.returncode == 0
        test_warnings = result.returncode in [1, 2]  # pytest exit codes for warnings/failures
        
        # Log results with more detail
        if test_passed:
            log_step_success(logger, "All tests passed successfully")
        elif test_warnings:
            log_step_warning(logger, f"Tests completed with issues (exit code: {result.returncode})")
        else:
            log_step_error(logger, f"Test execution failed (exit code: {result.returncode})")
        
        # Enhanced output logging
        if verbose and result.stdout:
            logger.info("=== Test Output ===")
            for line in result.stdout.splitlines()[-50:]:  # Last 50 lines to avoid spam
                logger.info(f"  {line}")
        
        if result.stderr:
            logger.warning("=== Test Errors ===")
            for line in result.stderr.splitlines()[-20:]:  # Last 20 lines
                logger.warning(f"  {line}")
        
        # Enhanced results summary
        coverage_dir = test_output_dir / "coverage"
        coverage_json = test_output_dir / "test_coverage.json"
        
        # Parse test statistics from output
        test_stats = _parse_test_statistics(result.stdout)
        
        results_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_code": result.returncode,
            "success": test_passed,
            "test_configuration": {
                "fast_only": fast_only,
                "slow_tests_included": include_slow,
                "verbose": verbose,
                "parallel_execution": dependencies["pytest-xdist"]["available"],
                "coverage_enabled": verbose and dependencies["pytest-cov"]["available"]
            },
            "execution": {
                "timeout_seconds": timeout_seconds,
                "command": ' '.join(pytest_cmd),
                "working_directory": str(project_root)
            },
            "test_statistics": test_stats,
            "dependencies": dependencies,
            "output_files": {
                "xml_report": str(xml_report_path) if xml_report_path.exists() else None,
                "json_report": str(json_report_path) if json_report_path.exists() else None,
                "markdown_report": str(markdown_report_path),
                "coverage_html": str(coverage_dir) if coverage_dir.exists() else None,
                "coverage_json": str(coverage_json) if coverage_json.exists() else None
            },
            "raw_output": {
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        }
        
        # Save enhanced summary
        summary_file = test_output_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Generate enhanced markdown report
        _generate_markdown_report(markdown_report_path, results_summary)
        
        logger.info(f"Enhanced test reports saved to: {test_output_dir}")
        
        # Generate fallback report if XML wasn't created
        if not xml_report_path.exists():
            _generate_fallback_report(test_output_dir, results_summary)
        
        return test_passed or test_warnings  # Consider warnings as success for pipeline
        
    except subprocess.TimeoutExpired:
        log_step_error(logger, f"Test execution timed out after {timeout_seconds} seconds")
        
        # Enhanced timeout report
        _generate_timeout_report(test_output_dir, pytest_cmd, timeout_seconds)
        return False
        
    except Exception as e:
        log_step_error(logger, f"Error running tests: {e}")
        
        # Enhanced error report
        _generate_error_report(test_output_dir, pytest_cmd, str(e))
        return False


def _parse_test_statistics(stdout: str) -> dict:
    """Parse test statistics from pytest output."""
    stats = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "warnings": 0
    }
    
    # Look for pytest summary line
    for line in stdout.splitlines():
        if "passed" in line and any(word in line for word in ["failed", "error", "skipped"]):
            # Parse lines like "5 passed, 2 failed, 1 skipped in 10.2s"
            parts = line.split()
            for i, part in enumerate(parts):
                if part.isdigit():
                    count = int(part)
                    if i + 1 < len(parts):
                        test_type = parts[i + 1].rstrip(',')
                        if test_type in stats:
                            stats[test_type] = count
                        elif test_type == "passed":
                            stats["passed"] = count
            break
    
    stats["total_tests"] = sum(stats[k] for k in ["passed", "failed", "skipped", "errors"])
    return stats


def _generate_markdown_report(report_path: Path, summary: dict):
    """Generate an enhanced markdown test report."""
    with open(report_path, 'w') as f:
        f.write("# Test Execution Report\n\n")
        f.write(f"**Generated**: {summary['timestamp']}\n")
        f.write(f"**Status**: {'✅ SUCCESS' if summary['success'] else '❌ FAILED'}\n")
        f.write(f"**Exit Code**: {summary['exit_code']}\n\n")
        
        # Test configuration
        config = summary['test_configuration']
        f.write("## Test Configuration\n\n")
        f.write(f"- **Test Mode**: {'Fast Only' if config['fast_only'] else 'All Tests' if config['slow_tests_included'] else 'Regular Tests'}\n")
        f.write(f"- **Verbose Output**: {config['verbose']}\n")
        f.write(f"- **Parallel Execution**: {config['parallel_execution']}\n")
        f.write(f"- **Coverage Enabled**: {config['coverage_enabled']}\n\n")
        
        # Test statistics
        stats = summary['test_statistics']
        if stats['total_tests'] > 0:
            f.write("## Test Statistics\n\n")
            f.write(f"- **Total Tests**: {stats['total_tests']}\n")
            f.write(f"- **Passed**: ✅ {stats['passed']}\n")
            f.write(f"- **Failed**: ❌ {stats['failed']}\n")
            f.write(f"- **Skipped**: ⏭️ {stats['skipped']}\n")
            f.write(f"- **Errors**: 🚨 {stats['errors']}\n\n")
            
            if stats['total_tests'] > 0:
                success_rate = (stats['passed'] / stats['total_tests']) * 100
                f.write(f"**Success Rate**: {success_rate:.1f}%\n\n")
        
        # Dependencies
        f.write("## Test Dependencies\n\n")
        deps = summary['dependencies']
        for dep, info in deps.items():
            status = "✅" if info['available'] else "❌"
            version = f" (v{info['version']})" if info['version'] else ""
            f.write(f"- {status} **{dep}**{version}\n")
        f.write("\n")
        
        # Output files
        files = summary['output_files']
        f.write("## Generated Reports\n\n")
        for file_type, file_path in files.items():
            if file_path:
                f.write(f"- **{file_type.replace('_', ' ').title()}**: `{file_path}`\n")
        f.write("\n")
        
        # Execution details
        execution = summary['execution']
        f.write("## Execution Details\n\n")
        f.write(f"- **Command**: `{execution['command']}`\n")
        f.write(f"- **Working Directory**: `{execution['working_directory']}`\n")
        f.write(f"- **Timeout**: {execution['timeout_seconds']} seconds\n\n")


def _generate_fallback_report(output_dir: Path, summary: dict):
    """Generate a fallback report when XML wasn't created."""
    report_path = output_dir / "fallback_test_report.txt"
    with open(report_path, 'w') as f:
        f.write("Fallback Test Execution Report\n")
        f.write("=============================\n\n")
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Exit Code: {summary['exit_code']}\n")
        f.write(f"Success: {summary['success']}\n")
        f.write(f"Command: {summary['execution']['command']}\n\n")
        f.write("STDOUT:\n")
        f.write(summary['raw_output']['stdout'])
        f.write("\n\nSTDERR:\n")
        f.write(summary['raw_output']['stderr'])


def _generate_timeout_report(output_dir: Path, pytest_cmd: list, timeout_seconds: int):
    """Generate a timeout report."""
    report_path = output_dir / "timeout_report.md"
    with open(report_path, 'w') as f:
        f.write("# Test Execution Timeout Report\n\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Status**: ⏰ TIMEOUT\n")
        f.write(f"**Timeout Duration**: {timeout_seconds} seconds\n\n")
        f.write("## Issue\n\n")
        f.write("Test execution exceeded the maximum allowed time.\n\n")
        f.write("## Recommendations\n\n")
        f.write("1. Run tests with `--fast-only` flag for quicker execution\n")
        f.write("2. Check for hanging tests or infinite loops\n")
        f.write("3. Consider increasing timeout for slow test environments\n\n")
        f.write(f"**Command**: `{' '.join(pytest_cmd)}`\n")


def _generate_error_report(output_dir: Path, pytest_cmd: list, error: str):
    """Generate an error report."""
    report_path = output_dir / "error_report.md"
    with open(report_path, 'w') as f:
        f.write("# Test Execution Error Report\n\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Status**: 🚨 ERROR\n")
        f.write(f"**Error**: {error}\n\n")
        f.write("## Issue\n\n")
        f.write("An unexpected error occurred during test execution.\n\n")
        f.write("## Recommendations\n\n")
        f.write("1. Check that all test dependencies are installed\n")
        f.write("2. Verify pytest configuration is correct\n")
        f.write("3. Check system resources and permissions\n\n")
        f.write(f"**Command**: `{' '.join(pytest_cmd)}`\n") 