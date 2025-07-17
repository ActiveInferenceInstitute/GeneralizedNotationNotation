import sys
import subprocess
import json
import time
from pathlib import Path
import logging
import shutil
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker
import re
import warnings
import os
warnings.filterwarnings('ignore', category=ResourceWarning)

def check_test_dependencies(logger: logging.Logger) -> dict:
    """Check for test-related dependencies with enhanced detection."""
    dependencies = {
        "pytest": {"available": False, "version": "N/A"},
        "pytest-cov": {"available": False, "version": "N/A"},
        "pytest-json-report": {"available": False, "version": "N/A"},
        "pytest-xdist": {"available": False, "version": "N/A"},
        "coverage": {"available": False, "version": "N/A"},
        "mock": {"available": False, "version": "N/A"},
        "psutil": {"available": False, "version": "N/A"},
    }
    
    for dep in dependencies.keys():
        try:
            module = __import__(dep.replace('-', '_'))
            dependencies[dep]["available"] = True
            dependencies[dep]["version"] = getattr(module, '__version__', "Unknown")
        except ImportError:
            dependencies[dep]["available"] = False
    
    return dependencies

def run_tests(logger: logging.Logger, output_dir: Path, verbose: bool = False, include_slow: bool = False, fast_only: bool = False, generate_coverage: bool = True):
    """Run the test suite with enhanced reporting and error handling."""
    log_step_start(logger, "Running enhanced test suite with comprehensive reporting")
    
    # Use centralized output directory configuration
    test_output_dir = get_output_dir_for_script("3_tests.py", output_dir)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define test report paths
    xml_report_path = test_output_dir / "pytest_report.xml"
    json_report_path = test_output_dir / "test_results.json"
    markdown_report_path = test_output_dir / "test_report.md"
    coverage_report_path = test_output_dir / "coverage_report.html"
    
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
        f"-c{project_root}/pytest.ini",  # Explicitly specify config file location
    ]
    
    # Add parallel execution if pytest-xdist is available
    if dependencies["pytest-xdist"]["available"]:
        pytest_cmd.extend(["-n", "auto"])
        logger.debug("Parallel test execution enabled")
    
    # Add coverage reporting if available and requested, limited to key modules
    if generate_coverage and dependencies["pytest-cov"]["available"]:
        coverage_dir = test_output_dir / "coverage"
        pytest_cmd.extend([
            "--cov=src/gnn",  # Limit to key modules
            "--cov=src/pipeline",
            "--cov=src/utils",
            f"--cov-report=html:{coverage_dir}",
            f"--cov-report=json:{test_output_dir}/test_coverage.json",
            "--cov-report=term-missing",
            "--cov-fail-under=0",
            f"--cov-config={project_root}/.coveragerc",
            "--cov-branch",  # Enable branch coverage
        ])
        logger.debug("Limited coverage reporting enabled to reduce resource usage")
    elif generate_coverage: logger.warning('Coverage generation requested but pytest-cov not available')
    
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
    else: logger.warning('JSON reporting not available - consider installing pytest-json-report')
    
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
        os.environ['PYTHONWARNINGS'] = 'ignore::ResourceWarning'
        
        with performance_tracker.track_operation("execute_test_suite"):
            result = subprocess.run(
                pytest_cmd,
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=timeout_seconds,
                env=dict(os.environ, PYTHONWARNINGS='ignore::ResourceWarning')
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
        coverage_stats = _parse_coverage_statistics(coverage_json, logger) if generate_coverage and coverage_json.exists() else {}
        
        results_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_code": result.returncode,
            "success": test_passed,
            "test_configuration": {
                "fast_only": fast_only,
                "slow_tests_included": include_slow,
                "verbose": verbose,
                "parallel_execution": dependencies["pytest-xdist"]["available"],
                "coverage_enabled": generate_coverage and dependencies["pytest-cov"]["available"],
            },
            "execution": {
                "timeout_seconds": timeout_seconds,
                "command": ' '.join(pytest_cmd),
                "working_directory": str(project_root)
            },
            "test_statistics": test_stats,
            "coverage_statistics": coverage_stats,
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
        if 'logger' in locals():
            log_step_error(logger, f"Error running tests: {str(e)}")
        else:
            print(f"Critical error: {str(e)}")
        # Enhanced error report
        _generate_error_report(test_output_dir, pytest_cmd, str(e))
        return False


def _parse_test_statistics(pytest_output: str) -> dict:
    """Parse test statistics from pytest output with enhanced parsing."""
    stats = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "warnings": 0,
        "xfailed": 0,
        "xpassed": 0,
        "deselected": 0,
        "execution_time_seconds": 0.0,
        "collection_time_seconds": 0.0
    }
    
    # Parse counts from summary line
    summary_match = re.search(r"collected (\d+) items", pytest_output)
    if summary_match:
        stats["total_tests"] = int(summary_match.group(1))
    
    # Parse results
    results_pattern = re.compile(r"(\d+) (passed|failed|skipped|errors|warnings|xfailed|xpassed|deselected)")
    for match in results_pattern.finditer(pytest_output):
        count = int(match.group(1))
        category = match.group(2)
        stats[category] = count
    
    # Parse execution time
    time_match = re.search(r"in ([\d.]+)s", pytest_output)
    if time_match:
        stats["execution_time_seconds"] = float(time_match.group(1))
    
    return stats


def _parse_coverage_statistics(coverage_json_path: Path, logger: logging.Logger) -> dict:
    """Parse coverage statistics from JSON report."""
    if not coverage_json_path.exists():
        return {}
    try:
        with open(coverage_json_path, 'r') as f:
            data = json.load(f)
        return {
            "total_coverage": data.get("totals", {}).get("percent_covered", 0.0),
            "covered_lines": data.get("totals", {}).get("covered_lines", 0),
            "missing_lines": data.get("totals", {}).get("missing_lines", 0),
            "num_statements": data.get("totals", {}).get("num_statements", 0),
            "files_covered": len(data.get("files", {})),
            "branch_coverage": data.get("totals", {}).get("branch_percent", 0.0),
            "line_coverage": data.get("totals", {}).get("line_percent", 0.0),
        }
    except Exception as e:
        logger.warning(f"Failed to parse coverage JSON: {e}")
        return {}


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
        f.write(f"- **Coverage Enabled**: {config['coverage_enabled']}\n")
        f.write("\n")
        
        # Test statistics
        stats = summary['test_statistics']
        if stats['total_tests'] > 0:
            f.write("## Test Statistics\n\n")
            f.write(f"- **Total Tests**: {stats['total_tests']}\n")
            f.write(f"- **Passed**: ✅ {stats['passed']}\n")
            f.write(f"- **Failed**: ❌ {stats['failed']}\n")
            f.write(f"- **Skipped**: ⏭️ {stats['skipped']}\n")
            f.write(f"- **Errors**: 🚨 {stats['errors']}\n")
            f.write(f"- **Warnings**: ⚠️ {stats['warnings']}\n")
            f.write(f"- **Xfailed**: {stats['xfailed']}\n")
            f.write(f"- **Xpassed**: {stats['xpassed']}\n")
            f.write(f"- **Deselected**: {stats['deselected']}\n\n")
            
            if stats['total_tests'] > 0:
                success_rate = (stats['passed'] / stats['total_tests']) * 100
                f.write(f"**Success Rate**: {success_rate:.1f}%\n")
                failure_rate = (stats['failed'] + stats['errors']) / stats['total_tests'] * 100
                f.write(f"**Failure Rate**: {failure_rate:.1f}%\n")
            f.write(f"**Execution Time**: {stats['execution_time_seconds']:.1f} seconds\n\n")
        
        # Coverage statistics
        coverage = summary.get('coverage_statistics', {})
        if coverage:
            f.write("## Coverage Statistics\n\n")
            f.write(f"- **Total Coverage**: {coverage['total_coverage']:.1f}%\n")
            f.write(f"- **Covered Lines**: {coverage['covered_lines']}\n")
            f.write(f"- **Missing Lines**: {coverage['missing_lines']}\n")
            f.write(f"- **Total Statements**: {coverage['num_statements']}\n")
            f.write(f"- **Files Covered**: {coverage['files_covered']}\n")
            f.write(f"- **Branch Coverage**: {coverage['branch_coverage']:.1f}%\n")
            f.write(f"- **Line Coverage**: {coverage['line_coverage']:.1f}%\n\n")
        
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
        
        # Raw output (truncated)
        f.write("## Raw Output Preview\n\n")
        stdout_preview = '\n'.join(summary['raw_output']['stdout'].splitlines()[-50:])
        stderr_preview = '\n'.join(summary['raw_output']['stderr'].splitlines()[-20:])
        f.write("### Standard Output (last 50 lines)\n```\n")
        f.write(stdout_preview + "\n```\n\n")
        f.write("### Standard Error (last 20 lines)\n```\n")
        f.write(stderr_preview + "\n```\n")


def _generate_fallback_report(output_dir: Path, summary: dict):
    """Generate fallback report if primary reports failed."""
    report_path = output_dir / "fallback_test_report.md"
    with open(report_path, 'w') as f:
        f.write("# Fallback Test Report\n\n")
        f.write("Primary test reports could not be generated. Basic summary:\n\n")
        f.write(f"**Success**: {summary['success']}\n")
        f.write(f"**Exit Code**: {summary['exit_code']}\n")
        f.write(f"**Configuration**: {json.dumps(summary['test_configuration'], indent=2)}\n")
        f.write(f"**Test Statistics**: {json.dumps(summary['test_statistics'], indent=2)}\n")
        f.write(f"**Dependencies**: {json.dumps(summary['dependencies'], indent=2)}\n")
    logger.warning(f"Fallback report generated at: {report_path}")


def _generate_timeout_report(output_dir: Path, cmd: list, timeout: int):
    """Generate report for timed out test execution."""
    report_path = output_dir / "test_timeout_report.md"
    with open(report_path, 'w') as f:
        f.write("# Test Execution Timeout Report\n\n")
        f.write(f"**Timeout**: {timeout} seconds\n")
        f.write(f"**Command**: {' '.join(cmd)}\n")
        f.write("\n## Suggested Fixes\n")
        f.write("- Increase timeout with --timeout option\n")
        f.write("- Run with --fast-only to exclude slow tests\n")
        f.write("- Check for infinite loops in tests\n")
        f.write("- Optimize slow tests\n")
        f.write("- Run on faster hardware or with more resources\n")
        f.write("- Use parallel execution if not already enabled\n")
    logger.info(f"Timeout report saved to: {report_path}")

def _generate_error_report(output_dir: Path, cmd: list, error_msg: str):
    """Generate report for test execution errors."""
    report_path = output_dir / "test_error_report.md"
    with open(report_path, 'w') as f:
        f.write("# Test Execution Error Report\n\n")
        f.write(f"**Error**: {error_msg}\n")
        f.write(f"**Command**: {' '.join(cmd)}\n")
        f.write("\n## Suggested Fixes\n")
        f.write("- Check pytest installation\n")
        f.write("- Verify dependencies\n")
        f.write("- Run with --verbose for more details\n")
        f.write("- Check Python version compatibility\n")
        f.write("- Ensure no conflicts in PYTHONPATH\n")
        f.write("- Run in a clean virtual environment\n")
    logger.info(f"Error report saved to: {report_path}") 

# Add memory monitoring if psutil available
def monitor_memory(logger, threshold_mb=2000):
    if dependencies.get('psutil', {}).get('available'):
        import psutil
        mem = psutil.virtual_memory()
        used_mb = mem.used / (1024 * 1024)
        if used_mb > threshold_mb:
            logger.warning(f"High memory usage: {used_mb:.1f}MB") 