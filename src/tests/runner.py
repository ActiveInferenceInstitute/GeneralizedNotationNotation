import sys
import subprocess
import json
import time
from pathlib import Path
import logging
import shutil
from pipeline import get_output_dir_for_script
from utils.pipeline_template import log_step_start, log_step_success, log_step_warning, log_step_error
from utils import performance_tracker
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
    }
    try:
        import pytest
        dependencies["pytest"]["available"] = True
        dependencies["pytest"]["version"] = pytest.__version__
    except ImportError:
        logger.warning("pytest not found. Test execution will fail.")
    except Exception as e:
        logger.error(f"Error checking pytest dependency: {e}")

    # Check other dependencies
    for dep in ["pytest-cov", "pytest-json-report", "pytest-xdist"]:
        try:
            # Use a robust way to check for module availability
            __import__(dep.replace("-", "_"))
            dependencies[dep]["available"] = True
        except ImportError:
            logger.warning(f"{dep} not found. Some functionality may be limited.")
        except Exception as e:
            logger.error(f"Error checking {dep} dependency: {e}")
            
    return dependencies

def build_pytest_command(
    test_markers: list = None,
    timeout_seconds: int = 600,
    max_failures: int = 20,
    parallel: bool = True,
    verbose: bool = False,
    generate_coverage: bool = True,
    fast_only: bool = False,
    include_slow: bool = False
) -> list:
    """Build pytest command with specified configuration."""
    project_root = Path(__file__).parent.parent.parent
    venv_python = project_root / ".venv" / "bin" / "python"
    python_executable = str(venv_python) if venv_python.exists() else sys.executable
    
    # Base pytest command
    pytest_cmd = [
        python_executable, "-m", "pytest",
        "--verbose" if verbose else "--quiet",
        "--tb=short",
        f"--maxfail={max_failures}",
        "--durations=10",
        "--disable-warnings",
        f"-c{project_root}/pytest.ini",
    ]
    
    # Add marker-based test selection
    if test_markers:
        # Combine multiple markers with 'and' for proper pytest syntax
        if len(test_markers) == 1:
            pytest_cmd.extend(["-m", test_markers[0]])
        else:
            combined_markers = " and ".join(test_markers)
            pytest_cmd.extend(["-m", combined_markers])
    elif fast_only:
        pytest_cmd.extend(["-m", "fast"])
    elif not include_slow:
        pytest_cmd.extend(["-m", "not slow"])
    
    # Add parallel execution if available
    dependencies = check_test_dependencies(logging.getLogger("test_deps"))
    if parallel and dependencies["pytest-xdist"]["available"]:
        pytest_cmd.extend(["-n", "auto"])
    
    # Add coverage if requested and available
    if generate_coverage and dependencies["pytest-cov"]["available"]:
        pytest_cmd.extend([
            "--cov=src/gnn",
            "--cov=src/pipeline", 
            "--cov=src/utils",
            "--cov-report=term-missing",
            "--cov-fail-under=0",
            "--cov-branch",
        ])
    
    # Add test directory
    src_tests_dir = project_root / "src" / "tests"
    pytest_cmd.append(str(src_tests_dir))
    
    return pytest_cmd

def run_tests(logger: logging.Logger, output_dir: Path, verbose: bool = False, include_slow: bool = False, fast_only: bool = False, generate_coverage: bool = True) -> bool:
    """Execute tests using pytest with comprehensive error handling."""
    try:
        # Check if pytest is available
        if not shutil.which("pytest"):
            log_step_error(logger, "pytest executable not found. Please install pytest.")
            return False

        # Build pytest command
        cmd = ["pytest"]
        
        # Add verbosity
        if verbose:
            cmd.append("-v")

        # Handle test markers
        if fast_only:
            cmd.extend(["-m", "fast"])
        elif not include_slow:
            cmd.extend(["-m", "not slow and not performance"])
        else: # include all tests
            cmd.extend(["-m", "not performance"])

        # Add other options
        cmd.extend([
            f"--json-report --json-report-file={output_dir / 'report.json'}",
            f"--junitxml={output_dir / 'report.xml'}"
        ])

        if generate_coverage:
            cmd.extend([
                "--cov=src",
                f"--cov-report=html:{output_dir / 'coverage_html'}",
                f"--cov-report=xml:{output_dir / 'coverage.xml'}"
            ])

        # Execute pytest
        log_step_start(logger, f"Running pytest with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(output_dir.parent))

        # Log results
        if result.returncode == 0:
            log_step_success(logger, "Pytest execution completed successfully.")
        else:
            log_step_warning(logger, f"Pytest execution finished with return code {result.returncode}.")

        # Save output
        (output_dir / "pytest_stdout.txt").write_text(result.stdout)
        (output_dir / "pytest_stderr.txt").write_text(result.stderr)
        
        # Always return true to let the main script analyze results
        return True

    except Exception as e:
        log_step_error(logger, f"An unexpected error occurred in run_tests: {e}", exc_info=True)
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