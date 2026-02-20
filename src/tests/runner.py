"""
Test Runner for GNN Processing Pipeline.

This module provides comprehensive test execution capabilities for the GNN pipeline.
It implements the core test execution logic, following the thin orchestrator pattern
where 2_tests.py delegates all functionality to this module.

Architecture:
  The module provides multiple test execution modes:
  - run_tests() - Main entry point that routes to appropriate mode
  - run_fast_pipeline_tests() - Fast tests for quick pipeline validation (default)
  - run_comprehensive_tests() - All tests including slow/performance tests
  - run_fast_reliable_tests() - Essential tests fallback mode
  
  The ModularTestRunner class provides category-based execution with:
  - Resource monitoring (memory, CPU)
  - Timeout handling per category
  - Parallel execution support
  - Comprehensive error recovery

Key Features:
  - Staged test execution (fast, comprehensive, reliable)
  - Parallel test execution with resource monitoring
  - Comprehensive reporting and analytics (JSON, Markdown)
  - Graceful error handling and recovery
  - Performance regression detection
  - Memory usage tracking
  - Coverage analysis integration
  - Collection error detection (import/syntax errors)
  - Category-based test organization

Test Categories:
  Tests are organized into categories defined in MODULAR_TEST_CATEGORIES:
  - gnn, render, mcp, audio, visualization, pipeline, etc.
  Each category has its own timeout, max failures, and parallel execution settings.

Usage:
  from tests import run_tests
  from pathlib import Path
  import logging
  
  logger = logging.getLogger(__name__)
  success = run_tests(
      logger=logger,
      output_dir=Path("output/2_tests_output"),
      verbose=True,
      fast_only=True
  )

Dependencies:
  - pytest: Test framework
  - pytest-cov: Coverage analysis (optional)
  - pytest-timeout: Per-test timeouts (optional)
  - psutil: Resource monitoring (optional)
"""

import logging
import subprocess
import sys
import time
import json
import gc
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import signal
import re

# Import test categories from dedicated module
from .categories import MODULAR_TEST_CATEGORIES

# Import from infrastructure submodule (refactored components)
from .infrastructure import (
    TestExecutionConfig,
    TestExecutionResult,
    ResourceMonitor,
    TestRunner,
    PSUTIL_AVAILABLE,
    # Report generators
    _generate_markdown_report,
    _generate_fallback_report,
    _generate_timeout_report,
    _generate_error_report,
    # Utilities
    check_test_dependencies,
    build_pytest_command,
    _extract_collection_errors,
    _parse_test_statistics,
    _parse_coverage_statistics,
)

# Import test utilities
from utils.test_utils import TEST_DIR, PROJECT_ROOT
from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

# Calculate project root (don't import from conftest as it's a pytest fixture)
project_root = Path(__file__).parent.parent.parent

# psutil reference for legacy code
try:
    import psutil as _psutil  # type: ignore
except Exception:
    _psutil = None  # type: ignore

class TestRunner:
    """Test runner with comprehensive monitoring and reporting."""
    
    def __init__(self, config: TestExecutionConfig):
        self.config = config
        self.logger = logging.getLogger("test_runner")
        self.resource_monitor = ResourceMonitor(
            memory_limit_mb=config.memory_limit_mb,
            cpu_limit_percent=config.cpu_limit_percent
        )
        self.execution_history: List[TestExecutionResult] = []
        
    def run_tests(self, test_paths: List[Path], output_dir: Path) -> TestExecutionResult:
        """Execute tests with comprehensive monitoring."""
        start_time = time.time()
        
        try:
            # Start resource monitoring
            self.resource_monitor.start_monitoring()
            
            # Build pytest command
            cmd = self._build_pytest_command(test_paths, output_dir)
            
            # Execute tests
            result = self._execute_pytest(cmd, output_dir)
            
            # Stop monitoring and get stats
            self.resource_monitor.stop_monitoring()
            resource_stats = self.resource_monitor.get_stats()
            
            # Create execution result
            execution_result = TestExecutionResult(
                success=result["success"],
                tests_run=result["tests_run"],
                tests_passed=result["tests_passed"],
                tests_failed=result["tests_failed"],
                tests_skipped=result["tests_skipped"],
                execution_time=time.time() - start_time,
                memory_peak_mb=resource_stats["peak_memory_mb"],
                coverage_percentage=result.get("coverage_percentage"),
                error_message=result.get("error_message"),
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", "")
            )
            
            # Store in history
            self.execution_history.append(execution_result)
            
            return execution_result
            
        except Exception as e:
            self.resource_monitor.stop_monitoring()
            return TestExecutionResult(
                success=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                execution_time=time.time() - start_time,
                memory_peak_mb=0.0,
                error_message=str(e)
            )
    
    def _build_pytest_command(self, test_paths: List[Path], output_dir: Path) -> List[str]:
        """Build pytest command with appropriate options."""
        cmd = [
            sys.executable, "-m", "pytest",
            "--verbose",
            "--tb=short",
            f"--maxfail={self.config.max_failures}",
            "--durations=10",
            "--disable-warnings"
        ]
        
        # Add markers
        if self.config.markers:
            for marker in self.config.markers:
                cmd.extend(["-m", marker])
        
        # Add coverage if enabled
        if self.config.coverage:
            cov_json = output_dir / "coverage.json"
            cov_html = output_dir / "htmlcov"
            cmd.extend([
                "--cov=src",
                f"--cov-report=json:{cov_json}",
                f"--cov-report=html:{cov_html}",
                "--cov-report=term-missing"
            ])
        
        # Add parallel execution if enabled (disabled for now to avoid hanging)
        # if self.config.parallel:
        #     cmd.extend(["-n", "auto"])
        
        # Add test paths
        cmd.extend([str(path) for path in test_paths])
        
        return cmd
    
    def _execute_pytest(self, cmd: List[str], output_dir: Path) -> Dict[str, Any]:
        """Execute pytest command and capture results."""
        try:
            # Debug: Log what we're trying to do
            self.logger.debug(f"Creating output directory: {output_dir}")
            self.logger.debug(f"Output dir type: {type(output_dir)}")
            
            # Create output files
            stdout_file = output_dir / "pytest_stdout.txt"
            stderr_file = output_dir / "pytest_stderr.txt"
            
            # Execute command with streaming
            from utils.execution_utils import execute_command_streaming
            
            result = execute_command_streaming(
                cmd,
                cwd=project_root,
                timeout=self.config.timeout_seconds,
                print_stdout=True,
                print_stderr=True,
                capture_output=True
            )
            
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            
            # Save output to files
            with open(stdout_file, 'w') as f:
                f.write(stdout)
            with open(stderr_file, 'w') as f:
                f.write(stderr)
                
            if result["status"] == "TIMEOUT":
                 return {
                    "success": False,
                    "tests_run": 0,
                    "tests_passed": 0,
                    "tests_failed": 0,
                    "tests_skipped": 0,
                    "error_message": f"Test execution timed out after {self.config.timeout_seconds} seconds",
                    "stdout": stdout,
                    "stderr": stderr
                }
            
            # Parse results
            results = self._parse_pytest_output(stdout, stderr)
            results["stdout"] = stdout
            results["stderr"] = stderr
            
            return results
            
        except Exception as e:
            import traceback
            self.logger.error(f"Exception in _execute_pytest: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
                "error_message": str(e),
                "stdout": "",
                "stderr": ""
            }
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse pytest output to extract test statistics."""
        try:
            # Extract test counts from output
            lines = stdout.split('\n')
            tests_run = 0
            tests_passed = 0
            tests_failed = 0
            tests_skipped = 0

            # Look for the summary line at the end (e.g., "60 passed, 20 skipped in 1.85s")
            for line in reversed(lines):
                if "passed" in line or "failed" in line or "skipped" in line:
                    # Try to extract numbers from the summary line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if i > 0:  # Check previous part is a number
                            try:
                                num = int(parts[i-1])
                                if "passed" in part:
                                    tests_passed = num
                                elif "failed" in part:
                                    tests_failed = num
                                elif "skipped" in part:
                                    tests_skipped = num
                            except (ValueError, IndexError):
                                # Not a number, skip
                                continue
                    # Calculate total from summary
                    if tests_passed > 0 or tests_failed > 0:
                        tests_run = tests_passed + tests_failed + tests_skipped
                        break

            # Fallback: look for collected items line
            if tests_run == 0:
                for line in lines:
                    if "collected" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "collected" and i > 0:
                                try:
                                    tests_run = int(parts[i-1])
                                except ValueError:
                                    pass
                                break

            # Check for collection errors
            collection_errors = []
            for line in lines:
                if "ERROR collecting" in line or "ERROR: No tests collected" in line:
                    collection_errors.append(line)

            # Determine success - fail if no tests collected or collection errors
            success = tests_failed == 0 and tests_run > 0 and not collection_errors
            
            # Extract coverage if present
            coverage_percentage = None
            for line in lines:
                if "TOTAL" in line and "%" in line:
                    try:
                        coverage_percentage = float(line.split()[-1].replace('%', ''))
                    except (ValueError, IndexError):
                        pass
            
            return {
                "success": success,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "tests_skipped": tests_skipped,
                "coverage_percentage": coverage_percentage,
                "collection_errors": collection_errors
            }
            
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to parse pytest output: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
                "error_message": f"Failed to parse pytest output: {e}"
            }
    
    def generate_report(self, output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive test execution report."""
        if not self.execution_history:
            return {"error": "No test execution history available"}
        
        latest_result = self.execution_history[-1]
        
        report = {
            "execution_summary": asdict(latest_result),
            "resource_usage": self.resource_monitor.get_stats(),
            "execution_history": [asdict(result) for result in self.execution_history],
            "performance_metrics": {
                "average_execution_time": sum(r.execution_time for r in self.execution_history) / len(self.execution_history),
                "peak_memory_usage": max(r.memory_peak_mb for r in self.execution_history),
                "success_rate": sum(1 for r in self.execution_history if r.success) / len(self.execution_history) * 100
            }
        }
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report_file = output_dir / "test_execution_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def check_test_dependencies(logger: logging.Logger) -> Dict[str, Any]:
    """
    Check if required test dependencies are available.
    
    Verifies that pytest and optional dependencies (pytest-cov, pytest-timeout)
    are installed and available.
    
    Args:
        logger: Logger instance for reporting
    
    Returns:
        Dictionary with dependency status:
        {
            'pytest': bool,
            'pytest_cov': bool,
            'pytest_timeout': bool,
            'all_required': bool
        }
    """
    dependencies = {
        "pytest": False,
        "pytest-cov": False,
        "pytest-xdist": False,
        "psutil": False,
        "coverage": False
    }
    
    try:
        import pytest
        dependencies["pytest"] = True
    except ImportError:
        pass
    
    try:
        import pytest_cov
        dependencies["pytest-cov"] = True
    except ImportError:
        pass
    
    try:
        import xdist
        dependencies["pytest-xdist"] = True
    except ImportError:
        pass
    
    try:
        import psutil
        dependencies["psutil"] = True
    except ImportError:
        pass

    try:
        import coverage
        dependencies["coverage"] = True
    except ImportError:
        pass
    
    # Log results
    missing_deps = [name for name, available in dependencies.items() if not available]
    if missing_deps:
        logger.warning(f"⚠️ Missing test dependencies: {missing_deps}")
    else:
        logger.info("✅ All test dependencies available")
            
    return dependencies

def build_pytest_command(
    test_markers: List[str] = None,
    timeout_seconds: int = 600,
    max_failures: int = 20,
    parallel: bool = True,
    verbose: bool = False,
    generate_coverage: bool = True,
    fast_only: bool = False,
    include_slow: bool = False
) -> List[str]:
    """
    Build pytest command with appropriate options.
    
    Constructs a pytest command line with all necessary flags and options
    based on the provided parameters. Handles test filtering, timeout,
    coverage, and execution mode settings.
    
    Args:
        test_markers: List of pytest markers to include (e.g., ['fast', 'unit'])
        timeout_seconds: Maximum execution time per test (default: 600)
        max_failures: Maximum number of test failures before stopping (default: 20)
        parallel: Enable parallel test execution (default: True)
        verbose: Enable verbose output (default: False)
        generate_coverage: Generate coverage reports (default: True)
        fast_only: Run only fast tests, exclude slow tests (default: False)
        include_slow: Include slow tests (default: False)
    
    Returns:
        List of command arguments for subprocess.run()
    
    Example:
        cmd = build_pytest_command(
            test_markers=['fast'],
            timeout_seconds=120,
            max_failures=5,
            verbose=True,
            fast_only=True
        )
        # Returns: ['python', '-m', 'pytest', '--verbose', '--tb=short', ...]
    """
    cmd = [
        sys.executable, "-m", "pytest",
        "--verbose" if verbose else "--quiet",
        "--tb=short",
        f"--maxfail={max_failures}",
        "--durations=10",
        "--disable-warnings"
    ]
    
    # Add markers
    if test_markers:
        for marker in test_markers:
            cmd.extend(["-m", marker])
    
    # Add coverage if enabled
    if generate_coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=json",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add parallel execution if enabled
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add test path
    cmd.append(str(TEST_DIR))
    
    return cmd

def run_tests(
    logger: logging.Logger,
    output_dir: Path,
    verbose: bool = False,
    include_slow: bool = False,
    fast_only: bool = True,  # Default to fast tests for pipeline integration
    comprehensive: bool = False,
    generate_coverage: bool = False,  # Disable coverage by default for speed
    auto_fallback: bool = True  # Automatically fallback to comprehensive if no fast tests collected
) -> bool:
    """
    Run optimized test suite with improved performance and reliability.

    Args:
        logger: Logger instance
        output_dir: Output directory for test results
        verbose: Enable verbose output
        include_slow: Include slow tests (deprecated, use comprehensive)
        fast_only: Run only fast tests
        comprehensive: Run comprehensive test suite (all tests)
        generate_coverage: Generate coverage report
        auto_fallback: If fast tests collect 0 tests, automatically try comprehensive

    Returns:
        True if tests pass, False otherwise
    """
    try:
        log_step_start(logger, "Running optimized test suite")

        # Check dependencies
        dependencies = check_test_dependencies(logger)
        if not all(dependencies.values()):
            log_step_warning(logger, "Some test dependencies missing - functionality may be limited")

        # For pipeline integration, run a focused subset of tests
        if fast_only and not comprehensive:
            logger.info("🏃 Running fast pipeline test subset for quick validation")
            success = run_fast_pipeline_tests(logger, output_dir, verbose)

            # Auto-fallback: if no tests collected and fallback enabled, try comprehensive
            if not success and auto_fallback:
                if _check_zero_tests_collected(output_dir, logger):
                    logger.warning("⚠️ Fast test suite yielded 0 tests. Automatically falling back to comprehensive mode.")
                    return run_comprehensive_tests(logger, output_dir, verbose, generate_coverage)

            return success

        # For comprehensive mode, run all tests but with better timeout handling
        if comprehensive:
            logger.info("🔬 Running comprehensive test suite with enhanced monitoring")
            return run_comprehensive_tests(logger, output_dir, verbose, generate_coverage)

        # Default to fast tests with improved reliability
        logger.info("⚡ Running fast test suite with reliability improvements")
        return run_fast_reliable_tests(logger, output_dir, verbose)

    except Exception as e:
        log_step_error(logger, f"Test execution failed: {e}")
        return False


def _check_zero_tests_collected(output_dir: Path, logger: logging.Logger) -> bool:
    """Check if the test execution report shows zero tests collected."""
    try:
        summary_file = output_dir / "test_execution_report.json"
        if summary_file.exists():
            summary = json.loads(summary_file.read_text())
            tests_run = summary.get("execution_summary", {}).get("tests_run", 0)
            return tests_run == 0
    except Exception as e:
        logger.debug(f"Could not check test count: {e}")
    return False

# Re-export from test_runner_modes sub-module for backward compatibility
from .test_runner_modes import (
    run_fast_pipeline_tests,
    run_comprehensive_tests,
    run_fast_reliable_tests,
)

# Re-export from test_runner_modular sub-module for backward compatibility
from .test_runner_modular import (
    ModularTestRunner,
    create_test_runner,
    monitor_memory,
)
