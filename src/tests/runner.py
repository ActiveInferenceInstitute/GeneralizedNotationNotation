"""
Enhanced Test Runner for GNN Processing Pipeline.

This module provides comprehensive test execution capabilities with:
- Staged test execution (fast, standard, slow, performance)
- Parallel test execution with resource monitoring
- Comprehensive reporting and analytics
- Graceful error handling and recovery
- Performance regression detection
- Memory usage tracking
- Coverage analysis integration

Features:
- Real-time test progress monitoring
- Resource usage tracking and limits
- Test result aggregation and reporting
- Performance baseline comparison
- Automatic test discovery and organization
- Safe-to-fail test execution
"""

import logging
import subprocess
import sys
import time
import json
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import test utilities
from .conftest import project_root
from .test_utils import TEST_DIR
from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

@dataclass
class TestExecutionConfig:
    """Configuration for test execution."""
    timeout_seconds: int = 300
    max_failures: int = 10
    parallel: bool = True
    coverage: bool = True
    verbose: bool = False
    markers: List[str] = None
    memory_limit_mb: int = 2048
    cpu_limit_percent: int = 80

@dataclass
class TestExecutionResult:
    """Results from test execution."""
    success: bool
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    execution_time: float
    memory_peak_mb: float
    coverage_percentage: Optional[float] = None
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""

class ResourceMonitor:
    """Monitor system resources during test execution."""
    
    def __init__(self, memory_limit_mb: int = 2048, cpu_limit_percent: int = 80):
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_percent = cpu_limit_percent
        self.peak_memory = 0.0
        self.peak_cpu = 0.0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_resources(self):
        """Monitor system resources in background."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, memory_mb)
                
                # CPU usage
                cpu_percent = process.cpu_percent()
                self.peak_cpu = max(self.peak_cpu, cpu_percent)
                
                # Check limits
                if memory_mb > self.memory_limit_mb:
                    logging.warning(f"⚠️ Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.memory_limit_mb}MB)")
                
                if cpu_percent > self.cpu_limit_percent:
                    logging.warning(f"⚠️ CPU usage ({cpu_percent:.1f}%) exceeds limit ({self.cpu_limit_percent}%)")
                    
                time.sleep(0.5)  # Monitor every 500ms
                
            except Exception as e:
                logging.warning(f"Resource monitoring error: {e}")
                break
                
    def get_stats(self) -> Dict[str, float]:
        """Get current resource statistics."""
        return {
            "peak_memory_mb": self.peak_memory,
            "peak_cpu_percent": self.peak_cpu,
            "current_memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "current_cpu_percent": psutil.Process().cpu_percent()
        }

class TestRunner:
    """Enhanced test runner with comprehensive monitoring and reporting."""
    
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
            cmd = self._build_pytest_command(test_paths)
            
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
    
    def _build_pytest_command(self, test_paths: List[Path]) -> List[str]:
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
            cmd.extend([
                "--cov=src",
                "--cov-report=json",
                "--cov-report=html",
                "--cov-report=term-missing"
            ])
        
        # Add parallel execution if enabled
        if self.config.parallel:
            cmd.extend(["-n", "auto"])
        
        # Add test paths
        cmd.extend([str(path) for path in test_paths])
        
        return cmd
    
    def _execute_pytest(self, cmd: List[str], output_dir: Path) -> Dict[str, Any]:
        """Execute pytest command and capture results."""
        try:
            # Create output files
            stdout_file = output_dir / "pytest_stdout.txt"
            stderr_file = output_dir / "pytest_stderr.txt"
            
            # Execute command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=project_root
            )
            
            # Capture output with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.config.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                return {
                    "success": False,
                    "tests_run": 0,
                    "tests_passed": 0,
                    "tests_failed": 0,
                    "tests_skipped": 0,
                    "error_message": f"Test execution timed out after {self.config.timeout_seconds} seconds",
                    "stdout": "",
                    "stderr": ""
                }
            
            # Save output to files
            with open(stdout_file, 'w') as f:
                f.write(stdout)
            with open(stderr_file, 'w') as f:
                f.write(stderr)
            
            # Parse results
            results = self._parse_pytest_output(stdout, stderr)
            results["stdout"] = stdout
            results["stderr"] = stderr
            
            return results
            
        except Exception as e:
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
            
            for line in lines:
                if "collected" in line and "items" in line:
                    # Extract total tests
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "collected":
                            tests_run = int(parts[i-1])
                            break
                elif "passed" in line and "failed" in line:
                    # Extract passed/failed counts
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            tests_passed = int(parts[i-1])
                        elif part == "failed":
                            tests_failed = int(parts[i-1])
                        elif part == "skipped":
                            tests_skipped = int(parts[i-1])
            
            # Determine success
            success = tests_failed == 0 and tests_run > 0
            
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
                "coverage_percentage": coverage_percentage
            }
            
        except Exception as e:
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
        
        # Save report
        report_file = output_dir / "test_execution_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def check_test_dependencies(logger: logging.Logger) -> Dict[str, Any]:
    """Check that all test dependencies are available."""
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
    
    Args:
        test_markers: List of pytest markers to include
        timeout_seconds: Maximum execution time
        max_failures: Maximum number of test failures before stopping
        parallel: Enable parallel test execution
        verbose: Enable verbose output
        generate_coverage: Generate coverage reports
        fast_only: Run only fast tests
        include_slow: Include slow tests
    
    Returns:
        List of command arguments for pytest
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
    fast_only: bool = False,
    generate_coverage: bool = True
) -> bool:
    """
    Run comprehensive test suite.
    
    Args:
        logger: Logger instance
        output_dir: Output directory for test results
        verbose: Enable verbose output
        include_slow: Include slow tests
        fast_only: Run only fast tests
        generate_coverage: Generate coverage reports
    
    Returns:
        True if tests pass, False otherwise
    """
    try:
        log_step_start(logger, "Running comprehensive test suite")
        
        # Check dependencies
        dependencies = check_test_dependencies(logger)
        if not all(dependencies.values()):
            log_step_warning(logger, "Some test dependencies missing - functionality may be limited")

        # Create test configuration
        config = TestExecutionConfig(
            timeout_seconds=600,
            max_failures=20,
            parallel=True,
            coverage=generate_coverage,
            verbose=verbose,
            markers=["fast"] if fast_only else (["not slow"] if not include_slow else None)
        )

        # Create test runner
        runner = TestRunner(config)
        
        # Run tests
        result = runner.run_tests([TEST_DIR], output_dir)
        
        # Generate report
        report = runner.generate_report(output_dir)

        # Log results
        if result.success:
            log_step_success(logger, f"Tests passed: {result.tests_passed}/{result.tests_run} tests")
        else:
            log_step_error(logger, f"Tests failed: {result.tests_failed}/{result.tests_run} tests")
            if result.error_message:
                log_step_error(logger, f"Error: {result.error_message}")
        
        return result.success

    except Exception as e:
        log_step_error(logger, f"Test execution failed: {e}")
        return False

def _parse_test_statistics(pytest_output: str) -> Dict[str, int]:
    """Parse pytest output to extract test statistics."""
    stats = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0
    }
    
    try:
        lines = pytest_output.split('\n')
        
        for line in lines:
            if "collected" in line and "items" in line:
                # Extract total tests
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "collected":
                        stats["tests_run"] = int(parts[i-1])
                        break
            elif "passed" in line and "failed" in line:
                # Extract passed/failed counts
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed":
                        stats["tests_passed"] = int(parts[i-1])
                    elif part == "failed":
                        stats["tests_failed"] = int(parts[i-1])
                    elif part == "skipped":
                        stats["tests_skipped"] = int(parts[i-1])
        
    except Exception as e:
        logging.warning(f"Failed to parse test statistics: {e}")
    
    return stats

def _parse_coverage_statistics(coverage_json_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Parse coverage JSON file to extract coverage statistics."""
    try:
        if not coverage_json_path.exists():
            return {"error": "Coverage file not found"}
        
        with open(coverage_json_path) as f:
            coverage_data = json.load(f)
        
        # Extract overall coverage
        total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
        
        # Extract per-file coverage
        files_coverage = {}
        for file_path, file_data in coverage_data.get("files", {}).items():
            files_coverage[file_path] = file_data.get("summary", {}).get("percent_covered", 0.0)
        
        return {
            "total_coverage": total_coverage,
            "files_coverage": files_coverage,
            "files_count": len(files_coverage)
        }
        
    except Exception as e:
        logger.error(f"Failed to parse coverage statistics: {e}")
        return {"error": str(e)}

def _generate_markdown_report(report_path: Path, summary: Dict[str, Any]):
    """Generate markdown test report."""
    try:
        with open(report_path, 'w') as f:
            f.write("# Test Execution Report\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            f.write(f"- **Total Tests**: {summary.get('total_tests_run', 0)}\n")
            f.write(f"- **Passed**: {summary.get('total_tests_passed', 0)}\n")
            f.write(f"- **Failed**: {summary.get('total_tests_failed', 0)}\n")
            f.write(f"- **Skipped**: {summary.get('total_tests_skipped', 0)}\n")
            f.write(f"- **Success Rate**: {summary.get('success_rate', 0):.1f}%\n")
            f.write(f"- **Execution Time**: {summary.get('total_execution_time', 0):.2f}s\n\n")
        
            # Stage results
            if 'stage_results' in summary:
                f.write("## Stage Results\n\n")
                for stage_name, stage_data in summary['stage_results'].items():
                    f.write(f"### {stage_name.title()}\n")
                    f.write(f"- **Status**: {'✅ Passed' if stage_data.get('success') else '❌ Failed'}\n")
                    f.write(f"- **Duration**: {stage_data.get('duration_seconds', 0):.2f}s\n")
                    f.write(f"- **Tests**: {stage_data.get('tests_passed', 0)}/{stage_data.get('tests_run', 0)} passed\n\n")
        
            # Performance metrics
            if 'performance_metrics' in summary:
                f.write("## Performance Metrics\n\n")
                perf = summary['performance_metrics']
                f.write(f"- **Peak Memory**: {perf.get('peak_memory_mb', 0):.1f}MB\n")
                f.write(f"- **Average Execution Time**: {perf.get('average_execution_time', 0):.2f}s\n")
                f.write(f"- **Success Rate**: {perf.get('success_rate', 0):.1f}%\n\n")
        
        logging.info(f"✅ Markdown report generated: {report_path}")
        
    except Exception as e:
        logging.error(f"Failed to generate markdown report: {e}")

def _generate_fallback_report(output_dir: Path, summary: Dict[str, Any]):
    """Generate fallback report when main report generation fails."""
    try:
        fallback_file = output_dir / "test_report_fallback.txt"
        with open(fallback_file, 'w') as f:
            f.write("Test Execution Summary\n")
            f.write("=====================\n\n")
            f.write(f"Total Tests: {summary.get('total_tests_run', 0)}\n")
            f.write(f"Passed: {summary.get('total_tests_passed', 0)}\n")
            f.write(f"Failed: {summary.get('total_tests_failed', 0)}\n")
            f.write(f"Skipped: {summary.get('total_tests_skipped', 0)}\n")
            f.write(f"Execution Time: {summary.get('total_execution_time', 0):.2f}s\n")
        
        logging.info(f"✅ Fallback report generated: {fallback_file}")
        
    except Exception as e:
        logging.error(f"Failed to generate fallback report: {e}")

def _generate_timeout_report(output_dir: Path, cmd: List[str], timeout: int):
    """Generate report for timeout scenarios."""
    try:
        timeout_file = output_dir / "test_timeout_report.txt"
        with open(timeout_file, 'w') as f:
            f.write("Test Execution Timeout\n")
            f.write("=====================\n\n")
            f.write(f"Timeout: {timeout} seconds\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logging.warning(f"⚠️ Timeout report generated: {timeout_file}")
        
    except Exception as e:
        logging.error(f"Failed to generate timeout report: {e}")

def _generate_error_report(output_dir: Path, cmd: List[str], error_msg: str):
    """Generate report for error scenarios."""
    try:
        error_file = output_dir / "test_error_report.txt"
        with open(error_file, 'w') as f:
            f.write("Test Execution Error\n")
            f.write("===================\n\n")
            f.write(f"Error: {error_msg}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logging.error(f"❌ Error report generated: {error_file}")
        
    except Exception as e:
        logging.error(f"Failed to generate error report: {e}")

def monitor_memory(logger: logging.Logger, threshold_mb: int = 2000):
    """Monitor memory usage and log warnings if threshold exceeded."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        if memory_mb > threshold_mb:
            logger.warning(f"⚠️ High memory usage: {memory_mb:.1f}MB (threshold: {threshold_mb}MB)")
            
            # Suggest garbage collection
            gc.collect()
            
            # Check again after GC
            memory_info = process.memory_info()
            memory_mb_after = memory_info.rss / 1024 / 1024
            
            if memory_mb_after < memory_mb:
                logger.info(f"✅ Memory usage reduced to {memory_mb_after:.1f}MB after garbage collection")
            else:
                logger.warning(f"⚠️ Memory usage still high: {memory_mb_after:.1f}MB")
        
        return memory_mb
        
    except Exception as e:
        logger.warning(f"Failed to monitor memory: {e}")
        return 0.0 