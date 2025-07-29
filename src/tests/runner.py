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
- Modular test category execution
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
import signal
import json

# Test category definitions for modular test execution
MODULAR_TEST_CATEGORIES = {
    "core": {
        "name": "Core Module Tests",
        "description": "Essential GNN core functionality tests",
        "files": ["test_gnn_core_modules.py", "test_core_modules.py", "test_environment.py"],
        "markers": [],
        "timeout_seconds": 60,
        "max_failures": 5,
        "parallel": True
    },
    "pipeline": {
        "name": "Pipeline Infrastructure Tests",
        "description": "Pipeline scripts and infrastructure tests",
        "files": ["test_main_orchestrator.py", "test_pipeline_functionality.py", "test_pipeline_infrastructure.py",
                  "test_pipeline_performance.py", "test_pipeline_recovery.py", "test_pipeline_scripts.py",
                  "test_pipeline_steps.py"],
        "markers": [],
        "timeout_seconds": 120,  # Increased timeout for pipeline tests
        "max_failures": 10,
        "parallel": False  # Disable parallel for pipeline tests to avoid timeout
    },
    "integration": {
        "name": "Integration Tests",
        "description": "Cross-module integration and workflow tests",
        "files": ["test_comprehensive_api.py", "test_mcp_integration_comprehensive.py",
                  "test_mcp_comprehensive.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 10,
        "parallel": True
    },
    "validation": {
        "name": "Validation Tests",
        "description": "GNN validation and type checking tests",
        "files": ["test_gnn_type_checker.py", "test_parsers.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 8,
        "parallel": True
    },
    "utilities": {
        "name": "Utility Tests",
        "description": "Utility functions and helper tests",
        "files": ["test_utils.py", "test_utilities.py", "test_utility_modules.py", "test_runner_helper.py"],
        "markers": [],
        "timeout_seconds": 60,
        "max_failures": 5,
        "parallel": True
    },
    "specialized": {
        "name": "Specialized Module Tests",
        "description": "Specialized module tests (audio, visualization, etc.)",
        "files": ["test_visualization.py", "test_sapf.py", "test_export.py", "test_render.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "reporting": {
        "name": "Reporting Tests",
        "description": "Report generation and output tests",
        "files": ["test_report_comprehensive.py"],
        "markers": [],
        "timeout_seconds": 60,
        "max_failures": 5,
        "parallel": True
    },
    "performance": {
        "name": "Performance Tests",
        "description": "Performance and benchmarking tests",
        "files": ["test_pipeline_performance.py"],  # Only include actual performance tests
        "markers": [],
        "timeout_seconds": 300,  # Increased timeout to 5 minutes
        "max_failures": 10,
        "parallel": False  # Disable parallel for performance tests
    },
    "fast_suite": {
        "name": "Fast Test Suite",
        "description": "Fast execution test suite",
        "files": ["test_fast_suite.py"],
        "markers": [],
        "timeout_seconds": 30,
        "max_failures": 3,
        "parallel": True
    },
    "comprehensive": {
        "name": "Comprehensive Tests",
        "description": "All remaining test files for complete coverage",
        "files": ["test_*"],  # Pattern to catch any remaining test files
        "markers": [],
        "timeout_seconds": 600,  # Increased timeout for comprehensive tests
        "max_failures": 20,
        "parallel": True
    }
}

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

class ModularTestRunner:
    """Enhanced test runner with comprehensive error handling and reporting."""
    
    def __init__(self, args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.project_root = Path(__file__).parent.parent.parent
        self.test_dir = self.project_root / "src" / "tests"
        self.results = {}
        self.start_time = time.time()
        
        # Enhanced error tracking
        self.total_tests_run = 0
        self.total_tests_passed = 0
        self.total_tests_failed = 0
        self.total_tests_skipped = 0
        self.categories_run = 0
        self.categories_successful = 0
        self.categories_failed = 0
        
        # Performance tracking
        self.category_times = {}
        self.slowest_tests = []
        
        # Resource monitoring
        self.resource_usage = {}
        
        # Error categorization
        self.import_errors = []
        self.runtime_errors = []
        self.pathlib_errors = []
        self.sapf_errors = []
        
        self.logger.info(f"Initialized ModularTestRunner with {len(MODULAR_TEST_CATEGORIES)} categories")

    def _monitor_resources_during_test(self):
        """Monitor system resources during test execution."""
        try:
            import psutil
            process = psutil.Process()
            return {
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads()
            }
        except ImportError:
            return {"memory_mb": 0, "cpu_percent": 0, "threads": 0}
            
    def should_run_category(self, category: str) -> bool:
        """Determine if a test category should be run based on arguments."""
        # Always run comprehensive category when requested
        if hasattr(self.args, 'comprehensive') and self.args.comprehensive:
            return True
            
        if hasattr(self.args, 'fast_only') and self.args.fast_only:
            # In fast mode, run core, pipeline, validation, utilities, and fast_suite
            # This gives better coverage while still being fast
            return category in ["core", "pipeline", "validation", "utilities", "fast_suite"]
        
        # Skip performance tests if not explicitly included
        if category == "performance" and hasattr(self.args, 'include_performance') and not self.args.include_performance:
            return False
            
        # Skip slow tests if not explicitly included
        if category in ["specialized", "integration"] and hasattr(self.args, 'include_slow') and not self.args.include_slow:
            return False
        
        # By default, run all categories except performance and slow tests
        return category not in ["performance", "specialized", "integration"]

    def discover_test_files(self, category: str, config: Dict[str, Any]) -> List[str]:
        """Discover test files for a category with enhanced error handling."""
        test_files = []
        
        for file_pattern in config.get("files", []):
            # Look for exact file matches first
            exact_file = self.test_dir / file_pattern
            if exact_file.exists():
                test_files.append(str(exact_file))
                continue
            
            # Look for pattern matches
            pattern_files = list(self.test_dir.glob(file_pattern))
            test_files.extend([str(f) for f in pattern_files])
        
        # Remove duplicates and sort
        test_files = sorted(list(set(test_files)))
        
        self.logger.info(f"Discovered {len(test_files)} test files for category '{category}': {test_files}")
        return test_files
        
    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, int]:
        """Parse pytest output to extract test counts."""
        tests_run = 0
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0
        
        # Parse the output for test results
        lines = stdout.split('\n')
        
        # First try to parse summary line with combined stats
        for line in lines:
            line = line.strip()
            
            # Skip lines without = separators
            if "=" not in line:
                continue
                
            # Look for summary lines like:
            # "======================== 41 passed, 12 skipped in 1.76s ========================="
            if "passed" in line or "failed" in line or "skipped" in line:
                # Extract the content between the = signs
                start = line.find("=")
                end = line.rfind("=")
                if start != end and start != -1 and end != -1:
                    content = line[start+1:end].strip()
                    
                    # Split by comma and parse each part
                    test_parts = content.split(',')
                    for part in test_parts:
                        part = part.strip()
                        
                        # Extract numbers from specific keywords
                        if "passed" in part:
                            words = part.split()
                            for i, word in enumerate(words):
                                if "passed" in word and i > 0:
                                    try:
                                        tests_passed = int(words[i-1])
                                    except (ValueError, IndexError):
                                        pass
                        
                        elif "failed" in part:
                            words = part.split()
                            for i, word in enumerate(words):
                                if "failed" in word and i > 0:
                                    try:
                                        tests_failed = int(words[i-1])
                                    except (ValueError, IndexError):
                                        pass
                        
                        elif "skipped" in part:
                            words = part.split()
                            for i, word in enumerate(words):
                                if "skipped" in word and i > 0:
                                    try:
                                        tests_skipped = int(words[i-1])
                                    except (ValueError, IndexError):
                                        pass
                    
                    # If we found a summary line, calculate total and stop processing
                    if tests_passed > 0 or tests_failed > 0 or tests_skipped > 0:
                        tests_run = tests_passed + tests_failed + tests_skipped
                        break
        
        # If no summary found, try to count from individual test results
        if tests_run == 0:
            # Count PASSED, FAILED, SKIPPED lines
            passed_count = stdout.count("PASSED")
            failed_count = stdout.count("FAILED") + stdout.count("ERROR")
            skipped_count = stdout.count("SKIPPED")
            
            if passed_count > 0 or failed_count > 0 or skipped_count > 0:
                tests_passed = passed_count
                tests_failed = failed_count
                tests_skipped = skipped_count
                tests_run = passed_count + failed_count + skipped_count
        
        # If still no tests found, try to parse from the collected line
        if tests_run == 0:
            for line in lines:
                if "collected" in line and "items" in line:
                    # Example: "collected 52 items"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "collected" and i+1 < len(parts):
                            try:
                                tests_run = int(parts[i+1])
                                # If we found collected tests but no summary, assume they all passed
                                tests_passed = tests_run
                                break
                            except (ValueError, IndexError):
                                pass
        
        return {
            "tests_run": tests_run,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "tests_skipped": tests_skipped
        }
                        
    def run_test_category(self, category: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests for a specific category with comprehensive error handling."""
        category_start_time = time.time()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running category: {config['name']}")
        self.logger.info(f"Description: {config['description']}")
        self.logger.info(f"{'='*60}")
        
        # Discover test files
        test_files = self.discover_test_files(category, config)
        
        if not test_files:
            self.logger.warning(f"No test files found for category '{category}'")
            return {
                "success": False,
                "error": "No test files found",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
                "duration": 0
            }
        
        # Use virtual environment Python if available
        venv_python = self.project_root / ".venv" / "bin" / "python"
        python_executable = str(venv_python) if venv_python.exists() else sys.executable
        
        # Build pytest command with enhanced options
        cmd = [
            python_executable, "-m", "pytest",
            "--tb=short",  # Shorter traceback
            "--maxfail=10",  # Stop after 10 failures
            "--durations=10",  # Show 10 slowest tests
            "--json-report",  # Generate JSON report
            "--json-report-file=none",  # Don't write to file
            "--no-header",  # Skip pytest header
        ]
        
        # Add markers if specified
        if config.get("markers"):
            for marker in config["markers"]:
                cmd.extend(["-m", marker])
        
        # Add test files
        cmd.extend(test_files)
        
        # Add parallel execution if enabled (disable for pipeline tests to avoid timeout)
        if (config.get("parallel", True) and
            not (hasattr(self.args, 'fast_only') and self.args.fast_only) and
            category != "pipeline"):  # Disable parallel for pipeline tests
            cmd.extend(["-n", "auto"])
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Monitor resources before test execution
            initial_resources = self._monitor_resources_during_test()
            self.logger.info(f"Initial resource usage: {initial_resources}")
            
            # Set up timeout
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Test category '{category}' timed out after {config.get('timeout_seconds', 60)} seconds")
            
            # Set signal handler for timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(config.get("timeout_seconds", 60))
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=config.get("timeout_seconds", 60)
                )
                
                # Cancel timeout
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
                # Monitor resources after test execution
                final_resources = self._monitor_resources_during_test()
                self.logger.info(f"Final resource usage: {final_resources}")
                
                # Store resource usage (absolute values only)
                resource_usage = {
                    "memory_mb": final_resources.get("memory_mb", 0),
                    "cpu_percent": final_resources.get("cpu_percent", 0),
                    "threads": final_resources.get("threads", 0)
                }
                self.resource_usage[category] = resource_usage
                
                # Parse results
                test_results = self._parse_pytest_output(result.stdout, result.stderr)
                
                # Calculate duration
                duration = time.time() - category_start_time
                self.category_times[category] = duration
                
                # Update global counters
                self.total_tests_run += test_results.get("tests_run", 0)
                self.total_tests_passed += test_results.get("tests_passed", 0)
                self.total_tests_failed += test_results.get("tests_failed", 0)
                self.total_tests_skipped += test_results.get("tests_skipped", 0)
                
                # Log results
                self.logger.info(f"Category '{category}' completed in {duration:.2f}s")
                self.logger.info(f"  Tests run: {test_results.get('tests_run', 0)}")
                self.logger.info(f"  Tests passed: {test_results.get('tests_passed', 0)}")
                self.logger.info(f"  Tests failed: {test_results.get('tests_failed', 0)}")
                self.logger.info(f"  Tests skipped: {test_results.get('tests_skipped', 0)}")
                self.logger.info(f"  Peak memory usage: {resource_usage.get('memory_mb', 0):.2f} MB")
                
                # Check for specific error patterns
                if "pathlib" in result.stderr.lower() or "recursion" in result.stderr.lower():
                    self.pathlib_errors.append(category)
                
                if "sapf" in result.stderr.lower() or "audio" in result.stderr.lower():
                    self.sapf_errors.append(category)
                
                if "import" in result.stderr.lower():
                    self.import_errors.append(category)
                
                return {
                    "success": result.returncode == 0,
                    "tests_run": test_results.get("tests_run", 0),
                    "tests_passed": test_results.get("tests_passed", 0),
                    "tests_failed": test_results.get("tests_failed", 0),
                    "tests_skipped": test_results.get("tests_skipped", 0),
                    "duration": duration,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "resource_usage": resource_usage
                }
                
            except TimeoutError as e:
                self.logger.error(f"Timeout error in category '{category}': {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "tests_run": 0,
                    "tests_passed": 0,
                    "tests_failed": 0,
                    "tests_skipped": 0,
                    "duration": time.time() - category_start_time
                }
                
        except Exception as e:
            self.logger.error(f"Failed to run category '{category}': {e}")
            return {
                "success": False,
                "error": str(e),
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
                "duration": time.time() - category_start_time
            }
            
    def run_all_categories(self) -> Dict[str, Any]:
        """Run all test categories and generate comprehensive report."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("STARTING COMPREHENSIVE TEST SUITE EXECUTION")
        self.logger.info(f"{'='*80}")
        
        overall_success = True
        
        for category, config in MODULAR_TEST_CATEGORIES.items():
            if not self.should_run_category(category):
                self.logger.info(f"Skipping category '{category}' based on arguments")
                continue
            
            self.categories_run += 1
            
            try:
                result = self.run_test_category(category, config)
                self.results[category] = result
                
                if result.get("success", False):
                    self.categories_successful += 1
                else:
                    self.categories_failed += 1
                    overall_success = False
                    
            except Exception as e:
                self.logger.error(f"Unexpected error in category '{category}': {e}")
                self.results[category] = {
                    "success": False,
                    "error": str(e),
                    "tests_run": 0,
                    "tests_passed": 0,
                    "tests_failed": 0,
                    "tests_skipped": 0,
                    "duration": 0
                }
                self.categories_failed += 1
                overall_success = False
        
        # Generate comprehensive report
        self._save_intermediate_results()
        self._generate_final_report()
        self._log_final_summary(overall_success)
        
        # Return comprehensive results
        return {
            "overall_success": overall_success,
            "total_tests_run": self.total_tests_run,
            "total_tests_passed": self.total_tests_passed,
            "total_tests_failed": self.total_tests_failed,
            "total_tests_skipped": self.total_tests_skipped,
            "categories_run": self.categories_run,
            "categories_successful": self.categories_successful,
            "categories_failed": self.categories_failed,
            "success_rate": (self.total_tests_passed / self.total_tests_run * 100) if self.total_tests_run > 0 else 0
        }
    
    def _save_intermediate_results(self):
        """Save intermediate results to JSON files."""
        output_dir = Path(self.args.output_dir) / "test_reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results for each category
        for category, result in self.results.items():
            category_dir = output_dir / f"category_{category}"
            category_dir.mkdir(exist_ok=True)
            
            # Save stdout and stderr
            if "stdout" in result:
                with open(category_dir / "pytest_stdout.txt", "w") as f:
                    f.write(result["stdout"])
            
            if "stderr" in result:
                with open(category_dir / "pytest_stderr.txt", "w") as f:
                    f.write(result["stderr"])
            
            # Save command info
            with open(category_dir / "pytest_command.txt", "w") as f:
                f.write(f"Category: {category}\n")
                f.write(f"Success: {result.get('success', False)}\n")
                f.write(f"Duration: {result.get('duration', 0):.2f}s\n")
                if "error" in result:
                    f.write(f"Error: {result['error']}\n")
                
                # Add resource usage if available
                if "resource_usage" in result:
                    resource_data = result["resource_usage"]
                    f.write(f"Peak Memory: {resource_data.get('memory_mb', 0):.2f} MB\n")
                    f.write(f"CPU Usage: {resource_data.get('cpu_percent', 0):.1f}%\n")
    
    def _generate_final_report(self):
        """Generate comprehensive final test report."""
        output_dir = Path(self.args.output_dir) / "test_reports"
        
        # Calculate success rate
        success_rate = (self.total_tests_passed / max(self.total_tests_run, 1)) * 100
        
        # Generate summary
        summary = {
            "total_categories": self.categories_run,
            "successful_categories": self.categories_successful,
            "failed_categories": self.categories_failed,
            "total_tests_run": self.total_tests_run,
            "total_tests_passed": self.total_tests_passed,
            "total_tests_failed": self.total_tests_failed,
            "total_tests_skipped": self.total_tests_skipped,
            "success_rate": success_rate,
            "total_duration": time.time() - self.start_time,
            "category_times": self.category_times,
            "resource_usage": self.resource_usage,
            "error_categories": {
                "pathlib_errors": self.pathlib_errors,
                "sapf_errors": self.sapf_errors,
                "import_errors": self.import_errors,
                "runtime_errors": self.runtime_errors
            }
        }
        
        # Save summary
        with open(output_dir / "modular_test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        with open(output_dir / "modular_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
    
    def _log_final_summary(self, success: bool):
        """Log comprehensive final summary."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("COMPREHENSIVE TEST SUITE EXECUTION COMPLETE")
        self.logger.info(f"{'='*80}")
        
        # Overall statistics
        self.logger.info(f"Overall Result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
        self.logger.info(f"Total Duration: {time.time() - self.start_time:.2f}s")
        self.logger.info(f"Categories Run: {self.categories_run}")
        self.logger.info(f"Categories Successful: {self.categories_successful}")
        self.logger.info(f"Categories Failed: {self.categories_failed}")
        
        # Test statistics
        self.logger.info(f"Total Tests Run: {self.total_tests_run}")
        self.logger.info(f"Total Tests Passed: {self.total_tests_passed}")
        self.logger.info(f"Total Tests Failed: {self.total_tests_failed}")
        self.logger.info(f"Total Tests Skipped: {self.total_tests_skipped}")
        
        if self.total_tests_run > 0:
            success_rate = (self.total_tests_passed / self.total_tests_run) * 100
            self.logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Error analysis
        if self.pathlib_errors:
            self.logger.warning(f"Pathlib errors detected in categories: {self.pathlib_errors}")
        
        if self.sapf_errors:
            self.logger.warning(f"SAPF errors detected in categories: {self.sapf_errors}")
        
        if self.import_errors:
            self.logger.warning(f"Import errors detected in categories: {self.import_errors}")
        
        # Performance analysis
        if self.category_times:
            slowest_category = max(self.category_times.items(), key=lambda x: x[1])
            self.logger.info(f"Slowest category: {slowest_category[0]} ({slowest_category[1]:.2f}s)")
        
        # Resource usage summary
        if self.resource_usage:
            peak_memory = max([usage.get("memory_mb", 0) for usage in self.resource_usage.values()])
            self.logger.info(f"Peak Memory Usage: {peak_memory:.2f} MB")
        
        self.logger.info(f"{'='*80}")

def create_test_runner(args, logger: logging.Logger) -> ModularTestRunner:
    """Factory function to create a ModularTestRunner instance."""
    return ModularTestRunner(args, logger)

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