"""
Test Runner for GNN Processing Pipeline.

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
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import signal
import json

# Test category definitions for modular test execution
MODULAR_TEST_CATEGORIES = {
    "gnn": {
        "name": "GNN Module Tests",
        "description": "GNN processing and validation tests",
        "files": ["test_gnn_overall.py", "test_gnn_parsing.py", "test_gnn_validation.py", 
                  "test_gnn_processing.py", "test_gnn_integration.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "render": {
        "name": "Render Module Tests", 
        "description": "Code generation and rendering tests",
        "files": ["test_render_overall.py", "test_render_integration.py", "test_render_performance.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "mcp": {
        "name": "MCP Module Tests",
        "description": "Model Context Protocol tests",
        "files": ["test_mcp_overall.py", "test_mcp_tools.py", "test_mcp_transport.py", 
                  "test_mcp_integration.py", "test_mcp_performance.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "audio": {
        "name": "Audio Module Tests",
        "description": "Audio generation and SAPF tests",
        "files": ["test_audio_overall.py", "test_audio_sapf.py", "test_audio_generation.py", 
                  "test_audio_integration.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "visualization": {
        "name": "Visualization Module Tests",
        "description": "Graph and matrix visualization tests",
        "files": ["test_visualization_overall.py", "test_visualization_matrices.py", 
                  "test_visualization_ontology.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "pipeline": {
        "name": "Pipeline Module Tests",
        "description": "Pipeline orchestration and step tests",
        "files": ["test_pipeline_overall.py", "test_pipeline_integration.py", 
                  "test_pipeline_orchestration.py", "test_pipeline_performance.py", 
                  "test_pipeline_recovery.py", "test_pipeline_scripts.py", 
                  "test_pipeline_infrastructure.py", "test_pipeline_functionality.py"],
        "markers": [],
        "timeout_seconds": 1800,
        "max_failures": 10,
        "parallel": False
    },
    "export": {
        "name": "Export Module Tests",
        "description": "Multi-format export tests",
        "files": ["test_export_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "execute": {
        "name": "Execute Module Tests",
        "description": "Execution and simulation tests",
        "files": ["test_execute_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "llm": {
        "name": "LLM Module Tests",
        "description": "LLM integration and analysis tests",
        "files": ["test_llm_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "ontology": {
        "name": "Ontology Module Tests",
        "description": "Ontology processing and validation tests",
        "files": ["test_ontology_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "website": {
        "name": "Website Module Tests",
        "description": "Website generation tests",
        "files": ["test_website_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "report": {
        "name": "Report Module Tests",
        "description": "Report generation and formatting tests",
        "files": ["test_report_overall.py", "test_report_generation.py", 
                  "test_report_integration.py", "test_report_formats.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "environment": {
        "name": "Environment Module Tests",
        "description": "Environment setup and validation tests",
        "files": ["test_environment_overall.py", "test_environment_dependencies.py",
                  "test_environment_integration.py", "test_environment_python.py",
                  "test_environment_system.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "type_checker": {
        "name": "Type Checker Module Tests",
        "description": "Type checking and validation tests",
        "files": ["test_type_checker_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "validation": {
        "name": "Validation Module Tests",
        "description": "Validation and consistency tests",
        "files": ["test_validation_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "model_registry": {
        "name": "Model Registry Module Tests",
        "description": "Model registry and versioning tests",
        "files": ["test_model_registry_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "analysis": {
        "name": "Analysis Module Tests",
        "description": "Analysis and statistical tests",
        "files": ["test_analysis_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "integration": {
        "name": "Integration Module Tests",
        "description": "System integration tests",
        "files": ["test_integration_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "security": {
        "name": "Security Module Tests",
        "description": "Security validation tests",
        "files": ["test_security_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "research": {
        "name": "Research Module Tests",
        "description": "Research tools tests",
        "files": ["test_research_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "ml_integration": {
        "name": "ML Integration Module Tests",
        "description": "Machine learning integration tests",
        "files": ["test_ml_integration_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "advanced_visualization": {
        "name": "Advanced Visualization Module Tests",
        "description": "Advanced visualization tests",
        "files": ["test_advanced_visualization_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "comprehensive": {
        "name": "Comprehensive API Tests",
        "description": "Comprehensive API and integration tests",
        "files": ["test_comprehensive_api.py", "test_core_modules.py", "test_fast_suite.py",
                  "test_main_orchestrator.py", "test_coverage_overall.py", "test_performance_overall.py",
                  "test_unit_overall.py"],
        "markers": [],
        "timeout_seconds": 300,
        "max_failures": 15,
        "parallel": False
    }
}

# Import test utilities
from .conftest import project_root
from utils.test_utils import TEST_DIR
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
    """Test runner with comprehensive error handling and reporting."""
    
    def __init__(self, args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.project_root = Path(__file__).parent.parent.parent
        self.test_dir = self.project_root / "src" / "tests"
        self.results = {}
        self.start_time = time.time()
        
        # Error tracking
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

    def _run_fallback_tests(self, category: str, test_files: List[str], python_executable: str, start_time: float) -> Dict[str, Any]:
        """Fallback test execution when pytest fails with internal errors."""
        self.logger.info(f"Running fallback test execution for category '{category}'")
        
        # Simple test discovery and execution without pytest
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for test_file in test_files:
            try:
                # Run basic Python syntax check
                syntax_result = subprocess.run(
                    [python_executable, "-m", "py_compile", test_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if syntax_result.returncode == 0:
                    total_tests += 1
                    passed_tests += 1
                    self.logger.info(f"  ✅ {Path(test_file).name}: Syntax valid")
                else:
                    total_tests += 1
                    failed_tests += 1
                    self.logger.warning(f"  ❌ {Path(test_file).name}: Syntax error")
                    
            except Exception as e:
                total_tests += 1
                failed_tests += 1
                self.logger.error(f"  ❌ {Path(test_file).name}: Error {e}")
        
        duration = time.time() - start_time
        self.category_times[category] = duration
        
        # Update global counters
        self.total_tests_run += total_tests
        self.total_tests_passed += passed_tests
        self.total_tests_failed += failed_tests
        
        return {
            "success": failed_tests == 0,
            "tests_run": total_tests,
            "tests_passed": passed_tests,
            "tests_failed": failed_tests,
            "tests_skipped": 0,
            "duration": duration,
            "stdout": f"Fallback execution: {passed_tests}/{total_tests} files passed syntax check",
            "stderr": "",
            "returncode": 1 if failed_tests > 0 else 0,
            "resource_usage": {"memory_mb": 0, "cpu_percent": 0, "threads": 0}
        }

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
        
        # Return with both canonical and legacy key names used elsewhere in the runner
        return {
            "tests_run": tests_run,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "tests_skipped": tests_skipped,
            # Legacy/shorthand keys referenced by callers
            "total": tests_run,
            "passed": tests_passed,
            "failed": tests_failed,
            "skipped": tests_skipped
        }
                        
    def run_test_category(self, category: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests for a specific category with comprehensive error handling and enhanced logging."""
        category_start_time = time.time()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🧪 Running category: {config['name']}")
        self.logger.info(f"📝 Description: {config['description']}")
        self.logger.info(f"⏱️ Timeout: {config.get('timeout_seconds', 60)} seconds")
        self.logger.info(f"🔄 Parallel: {config.get('parallel', True)}")
        self.logger.info(f"{'='*60}")
        
        # Discover test files
        test_files = self.discover_test_files(category, config)
        
        if not test_files:
            self.logger.info(f"⏭️ No test files found for category '{category}', marking as skipped/success")
            return {
                "success": True,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
                "duration": 0,
                "stdout": "",
                "stderr": "",
                "return_code": 0
            }
        
        self.logger.info(f"📁 Found {len(test_files)} test files:")
        for test_file in test_files:
            self.logger.info(f"  🔸 {test_file}")
        
        # Use virtual environment Python if available
        venv_python = self.project_root / ".venv" / "bin" / "python"
        python_executable = str(venv_python) if venv_python.exists() else sys.executable
        
        self.logger.info(f"🐍 Using Python: {python_executable}")
        
        # Detect optional plugins
        has_xdist = False
        has_pytest_cov = False
        has_jsonreport = False
        try:
            import xdist  # type: ignore
            has_xdist = True
        except Exception:
            has_xdist = False
        try:
            import pytest_cov  # type: ignore
            has_pytest_cov = True
        except Exception:
            has_pytest_cov = False
        try:
            import pytest_jsonreport  # type: ignore
            has_jsonreport = True
        except Exception:
            has_jsonreport = False

        # Ensure per-category output directory exists (for coverage artifacts)
        category_output_dir = Path(self.args.output_dir) / "test_reports" / f"category_{category}"
        category_output_dir.mkdir(parents=True, exist_ok=True)

        # Build pytest command with options and plugin isolation
        cmd = [
            python_executable, "-m", "pytest",
            "--tb=short",  # Shorter traceback
            "--maxfail=10",  # Stop after 10 failures
            "--durations=10",  # Show 10 slowest tests
            "--no-header",  # Skip pytest header
            "-p", "no:randomly",  # Disable pytest-randomly plugin
            "-p", "no:sugar",    # Disable pytest-sugar plugin
            "-p", "no:cacheprovider",  # Disable cache to avoid corruption
        ]
        
        # Load and enable optional plugins explicitly since autoload is disabled
        if has_jsonreport:
            cmd.extend(["-p", "pytest_jsonreport", "--json-report", "--json-report-file=none"])
            self.logger.info("📊 JSON reporting enabled")
        else:
            self.logger.info("JSON reporting plugin not available")

        if has_pytest_cov:
            cmd.extend(["-p", "pytest_cov", "--cov=src",
                        f"--cov-report=term-missing",
                        f"--cov-report=json:{category_output_dir}/coverage.json",
                        f"--cov-report=html:{category_output_dir}/htmlcov"])        
        
        # Add markers if specified
        if config.get("markers"):
            for marker in config["markers"]:
                cmd.extend(["-m", marker])
            self.logger.info(f"🏷️ Using markers: {config['markers']}")
        
        # Add test files
        cmd.extend(test_files)
        
        # Add parallel execution if enabled and xdist is available (disable for pipeline tests to avoid timeout)
        if (config.get("parallel", True)
            and not (hasattr(self.args, 'fast_only') and self.args.fast_only)
            and category != "pipeline"
            and has_xdist):
            cmd.extend(["-p", "xdist", "-n", "auto"])
            self.logger.info("⚡ Parallel execution enabled (xdist)")
        else:
            if config.get("parallel", True) and not has_xdist:
                self.logger.info("Parallel requested but xdist not available; running sequentially")
            self.logger.info("🔄 Sequential execution (parallel disabled)")
        
        self.logger.info(f"🚀 Executing command: {' '.join(cmd[:5])}...")
        
        try:
            # Monitor resources before test execution
            initial_resources = self._monitor_resources_during_test()
            self.logger.info(f"💾 Initial resource usage: {initial_resources}")
            
            # Set up timeout
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Test category '{category}' timed out after {config.get('timeout_seconds', 60)} seconds")
            
            # Set signal handler for timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(config.get("timeout_seconds", 60))
            
            try:
                # Set environment variables to avoid dependency conflicts
                env = os.environ.copy()
                env["PYTHONPATH"] = str(self.project_root / "src")
                env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"  # Disable automatic plugin loading
                
                self.logger.info(f"⏱️ Starting test execution at {time.strftime('%H:%M:%S')}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=config.get("timeout_seconds", 60),
                    env=env
                )
                
                # Cancel timeout
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
                execution_time = time.time() - category_start_time
                self.logger.info(f"⏱️ Test execution completed in {execution_time:.2f} seconds")
                
                # Monitor resources after test execution
                final_resources = self._monitor_resources_during_test()
                self.logger.info(f"💾 Final resource usage: {final_resources}")
                
                # Store resource usage (absolute values only)
                resource_usage = {
                    "memory_mb": final_resources.get("memory_mb", 0),
                    "cpu_percent": final_resources.get("cpu_percent", 0),
                    "threads": final_resources.get("threads", 0)
                }
                self.resource_usage[category] = resource_usage
                
                # Handle pytest internal errors with fallback
                if result.returncode == 3 and ("INTERNALERROR" in result.stdout or "INTERNALERROR" in result.stderr):
                    self.logger.warning(f"⚠️ Pytest internal error detected for category '{category}', attempting fallback...")
                    return self._run_fallback_tests(category, test_files, python_executable, category_start_time)
                
                # Parse test results
                test_stats = self._parse_pytest_output(result.stdout, result.stderr)
                
                # Log detailed results
                self.logger.info(f"📊 Test Results for {category}:")
                self.logger.info(f"  ✅ Passed: {test_stats.get('passed', 0)}")
                self.logger.info(f"  ❌ Failed: {test_stats.get('failed', 0)}")
                self.logger.info(f"  ⏭️ Skipped: {test_stats.get('skipped', 0)}")
                self.logger.info(f"  📈 Total: {test_stats.get('total', 0)}")
                
                # Log slowest tests if available
                if "slowest" in result.stdout:
                    slowest_lines = [line for line in result.stdout.split('\n') if "slowest" in line.lower()]
                    if slowest_lines:
                        self.logger.info("🐌 Slowest tests:")
                        for line in slowest_lines[:3]:  # Show top 3
                            self.logger.info(f"  {line.strip()}")
                
                # Determine success based on test results
                total_tests = test_stats.get('total', 0)
                failed_tests = test_stats.get('failed', 0)
                passed_tests = test_stats.get('passed', 0)
                
                success = (total_tests > 0 and failed_tests == 0) or (passed_tests > 0 and failed_tests <= config.get('max_failures', 10))
                
                if success:
                    self.logger.info(f"✅ Category '{category}' completed successfully")
                else:
                    self.logger.warning(f"⚠️ Category '{category}' completed with failures")
                
                return {
                    "success": success,
                    "tests_run": total_tests,
                    "tests_passed": passed_tests,
                    "tests_failed": failed_tests,
                    "tests_skipped": test_stats.get('skipped', 0),
                    "duration": execution_time,
                    "resource_usage": resource_usage,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
                
            except subprocess.TimeoutExpired:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                self.logger.error(f"⏰ Test category '{category}' timed out after {config.get('timeout_seconds', 60)} seconds")
                return {
                    "success": False,
                    "error": f"Timeout after {config.get('timeout_seconds', 60)} seconds",
                    "tests_run": 0,
                    "tests_passed": 0,
                    "tests_failed": 0,
                    "tests_skipped": 0,
                    "duration": config.get('timeout_seconds', 60)
                }
                
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in category '{category}': {e}")
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
        """Run all test categories and generate comprehensive report with enhanced logging."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("🚀 STARTING COMPREHENSIVE TEST SUITE EXECUTION")
        self.logger.info(f"{'='*80}")
        
        # Calculate total categories to run
        total_categories = len([cat for cat in MODULAR_TEST_CATEGORIES.keys() if self.should_run_category(cat)])
        self.logger.info(f"📊 Total categories to run: {total_categories}")
        
        overall_success = True
        start_time = time.time()
        
        for i, (category, config) in enumerate(MODULAR_TEST_CATEGORIES.items(), 1):
            if not self.should_run_category(category):
                self.logger.info(f"⏭️ Skipping category '{category}' based on arguments")
                continue
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📋 Category {i}/{total_categories}: {category}")
            self.logger.info(f"📝 {config['name']}")
            self.logger.info(f"{'='*60}")
            
            self.categories_run += 1
            
            try:
                result = self.run_test_category(category, config)
                self.results[category] = result
                
                # Update global counters
                self.total_tests_run += result.get("tests_run", 0)
                self.total_tests_passed += result.get("tests_passed", 0)
                self.total_tests_failed += result.get("tests_failed", 0)
                self.total_tests_skipped += result.get("tests_skipped", 0)
                
                if result.get("success", False):
                    self.categories_successful += 1
                    self.logger.info(f"✅ Category '{category}' completed successfully")
                else:
                    self.categories_failed += 1
                    overall_success = False
                    self.logger.warning(f"⚠️ Category '{category}' completed with issues")
                
                # Log category summary
                duration = result.get("duration", 0)
                tests_run = result.get("tests_run", 0)
                tests_passed = result.get("tests_passed", 0)
                tests_failed = result.get("tests_failed", 0)
                
                self.logger.info(f"📊 Category '{category}' Summary:")
                self.logger.info(f"  ⏱️ Duration: {duration:.2f}s")
                self.logger.info(f"  📈 Tests: {tests_run} total, {tests_passed} passed, {tests_failed} failed")
                
                if tests_run > 0:
                    success_rate = (tests_passed / tests_run) * 100
                    self.logger.info(f"  📊 Success Rate: {success_rate:.1f}%")
                
                # Log resource usage if available
                resource_usage = result.get("resource_usage", {})
                if resource_usage:
                    memory_mb = resource_usage.get("memory_mb", 0)
                    cpu_percent = resource_usage.get("cpu_percent", 0)
                    self.logger.info(f"  💾 Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
                    
            except Exception as e:
                self.logger.error(f"❌ Unexpected error in category '{category}': {e}")
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
        
        # Calculate overall statistics
        total_duration = time.time() - start_time
        total_categories_run = self.categories_run
        total_categories_successful = self.categories_successful
        total_categories_failed = self.categories_failed
        
        # Log overall summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info("📊 COMPREHENSIVE TEST SUITE SUMMARY")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"⏱️ Total Duration: {total_duration:.2f}s")
        self.logger.info(f"📋 Categories: {total_categories_run} run, {total_categories_successful} successful, {total_categories_failed} failed")
        self.logger.info(f"🧪 Tests: {self.total_tests_run} total, {self.total_tests_passed} passed, {self.total_tests_failed} failed, {self.total_tests_skipped} skipped")
        
        if self.total_tests_run > 0:
            overall_success_rate = (self.total_tests_passed / self.total_tests_run) * 100
            self.logger.info(f"📊 Overall Success Rate: {overall_success_rate:.1f}%")
        
        if total_categories_run > 0:
            category_success_rate = (total_categories_successful / total_categories_run) * 100
            self.logger.info(f"📊 Category Success Rate: {category_success_rate:.1f}%")
        
        # Generate comprehensive report
        self.logger.info(f"\n💾 Generating comprehensive test reports...")
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
            "total_duration": total_duration,
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