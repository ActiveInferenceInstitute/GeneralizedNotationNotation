#!/usr/bin/env python3
"""
Step 2: Test Suite Execution (Robust Version)

This script runs comprehensive tests for the GNN pipeline in staged execution (fast, standard, slow, performance).
It is robust to invocation from both main.py and CLI, and provides actionable logging and output.

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
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

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

# --- Test Stage Configuration ---
class TestStage(Enum):
    """Test execution stages with different characteristics."""
    FAST = "fast"
    STANDARD = "standard" 
    SLOW = "slow"
    PERFORMANCE = "performance"

@dataclass
class TestStageConfig:
    """Configuration for a test execution stage."""
    name: str
    markers: List[str]
    timeout_seconds: int
    description: str
    max_failures: int = 10
    parallel: bool = True
    coverage: bool = True

# --- Test Stage Configuration - Modular Approach ---
TEST_STAGES = {
    TestStage.FAST: TestStageConfig(
        name="Fast Tests", 
        markers=["fast"],
        timeout_seconds=120,  # 2 minutes - quick validation
        description="Quick validation tests for core functionality",
        max_failures=10,
        parallel=True,
        coverage=False  # Skip coverage for speed
    ),
    TestStage.STANDARD: TestStageConfig(
        name="Standard Tests", 
        markers=["not slow", "not performance"],
        timeout_seconds=180,  # 3 minutes - reasonable for standard tests
        description="Comprehensive module and integration tests",
        max_failures=20,
        parallel=True,
        coverage=True
    ),
    TestStage.SLOW: TestStageConfig(
        name="Slow Tests",
        markers=["slow"],
        timeout_seconds=300,  # 5 minutes - for complex scenarios
        description="Integration tests and complex scenarios",
        max_failures=20,
        parallel=False,  # May have resource conflicts
        coverage=True
    ),
    TestStage.PERFORMANCE: TestStageConfig(
        name="Performance Tests",
        markers=["performance"],
        timeout_seconds=240,  # 4 minutes - performance benchmarks
        description="Performance benchmarks and resource usage tests",
        max_failures=5,
        parallel=False,
        coverage=False
    )
}

# --- Modular Test Categories with Improved Discovery ---
MODULAR_TEST_CATEGORIES = {
    "core": {
        "name": "Core Module Tests",
        "description": "Essential GNN core functionality tests",
        "files": ["test_gnn_core_modules.py", "test_core_modules.py"],
        "markers": [],
        "timeout_seconds": 60,
        "max_failures": 5
    },
    "pipeline": {
        "name": "Pipeline Infrastructure Tests", 
        "description": "Pipeline scripts and infrastructure tests",
        "files": ["test_pipeline_scripts.py", "test_main_orchestrator.py", "test_pipeline_infrastructure.py"],
        "markers": [],
        "timeout_seconds": 60,
        "max_failures": 5
    },
    "integration": {
        "name": "Integration Tests",
        "description": "Cross-module integration and workflow tests",
        "files": ["integration_tests.py", "test_comprehensive_api.py"],
        "markers": ["not slow"],
        "timeout_seconds": 120,
        "max_failures": 10
    },
    "validation": {
        "name": "Validation Tests",
        "description": "GNN validation and type checking tests",
        "files": ["test_gnn_type_checker.py"],
        "markers": ["fast"],
        "timeout_seconds": 90,
        "max_failures": 8
    },
    "utilities": {
        "name": "Utility Tests",
        "description": "Utility functions and helper tests",
        "files": ["test_utils.py", "test_utilities.py", "test_utility_modules.py"],
        "markers": ["fast"],
        "timeout_seconds": 60,
        "max_failures": 5
    },
    "specialized": {
        "name": "Specialized Module Tests",
        "description": "Specialized module tests (audio, visualization, etc.)",
        "files": ["test_visualization.py", "test_sapf.py", "test_export.py", "test_render.py"],
        "markers": ["not slow"],
        "timeout_seconds": 120,
        "max_failures": 8
    },
    "reporting": {
        "name": "Reporting Tests",
        "description": "Report generation and output tests",
        "files": ["test_report_comprehensive.py"],
        "markers": ["fast"],
        "timeout_seconds": 60,
        "max_failures": 5
    }
}

# --- Modular Test Runner with Enhanced Error Handling ---
class ModularTestRunner:
    """Enhanced test runner with comprehensive error handling and reporting."""
    
    def __init__(self, args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.project_root = Path(__file__).parent.parent
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
        
        # Error categorization
        self.import_errors = []
        self.runtime_errors = []
        self.pathlib_errors = []
        self.sapf_errors = []
        
        self.logger.info(f"Initialized ModularTestRunner with {len(MODULAR_TEST_CATEGORIES)} categories")

    def should_run_category(self, category: str) -> bool:
        """Determine if a test category should be run based on arguments."""
        if self.args.fast_only:
            # In fast mode, only run core and pipeline tests
            return category in ["core", "pipeline"]
        
        if not self.args.include_slow:
            # Skip slow tests unless explicitly included
            return category not in ["specialized"]
        
        if not self.args.include_performance:
            # Skip performance tests unless explicitly included
            return category not in ["performance"]
        
        return True

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
        
        # Add coverage if enabled
        if config.get("coverage", True) and not self.args.fast_only:
            cmd.extend(["--cov=src", "--cov-report=term-missing"])
        
        # Add parallel execution if enabled
        if config.get("parallel", True) and not self.args.fast_only:
            cmd.extend(["-n", "auto"])
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run pytest with timeout
            import subprocess
            import signal
            
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
                    "returncode": result.returncode
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

    def run_all_categories(self) -> bool:
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
        
        return overall_success

    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, int]:
        """Parse pytest output to extract test counts."""
        tests_run = 0
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0
        
        # Parse the output for test results
        lines = stdout.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for test result patterns - handle various formats
            if "passed" in line and ("skipped" in line or "failed" in line):
                # Parse summary lines like:
                # "======================== 41 passed, 12 skipped in 1.76s ========================="
                # "================== 10 failed, 14 passed, 3 warnings in 0.11s ==================="
                
                # Extract the content between the = signs
                if "=" in line:
                    # Find the content between the first and last = signs
                    start = line.find("=")
                    end = line.rfind("=")
                    if start != end and start != -1 and end != -1:
                        content = line[start+1:end].strip()
                        
                        # Split by comma and parse each part
                        test_parts = content.split(',')
                        for part in test_parts:
                            part = part.strip()
                            # Extract the number before the keyword
                            words = part.split()
                            for i, word in enumerate(words):
                                if "passed" in word and i > 0:
                                    try:
                                        tests_passed = int(words[i-1])
                                    except (ValueError, IndexError):
                                        pass
                                elif "failed" in word and i > 0:
                                    try:
                                        tests_failed = int(words[i-1])
                                    except (ValueError, IndexError):
                                        pass
                                elif "skipped" in word and i > 0:
                                    try:
                                        tests_skipped = int(words[i-1])
                                    except (ValueError, IndexError):
                                        pass
                        
                        tests_run = tests_passed + tests_failed + tests_skipped
                        break
            
            # Also handle cases with only passed tests
            elif "passed" in line and "failed" not in line and "skipped" not in line:
                # Parse summary line like "======================== 30 passed in 1.50s ========================="
                if "=" in line:
                    start = line.find("=")
                    end = line.rfind("=")
                    if start != end and start != -1 and end != -1:
                        content = line[start+1:end].strip()
                        test_parts = content.split(',')
                        for part in test_parts:
                            part = part.strip()
                            if "passed" in part:
                                try:
                                    tests_passed = int(part.split()[0])
                                except (ValueError, IndexError):
                                    pass
                        
                        tests_run = tests_passed
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
                        if part == "collected" and i > 0:
                            try:
                                tests_run = int(parts[i-1])
                                # If we found collected tests but no summary, assume they all passed
                                tests_passed = tests_run
                            except ValueError:
                                pass
        
        return {
            "tests_run": tests_run,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "tests_skipped": tests_skipped
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
        
        self.logger.info(f"{'='*80}")

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
        # Create test runner
        runner = ModularTestRunner(args, logger)
        
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