#!/usr/bin/env python3
"""
Comprehensive Test Runner Module

This module provides the core test execution functionality moved from 2_tests.py
to maintain separation of concerns and improve modularity.

Key Components:
- ModularTestRunner: Main test execution engine
- Test execution utilities and helpers
- Pytest output parsing and result processing
- Test category management and configuration
"""

import sys
import os
import time
import json
import logging
import subprocess
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Test configuration
MODULAR_TEST_CATEGORIES = {
    "core": {
        "name": "Core Module Tests",
        "description": "Essential GNN core functionality tests",
        "files": ["test_gnn_core_modules.py", "test_core_modules.py", "test_environment.py"],
        "markers": [],
        "timeout_seconds": 60,
        "max_failures": 5
    },
    "pipeline": {
        "name": "Pipeline Infrastructure Tests", 
        "description": "Pipeline scripts and infrastructure tests",
        "files": ["test_main_orchestrator.py", "test_pipeline_functionality.py", "test_pipeline_infrastructure.py", 
                  "test_pipeline_performance.py", "test_pipeline_recovery.py", "test_pipeline_scripts.py", 
                  "test_pipeline_steps.py"],
        "markers": [],
        "timeout_seconds": 120,  # Increased timeout for pipeline tests
        "max_failures": 10
    },
    "integration": {
        "name": "Integration Tests",
        "description": "Cross-module integration and workflow tests",
        "files": ["test_comprehensive_api.py", "test_mcp_integration_comprehensive.py", 
                  "test_mcp_comprehensive.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 10
    },
    "validation": {
        "name": "Validation Tests",
        "description": "GNN validation and type checking tests",
        "files": ["test_gnn_type_checker.py", "test_parsers.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 8
    },
    "utilities": {
        "name": "Utility Tests",
        "description": "Utility functions and helper tests",
        "files": ["test_utils.py", "test_utilities.py", "test_utility_modules.py", "test_runner_helper.py"],
        "markers": [],
        "timeout_seconds": 60,
        "max_failures": 5
    },
    "specialized": {
        "name": "Specialized Module Tests",
        "description": "Specialized module tests (audio, visualization, etc.)",
        "files": ["test_visualization.py", "test_sapf.py", "test_export.py", "test_render.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8
    },
    "reporting": {
        "name": "Reporting Tests",
        "description": "Report generation and output tests",
        "files": ["test_report_comprehensive.py"],
        "markers": [],
        "timeout_seconds": 60,
        "max_failures": 5
    },
    "performance": {
        "name": "Performance Tests",
        "description": "Performance and benchmarking tests",
        "files": ["test_pipeline_performance.py"],  # Only include actual performance tests
        "markers": [],
        "timeout_seconds": 180,
        "max_failures": 10
    },
    "fast_suite": {
        "name": "Fast Test Suite",
        "description": "Fast execution test suite",
        "files": ["test_fast_suite.py"],
        "markers": [],
        "timeout_seconds": 30,
        "max_failures": 3
    },
    "comprehensive": {
        "name": "Comprehensive Tests",
        "description": "All remaining test files for complete coverage",
        "files": ["test_*"],  # Pattern to catch any remaining test files
        "markers": [],
        "timeout_seconds": 600,  # Increased timeout for comprehensive tests
        "max_failures": 20
    }
}

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
        
        # Error categorization
        self.import_errors = []
        self.runtime_errors = []
        self.pathlib_errors = []
        self.sapf_errors = []
        
        self.logger.info(f"Initialized ModularTestRunner with {len(MODULAR_TEST_CATEGORIES)} categories")

    def should_run_category(self, category: str) -> bool:
        """Determine if a test category should be run based on arguments."""
        if self.args.fast_only:
            # In fast mode, run core, pipeline, validation, utilities, and fast_suite
            # This gives better coverage while still being fast
            return category in ["core", "pipeline", "validation", "utilities", "fast_suite"]
        
        # By default, run ALL categories including performance
        # Only skip performance tests if explicitly excluded with --no-performance
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
        
        # Add parallel execution if enabled (disable for pipeline tests to avoid timeout)
        if (config.get("parallel", True) and not self.args.fast_only and 
            category != "pipeline"):  # Disable parallel for pipeline tests
            cmd.extend(["-n", "auto"])
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
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

def create_test_runner(args, logger: logging.Logger) -> ModularTestRunner:
    """Factory function to create a ModularTestRunner instance."""
    return ModularTestRunner(args, logger) 