"""
Tests module for GNN Processing Pipeline.

This module provides comprehensive testing capabilities for the GNN pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

def run_all_tests(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False
) -> bool:
    """
    Run comprehensive test suite for the GNN pipeline.
    
    Args:
        target_dir: Directory containing GNN files to test
        output_dir: Directory to save test results
        verbose: Enable verbose output
        
    Returns:
        True if all tests pass, False otherwise
    """
    logger = logging.getLogger("tests")
    
    try:
        log_step_start(logger, "Running comprehensive test suite")
        
        # Create test results directory
        test_results_dir = output_dir / "test_results"
        test_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run different types of tests
        test_results = {}
        
        # Unit tests
        test_results["unit_tests"] = run_unit_tests(test_results_dir, verbose)
        
        # Integration tests
        test_results["integration_tests"] = run_integration_tests(target_dir, test_results_dir, verbose)
        
        # Performance tests
        test_results["performance_tests"] = run_performance_tests(target_dir, test_results_dir, verbose)
        
        # Coverage tests
        test_results["coverage_tests"] = run_coverage_tests(test_results_dir, verbose)
        
        # Save test results
        import json
        results_file = test_results_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Check overall success
        all_passed = all(test_results.values())
        
        if all_passed:
            log_step_success(logger, "All tests passed successfully")
        else:
            failed_tests = [name for name, passed in test_results.items() if not passed]
            log_step_error(logger, f"Some tests failed: {failed_tests}")
        
        return all_passed
        
    except Exception as e:
        log_step_error(logger, f"Test execution failed: {e}")
        return False

def run_unit_tests(test_results_dir: Path, verbose: bool) -> bool:
    """Run unit tests."""
    logger = logging.getLogger("tests.unit")
    
    try:
        log_step_start(logger, "Running unit tests")
        
        # Import and run unit tests
        from .unit_tests import run_unit_test_suite
        
        success = run_unit_test_suite(test_results_dir, verbose)
        
        if success:
            log_step_success(logger, "Unit tests passed")
        else:
            log_step_error(logger, "Unit tests failed")
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Unit test execution failed: {e}")
        return False

def run_integration_tests(target_dir: Path, test_results_dir: Path, verbose: bool) -> bool:
    """Run integration tests."""
    logger = logging.getLogger("tests.integration")
    
    try:
        log_step_start(logger, "Running integration tests")
        
        # Import and run integration tests
        from .integration_tests import run_integration_test_suite
        
        success = run_integration_test_suite(target_dir, test_results_dir, verbose)
        
        if success:
            log_step_success(logger, "Integration tests passed")
        else:
            log_step_error(logger, "Integration tests failed")
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Integration test execution failed: {e}")
        return False

def run_performance_tests(target_dir: Path, test_results_dir: Path, verbose: bool) -> bool:
    """Run performance tests."""
    logger = logging.getLogger("tests.performance")
    
    try:
        log_step_start(logger, "Running performance tests")
        
        # Import and run performance tests
        from .performance_tests import run_performance_test_suite
        
        success = run_performance_test_suite(target_dir, test_results_dir, verbose)
        
        if success:
            log_step_success(logger, "Performance tests passed")
        else:
            log_step_error(logger, "Performance tests failed")
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Performance test execution failed: {e}")
        return False

def run_coverage_tests(test_results_dir: Path, verbose: bool) -> bool:
    """Run coverage tests."""
    logger = logging.getLogger("tests.coverage")
    
    try:
        log_step_start(logger, "Running coverage tests")
        
        # Import and run coverage tests
        from .coverage_tests import run_coverage_test_suite
        
        success = run_coverage_test_suite(test_results_dir, verbose)
        
        if success:
            log_step_success(logger, "Coverage tests passed")
        else:
            log_step_error(logger, "Coverage tests failed")
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Coverage test execution failed: {e}")
        return False

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "Comprehensive testing for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'unit_tests': True,
    'integration_tests': True,
    'performance_tests': True,
    'coverage_tests': True,
    'test_reporting': True
}

__all__ = [
    'run_all_tests',
    'run_unit_tests',
    'run_integration_tests',
    'run_performance_tests',
    'run_coverage_tests',
    'FEATURES',
    '__version__'
] 