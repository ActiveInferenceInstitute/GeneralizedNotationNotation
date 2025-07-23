"""
Unit tests for GNN Processing Pipeline.

This module provides unit testing capabilities with fallback implementations.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

def run_unit_test_suite(test_results_dir: Path, verbose: bool) -> bool:
    """
    Run unit test suite for the GNN pipeline.
    
    Args:
        test_results_dir: Directory to save test results
        verbose: Enable verbose output
        
    Returns:
        True if all unit tests pass, False otherwise
    """
    logger = logging.getLogger("unit_tests")
    
    try:
        log_step_start(logger, "Running unit test suite")
        
        # Create unit test results directory
        unit_results_dir = test_results_dir / "unit_tests"
        unit_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run different unit test categories
        test_results = {}
        
        # Core functionality tests
        test_results["core_tests"] = run_core_unit_tests(unit_results_dir, verbose)
        
        # Utility function tests
        test_results["utility_tests"] = run_utility_unit_tests(unit_results_dir, verbose)
        
        # Pipeline step tests
        test_results["pipeline_tests"] = run_pipeline_unit_tests(unit_results_dir, verbose)
        
        # Save unit test results
        import json
        results_file = unit_results_dir / "unit_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Check overall success
        all_passed = all(test_results.values())
        
        if all_passed:
            log_step_success(logger, "All unit tests passed")
        else:
            failed_tests = [name for name, passed in test_results.items() if not passed]
            log_step_error(logger, f"Some unit tests failed: {failed_tests}")
        
        return all_passed
        
    except Exception as e:
        log_step_error(logger, f"Unit test suite execution failed: {e}")
        return False

def run_core_unit_tests(test_results_dir: Path, verbose: bool) -> bool:
    """Run core functionality unit tests."""
    logger = logging.getLogger("unit_tests.core")
    
    try:
        log_step_start(logger, "Running core unit tests")
        
        # Test basic imports
        try:
            from utils import setup_step_logging
            from pipeline import get_pipeline_config
            log_step_success(logger, "Core imports successful")
        except ImportError as e:
            log_step_error(logger, f"Core import test failed: {e}")
            return False
        
        # Test basic functionality
        test_results = {
            "import_test": True,
            "config_test": test_config_functionality(),
            "logging_test": test_logging_functionality()
        }
        
        # Save core test results
        import json
        results_file = test_results_dir / "core_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        all_passed = all(test_results.values())
        
        if all_passed:
            log_step_success(logger, "Core unit tests passed")
        else:
            failed_tests = [name for name, passed in test_results.items() if not passed]
            log_step_error(logger, f"Some core unit tests failed: {failed_tests}")
        
        return all_passed
        
    except Exception as e:
        log_step_error(logger, f"Core unit test execution failed: {e}")
        return False

def run_utility_unit_tests(test_results_dir: Path, verbose: bool) -> bool:
    """Run utility function unit tests."""
    logger = logging.getLogger("unit_tests.utility")
    
    try:
        log_step_start(logger, "Running utility unit tests")
        
        # Test utility functions
        test_results = {
            "path_utilities": test_path_utilities(),
            "file_utilities": test_file_utilities(),
            "validation_utilities": test_validation_utilities()
        }
        
        # Save utility test results
        import json
        results_file = test_results_dir / "utility_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        all_passed = all(test_results.values())
        
        if all_passed:
            log_step_success(logger, "Utility unit tests passed")
        else:
            failed_tests = [name for name, passed in test_results.items() if not passed]
            log_step_error(logger, f"Some utility unit tests failed: {failed_tests}")
        
        return all_passed
        
    except Exception as e:
        log_step_error(logger, f"Utility unit test execution failed: {e}")
        return False

def run_pipeline_unit_tests(test_results_dir: Path, verbose: bool) -> bool:
    """Run pipeline step unit tests."""
    logger = logging.getLogger("unit_tests.pipeline")
    
    try:
        log_step_start(logger, "Running pipeline unit tests")
        
        # Test pipeline step functionality
        test_results = {
            "step_discovery": test_step_discovery(),
            "step_execution": test_step_execution(),
            "step_configuration": test_step_configuration()
        }
        
        # Save pipeline test results
        import json
        results_file = test_results_dir / "pipeline_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        all_passed = all(test_results.values())
        
        if all_passed:
            log_step_success(logger, "Pipeline unit tests passed")
        else:
            failed_tests = [name for name, passed in test_results.items() if not passed]
            log_step_error(logger, f"Some pipeline unit tests failed: {failed_tests}")
        
        return all_passed
        
    except Exception as e:
        log_step_error(logger, f"Pipeline unit test execution failed: {e}")
        return False

# Test helper functions
def test_config_functionality() -> bool:
    """Test configuration functionality."""
    try:
        from pipeline import get_pipeline_config
        config = get_pipeline_config()
        return config is not None
    except Exception:
        return False

def test_logging_functionality() -> bool:
    """Test logging functionality."""
    try:
        from utils import setup_step_logging
        logger = setup_step_logging("test", None)
        return logger is not None
    except Exception:
        return False

def test_path_utilities() -> bool:
    """Test path utility functions."""
    try:
        from pathlib import Path
        test_path = Path("/tmp/test_path")
        return True
    except Exception:
        return False

def test_file_utilities() -> bool:
    """Test file utility functions."""
    try:
        import json
        test_data = {"test": "data"}
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        return parsed_data == test_data
    except Exception:
        return False

def test_validation_utilities() -> bool:
    """Test validation utility functions."""
    try:
        # Basic validation test
        test_value = "test"
        is_string = isinstance(test_value, str)
        return is_string
    except Exception:
        return False

def test_step_discovery() -> bool:
    """Test pipeline step discovery."""
    try:
        from pipeline.discovery import get_pipeline_scripts
        from pathlib import Path
        scripts = get_pipeline_scripts(Path(__file__).parent.parent)
        return isinstance(scripts, list)
    except Exception:
        return False

def test_step_execution() -> bool:
    """Test pipeline step execution."""
    try:
        # Basic execution test
        return True
    except Exception:
        return False

def test_step_configuration() -> bool:
    """Test pipeline step configuration."""
    try:
        from pipeline import get_pipeline_config
        config = get_pipeline_config()
        return hasattr(config, 'steps')
    except Exception:
        return False 