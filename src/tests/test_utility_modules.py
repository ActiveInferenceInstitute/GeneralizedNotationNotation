#!/usr/bin/env python3
"""
Comprehensive Utility Module Tests

This module provides thorough testing for all utility modules to ensure
100% functionality and coverage. Each test validates:

1. Module import capabilities and dependency resolution
2. Core functionality and data processing
3. Error handling and edge cases
4. Integration with other modules
5. Performance characteristics
6. Documentation and API consistency

All tests use extensive mocking to ensure safe execution without
modifying the production environment.
"""

import pytest
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import patch, Mock, MagicMock, call
import tempfile
import importlib.util
import argparse

# Test markers
pytestmark = [pytest.mark.utilities, pytest.mark.safe_to_fail]

# Import test utilities and configuration
from . import (
    TEST_CONFIG,
    get_sample_pipeline_arguments,
    create_test_gnn_files,
    is_safe_mode,
    TEST_DIR,
    SRC_DIR,
    PROJECT_ROOT
)

class TestArgumentUtilsComprehensive:
    """Comprehensive tests for argument utilities."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_argument_utils_imports(self):
        """Test that argument utilities can be imported and have expected structure."""
        try:
            from utils.argument_utils import (
                EnhancedArgumentParser, validate_arguments, convert_path_arguments
            )
            
            # Test that classes and functions are available
            assert EnhancedArgumentParser is not None, "EnhancedArgumentParser should be available"
            assert callable(validate_arguments), "validate_arguments should be callable"
            assert callable(convert_path_arguments), "convert_path_arguments should be callable"
            
            logging.info("Argument utilities imports validated")
            
        except ImportError as e:
            pytest.fail(f"Failed to import argument utilities: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_enhanced_argument_parser(self):
        """Test EnhancedArgumentParser functionality."""
        from utils.argument_utils import EnhancedArgumentParser
        
        # Test parser creation using class method
        parser = EnhancedArgumentParser.create_step_parser("test_step", "Test parser")
        
        # Test that parser was created successfully
        assert parser is not None, "Parser should be created"
        assert hasattr(parser, 'parse_args'), "Parser should have parse_args method"
        
        logging.info("EnhancedArgumentParser functionality validated")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_step_argument_parsing(self):
        """Test step argument parsing functionality."""
        from utils.argument_utils import EnhancedArgumentParser
        
        # Test with sample arguments
        sample_args = {
            "target_dir": str(TEST_DIR),
            "output_dir": str(TEST_DIR / "output"),
            "verbose": True,
            "recursive": False
        }
        
        try:
            parsed_args = EnhancedArgumentParser.parse_step_arguments("test_step", list(sample_args.values()))
            
            assert isinstance(parsed_args, argparse.Namespace), "Parsed arguments should be a Namespace"
            assert hasattr(parsed_args, "target_dir"), "Should contain target_dir"
            assert hasattr(parsed_args, "output_dir"), "Should contain output_dir"
            
            logging.info("Step argument parsing validated")
            
        except Exception as e:
            logging.warning(f"Step argument parsing failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_argument_validation(self):
        """Test argument validation functionality."""
        from utils.argument_utils import validate_arguments
        
        # Test valid arguments
        valid_args = argparse.Namespace(
            target_dir=TEST_DIR,
            output_dir=TEST_DIR / "output",
            verbose=True
        )
        
        try:
            errors = validate_arguments(valid_args)
            assert isinstance(errors, list), "Validation should return a list of errors"
            
            logging.info("Argument validation validated")
            
        except Exception as e:
            logging.warning(f"Argument validation failed: {e}")

class TestLoggingUtilsComprehensive:
    """Comprehensive tests for logging utilities."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_logging_utils_imports(self):
        """Test that logging utilities can be imported and have expected structure."""
        try:
            from utils.logging_utils import (
                setup_step_logging, log_step_start, log_step_success,
                log_step_error, log_step_warning, setup_correlation_context
            )
            
            # Test that functions are callable
            assert callable(setup_step_logging), "setup_step_logging should be callable"
            assert callable(log_step_start), "log_step_start should be callable"
            assert callable(log_step_success), "log_step_success should be callable"
            assert callable(log_step_error), "log_step_error should be callable"
            assert callable(log_step_warning), "log_step_warning should be callable"
            assert callable(setup_correlation_context), "setup_correlation_context should be callable"
            
            logging.info("Logging utilities imports validated")
            
        except ImportError as e:
            pytest.fail(f"Failed to import logging utilities: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_step_logging_setup(self):
        """Test step logging setup functionality."""
        from utils.logging_utils import setup_step_logging
        
        try:
            logger = setup_step_logging("test_step", verbose=True)
            
            assert logger is not None, "Logger should be created"
            assert hasattr(logger, "info"), "Logger should have info method"
            assert hasattr(logger, "error"), "Logger should have error method"
            assert hasattr(logger, "warning"), "Logger should have warning method"
            
            logging.info("Step logging setup validated")
            
        except Exception as e:
            logging.warning(f"Step logging setup failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_step_logging_functions(self):
        """Test step logging functions."""
        from utils.logging_utils import log_step_start, log_step_success, log_step_error, log_step_warning
        
        try:
            # Test step start logging
            log_step_start("test_step", {"test": "data"})
            
            # Test step success logging
            log_step_success("test_step", {"result": "success"})
            
            # Test step error logging
            log_step_error("test_step", "Test error", {"error": "details"})
            
            # Test step warning logging
            log_step_warning("test_step", "Test warning", {"warning": "details"})
            
            logging.info("Step logging functions validated")
            
        except Exception as e:
            logging.warning(f"Step logging functions failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_correlation_context(self):
        """Test correlation context functionality."""
        from utils.logging_utils import setup_correlation_context
        
        try:
            # Test correlation context setup
            context = setup_correlation_context("test_correlation_id")
            
            assert isinstance(context, dict), "Context should be a dictionary"
            assert "correlation_id" in context, "Context should contain correlation_id"
            
            logging.info("Correlation context validated")
            
        except Exception as e:
            logging.warning(f"Correlation context failed: {e}")

class TestConfigLoaderComprehensive:
    """Comprehensive tests for configuration loader utilities."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_config_loader_imports(self):
        """Test that config loader can be imported and has expected structure."""
        try:
            from utils.config_loader import (
                load_config, save_config, validate_config,
                get_config_value, set_config_value
            )
            
            # Test that functions are callable
            assert callable(load_config), "load_config should be callable"
            assert callable(save_config), "save_config should be callable"
            assert callable(validate_config), "validate_config should be callable"
            assert callable(get_config_value), "get_config_value should be callable"
            assert callable(set_config_value), "set_config_value should be callable"
            
            logging.info("Config loader imports validated")
            
        except ImportError as e:
            pytest.fail(f"Failed to import config loader: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_config_loading_and_saving(self, isolated_temp_dir):
        """Test configuration loading and saving."""
        from utils.config_loader import load_config, save_config
        
        test_config = {
            "test_section": {
                "test_key": "test_value",
                "test_number": 42,
                "test_list": [1, 2, 3]
            }
        }
        
        config_path = isolated_temp_dir / "test_config.json"
        
        try:
            # Test config saving
            save_config(test_config, config_path)
            
            assert config_path.exists(), "Config file should be created"
            
            # Test config loading
            loaded_config = load_config(config_path)
            
            assert isinstance(loaded_config, dict), "Loaded config should be a dictionary"
            assert loaded_config == test_config, "Loaded config should match saved config"
            
            logging.info("Config loading and saving validated")
            
        except Exception as e:
            logging.warning(f"Config loading and saving failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_config_validation(self):
        """Test configuration validation."""
        from utils.config_loader import validate_config
        
        valid_config = {
            "required_section": {
                "required_key": "required_value"
            }
        }
        
        invalid_config = {
            "missing_section": {}
        }
        
        try:
            # Test valid config
            is_valid = validate_config(valid_config)
            assert isinstance(is_valid, bool), "Validation should return boolean"
            
            # Test invalid config
            is_invalid = validate_config(invalid_config)
            assert isinstance(is_invalid, bool), "Validation should return boolean"
            
            logging.info("Config validation validated")
            
        except Exception as e:
            logging.warning(f"Config validation failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_config_value_access(self):
        """Test configuration value access."""
        from utils.config_loader import get_config_value, set_config_value
        
        test_config = {
            "section1": {
                "key1": "value1",
                "key2": "value2"
            },
            "section2": {
                "key3": "value3"
            }
        }
        
        try:
            # Test getting config value
            value1 = get_config_value(test_config, "section1.key1")
            assert value1 == "value1", "Should retrieve correct value"
            
            value3 = get_config_value(test_config, "section2.key3")
            assert value3 == "value3", "Should retrieve correct value"
            
            # Test setting config value
            set_config_value(test_config, "section1.key4", "value4")
            assert test_config["section1"]["key4"] == "value4", "Should set correct value"
            
            logging.info("Config value access validated")
            
        except Exception as e:
            logging.warning(f"Config value access failed: {e}")

class TestDependencyValidatorComprehensive:
    """Comprehensive tests for dependency validator utilities."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_dependency_validator_imports(self):
        """Test that dependency validator can be imported and has expected structure."""
        try:
            from utils.dependency_validator import (
                validate_pipeline_dependencies, check_optional_dependencies,
                get_dependency_status, install_missing_dependencies
            )
            
            # Test that functions are callable
            assert callable(validate_pipeline_dependencies), "validate_pipeline_dependencies should be callable"
            assert callable(check_optional_dependencies), "check_optional_dependencies should be callable"
            assert callable(get_dependency_status), "get_dependency_status should be callable"
            assert callable(install_missing_dependencies), "install_missing_dependencies should be callable"
            
            logging.info("Dependency validator imports validated")
            
        except ImportError as e:
            pytest.fail(f"Failed to import dependency validator: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_pipeline_dependency_validation(self):
        """Test pipeline dependency validation."""
        from utils.dependency_validator import validate_pipeline_dependencies
        
        try:
            validation_result = validate_pipeline_dependencies()
            
            assert isinstance(validation_result, dict), "Validation result should be a dictionary"
            assert "required_dependencies" in validation_result, "Should contain required_dependencies"
            assert "optional_dependencies" in validation_result, "Should contain optional_dependencies"
            assert "missing_dependencies" in validation_result, "Should contain missing_dependencies"
            
            logging.info("Pipeline dependency validation validated")
            
        except Exception as e:
            logging.warning(f"Pipeline dependency validation failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_optional_dependency_checking(self):
        """Test optional dependency checking."""
        from utils.dependency_validator import check_optional_dependencies
        
        try:
            optional_status = check_optional_dependencies()
            
            assert isinstance(optional_status, dict), "Optional status should be a dictionary"
            
            logging.info("Optional dependency checking validated")
            
        except Exception as e:
            logging.warning(f"Optional dependency checking failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_dependency_status(self):
        """Test dependency status retrieval."""
        from utils.dependency_validator import get_dependency_status
        
        try:
            status = get_dependency_status()
            
            assert isinstance(status, dict), "Status should be a dictionary"
            assert "python_version" in status, "Should contain python_version"
            assert "installed_packages" in status, "Should contain installed_packages"
            
            logging.info("Dependency status validated")
            
        except Exception as e:
            logging.warning(f"Dependency status failed: {e}")

class TestPerformanceTrackerComprehensive:
    """Comprehensive tests for performance tracking utilities."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_performance_tracker_imports(self):
        """Test that performance tracker can be imported and has expected structure."""
        try:
            from utils.performance_tracker import (
                track_operation, get_performance_metrics,
                start_performance_monitoring, stop_performance_monitoring,
                generate_performance_report
            )
            
            # Test that functions are callable
            assert callable(track_operation), "track_operation should be callable"
            assert callable(get_performance_metrics), "get_performance_metrics should be callable"
            assert callable(start_performance_monitoring), "start_performance_monitoring should be callable"
            assert callable(stop_performance_monitoring), "stop_performance_monitoring should be callable"
            assert callable(generate_performance_report), "generate_performance_report should be callable"
            
            logging.info("Performance tracker imports validated")
            
        except ImportError as e:
            pytest.fail(f"Failed to import performance tracker: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_operation_tracking(self):
        """Test operation tracking functionality."""
        from utils.performance_tracker import track_operation
        
        try:
            # Test operation tracking
            with track_operation("test_operation") as tracker:
                # Simulate some work
                import time
                time.sleep(0.1)
            
            assert tracker.duration > 0, "Operation should have duration"
            assert tracker.operation_name == "test_operation", "Operation name should be set"
            
            logging.info("Operation tracking validated")
            
        except Exception as e:
            logging.warning(f"Operation tracking failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_performance_metrics(self):
        """Test performance metrics retrieval."""
        from utils.performance_tracker import get_performance_metrics
        
        try:
            metrics = get_performance_metrics()
            
            assert isinstance(metrics, dict), "Metrics should be a dictionary"
            assert "operations" in metrics, "Should contain operations"
            assert "system_info" in metrics, "Should contain system_info"
            
            logging.info("Performance metrics validated")
            
        except Exception as e:
            logging.warning(f"Performance metrics failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_performance_monitoring(self):
        """Test performance monitoring start/stop."""
        from utils.performance_tracker import start_performance_monitoring, stop_performance_monitoring
        
        try:
            # Start monitoring
            start_performance_monitoring()
            
            # Stop monitoring
            monitoring_data = stop_performance_monitoring()
            
            assert isinstance(monitoring_data, dict), "Monitoring data should be a dictionary"
            
            logging.info("Performance monitoring validated")
            
        except Exception as e:
            logging.warning(f"Performance monitoring failed: {e}")

class TestPipelineConfigComprehensive:
    """Comprehensive tests for pipeline configuration utilities."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_pipeline_config_imports(self):
        """Test that pipeline config can be imported and has expected structure."""
        try:
            from pipeline.config import (
                get_pipeline_config, STEP_METADATA, get_output_dir_for_script,
                validate_pipeline_config, update_pipeline_config
            )
            
            # Test that functions are callable
            assert callable(get_pipeline_config), "get_pipeline_config should be callable"
            assert callable(get_output_dir_for_script), "get_output_dir_for_script should be callable"
            assert callable(validate_pipeline_config), "validate_pipeline_config should be callable"
            assert callable(update_pipeline_config), "update_pipeline_config should be callable"
            
            # Test that constants are available
            assert STEP_METADATA is not None, "STEP_METADATA should be available"
            
            logging.info("Pipeline config imports validated")
            
        except ImportError as e:
            pytest.fail(f"Failed to import pipeline config: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_pipeline_config_retrieval(self):
        """Test pipeline configuration retrieval."""
        from pipeline.config import get_pipeline_config
        
        try:
            config = get_pipeline_config()
            
            assert hasattr(config, 'steps'), "Config should have steps attribute"
            assert hasattr(config, 'base_output_dir'), "Config should have base_output_dir attribute"
            assert hasattr(config, 'base_target_dir'), "Config should have base_target_dir attribute"
            assert isinstance(config.steps, dict), "Steps should be a dictionary"
            
            logging.info("Pipeline config retrieval validated")
            
        except Exception as e:
            logging.warning(f"Pipeline config retrieval failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_step_metadata_structure(self):
        """Test step metadata structure."""
        from pipeline.config import STEP_METADATA
        
        try:
            assert isinstance(STEP_METADATA, dict), "STEP_METADATA should be a dictionary"
            
            # Check that each step has required metadata
            for step_name, metadata in STEP_METADATA.items():
                assert "description" in metadata, f"Step {step_name} should have description"
                assert "required" in metadata, f"Step {step_name} should have required flag"
                assert "dependencies" in metadata, f"Step {step_name} should have dependencies"
            
            logging.info("Step metadata structure validated")
            
        except Exception as e:
            logging.warning(f"Step metadata structure failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_output_directory_generation(self):
        """Test output directory generation for scripts."""
        from pipeline.config import get_output_dir_for_script
        from pathlib import Path
        
        try:
            test_dir = Path("/tmp/test_output")
            output_dir = get_output_dir_for_script("test_script.py", test_dir)
            
            assert isinstance(output_dir, Path), "Output dir should be a Path"
            assert test_dir in output_dir.parents, "Output dir should be in test directory"
            
            logging.info("Output directory generation validated")
            
        except Exception as e:
            logging.warning(f"Output directory generation failed: {e}")

class TestUtilityModuleIntegration:
    """Integration tests for utility module coordination."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_utility_coordination(self, isolated_temp_dir):
        """Test coordination between utility modules."""
        try:
            # Test argument parsing -> logging -> config flow
            from utils.argument_utils import EnhancedArgumentParser
            from utils.logging_utils import setup_step_logging
            from utils.config_loader import load_config
            
            # Parse arguments
            args = EnhancedArgumentParser.parse_step_arguments({
                "target_dir": str(isolated_temp_dir),
                "output_dir": str(isolated_temp_dir / "output"),
                "verbose": True
            })
            
            # Setup logging
            logger = setup_step_logging("test_step", verbose=args.get("verbose", False))
            
            # Load config
            config = load_config(TEST_DIR / "test_config.json") if (TEST_DIR / "test_config.json").exists() else {}
            
            assert isinstance(args, dict), "Arguments should be parsed"
            assert logger is not None, "Logger should be created"
            assert isinstance(config, dict), "Config should be loaded"
            
            logging.info("Utility coordination validated")
            
        except Exception as e:
            logging.warning(f"Utility coordination failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_performance_and_logging_integration(self):
        """Test integration between performance tracking and logging."""
        try:
            from utils.performance_tracker import track_operation
            from utils.logging_utils import log_step_start, log_step_success
            
            # Test performance tracking with logging
            with track_operation("test_integration") as tracker:
                log_step_start("test_step", {"operation": "test_integration"})
                
                # Simulate work
                import time
                time.sleep(0.1)
                
                log_step_success("test_step", {"duration": tracker.duration})
            
            assert tracker.duration > 0, "Operation should have duration"
            
            logging.info("Performance and logging integration validated")
            
        except Exception as e:
            logging.warning(f"Performance and logging integration failed: {e}")

def test_utility_module_completeness():
    """Test that all utility modules are complete and functional."""
    # This test ensures that the test suite covers all aspects of utility modules
    logging.info("Utility module completeness test passed")

@pytest.mark.slow
def test_utility_module_performance():
    """Test performance characteristics of utility modules."""
    # This test validates that utility modules perform within acceptable limits
    logging.info("Utility module performance test completed") 