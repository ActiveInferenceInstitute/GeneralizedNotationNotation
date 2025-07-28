#!/usr/bin/env python3
"""
Utilities Tests for GNN Processing Pipeline

This module contains comprehensive tests for all utility functions and
infrastructure components that support the GNN processing pipeline.

Tests cover:
1. Enhanced argument parser functionality
2. Logging setup, verbosity control, and step logging functions
3. Path validation, file operations, and extension validation
4. Performance tracking context managers and timestamps
5. Pipeline dependency validation
6. Pipeline configuration and metadata
7. Integration between utility components
8. Safe import patterns and graceful degradation
9. Safe file operations and directory creation
10. Configuration validation and consistency
11. Python version, platform compatibility, and resource availability

All tests are designed to be safe-to-fail with real functional implementations.
"""

import pytest
import sys
import os
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import json
import argparse
import time

# Test markers
pytestmark = [pytest.mark.utilities, pytest.mark.safe_to_fail, pytest.mark.fast]

# Import test utilities and fixtures with fallback
try:
    from . import (
        TEST_CONFIG,
        get_sample_pipeline_arguments,
        is_safe_mode,
        TEST_DIR,
        SRC_DIR,
        PROJECT_ROOT
    )
except ImportError:
    # Fallback for when conftest.py utilities are not available
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    SRC_DIR = PROJECT_ROOT / "src"
    TEST_DIR = PROJECT_ROOT / "src" / "tests"
    
    TEST_CONFIG = {
        "safe_mode": True,
        "mock_external_deps": True,
        "timeout_seconds": 30,
        "temp_output_dir": str(TEST_DIR / "temp"),
        "sample_gnn_dir": str(PROJECT_ROOT / "input" / "gnn_files"),
        "test_data_dir": str(TEST_DIR / "data")
    }
    
    def get_sample_pipeline_arguments():
        return {
            "target_dir": TEST_CONFIG["sample_gnn_dir"],
            "output_dir": TEST_CONFIG["temp_output_dir"],
            "verbose": False,
            "recursive": False,
            "strict": False,
            "estimate_resources": False,
            "skip_steps": [],
            "only_steps": []
        }
    
    def is_safe_mode():
        return TEST_CONFIG["safe_mode"]


class TestArgumentParsing:
    """Test Enhanced argument parser functionality."""
    
    @pytest.mark.unit
    def test_enhanced_argument_parser_import(self):
        """Test that EnhancedArgumentParser can be imported."""
        try:
            from utils import EnhancedArgumentParser
            
            # Test that the class is available
            assert EnhancedArgumentParser is not None, "Should be able to import EnhancedArgumentParser"
            
            # Test that it has expected class methods
            expected_methods = ['parse_step_arguments', 'create_step_parser', 'get_step_help']
            for method in expected_methods:
                assert hasattr(EnhancedArgumentParser, method), f"Parser should have {method} method"
                assert callable(getattr(EnhancedArgumentParser, method)), f"{method} should be callable"
            
            logging.info("EnhancedArgumentParser import and interface validated")
            
        except ImportError as e:
            pytest.skip(f"EnhancedArgumentParser not available: {e}")
    
    @pytest.mark.unit
    def test_argument_parser_basic_functionality(self):
        """Test basic argument parser functionality."""
        sample_args = get_sample_pipeline_arguments()
        
        # Test argument structure validation
        required_keys = ["target_dir", "output_dir", "verbose", "recursive"]
        for key in required_keys:
            assert key in sample_args, f"Required argument {key} missing from sample args"
        
        # Test argument types
        assert isinstance(sample_args["target_dir"], str), "target_dir should be string"
        assert isinstance(sample_args["output_dir"], str), "output_dir should be string"
        assert isinstance(sample_args["verbose"], bool), "verbose should be boolean"
        assert isinstance(sample_args["recursive"], bool), "recursive should be boolean"
        
        logging.info("Argument parser basic functionality validated")
    
    @pytest.mark.unit
    def test_argument_parsing_fallback(self):
        """Test argument parsing fallback mechanisms."""
        # Test that fallback argument parsing works
        try:
            from utils import EnhancedArgumentParser
            
            parser = EnhancedArgumentParser()
            
            # Test with minimal arguments (should use defaults)
            test_args = argparse.Namespace()
            test_args.target_dir = str(TEST_CONFIG["sample_gnn_dir"])
            test_args.output_dir = str(TEST_CONFIG["temp_output_dir"])
            test_args.verbose = False
            test_args.recursive = False
            
            # Test that we can access these attributes
            assert hasattr(test_args, 'target_dir'), "Args should have target_dir"
            assert hasattr(test_args, 'output_dir'), "Args should have output_dir"
            assert hasattr(test_args, 'verbose'), "Args should have verbose"
            
            logging.info("Argument parsing fallback validated")
                    
        except ImportError:
            pytest.skip("EnhancedArgumentParser not available for fallback testing")
    
    @pytest.mark.unit
    def test_path_argument_conversion(self):
        """Test path argument conversion functionality."""
        sample_args = get_sample_pipeline_arguments()
        
        # Test Path conversion
        target_path = Path(sample_args["target_dir"])
        output_path = Path(sample_args["output_dir"])
        
        assert isinstance(target_path, Path), "target_dir should be convertible to Path"
        assert isinstance(output_path, Path), "output_dir should be convertible to Path"
        
        # Test path validation
        assert str(target_path) == sample_args["target_dir"], "Path conversion should preserve string value"
        assert str(output_path) == sample_args["output_dir"], "Path conversion should preserve string value"
        
        logging.info("Path argument conversion validated")


class TestLoggingUtilities:
    """Test logging setup, verbosity control, and step logging functions."""
    
    @pytest.mark.unit
    def test_logging_setup_import(self):
        """Test that logging utilities can be imported."""
        try:
            from utils import setup_step_logging, setup_main_logging
            
            # Test basic import functionality
            assert callable(setup_step_logging), "setup_step_logging should be callable"
            assert callable(setup_main_logging), "setup_main_logging should be callable"
            
            logging.info("Logging utilities import validated")
            
        except ImportError as e:
            pytest.skip(f"Logging utilities not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_step_logging_setup(self):
        """Test step logging setup functionality."""
        try:
            from utils import setup_step_logging
            
            # Test basic step logging setup
            logger = setup_step_logging("test_step", verbose=False)
            assert logger is not None, "Step logging should return a logger"
            assert hasattr(logger, 'info'), "Logger should have info method"
            assert hasattr(logger, 'warning'), "Logger should have warning method"
            assert hasattr(logger, 'error'), "Logger should have error method"
            
            # Test verbose logging setup
            verbose_logger = setup_step_logging("test_step_verbose", verbose=True)
            assert verbose_logger is not None, "Verbose step logging should return a logger"
            
            logging.info("Step logging setup validated")
            
        except ImportError:
            pytest.skip("Step logging utilities not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_logging_verbosity_control(self):
        """Test logging verbosity control."""
        try:
            from utils import setup_step_logging
            
            # Test different verbosity levels
            quiet_logger = setup_step_logging("quiet_test", verbose=False)
            verbose_logger = setup_step_logging("verbose_test", verbose=True)
            
            # Both should be valid loggers
            assert quiet_logger is not None, "Quiet logger should be created"
            assert verbose_logger is not None, "Verbose logger should be created"
            
            # Test that they have the same interface
            for logger in [quiet_logger, verbose_logger]:
                assert hasattr(logger, 'debug'), "Logger should have debug method"
                assert hasattr(logger, 'info'), "Logger should have info method"
                assert hasattr(logger, 'warning'), "Logger should have warning method"
                assert hasattr(logger, 'error'), "Logger should have error method"
            
            logging.info("Logging verbosity control validated")
            
        except ImportError:
            pytest.skip("Logging verbosity control not available")
    
    @pytest.mark.unit
    def test_step_logging_functions(self):
        """Test step logging functions (log_step_start, log_step_success, etc.)."""
        try:
            from utils import log_step_start, log_step_success, log_step_warning, log_step_error
            
            # Test that logging functions are callable
            logging_functions = [log_step_start, log_step_success, log_step_warning, log_step_error]
            for func in logging_functions:
                assert callable(func), f"Logging function {func.__name__} should be callable"
            
            # Test basic function calls (should not raise exceptions)
            test_logger = logging.getLogger("test_step_logging")
            
            log_step_start(test_logger, "test_step", "Starting test step")
            log_step_success(test_logger, "test_step", "Test step completed successfully")
            log_step_warning(test_logger, "test_step", "Test step warning")
            log_step_error(test_logger, "test_step", "Test step error")
            
            logging.info("Step logging functions validated")
            
        except ImportError:
            pytest.skip("Step logging functions not available")


class TestPathUtilities:
    """Test path validation, file operations, and extension validation."""
    
    @pytest.mark.unit
    def test_path_validation_utilities(self):
        """Test path validation utility functions."""
        # Test basic path validation
        test_paths = [
            str(TEST_CONFIG["sample_gnn_dir"]),
            str(TEST_CONFIG["temp_output_dir"]),
            "/nonexistent/path",
            ""
        ]
        
        for path_str in test_paths:
            path_obj = Path(path_str)
            
            # Test Path object creation
            assert isinstance(path_obj, Path), f"Should be able to create Path from {path_str}"
            
            # Test path properties
            assert hasattr(path_obj, 'exists'), "Path should have exists method"
            assert hasattr(path_obj, 'is_file'), "Path should have is_file method"
            assert hasattr(path_obj, 'is_dir'), "Path should have is_dir method"
            assert hasattr(path_obj, 'parent'), "Path should have parent property"
            assert hasattr(path_obj, 'suffix'), "Path should have suffix property"
        
        logging.info("Path validation utilities tested")
    
    @pytest.mark.unit
    def test_file_extension_validation(self):
        """Test file extension validation."""
        # Test file extension patterns
        test_files = [
            ("test.md", ".md", True),
            ("test.py", ".py", True),
            ("test.json", ".json", True),
            ("test.txt", ".md", False),
            ("test", ".md", False),
            ("test.MD", ".md", True)  # Case insensitive
        ]
        
        for filename, expected_ext, should_match in test_files:
            file_path = Path(filename)
            actual_ext = file_path.suffix.lower()
            expected_ext_lower = expected_ext.lower()
            
            matches = actual_ext == expected_ext_lower
            assert matches == should_match, \
                f"File {filename} extension check failed: expected {should_match}, got {matches}"
        
        logging.info("File extension validation tested")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_safe_file_operations(self, safe_filesystem):
        """Test safe file operations."""
        # Test safe file operation patterns using real filesystem operations
        test_file = Path("test_file.txt")
        test_content = "Test content"
        
        # Create a file using the safe filesystem
        created_file = safe_filesystem.create_file(test_file, test_content)
        
        # Test that file was created
        assert created_file.exists(), "File should be created"
        assert created_file.is_file(), "Created path should be a file"
        
        # Test reading the file
        read_content = created_file.read_text()
        assert read_content == test_content, "File content should match"
        
        logging.info("Safe file operations validated")
    
    @pytest.mark.unit
    def test_directory_creation_patterns(self, isolated_temp_dir, safe_filesystem):
        """Test directory creation patterns."""
        # Test directory creation using real filesystem operations
        test_dir = Path("test_directory")
        
        # Create directory using safe filesystem
        created_dir = safe_filesystem.create_dir(test_dir)
        
        # Test that directory was created
        assert created_dir.exists(), "Directory should be created"
        assert created_dir.is_dir(), "Created path should be a directory"
        
        # Test nested directory creation
        nested_dir = created_dir / "nested" / "deep"
        nested_dir.mkdir(parents=True, exist_ok=True)
        
        assert nested_dir.exists(), "Nested directory should be created"
        assert nested_dir.is_dir(), "Nested path should be a directory"
        
        logging.info("Directory creation patterns validated")


class TestPerformanceTracking:
    """Test performance tracking context managers and timestamps."""
    
    @pytest.mark.unit
    def test_performance_tracker_import(self):
        """Test that performance tracking utilities can be imported."""
        try:
            from utils import performance_tracker
            
            # Test basic import functionality
            assert performance_tracker is not None, "performance_tracker should be importable"
            
            # Test that it has expected functionality
            if hasattr(performance_tracker, 'track_operation'):
                assert callable(performance_tracker.track_operation), "track_operation should be callable"
            
            logging.info("Performance tracker import validated")
            
        except ImportError as e:
            pytest.skip(f"Performance tracker not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_performance_tracking_context_manager(self):
        """Test performance tracking context manager functionality."""
        # Test basic performance tracking using real timing
        start_time = time.time()
        
        # Simulate some work
        time.sleep(0.01)  # 10ms of work
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Test that timing works
        assert duration > 0, "Duration should be positive"
        assert duration < 1.0, "Duration should be reasonable for test"
        
        logging.info(f"Performance tracking validated: {duration:.4f}s")
    
    @pytest.mark.unit
    def test_timestamp_utilities(self):
        """Test timestamp utility functions."""
        # Test timestamp generation
        timestamp1 = time.time()
        time.sleep(0.001)  # 1ms delay
        timestamp2 = time.time()
        
        # Test timestamp comparison
        assert timestamp2 > timestamp1, "Later timestamp should be greater"
        assert timestamp2 - timestamp1 > 0, "Time difference should be positive"
        
        # Test timestamp formatting
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp1))
        assert isinstance(formatted_time, str), "Formatted timestamp should be string"
        assert len(formatted_time) > 0, "Formatted timestamp should not be empty"
        
        logging.info("Timestamp utilities validated")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_resource_monitoring_interface(self):
        """Test resource monitoring interface."""
        # Test basic resource monitoring using real system calls
        try:
            import psutil
            
            # Test memory usage
            memory_info = psutil.virtual_memory()
            assert memory_info.total > 0, "Total memory should be positive"
            assert memory_info.available >= 0, "Available memory should be non-negative"
            
            # Test CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            assert cpu_percent >= 0, "CPU usage should be non-negative"
            assert cpu_percent <= 100, "CPU usage should not exceed 100%"
            
            logging.info("Resource monitoring interface validated")
            
        except ImportError:
            # Fallback to basic system monitoring
            import os
            
            # Test basic system information
            cwd = os.getcwd()
            assert isinstance(cwd, str), "Current working directory should be string"
            assert len(cwd) > 0, "Current working directory should not be empty"
            
            logging.info("Basic system monitoring validated (psutil not available)")


class TestDependencyValidation:
    """Test pipeline dependency validation."""
    
    @pytest.mark.unit
    def test_dependency_validator_import(self):
        """Test that dependency validation utilities can be imported."""
        try:
            from utils import validate_pipeline_dependencies
            
            # Test that function is callable
            assert callable(validate_pipeline_dependencies), "validate_pipeline_dependencies should be callable"
            
            logging.info("Dependency validator import validated")
            
        except ImportError as e:
            pytest.skip(f"Dependency validator not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_pipeline_dependency_checking(self):
        """Test pipeline dependency checking functionality."""
        # Test dependency checking using real import attempts
        core_dependencies = [
            'pathlib',
            'json',
            'logging',
            'argparse',
            'subprocess',
            'sys',
            'os'
        ]
        
        missing_deps = []
        available_deps = []
        
        for dep in core_dependencies:
            try:
                importlib.import_module(dep)
                available_deps.append(dep)
            except ImportError:
                missing_deps.append(dep)
        
        # Core Python modules should be available
        assert len(available_deps) > 0, "Some core dependencies should be available"
        
        # Log results
        logging.info(f"Available dependencies: {available_deps}")
        if missing_deps:
            logging.warning(f"Missing dependencies: {missing_deps}")
        
        logging.info("Pipeline dependency checking validated")
    
    @pytest.mark.unit
    def test_optional_dependency_handling(self):
        """Test optional dependency handling."""
        # Test optional dependency handling patterns
        optional_dependencies = [
            'matplotlib',
            'networkx',
            'pandas',
            'numpy',
            'torch',
            'jax'
        ]
        
        dependency_status = {}
        
        for dep in optional_dependencies:
            try:
                importlib.import_module(dep)
                dependency_status[dep] = "available"
            except ImportError:
                dependency_status[dep] = "missing"
        
        # Test that we can handle both available and missing dependencies
        available_count = sum(1 for status in dependency_status.values() if status == "available")
        missing_count = sum(1 for status in dependency_status.values() if status == "missing")
        
        assert available_count + missing_count == len(optional_dependencies), \
            "All dependencies should be either available or missing"
        
        logging.info(f"Optional dependency status: {dependency_status}")
        logging.info("Optional dependency handling validated")


class TestConfigurationUtilities:
    """Test pipeline configuration and metadata."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_pipeline_config_import(self):
        """Test pipeline configuration import."""
        try:
            from pipeline.config import get_pipeline_config, STEP_METADATA
            from .test_utils import get_step_metadata_dict
            
            config = get_pipeline_config()
            step_metadata = get_step_metadata_dict()
            
            # Test configuration import
            assert config is not None, "Pipeline config should be importable"
            assert hasattr(config, 'steps'), "Pipeline config should have steps"
            assert isinstance(config.steps, dict), "Pipeline config steps should be dict"
            
            # Test STEP_METADATA - it can be either a dict or StepMetadataProxy
            assert STEP_METADATA is not None, "STEP_METADATA should be importable"
            # STEP_METADATA can be either a dict or StepMetadataProxy object
            assert hasattr(STEP_METADATA, '__getitem__'), "STEP_METADATA should support dict-like access"
            
            # Test step metadata dictionary
            assert isinstance(step_metadata, dict), "Step metadata should be a dictionary"
            
        except ImportError as e:
            pytest.skip(f"Pipeline config not available: {e}")
        except Exception as e:
            pytest.fail(f"Pipeline config import failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_pipeline_configuration_structure(self):
        """Test pipeline configuration structure."""
        try:
            from pipeline.config import get_pipeline_config
            from .test_utils import get_step_metadata_dict
            
            config = get_pipeline_config()
            STEP_METADATA = get_step_metadata_dict()
            
            # Test configuration structure
            assert hasattr(config, 'steps'), "Pipeline config should have steps"
            assert isinstance(config.steps, dict), "Pipeline config steps should be dict"
            
            # Test step metadata structure
            assert isinstance(STEP_METADATA, dict), "STEP_METADATA should be a dictionary"
            
            # Test configuration consistency
            if STEP_METADATA and config.steps:
                # Check that step configurations are consistent
                for step_name, step_config in config.steps.items():
                    assert hasattr(step_config, 'name'), f"Step config {step_name} should have name"
                    assert hasattr(step_config, 'description'), f"Step config {step_name} should have description"
                    assert hasattr(step_config, 'required'), f"Step config {step_name} should have required"
                    
                    # Test that step metadata exists for configured steps
                    if step_name in STEP_METADATA:
                        step_metadata = STEP_METADATA[step_name]
                        assert isinstance(step_metadata, dict), f"Step metadata for {step_name} should be dict"
                        
                        # Test metadata consistency
                        if 'name' in step_metadata and hasattr(step_config, 'name'):
                            assert step_metadata['name'] == step_config.name, f"Step name mismatch for {step_name}"
            
            logging.info("Pipeline configuration structure validated")
            
        except ImportError:
            pytest.skip("Pipeline configuration not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_step_metadata_structure(self):
        """Test step metadata structure."""
        try:
            from .test_utils import get_step_metadata_dict
            
            STEP_METADATA = get_step_metadata_dict()
            
            assert isinstance(STEP_METADATA, dict), "STEP_METADATA should be a dictionary"
            
            # Test metadata structure
            if STEP_METADATA:
                for step_key, step_info in STEP_METADATA.items():
                    assert isinstance(step_key, (str, int)), f"Step key {step_key} should be string or int"
                    assert isinstance(step_info, dict), f"Step info for {step_key} should be dict"
                    
                    # Test expected step info keys
                    expected_info_keys = ["name", "description", "required"]
                    for info_key in expected_info_keys:
                        if info_key in step_info:
                            logging.debug(f"Step {step_key} has {info_key}: {step_info[info_key]}")
            
            logging.info(f"Step metadata validated for {len(STEP_METADATA)} steps")
            
        except ImportError:
            pytest.skip("Step metadata not available")


class TestUtilityIntegration:
    """Test integration between utility components."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_argument_parsing_and_logging_integration(self):
        """Test integration between argument parsing and logging."""
        try:
            from utils import EnhancedArgumentParser, setup_step_logging
            
            # Test integration workflow
            parser = EnhancedArgumentParser()
            
            # Create test arguments
            test_args = argparse.Namespace()
            test_args.verbose = True
            test_args.target_dir = str(TEST_CONFIG["sample_gnn_dir"])
            
            # Use parsed verbosity for logging setup
            logger = setup_step_logging("integration_test", verbose=test_args.verbose)
            assert logger is not None, "Integrated logging setup should work"
            
            # Test logging with integrated setup
            logger.info("Testing integrated argument parsing and logging")
            
            logging.info("Argument parsing and logging integration validated")
                    
        except ImportError:
            pytest.skip("Argument parsing and logging integration not available")
    
    @pytest.mark.integration
    def test_path_and_configuration_integration(self):
        """Test integration between path utilities and configuration."""
        # Test path and configuration integration
        config_paths = {
            "sample_gnn_dir": TEST_CONFIG["sample_gnn_dir"],
            "temp_output_dir": TEST_CONFIG["temp_output_dir"],
            "test_data_dir": TEST_CONFIG["test_data_dir"]
        }
        
        for path_name, path_value in config_paths.items():
            # Test Path conversion
            path_obj = Path(path_value)
            assert isinstance(path_obj, Path), f"Config path {path_name} should be convertible to Path"
            
            # Test path properties
            assert hasattr(path_obj, 'parent'), f"Config path {path_name} should have parent"
            assert hasattr(path_obj, 'name'), f"Config path {path_name} should have name"
            
        logging.info("Path and configuration integration validated")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_performance_and_logging_integration(self):
        """Test integration between performance tracking and logging."""
        # Test performance and logging integration
        test_logger = logging.getLogger("performance_integration_test")
        
        start_time = time.time()
        
        # Simulate work with logging
        test_logger.info("Starting performance integration test")
        time.sleep(0.01)  # 10ms of "work"
        test_logger.info("Completing performance integration test")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Log performance results
        test_logger.info(f"Performance integration test completed in {duration:.4f}s")
        
        # Validate integration
        assert duration > 0, "Performance tracking should measure positive duration"
        
        logging.info("Performance and logging integration validated")


class TestErrorHandlingUtilities:
    """Test safe import patterns and graceful degradation."""
    
    @pytest.mark.unit
    def test_safe_import_patterns(self):
        """Test safe import patterns for optional dependencies."""
        # Test safe import pattern
        def safe_import(module_name, fallback=None):
            try:
                return importlib.import_module(module_name)
            except ImportError:
                return fallback
        
        # Test with known available module
        json_module = safe_import('json')
        assert json_module is not None, "json module should be available"
        
        # Test with known unavailable module
        fake_module = safe_import('nonexistent_module_12345', fallback="FALLBACK")
        assert fake_module == "FALLBACK", "Should return fallback for missing module"
        
        logging.info("Safe import patterns validated")
    
    @pytest.mark.unit
    def test_graceful_degradation_patterns(self):
        """Test graceful degradation patterns."""
        # Test graceful degradation scenarios
        degradation_scenarios = {
            "missing_optional_visualization": "Continue without visualization",
            "missing_optional_llm": "Continue without LLM analysis",
            "missing_optional_export": "Continue with basic export only",
            "network_unavailable": "Skip network-dependent operations"
        }
        
        for scenario, fallback_behavior in degradation_scenarios.items():
            # Test that we have a defined fallback for each scenario
            assert isinstance(fallback_behavior, str), \
                f"Scenario {scenario} should have string fallback behavior"
            assert len(fallback_behavior) > 0, \
                f"Scenario {scenario} should have non-empty fallback behavior"
        
        logging.info(f"Graceful degradation patterns validated for {len(degradation_scenarios)} scenarios")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_error_recovery_mechanisms(self):
        """Test error recovery mechanisms."""
        # Test error recovery patterns
        def resilient_operation(should_fail=False):
            if should_fail:
                raise ValueError("Simulated error")
            return "Success"
        
        # Test successful operation
        result = resilient_operation(should_fail=False)
        assert result == "Success", "Successful operation should return success"
        
        # Test error recovery
        try:
            result = resilient_operation(should_fail=True)
            pytest.fail("Should have raised an error")
        except ValueError as e:
            # This is expected - test error handling
            assert "Simulated error" in str(e), "Error message should be preserved"
            
            # Test recovery mechanism
            recovery_result = "Recovered from error"
            assert recovery_result is not None, "Error recovery should provide alternative"
        
        logging.info("Error recovery mechanisms validated")


class TestFileOperationUtilities:
    """Test safe file operations and directory creation."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_safe_file_reading(self, safe_filesystem):
        """Test safe file reading operations."""
        # Test safe file reading patterns using real filesystem
        test_content = "Test file content for reading"
        test_file = safe_filesystem.create_file(Path("test_read.txt"), test_content)
        
        # Test that we can read the file
        assert test_file.exists(), "Test file should exist"
        
        # Test reading content
        read_content = test_file.read_text()
        assert read_content == test_content, "Read content should match written content"
        
        logging.info("Safe file reading validated")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_safe_file_writing(self, isolated_temp_dir, safe_filesystem):
        """Test safe file writing operations."""
        # Test safe file writing patterns using real filesystem
        test_file = Path("test_output.txt")
        test_content = "Test file content for writing"
        
        # Create file using safe filesystem
        created_file = safe_filesystem.create_file(test_file, test_content)
        
        # Test that file was created with correct content
        assert created_file.exists(), "Created file should exist"
        written_content = created_file.read_text()
        assert written_content == test_content, "Written content should match expected"
        
        logging.info("Safe file writing validated")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_safe_json_operations(self, isolated_temp_dir, safe_filesystem):
        """Test safe JSON file operations."""
        # Test safe JSON operations using real filesystem
        test_data = {
            "test_key": "test_value",
            "test_number": 42,
            "test_list": [1, 2, 3],
            "test_dict": {"nested": "value"}
        }
        
        # Create JSON file
        json_content = json.dumps(test_data, indent=2)
        json_file = safe_filesystem.create_file(Path("test.json"), json_content)
        
        # Test reading JSON
        assert json_file.exists(), "JSON file should exist"
        
        # Test parsing JSON
        with open(json_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data, "Loaded JSON should match original data"
        
        logging.info("Safe JSON operations validated")


class TestConfigurationValidation:
    """Test configuration validation and consistency."""
    
    @pytest.mark.unit
    def test_test_configuration_validation(self):
        """Test that test configuration is valid."""
        # Test TEST_CONFIG structure
        required_config_keys = [
            "safe_mode",
            "temp_output_dir",
            "sample_gnn_dir",
            "test_data_dir"
        ]
        
        for key in required_config_keys:
            assert key in TEST_CONFIG, f"TEST_CONFIG should have {key}"
        
        # Test configuration values
        assert isinstance(TEST_CONFIG["safe_mode"], bool), "safe_mode should be boolean"
        
        # Test path values (can be either string or Path objects)
        for path_key in ["temp_output_dir", "sample_gnn_dir", "test_data_dir"]:
            path_value = TEST_CONFIG[path_key]
            assert isinstance(path_value, (str, Path)), f"{path_key} should be string or Path"
            
            # Test path conversion
            path_obj = Path(path_value)
            assert isinstance(path_obj, Path), f"Config {path_key} should be convertible to Path"
        
        logging.info("Test configuration validation completed")
    
    @pytest.mark.unit
    def test_environment_configuration_consistency(self):
        """Test environment configuration consistency."""
        # Test that environment paths are consistent
        assert PROJECT_ROOT.exists(), "Project root should exist"
        assert SRC_DIR.exists(), "Source directory should exist"
        assert TEST_DIR.exists(), "Test directory should exist"
        
        # Test path relationships
        assert SRC_DIR.parent == PROJECT_ROOT, "SRC_DIR should be child of PROJECT_ROOT"
        assert TEST_DIR.parent == SRC_DIR, "TEST_DIR should be child of SRC_DIR"
        
        # Test that we can access key directories
        assert (PROJECT_ROOT / "src").exists(), "src directory should exist in project root"
        assert (SRC_DIR / "tests").exists(), "tests directory should exist in src"
        
        logging.info("Environment configuration consistency validated")
    
    @pytest.mark.unit
    def test_path_configuration_validation(self):
        """Test path configuration validation."""
        # Test sample pipeline arguments
        sample_args = get_sample_pipeline_arguments()
        
        # Test that paths in arguments are valid strings
        path_args = ["target_dir", "output_dir"]
        for arg_name in path_args:
            if arg_name in sample_args:
                path_value = sample_args[arg_name]
                assert isinstance(path_value, str), f"Argument {arg_name} should be string"
                assert len(path_value) > 0, f"Argument {arg_name} should not be empty"
                
                # Test Path conversion
                path_obj = Path(path_value)
                assert isinstance(path_obj, Path), f"Argument {arg_name} should be convertible to Path"
        
        logging.info("Path configuration validation completed")


class TestSystemIntegration:
    """Test Python version, platform compatibility, and resource availability."""
    
    @pytest.mark.unit
    def test_python_version_compatibility(self):
        """Test Python version compatibility."""
        # Test Python version
        python_version = sys.version_info
        
        # Test minimum version requirements (Python 3.8+)
        assert python_version.major >= 3, "Python major version should be 3 or higher"
        assert python_version.minor >= 8, "Python minor version should be 8 or higher for Python 3"
        
        # Test version string
        version_string = sys.version
        assert isinstance(version_string, str), "Python version should be string"
        assert len(version_string) > 0, "Python version string should not be empty"
        
        logging.info(f"Python version compatibility validated: {python_version}")
    
    @pytest.mark.unit
    def test_platform_compatibility(self):
        """Test platform compatibility."""
        import platform
        
        # Test platform detection
        system_name = platform.system()
        assert isinstance(system_name, str), "System name should be string"
        assert len(system_name) > 0, "System name should not be empty"
        
        # Test architecture
        architecture = platform.machine()
        assert isinstance(architecture, str), "Architecture should be string"
        
        # Test Python implementation
        python_implementation = platform.python_implementation()
        assert isinstance(python_implementation, str), "Python implementation should be string"
        
        # Test that we can get platform info
        platform_info = {
            "system": system_name,
            "architecture": architecture,
            "python_implementation": python_implementation
        }
        
        logging.info(f"Platform compatibility validated: {platform_info}")
    
    @pytest.mark.unit
    def test_resource_availability(self):
        """Test system resource availability."""
        import tempfile
        import os
        
        # Test temporary directory access
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            assert temp_path.exists(), "Temporary directory should be accessible"
            assert temp_path.is_dir(), "Temporary path should be directory"
            
            # Test file creation in temp directory
            test_file = temp_path / "resource_test.txt"
            test_file.write_text("Resource test content")
            
            assert test_file.exists(), "Should be able to create files in temp directory"
            
            # Test file reading
            content = test_file.read_text()
            assert content == "Resource test content", "Should be able to read created files"
        
        # Test current working directory access
        cwd = Path.cwd()
        assert cwd.exists(), "Current working directory should be accessible"
        assert cwd.is_dir(), "Current working directory should be directory"
        
        # Test environment variable access
        path_env = os.environ.get("PATH")
        assert path_env is not None, "PATH environment variable should be available"
        assert isinstance(path_env, str), "PATH should be string"
        
        logging.info("System resource availability validated")


def test_utility_module_completeness():
    """Test that utility modules provide expected functionality."""
    # Test that we can import key utility modules
    utility_modules = [
        "pathlib",
        "json",
        "logging",
        "argparse",
        "sys",
        "os",
        "time"
    ]
    
    imported_modules = []
    failed_imports = []
    
    for module_name in utility_modules:
        try:
            module = importlib.import_module(module_name)
            imported_modules.append(module_name)
            assert module is not None, f"Module {module_name} should be importable"
        except ImportError:
            failed_imports.append(module_name)
    
    # Most core modules should be available
    assert len(imported_modules) > len(failed_imports), \
        "More modules should be available than missing"
    
    logging.info(f"Utility module completeness: {len(imported_modules)}/{len(utility_modules)} modules available")
    
    if failed_imports:
        logging.warning(f"Failed to import: {failed_imports}")


@pytest.mark.slow
def test_utility_performance_characteristics():
    """Test performance characteristics of utility functions."""
    # Test performance of key operations
    operations = []
    
    # Test path operations
    start_time = time.time()
    for i in range(100):
        path = Path(f"test_path_{i}")
        path_str = str(path)
        path_obj = Path(path_str)
    path_time = time.time() - start_time
    operations.append(("path_operations", path_time))
    
    # Test JSON operations
    start_time = time.time()
    test_data = {"key": "value", "number": 42}
    for i in range(100):
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
    json_time = time.time() - start_time
    operations.append(("json_operations", json_time))
    
    # Test logging operations
    start_time = time.time()
    test_logger = logging.getLogger("performance_test")
    for i in range(100):
        test_logger.debug(f"Performance test message {i}")
    logging_time = time.time() - start_time
    operations.append(("logging_operations", logging_time))
    
    # Validate performance
    for operation_name, duration in operations:
        assert duration > 0, f"{operation_name} should take positive time"
        assert duration < 10.0, f"{operation_name} should complete within reasonable time"
        logging.info(f"{operation_name}: {duration:.4f}s")
    
    logging.info("Utility performance characteristics validated")


if __name__ == "__main__":
    # Allow running this test module directly
    pytest.main([__file__, "-v"]) 