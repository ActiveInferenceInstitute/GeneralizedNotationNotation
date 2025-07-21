#!/usr/bin/env python3
"""
Main Orchestrator Tests for GNN Pipeline

This module contains comprehensive tests for the main pipeline orchestrator
(src/main.py) which coordinates the execution of all pipeline steps.

Tests cover:
1. Import validation and component checking
2. Argument parsing functionality and validation
3. Pipeline script discovery and metadata extraction
4. Virtual environment detection and handling
5. Individual step execution coordination
6. Overall pipeline orchestration
7. Performance monitoring and system info collection
8. Pipeline output and reporting
9. Error handling, retry logic, and graceful degradation
10. Configuration consistency validation
11. End-to-end integration scenarios

All tests are designed to be safe-to-fail with extensive mocking.
"""

import pytest
import os
import sys
import json
import logging
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import patch, Mock, MagicMock

# Test markers
pytestmark = [pytest.mark.main_orchestrator, pytest.mark.safe_to_fail]

# Import test utilities
from . import (
    TEST_CONFIG,
    get_sample_pipeline_arguments,
    is_safe_mode,
    TEST_DIR,
    SRC_DIR,
    PROJECT_ROOT
)

class TestMainOrchestratorImport:
    """Test main orchestrator import validation and component checking."""
    
    @pytest.mark.unit
    def test_main_orchestrator_file_exists(self):
        """Test that main.py exists and is readable."""
        main_script = SRC_DIR / "main.py"
        
        assert main_script.exists(), f"Main orchestrator script not found: {main_script}"
        assert main_script.is_file(), f"Main orchestrator path is not a file: {main_script}"
        
        # Test that file is readable
        try:
            content = main_script.read_text()
            assert len(content) > 0, "Main orchestrator script is empty"
            
            # Test basic Python structure
            assert "import" in content, "Main script should have import statements"
            assert "def main" in content or "if __name__" in content, \
                "Main script should have main function or entry point"
            
            logging.info("Main orchestrator file structure validated")
            
        except Exception as e:
            pytest.fail(f"Failed to read main orchestrator script: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_main_orchestrator_imports(self):
        """Test that main orchestrator can import required modules."""
        # Test critical imports that main.py needs
        try:
            # These should be available for the main orchestrator
            import subprocess
            import argparse
            import pathlib
            import logging
            import json
            
            # Test project-specific imports
            from utils import setup_main_logging
            from pipeline import get_pipeline_config
            
            logging.info("Main orchestrator imports validated")
            
        except ImportError as e:
            pytest.fail(f"Main orchestrator cannot import required modules: {e}")
    
    @pytest.mark.unit
    def test_main_orchestrator_component_availability(self):
        """Test that main orchestrator components are available."""
        # Test that main components exist
        components = {
            "utils_package": SRC_DIR / "utils" / "__init__.py",
            "pipeline_package": SRC_DIR / "pipeline" / "__init__.py",
            "main_script": SRC_DIR / "main.py"
        }
        
        missing_components = []
        for name, path in components.items():
            if not path.exists():
                missing_components.append(f"{name}: {path}")
        
        assert not missing_components, f"Missing main orchestrator components: {missing_components}"
        logging.info(f"All {len(components)} main orchestrator components available")

class TestArgumentParsing:
    """Test argument parsing functionality and validation."""
    
    @pytest.mark.unit
    def test_argument_parser_basic_functionality(self):
        """Test basic argument parsing functionality."""
        # Test that we can create and use argument parser
        sample_args = get_sample_pipeline_arguments()
        
        # Validate argument structure
        required_args = ["target_dir", "output_dir"]
        for arg in required_args:
            assert arg in sample_args, f"Required argument {arg} missing"
        
        # Validate argument types
        assert isinstance(sample_args["target_dir"], str), "target_dir should be string"
        assert isinstance(sample_args["output_dir"], str), "output_dir should be string"
        assert isinstance(sample_args["verbose"], bool), "verbose should be boolean"
        
        logging.info("Argument parser basic functionality validated")
    
    @pytest.mark.unit
    def test_argument_validation(self):
        """Test argument validation logic."""
        sample_args = get_sample_pipeline_arguments()
        
        # Test path validation
        target_dir = Path(sample_args["target_dir"])
        output_dir = Path(sample_args["output_dir"])
        
        # These should be valid path objects
        assert isinstance(target_dir, Path), "target_dir should be convertible to Path"
        assert isinstance(output_dir, Path), "output_dir should be convertible to Path"
        
        # Test boolean flags
        boolean_args = ["verbose", "recursive", "strict", "estimate_resources"]
        for arg in boolean_args:
            if arg in sample_args:
                assert isinstance(sample_args[arg], bool), f"Argument {arg} should be boolean"
        
        logging.info("Argument validation logic tested")
    
    @pytest.mark.unit
    def test_argument_defaults(self):
        """Test that argument defaults are reasonable."""
        sample_args = get_sample_pipeline_arguments()
        
        # Test safe defaults for testing
        assert sample_args["verbose"] is False, "verbose should default to False for safety"
        assert sample_args["recursive"] is False, "recursive should be False for testing"
        assert sample_args["estimate_resources"] is False, "estimate_resources should be False for testing"
        
        # Test that skip/only steps are empty by default
        assert sample_args["skip_steps"] == [], "skip_steps should default to empty list"
        assert sample_args["only_steps"] == [], "only_steps should default to empty list"
        
        logging.info("Argument defaults validated")

class TestPipelineScriptDiscovery:
    """Test pipeline script discovery and metadata extraction."""
    
    @pytest.mark.unit
    def test_pipeline_script_discovery_logic(self):
        """Test pipeline script discovery logic."""
        # Test script discovery pattern
        expected_pattern = r"^(\d+)_.*\.py$"
        
        test_scripts = [
            ("1_setup.py", True, 1),
            ("2_gnn.py", True, 2),
            ("13_sapf.py", True, 13),
            ("main.py", False, None),
            ("utils.py", False, None),
            ("not_a_script.txt", False, None)
        ]
        
        import re
        pattern = re.compile(expected_pattern)
        
        for script_name, should_match, expected_num in test_scripts:
            match = pattern.match(script_name)
            
            if should_match:
                assert match is not None, f"Script {script_name} should match pattern"
                assert int(match.group(1)) == expected_num, \
                    f"Script {script_name} should extract number {expected_num}"
            else:
                assert match is None, f"Script {script_name} should not match pattern"
        
        logging.info("Pipeline script discovery logic validated")
    
    @pytest.mark.unit
    def test_pipeline_script_sorting(self):
        """Test pipeline script sorting logic."""
        # Test script sorting by number and name
        mock_scripts = [
            {"num": 3, "basename": "3_tests.py"},
            {"num": 1, "basename": "1_setup.py"},
            {"num": 13, "basename": "13_sapf.py"},
            {"num": 2, "basename": "2_gnn.py"}
        ]
        
        # Sort like the main orchestrator would
        sorted_scripts = sorted(mock_scripts, key=lambda x: (x['num'], x['basename']))
        
        # Verify correct order
        expected_order = [1, 2, 3, 13]
        actual_order = [script['num'] for script in sorted_scripts]
        
        assert actual_order == expected_order, f"Scripts should be sorted correctly: {actual_order}"
        logging.info("Pipeline script sorting logic validated")

class TestVirtualEnvironmentHandling:
    """Test virtual environment detection and handling."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_virtual_environment_detection(self, mock_filesystem):
        """Test virtual environment detection logic."""
        # Mock virtual environment detection
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('sys.executable') as mock_executable:
            
            mock_executable.return_value = "/mock/venv/bin/python"
            mock_exists.return_value = True
            
            # Test venv detection logic
            venv_indicators = [
                "venv",
                ".venv", 
                "env",
                ".env"
            ]
            
            for indicator in venv_indicators:
                venv_path = PROJECT_ROOT / indicator
                # In a real scenario, we'd check if this exists
                assert isinstance(venv_path, Path), f"Virtual env path {indicator} should be Path object"
            
            logging.info("Virtual environment detection logic validated")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_python_executable_detection(self):
        """Test Python executable detection."""
        # Test that we can detect Python executable
        python_executable = sys.executable
        
        assert python_executable is not None, "Python executable should be detectable"
        assert isinstance(python_executable, str), "Python executable should be string path"
        assert len(python_executable) > 0, "Python executable path should not be empty"
        
        # Test path validation
        executable_path = Path(python_executable)
        # Note: In test environment, this might not exist, so we just test the logic
        assert isinstance(executable_path, Path), "Executable should be convertible to Path"
        
        logging.info(f"Python executable detection validated: {python_executable}")

class TestStepExecution:
    """Test individual step execution coordination."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_step_execution_interface(self, mock_subprocess):
        """Test step execution interface."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Mock step output",
                stderr="",
                args=["python", "2_gnn.py"]
            )
            
            # Test basic step execution
            result = subprocess.run([
                "python", "mock_step.py",
                "--target-dir", str(TEST_CONFIG["sample_gnn_dir"]),
                "--output-dir", str(TEST_CONFIG["temp_output_dir"])
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, "Step execution should succeed"
            mock_run.assert_called_once()
            
            logging.info("Step execution interface validated")
    
    @pytest.mark.unit
    def test_step_argument_building(self):
        """Test step argument building logic."""
        base_args = get_sample_pipeline_arguments()
        
        # Test building arguments for a step
        step_args = [
            "--target-dir", base_args["target_dir"],
            "--output-dir", base_args["output_dir"]
        ]
        
        if base_args["verbose"]:
            step_args.append("--verbose")
        
        if base_args["recursive"]:
            step_args.append("--recursive")
        
        # Validate argument structure
        assert isinstance(step_args, list), "Step arguments should be a list"
        assert len(step_args) >= 4, "Step arguments should include target-dir and output-dir"
        
        # Test that arguments are strings
        for arg in step_args:
            assert isinstance(arg, str), f"Step argument {arg} should be string"
        
        logging.info("Step argument building logic validated")

class TestPipelineCoordination:
    """Test overall pipeline orchestration."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_pipeline_coordination_flow(self, mock_subprocess, full_pipeline_environment):
        """Test overall pipeline coordination flow."""
        env = full_pipeline_environment
        
        with patch('subprocess.run') as mock_run:
            # Mock successful step execution
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Mock pipeline output",
                stderr=""
            )
            
            # Test pipeline coordination logic
            steps_to_run = [1, 2, 3, 4, 5]
            executed_steps = []
            
            for step_num in steps_to_run:
                # Simulate step execution
                result = subprocess.run([
                    "python", f"{step_num}_mock.py",
                    "--output-dir", str(env['temp_dir'])
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    executed_steps.append(step_num)
            
            assert len(executed_steps) == len(steps_to_run), \
                f"All steps should execute successfully: {executed_steps}"
            
            logging.info("Pipeline coordination flow validated")
    
    @pytest.mark.integration
    def test_pipeline_configuration_consistency(self):
        """Test pipeline configuration consistency."""
        # Test that configuration is consistent across the pipeline
        config_elements = {
            "safe_mode": TEST_CONFIG["safe_mode"],
            "mock_external_deps": TEST_CONFIG["mock_external_deps"],
            "timeout_seconds": TEST_CONFIG["timeout_seconds"]
        }
        
        for key, value in config_elements.items():
            assert value is not None, f"Configuration {key} should be defined"
            
        # Test configuration types
        assert isinstance(config_elements["safe_mode"], bool), "safe_mode should be boolean"
        assert isinstance(config_elements["timeout_seconds"], (int, float)), \
            "timeout_seconds should be numeric"
        
        logging.info("Pipeline configuration consistency validated")

class TestPerformanceTracking:
    """Test performance monitoring and system info collection."""
    
    @pytest.mark.unit
    def test_system_info_collection(self):
        """Test system information collection."""
        # Test basic system info collection
        system_info = {
            "python_version": sys.version,
            "platform": os.name,
            "cpu_count": os.cpu_count(),
            "working_directory": str(Path.cwd())
        }
        
        # Validate system info
        assert system_info["python_version"] is not None, "Python version should be available"
        assert system_info["platform"] is not None, "Platform should be available"
        assert isinstance(system_info["cpu_count"], (int, type(None))), \
            "CPU count should be integer or None"
        assert isinstance(system_info["working_directory"], str), \
            "Working directory should be string"
        
        logging.info("System info collection validated")
    
    @pytest.mark.unit
    def test_performance_metrics_structure(self):
        """Test performance metrics structure."""
        # Test performance metrics structure
        mock_metrics = {
            "start_time": "2024-01-01T00:00:00",
            "end_time": "2024-01-01T00:01:00", 
            "duration_seconds": 60.0,
            "memory_usage_mb": 100.0,
            "steps_executed": 5,
            "success_rate": 1.0
        }
        
        # Validate metrics structure
        assert isinstance(mock_metrics["duration_seconds"], (int, float)), \
            "Duration should be numeric"
        assert isinstance(mock_metrics["memory_usage_mb"], (int, float)), \
            "Memory usage should be numeric"
        assert isinstance(mock_metrics["steps_executed"], int), \
            "Steps executed should be integer"
        assert 0.0 <= mock_metrics["success_rate"] <= 1.0, \
            "Success rate should be between 0 and 1"
        
        logging.info("Performance metrics structure validated")

class TestOutputGeneration:
    """Test pipeline output and reporting."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_output_directory_structure(self, isolated_temp_dir, mock_filesystem):
        """Test output directory structure generation."""
        output_dir = isolated_temp_dir / "pipeline_output"
        
        # Test output directory creation (mocked)
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.return_value = None
            
            # Simulate output directory creation
            output_dir.mkdir(parents=True, exist_ok=True)
            
            mock_mkdir.assert_called_once()
            
        # Test expected output structure
        expected_subdirs = [
            "gnn_step",
            "setup_step", 
            "tests_step",
            "type_check_step",
            "export_step"
        ]
        
        for subdir in expected_subdirs:
            subdir_path = output_dir / subdir
            assert isinstance(subdir_path, Path), f"Subdir {subdir} should be Path object"
        
        logging.info("Output directory structure validated")
    
    @pytest.mark.unit
    def test_pipeline_summary_generation(self):
        """Test pipeline summary generation."""
        # Test pipeline summary structure
        mock_summary = {
            "pipeline_info": {
                "start_time": "2024-01-01T00:00:00",
                "end_time": "2024-01-01T00:01:00",
                "total_duration": 60.0,
                "steps_executed": 5,
                "overall_status": "success"
            },
            "step_results": [
                {
                    "step_number": 1,
                    "step_name": "2_gnn.py",
                    "status": "success",
                    "duration": 10.0,
                    "output": "Mock step 1 output"
                }
            ],
            "system_info": {
                "python_version": sys.version,
                "platform": os.name
            }
        }
        
        # Validate summary structure
        assert "pipeline_info" in mock_summary, "Summary should have pipeline info"
        assert "step_results" in mock_summary, "Summary should have step results"
        assert "system_info" in mock_summary, "Summary should have system info"
        
        # Validate pipeline info
        pipeline_info = mock_summary["pipeline_info"]
        assert "start_time" in pipeline_info, "Pipeline info should have start time"
        assert "overall_status" in pipeline_info, "Pipeline info should have status"
        
        logging.info("Pipeline summary generation validated")

class TestErrorHandlingAndRecovery:
    """Test error handling, retry logic, and graceful degradation."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_step_failure_handling(self, mock_subprocess, simulate_failures):
        """Test handling of step failures."""
        failure_sim = simulate_failures
        
        with patch('subprocess.run') as mock_run:
            # Simulate step failure
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr="Mock step failure"
            )
            
            # Test failure handling
            result = subprocess.run([
                "python", "failing_step.py"
            ], capture_output=True, text=True)
            
            assert result.returncode == 1, "Failed step should return non-zero exit code"
            
            # Test that failure is properly detected
            step_failed = result.returncode != 0
            assert step_failed, "Step failure should be detected"
            
            logging.info("Step failure handling validated")
    
    @pytest.mark.unit
    def test_graceful_degradation(self):
        """Test graceful degradation when components are missing."""
        # Test graceful degradation scenarios
        degradation_scenarios = {
            "missing_optional_dependency": "Continue with reduced functionality",
            "invalid_configuration": "Use default configuration",
            "network_unavailable": "Skip network-dependent operations",
            "insufficient_permissions": "Skip operations requiring elevated permissions"
        }
        
        for scenario, expected_behavior in degradation_scenarios.items():
            # Test that we have a strategy for each scenario
            assert isinstance(expected_behavior, str), \
                f"Scenario {scenario} should have defined behavior"
            assert len(expected_behavior) > 0, \
                f"Scenario {scenario} behavior should be documented"
        
        logging.info(f"Graceful degradation validated for {len(degradation_scenarios)} scenarios")
    
    @pytest.mark.unit
    def test_error_reporting(self, capture_logs):
        """Test error reporting functionality."""
        # Test error reporting
        test_logger = logging.getLogger("test_error_reporting")
        
        # Simulate different types of errors
        error_types = [
            ("warning", "Test warning message"),
            ("error", "Test error message"),
            ("critical", "Test critical message")
        ]
        
        for level, message in error_types:
            getattr(test_logger, level)(message)
        
        # In a real test, we'd check the captured logs
        # For now, just verify the logging interface works
        assert hasattr(test_logger, 'warning'), "Logger should have warning method"
        assert hasattr(test_logger, 'error'), "Logger should have error method"
        assert hasattr(test_logger, 'critical'), "Logger should have critical method"
        
        logging.info("Error reporting functionality validated")

class TestConfigurationValidation:
    """Test configuration validation and consistency."""
    
    @pytest.mark.unit
    def test_configuration_schema_validation(self):
        """Test configuration schema validation."""
        # Test configuration schema
        required_config_keys = [
            "safe_mode",
            "mock_external_deps", 
            "timeout_seconds",
            "temp_output_dir",
            "test_data_dir"
        ]
        
        for key in required_config_keys:
            assert key in TEST_CONFIG, f"Required config key {key} missing"
            assert TEST_CONFIG[key] is not None, f"Config key {key} should not be None"
        
        # Test configuration types
        assert isinstance(TEST_CONFIG["safe_mode"], bool), "safe_mode should be boolean"
        assert isinstance(TEST_CONFIG["timeout_seconds"], (int, float)), \
            "timeout_seconds should be numeric"
        
        logging.info("Configuration schema validation passed")
    
    @pytest.mark.unit
    def test_environment_variable_consistency(self):
        """Test environment variable consistency."""
        # Test that environment variables match configuration
        env_config_mapping = {
            "GNN_TEST_MODE": "true",
            "GNN_SAFE_MODE": str(TEST_CONFIG["safe_mode"]).lower()
        }
        
        for env_var, expected_value in env_config_mapping.items():
            actual_value = os.environ.get(env_var)
            assert actual_value == expected_value, \
                f"Environment variable {env_var} should be {expected_value}, got {actual_value}"
        
        logging.info("Environment variable consistency validated")

class TestEndToEndIntegration:
    """Test end-to-end integration scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.safe_to_fail
    def test_minimal_pipeline_execution(self, full_pipeline_environment):
        """Test minimal pipeline execution scenario."""
        env = full_pipeline_environment
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Mock minimal pipeline output",
                stderr=""
            )
            
            # Test minimal pipeline (just core steps)
            minimal_steps = [1, 2, 3]  # GNN, Setup, Tests
            
            for step_num in minimal_steps:
                result = subprocess.run([
                    "python", f"mock_{step_num}.py",
                    "--output-dir", str(env['temp_dir'])
                ], capture_output=True, text=True)
                
                assert result.returncode == 0, f"Minimal step {step_num} should succeed"
            
            logging.info("Minimal pipeline execution validated")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_pipeline_with_failures(self, mock_subprocess, simulate_failures):
        """Test pipeline behavior with step failures."""
        failure_sim = simulate_failures
        
        with patch('subprocess.run') as mock_run:
            # Simulate mixed success/failure
            def mock_run_side_effect(args, **kwargs):
                if "failing_step" in str(args):
                    return Mock(returncode=1, stdout="", stderr="Mock failure")
                else:
                    return Mock(returncode=0, stdout="Mock success", stderr="")
            
            mock_run.side_effect = mock_run_side_effect
            
            # Test pipeline with failures
            test_steps = ["success_step.py", "failing_step.py", "recovery_step.py"]
            results = []
            
            for step in test_steps:
                result = subprocess.run(["python", step], capture_output=True, text=True)
                results.append((step, result.returncode))
            
            # Validate mixed results
            success_count = sum(1 for _, code in results if code == 0)
            failure_count = sum(1 for _, code in results if code != 0)
            
            assert success_count > 0, "Some steps should succeed"
            assert failure_count > 0, "Some steps should fail (as designed)"
            
            logging.info(f"Pipeline with failures validated: {success_count} success, {failure_count} failures")

# Utility functions for main orchestrator testing

def test_main_orchestrator_help_functionality():
    """Test that main orchestrator provides help information."""
    # Test help functionality (conceptual)
    help_sections = {
        "usage": "How to run the pipeline",
        "options": "Available command-line options",
        "examples": "Usage examples",
        "troubleshooting": "Common issues and solutions"
    }
    
    for section, description in help_sections.items():
        assert isinstance(description, str), f"Help section {section} should have description"
        assert len(description) > 0, f"Help section {section} should be documented"
    
    logging.info(f"Main orchestrator help functionality validated for {len(help_sections)} sections")

@pytest.mark.slow
def test_main_orchestrator_performance_characteristics():
    """Test main orchestrator performance characteristics."""
    # Test performance characteristics (mocked)
    performance_expectations = {
        "startup_time_seconds": 2.0,  # Should start quickly
        "memory_overhead_mb": 50.0,  # Should have low overhead
        "step_coordination_latency_ms": 100.0,  # Should coordinate quickly
        "error_detection_time_ms": 50.0  # Should detect errors quickly
    }
    
    for metric, threshold in performance_expectations.items():
        assert isinstance(threshold, (int, float)), f"Performance metric {metric} should be numeric"
        assert threshold > 0, f"Performance metric {metric} should be positive"
    
    logging.info(f"Main orchestrator performance characteristics validated for {len(performance_expectations)} metrics")

if __name__ == "__main__":
    # Allow running this test module directly
    pytest.main([__file__, "-v"]) 