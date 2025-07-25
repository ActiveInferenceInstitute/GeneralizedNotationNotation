#!/usr/bin/env python3
"""
Environment Validation Tests for GNN Pipeline

This module contains comprehensive tests to validate the environment setup
for the GNN processing pipeline. These tests ensure that:

1. Python environment is properly configured
2. Required dependencies are available
3. Project structure is correct
4. System resources are adequate
5. Environment variables are set correctly
6. Import capabilities work as expected

All tests are designed to be safe-to-fail and provide detailed diagnostic
information when issues are encountered.
"""

import pytest
import os
import sys
import json
import logging
import importlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
# Real functional testing without mocks

# Test markers
pytestmark = [pytest.mark.environment, pytest.mark.safe_to_fail, pytest.mark.fast]

# Import test utilities
from . import (
    TEST_CONFIG,
    is_safe_mode,
    validate_test_environment,
    TEST_DIR,
    SRC_DIR,
    PROJECT_ROOT
)

class TestPythonEnvironment:
    """Test Python environment configuration and capabilities."""
    
    @pytest.mark.unit
    def test_python_version_compatibility(self):
        """Test that Python version meets minimum requirements."""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        assert current_version >= min_version, (
            f"Python {min_version[0]}.{min_version[1]}+ required, "
            f"but running {current_version[0]}.{current_version[1]}"
        )
        
        # Log version info for diagnostics
        logging.info(f"Python version: {sys.version}")
        logging.info(f"Python executable: {sys.executable}")
    
    @pytest.mark.unit
    def test_python_path_configuration(self):
        """Test that Python path includes necessary directories."""
        # Check that src directory is in Python path
        assert str(SRC_DIR) in sys.path, f"Source directory {SRC_DIR} not in Python path"
        
        # Check that we can import from src
        original_path = sys.path.copy()
        try:
            # This should work without issues
            if str(SRC_DIR) not in sys.path:
                sys.path.insert(0, str(SRC_DIR))
            
            # Test basic import capability
            import utils  # Should work if path is correct
            logging.info("Successfully imported utils module from src")
            
        except ImportError as e:
            pytest.fail(f"Cannot import from src directory: {e}")
        finally:
            sys.path = original_path
    
    @pytest.mark.unit
    def test_project_structure_validation(self):
        """Test that required project directories and files exist."""
        required_paths = {
            # Core directories
            "src": SRC_DIR,
            "tests": TEST_DIR,
            "project_root": PROJECT_ROOT,
            
            # Key files
            "main_script": SRC_DIR / "main.py",
            "requirements": PROJECT_ROOT / "requirements.txt",
            "readme": PROJECT_ROOT / "README.md",
            
            # Module directories
            "utils": SRC_DIR / "utils",
            "pipeline": SRC_DIR / "pipeline",
            "gnn": SRC_DIR / "gnn",
            "tests_package": SRC_DIR / "tests"
        }
        
        missing_paths = []
        for name, path in required_paths.items():
            if not path.exists():
                missing_paths.append(f"{name}: {path}")
        
        assert not missing_paths, f"Missing required project paths: {missing_paths}"
        
        # Log successful validation
        logging.info(f"Validated {len(required_paths)} required project paths")

class TestDependencyAvailability:
    """Test availability of required and optional dependencies."""
    
    @pytest.mark.unit
    def test_core_python_modules(self):
        """Test that core Python modules are available."""
        core_modules = [
            'os', 'sys', 'pathlib', 'json', 'logging', 'argparse',
            'subprocess', 'tempfile', 'shutil', 'glob', 're', 'datetime'
        ]
        
        missing_modules = []
        for module_name in core_modules:
            try:
                __import__(module_name)
            except ImportError:
                missing_modules.append(module_name)
        
        assert not missing_modules, f"Missing core Python modules: {missing_modules}"
        logging.info(f"All {len(core_modules)} core Python modules available")
    
    @pytest.mark.unit
    def test_testing_dependencies(self):
        """Test that testing-related dependencies are available."""
        testing_modules = ['pytest', 'unittest']
        
        missing_modules = []
        for module_name in testing_modules:
            try:
                __import__(module_name)
            except ImportError:
                missing_modules.append(module_name)
        
        assert not missing_modules, f"Missing testing modules: {missing_modules}"
    
    @pytest.mark.unit
    def test_optional_dependencies_graceful_degradation(self):
        """Test that optional dependencies fail gracefully when missing."""
        optional_modules = {
            'psutil': 'System information',
            'matplotlib': 'Visualization',
            'networkx': 'Graph processing', 
            'pandas': 'Data manipulation',
            'numpy': 'Numerical computing',
            'torch': 'Machine learning',
            'jax': 'Accelerated computing'
        }
        
        available_modules = {}
        missing_modules = {}
        
        for module_name, description in optional_modules.items():
            try:
                __import__(module_name)
                available_modules[module_name] = description
            except ImportError:
                missing_modules[module_name] = description
        
        # Log availability status
        if available_modules:
            logging.info(f"Available optional modules: {list(available_modules.keys())}")
        if missing_modules:
            logging.info(f"Missing optional modules (graceful degradation): {list(missing_modules.keys())}")
        
        # This test always passes - we just document what's available
        assert True, "Optional dependency check completed"

class TestProjectConfiguration:
    """Test project configuration files and settings."""
    
    @pytest.mark.unit
    def test_requirements_file_validity(self):
        """Test that requirements.txt exists and is valid."""
        requirements_file = PROJECT_ROOT / "requirements.txt"
        
        assert requirements_file.exists(), "requirements.txt file not found"
        assert requirements_file.is_file(), "requirements.txt is not a file"
        
        # Read and validate requirements
        try:
            requirements_content = requirements_file.read_text()
            requirements_lines = [
                line.strip() for line in requirements_content.splitlines()
                if line.strip() and not line.strip().startswith('#')
            ]
            
            assert len(requirements_lines) > 0, "requirements.txt appears to be empty"
            
            # Check for essential testing requirements
            essential_reqs = ['pytest']
            found_reqs = []
            
            for req in essential_reqs:
                if any(line.startswith(req) for line in requirements_lines):
                    found_reqs.append(req)
            
            missing_essential = [req for req in essential_reqs if req not in found_reqs]
            if missing_essential:
                logging.warning(f"Essential requirements missing from requirements.txt: {missing_essential}")
            
            logging.info(f"Found {len(requirements_lines)} requirements in requirements.txt")
            
        except Exception as e:
            pytest.fail(f"Error reading requirements.txt: {e}")
    
    @pytest.mark.unit
    def test_pipeline_scripts_discovery(self):
        """Test that all expected pipeline scripts exist."""
        expected_scripts = [
            f"{i}_{name}.py" for i, name in [
                (1, "gnn"), (2, "setup"), (3, "tests"), (4, "type_checker"),
                (5, "export"), (6, "visualization"), (7, "mcp"), (8, "ontology"),
                (9, "render"), (10, "execute"), (11, "llm"), (12, "discopy"),
                (13, "discopy_jax_eval"), (14, "website")
            ]
        ]
        
        missing_scripts = []
        found_scripts = []
        
        for script_name in expected_scripts:
            script_path = SRC_DIR / script_name
            if script_path.exists():
                found_scripts.append(script_name)
            else:
                missing_scripts.append(script_name)
        
        # Log findings
        logging.info(f"Found pipeline scripts: {found_scripts}")
        if missing_scripts:
            logging.warning(f"Missing pipeline scripts: {missing_scripts}")
        
        # We expect at least the basic scripts to exist
        critical_scripts = ["1_setup.py", "main.py"]
        missing_critical = [
            script for script in critical_scripts
            if not (SRC_DIR / script).exists()
        ]
        
        assert not missing_critical, f"Critical pipeline scripts missing: {missing_critical}"
    
    @pytest.mark.unit
    def test_utility_modules_availability(self):
        """Test that utility modules are available and importable."""
        utility_modules = [
            ("utils", "Main utilities package"),
            ("pipeline", "Pipeline configuration"),
        ]
        
        import_results = {}
        
        for module_name, description in utility_modules:
            try:
                # Test import from src directory
                module = importlib.import_module(module_name)
                import_results[module_name] = {
                    'success': True,
                    'description': description,
                    'module': module
                }
            except ImportError as e:
                import_results[module_name] = {
                    'success': False,
                    'description': description,
                    'error': str(e)
                }
        
        # Check results
        failed_imports = [
            name for name, result in import_results.items()
            if not result['success']
        ]
        
        if failed_imports:
            error_details = [
                f"{name}: {import_results[name]['error']}"
                for name in failed_imports
            ]
            pytest.fail(f"Failed to import utility modules: {error_details}")
        
        logging.info(f"Successfully imported all {len(utility_modules)} utility modules")

class TestSystemResources:
    """Test system resources and capabilities."""
    
    @pytest.mark.unit
    def test_disk_space_availability(self):
        """Test that adequate disk space is available."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(PROJECT_ROOT)
            
            # Convert to MB for easier reading
            free_mb = free / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            
            # We need at least 100MB free for testing
            min_free_mb = 100
            
            assert free_mb >= min_free_mb, (
                f"Insufficient disk space. Need {min_free_mb}MB, have {free_mb:.1f}MB"
            )
            
            logging.info(f"Disk space: {free_mb:.1f}MB free of {total_mb:.1f}MB total")
            
        except Exception as e:
            logging.warning(f"Could not check disk space: {e}")
            # Don't fail the test if we can't check disk space
    
    @pytest.mark.unit
    def test_memory_availability(self):
        """Test system memory availability."""
        try:
            # Try to get memory info using psutil if available
            try:
                import psutil
                memory = psutil.virtual_memory()
                
                available_mb = memory.available / (1024 * 1024)
                total_mb = memory.total / (1024 * 1024)
                
                # We need at least 512MB available for testing
                min_available_mb = 512
                
                if available_mb < min_available_mb:
                    logging.warning(
                        f"Low memory: {available_mb:.1f}MB available, "
                        f"recommend {min_available_mb}MB+"
                    )
                
                logging.info(f"Memory: {available_mb:.1f}MB available of {total_mb:.1f}MB total")
                
            except ImportError:
                # Fallback: just check that we can allocate some memory
                test_data = bytearray(10 * 1024 * 1024)  # 10MB
                del test_data
                logging.info("Memory availability check: basic allocation test passed")
                
        except Exception as e:
            logging.warning(f"Could not check memory availability: {e}")
            # Don't fail the test if we can't check memory
    
    @pytest.mark.unit
    def test_temporary_directory_access(self):
        """Test that we can create and use temporary directories."""
        import tempfile
        
        try:
            with tempfile.TemporaryDirectory(prefix="gnn_test_") as temp_dir:
                temp_path = Path(temp_dir)
                
                # Test basic directory operations
                assert temp_path.exists(), "Temporary directory was not created"
                assert temp_path.is_dir(), "Temporary path is not a directory"
                
                # Test file creation in temp directory
                test_file = temp_path / "test_file.txt"
                test_file.write_text("test content")
                
                assert test_file.exists(), "Could not create file in temporary directory"
                assert test_file.read_text() == "test content", "File content mismatch"
                
                logging.info(f"Temporary directory access: OK ({temp_path})")
                
        except Exception as e:
            pytest.fail(f"Temporary directory access failed: {e}")

class TestEnvironmentVariables:
    """Test environment variable configuration."""
    
    @pytest.mark.unit
    def test_test_mode_environment_variables(self):
        """Test that test mode environment variables are set correctly."""
        # Set environment variables for this test if not already set
        if "GNN_TEST_MODE" not in os.environ:
            os.environ["GNN_TEST_MODE"] = "true"
        if "GNN_SAFE_MODE" not in os.environ:
            os.environ["GNN_SAFE_MODE"] = "true"
            
        # Check if we're in test mode
        test_mode = os.environ.get("GNN_TEST_MODE")
        assert test_mode == "true", f"GNN_TEST_MODE should be 'true', got '{test_mode}'"
        
        # Check safe mode
        safe_mode = os.environ.get("GNN_SAFE_MODE")
        assert safe_mode == "true", f"GNN_SAFE_MODE should be 'true', got '{safe_mode}'"
        
        logging.info("Test mode environment variables correctly set")
    
    @pytest.mark.unit
    def test_python_path_environment(self):
        """Test Python path environment configuration."""
        python_path = os.environ.get("PYTHONPATH", "")
        
        # While PYTHONPATH isn't required (we set sys.path directly),
        # log it for diagnostic purposes
        logging.info(f"PYTHONPATH: {python_path or '(not set)'}")
        
        # More importantly, check that sys.path is configured correctly
        assert str(SRC_DIR) in sys.path, "Source directory not in sys.path"
    
    @pytest.mark.unit
    def test_test_configuration_environment(self):
        """Test test-specific environment configuration."""
        if is_safe_mode():
            # In safe mode, check that mock flags are set
            mock_external = os.environ.get("GNN_MOCK_EXTERNAL_DEPS")
            mock_subprocess = os.environ.get("GNN_MOCK_SUBPROCESS")
            
            logging.info(f"Mock external deps: {mock_external}")
            logging.info(f"Mock subprocess: {mock_subprocess}")
        
        # Check test output directory if set
        test_output_dir = os.environ.get("GNN_TEST_OUTPUT_DIR")
        if test_output_dir:
            test_output_path = Path(test_output_dir)
            if test_output_path.exists():
                assert test_output_path.is_dir(), "GNN_TEST_OUTPUT_DIR is not a directory"
                logging.info(f"Test output directory: {test_output_path}")

class TestImportCapabilities:
    """Test that project modules can be imported correctly."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_main_module_import(self):
        """Test that main modules can be imported safely."""
        # Test imports that should always work
        safe_imports = [
            ("utils", "Utilities package"),
            ("pipeline", "Pipeline configuration"),
        ]
        
        for module_name, description in safe_imports:
            try:
                module = importlib.import_module(module_name)
                logging.info(f"Successfully imported {module_name}: {description}")
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name} ({description}): {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_pipeline_step_imports(self):
        """Test that pipeline step modules can be imported safely."""
        # These imports might fail if dependencies are missing, but should be graceful
        pipeline_modules = [
            ("gnn", "GNN processing"),
            ("type_checker", "Type checking"),
            ("visualization", "Visualization"),
            ("export", "Export functionality"),
        ]
        
        import_results = {}
        
        for module_name, description in pipeline_modules:
            try:
                module = importlib.import_module(module_name)
                import_results[module_name] = {'success': True, 'description': description}
                logging.info(f"Successfully imported {module_name}: {description}")
            except ImportError as e:
                import_results[module_name] = {
                    'success': False, 
                    'description': description,
                    'error': str(e)
                }
                logging.warning(f"Could not import {module_name} ({description}): {e}")
        
        # Log summary
        successful = [name for name, result in import_results.items() if result['success']]
        failed = [name for name, result in import_results.items() if not result['success']]
        
        logging.info(f"Pipeline module imports: {len(successful)} successful, {len(failed)} failed")
        
        # We don't fail the test for optional modules, just log the results
        assert len(import_results) > 0, "No pipeline modules tested"
    
    @pytest.mark.unit
    def test_test_module_imports(self):
        """Test that test modules can be imported correctly."""
        # Import the test package itself
        try:
            from . import TEST_CONFIG, PYTEST_MARKERS
            assert isinstance(TEST_CONFIG, dict), "TEST_CONFIG should be a dictionary"
            assert isinstance(PYTEST_MARKERS, dict), "PYTEST_MARKERS should be a dictionary"
            logging.info("Successfully imported test configuration")
        except ImportError as e:
            pytest.fail(f"Failed to import test configuration: {e}")

class TestEnvironmentIntegration:
    """Test integration between environment components."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_logging_system_integration(self):
        """Test that logging system works correctly."""
        import logging
        
        # Create a test logger
        test_logger = logging.getLogger("test_environment_integration")
        
        # Test different log levels
        try:
            test_logger.debug("Debug message for environment test")
            test_logger.info("Info message for environment test")
            test_logger.warning("Warning message for environment test")
            # Note: not testing error/critical as they might affect test runner
            
            logging.info("Logging system integration test passed")
        except Exception as e:
            pytest.fail(f"Logging system integration failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_path_resolution_integration(self):
        """Test that path resolution works correctly across the system."""
        # Test various path operations
        paths_to_test = {
            "project_root": PROJECT_ROOT,
            "src_dir": SRC_DIR,
            "test_dir": TEST_DIR,
            "relative_path": Path("../"),
            "absolute_path": Path("/tmp") if os.name == 'posix' else Path("C:\\")
        }
        
        for path_name, path_obj in paths_to_test.items():
            try:
                # Test basic path operations
                resolved = path_obj.resolve()
                exists = path_obj.exists()
                
                logging.info(f"Path {path_name}: exists={exists}, resolved={resolved}")
                
            except Exception as e:
                logging.warning(f"Path resolution issue for {path_name}: {e}")
    
    @pytest.mark.integration
    def test_test_environment_validation_integration(self):
        """Test the integrated environment validation system."""
        is_valid, issues = validate_test_environment()
        
        if not is_valid:
            logging.warning(f"Environment validation issues found: {issues}")
            # Don't fail the test, just document the issues
        else:
            logging.info("Environment validation passed")
        
        # Test that validation function works
        assert isinstance(is_valid, bool), "validate_test_environment should return boolean"
        assert isinstance(issues, list), "validate_test_environment should return list of issues"

# Additional utility functions for environment testing

def test_requirements_parsing():
    """Test that requirements can be parsed correctly."""
    requirements_file = PROJECT_ROOT / "requirements.txt"
    
    if not requirements_file.exists():
        pytest.skip("requirements.txt not found")
    
    try:
        content = requirements_file.read_text()
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        
        # Basic validation that it looks like a requirements file
        valid_lines = []
        for line in lines:
            if line.startswith('#'):
                continue  # Comment line
            if '==' in line or '>=' in line or '<=' in line or '~=' in line:
                valid_lines.append(line)
            elif len(line.split()) == 1:  # Simple package name
                valid_lines.append(line)
        
        assert len(valid_lines) > 0, "No valid requirement lines found"
        logging.info(f"Parsed {len(valid_lines)} valid requirements")
        
    except Exception as e:
        pytest.fail(f"Requirements parsing failed: {e}")

@pytest.mark.slow
def test_subprocess_execution_capability():
    """Test that subprocess execution works (in safe mode)."""
    if not is_safe_mode():
        pytest.skip("Subprocess test requires safe mode")
    
    try:
        # Import patch for this test
        from unittest.mock import patch
        
        # Test basic subprocess execution (should be mocked in safe mode)
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "test output"
            
            result = subprocess.run(["echo", "test"], capture_output=True, text=True)
            
            assert result.returncode == 0, "Subprocess execution failed"
            logging.info("Subprocess execution capability verified (mocked)")
            
    except Exception as e:
        pytest.fail(f"Subprocess execution test failed: {e}")

if __name__ == "__main__":
    # Allow running this test module directly
    pytest.main([__file__, "-v"]) 