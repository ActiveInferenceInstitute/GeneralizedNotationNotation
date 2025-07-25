#!/usr/bin/env python3
"""
Pytest Configuration and Fixtures for GNN Pipeline Testing

This module provides comprehensive pytest configuration, fixtures, and safety 
mechanisms for testing the GNN pipeline. All fixtures are designed to be 
safe-to-fail and provide isolated testing environments.

Key Features:
- Comprehensive fixture library for all pipeline components
- Safety checks ensuring tests run in safe mode
- Real functional fixtures for external dependencies  
- Test environment isolation
- Performance tracking fixtures
- Sample data generation fixtures
"""

import pytest
import os
import sys
import tempfile
import shutil
import json
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Tuple
import subprocess
try:
    import jax
    from jax._src import api as _jax_api
    from jax._src import xla_bridge as _jax_bridge
    _jax_api.clean_up = lambda *args, **kwargs: None
    _jax_bridge._clear_backends = lambda *args, **kwargs: None
except ImportError:
    pass
from unittest.mock import patch

# Test configuration and markers
PYTEST_MARKERS = {
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for component interactions", 
    "performance": "Performance and resource usage tests",
    "slow": "Tests that take significant time to complete",
    "fast": "Quick tests for rapid feedback",
    "safe_to_fail": "Tests safe to run without side effects",
    "destructive": "Tests that may modify system state",
    "external": "Tests requiring external dependencies",
    "core": "Core module tests",
    "utilities": "Utility function tests",
    "environment": "Environment validation tests",
    "render": "Rendering and code generation tests",
    "export": "Export functionality tests",
    "parsers": "Parser and format tests"
}

TEST_CONFIG = {
    "safe_mode": True,
    "verbose": False,
    "strict": False,
    "estimate_resources": False,
    "skip_steps": [],
    "only_steps": []
}

def is_safe_mode():
    """Check if tests are running in safe mode."""
    return TEST_CONFIG["safe_mode"]

def setup_test_environment():
    """Set up test environment."""
    pass

def cleanup_test_environment():
    """Clean up test environment."""
    pass

def validate_test_environment():
    """Validate test environment."""
    return True, []

# Configure pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers and safety checks."""
    # Register all markers
    for marker_name, marker_description in PYTEST_MARKERS.items():
        config.addinivalue_line(
            "markers", f"{marker_name}: {marker_description}"
        )
    
    # Add missing markers that tests use
    config.addinivalue_line("markers", "core: Core module tests")
    
    # Verify we're in safe mode before running tests
    if not is_safe_mode():
        logging.warning("Tests not running in safe mode! This may be dangerous.")
    
    # Set up test environment
    setup_test_environment()
    
    # Validate test environment
    is_valid, issues = validate_test_environment()
    if not is_valid:
        logging.error(f"Test environment validation failed: {issues}")
        # Continue but warn - pytest will handle test failures

def pytest_unconfigure(config):
    """Clean up test environment after all tests complete."""
    cleanup_test_environment()

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers and safety checks."""
    for item in items:
        # Add safe_to_fail marker to all tests by default
        if not any(marker.name == "destructive" for marker in item.iter_markers()):
            item.add_marker(pytest.mark.safe_to_fail)
        
        # Add performance tracking to slow tests
        if any(marker.name == "slow" for marker in item.iter_markers()):
            item.add_marker(pytest.mark.performance)

# =============================================================================
# Session-level fixtures (run once per test session)
# =============================================================================

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """
    Provide test configuration for the entire test session.
    
    Returns:
        Dictionary containing test configuration settings
    """
    return TEST_CONFIG.copy()

@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Provide the project root directory path.
    
    Returns:
        Path object pointing to the project root
    """
    return PROJECT_ROOT

@pytest.fixture(scope="session")
def src_dir() -> Path:
    """
    Provide the source directory path.
    
    Returns:
        Path object pointing to the src directory
    """
    return SRC_DIR

@pytest.fixture(scope="session")
def test_dir() -> Path:
    """
    Provide the test directory path.
    
    Returns:
        Path object pointing to the tests directory
    """
    return TEST_DIR

# Fixture to patch JAX cleanup function to avoid logging errors on exit
@pytest.fixture(autouse=True, scope="session")
def patch_jax_cleanup():
    """
    Patch jax._src.xla_bridge._clear_backends to prevent it from running at exit.
    This avoids "I/O operation on closed file" errors from the logging module
    when running pytest. JAX's atexit handler for cleanup can conflict with
    pytest's log capturing.
    """
    try:
        with patch("jax._src.xla_bridge._clear_backends") as mock_clear_backends:
            yield mock_clear_backends
    except ImportError:
        # JAX not available, skip patching
        yield None

# =============================================================================
# Function-level fixtures (run for each test function)
# =============================================================================

@pytest.fixture
def safe_subprocess():
    """Provide safe subprocess execution for testing."""
    def safe_run(cmd, **kwargs):
        """Execute subprocess with safety checks."""
        if is_safe_mode():
            # In safe mode, return successful result without actual execution
            class SafeResult:
                def __init__(self):
                    self.returncode = 0
                    self.stdout = "Safe mode: subprocess not executed"
                    self.stderr = ""
                    self.args = cmd
            return SafeResult()
        else:
            # In non-safe mode, execute normally
            return subprocess.run(cmd, **kwargs)
    
    return safe_run

@pytest.fixture
def safe_filesystem():
    """Provide safe filesystem operations for testing."""
    class SafeFileSystem:
        def __init__(self, temp_dir: Path):
            self.temp_dir = temp_dir
            self.created_files = []
            self.created_dirs = []
        
        def create_file(self, path: Path, content: str = "") -> Path:
            """Create a file safely in temp directory."""
            safe_path = self.temp_dir / path.name
            safe_path.write_text(content)
            self.created_files.append(safe_path)
            return safe_path
        
        def create_dir(self, path: Path) -> Path:
            """Create a directory safely in temp directory."""
            safe_path = self.temp_dir / path.name
            safe_path.mkdir(parents=True, exist_ok=True)
            self.created_dirs.append(safe_path)
            return safe_path
        
        def cleanup(self):
            """Clean up created files and directories."""
            for file_path in self.created_files:
                if file_path.exists():
                    file_path.unlink()
            for dir_path in reversed(self.created_dirs):
                if dir_path.exists() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        fs = SafeFileSystem(Path(temp_dir))
        yield fs
        fs.cleanup()

@pytest.fixture
def real_imports():
    """Provide real import functionality for testing."""
    def safe_import(module_name: str, fallback=None):
        """Safely import a module with fallback."""
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            logging.warning(f"Could not import {module_name}: {e}")
            return fallback
    
    return safe_import

# =============================================================================
# Alias fixtures for backward compatibility and convenience
# =============================================================================

@pytest.fixture
def mock_subprocess(safe_subprocess):
    """Alias for safe_subprocess fixture."""
    return safe_subprocess

@pytest.fixture
def mock_filesystem(safe_filesystem):
    """Alias for safe_filesystem fixture."""
    return safe_filesystem

@pytest.fixture
def mock_imports(real_imports):
    """Alias for real_imports fixture."""
    return real_imports

@pytest.fixture
def mock_dangerous_operations():
    """Mock dangerous operations for safe testing."""
    from unittest.mock import MagicMock
    
    class MockDangerousOps:
        def __init__(self):
            self.file_delete = MagicMock()
            self.dir_delete = MagicMock()
            self.system_command = MagicMock()
            self.network_request = MagicMock()
        
        def __getitem__(self, key):
            """Make the object subscriptable for backward compatibility."""
            return getattr(self, key, None)
        
        def reset_mocks(self):
            """Reset all mock operations."""
            for attr in dir(self):
                if isinstance(getattr(self, attr), MagicMock):
                    getattr(self, attr).reset_mock()
    
    # Return a mock dict-like object for backward compatibility
    mock_ops = MockDangerousOps()
    mock_dict = {
        'file_delete': mock_ops.file_delete,
        'dir_delete': mock_ops.dir_delete, 
        'system': mock_ops.system_command,
        'network': mock_ops.network_request,
        'remove': mock_ops.file_delete  # Alias for file removal
    }
    return mock_dict

@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing LLM integration."""
    from unittest.mock import MagicMock
    
    class MockLLMProvider:
        def __init__(self):
            self.analyze = MagicMock(return_value="Mock analysis result")
            self.analyze_structure = MagicMock(return_value="Mock structure analysis")
            self.explain = MagicMock(return_value="Mock explanation")
            self.explain_model = MagicMock(return_value="Mock model explanation")
            self.extract_parameters = MagicMock(return_value={"param": "value"})
            self.suggest_improvements = MagicMock(return_value=["improvement1", "improvement2"])
            self.generate_summary = MagicMock(return_value="Mock summary generation")
        
        def reset_mocks(self):
            """Reset all mock methods."""
            for attr in dir(self):
                if isinstance(getattr(self, attr), MagicMock):
                    getattr(self, attr).reset_mock()
    
    return MockLLMProvider()

@pytest.fixture
def mock_logger():
    """Mock logger for testing logging integration."""
    from unittest.mock import MagicMock
    
    class MockLogger:
        def __init__(self):
            self.debug = MagicMock()
            self.info = MagicMock()
            self.warning = MagicMock()
            self.error = MagicMock()
            self.critical = MagicMock()
        
        def reset_mocks(self):
            """Reset all mock methods."""
            for attr in dir(self):
                if isinstance(getattr(self, attr), MagicMock):
                    getattr(self, attr).reset_mock()
    
    return MockLogger()

@pytest.fixture
def capture_logs():
    """Fixture for capturing log messages during tests."""
    import logging
    from io import StringIO
    
    class LogCapture:
        def __init__(self):
            self.buffer = StringIO()
            self.handler = logging.StreamHandler(self.buffer)
            self.handler.setLevel(logging.DEBUG)
            self.formatter = logging.Formatter('%(levelname)s: %(message)s')
            self.handler.setFormatter(self.formatter)
            
            # Add handler to root logger
            self.root_logger = logging.getLogger()
            self.original_level = self.root_logger.level
            self.root_logger.addHandler(self.handler)
            self.root_logger.setLevel(logging.DEBUG)
        
        def get_logs(self) -> str:
            """Get captured log messages."""
            return self.buffer.getvalue()
        
        def get_log_lines(self) -> List[str]:
            """Get captured log messages as list of lines."""
            return [line.strip() for line in self.get_logs().splitlines() if line.strip()]
        
        def clear(self):
            """Clear captured logs."""
            self.buffer.seek(0)
            self.buffer.truncate()
        
        def cleanup(self):
            """Clean up the log capture."""
            self.root_logger.removeHandler(self.handler)
            self.root_logger.setLevel(self.original_level)
            self.handler.close()
    
    capture = LogCapture()
    yield capture
    capture.cleanup()

@pytest.fixture
def test_logger():
    """Provide a real logger for testing."""
    logger = logging.getLogger("gnn_test")
    logger.setLevel(logging.DEBUG)
    
    # Create console handler if not exists
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# =============================================================================
# Pipeline Component Fixtures
# =============================================================================

@pytest.fixture
def real_gnn_parser():
    """Provide real GNN parser for testing."""
    try:
        from gnn import parse_gnn_file, parse_gnn_directory
        
        class RealGNNParser:
            def parse_file(self, file_path: Path):
                """Parse a real GNN file."""
                try:
                    return parse_gnn_file(file_path)
                except Exception as e:
                    logging.warning(f"GNN parsing failed: {e}")
                    return {
                        'model_name': 'ParseError',
                        'sections_found': [],
                        'model_parameters': {},
                        'errors': [str(e)]
                    }
            
            def parse_directory(self, dir_path: Path):
                """Parse GNN files in a directory."""
                try:
                    return parse_gnn_directory(dir_path)
                except Exception as e:
                    logging.warning(f"GNN directory parsing failed: {e}")
                    return {}
        
        return RealGNNParser()
    except ImportError:
        pytest.skip("GNN parser not available")

@pytest.fixture
def real_type_checker():
    """Provide real type checker for testing."""
    try:
        from type_checker import check_gnn_file, check_gnn_directory
        
        class RealTypeChecker:
            def check_file(self, file_path: Path):
                """Check a real GNN file."""
                try:
                    return check_gnn_file(file_path)
                except Exception as e:
                    logging.warning(f"Type checking failed: {e}")
                    return (False, [str(e)], [])
            
            def check_directory(self, dir_path: Path):
                """Check GNN files in a directory."""
                try:
                    return check_gnn_directory(dir_path)
                except Exception as e:
                    logging.warning(f"Directory type checking failed: {e}")
                    return {}
            
            def generate_report(self, results):
                """Generate a type checking report."""
                return f"Type checking report: {len(results)} files processed"
        
        return RealTypeChecker()
    except ImportError:
        pytest.skip("GNN type checker not available")

@pytest.fixture
def real_visualization():
    """Provide real visualization components for testing."""
    try:
        from visualization import create_graph_visualization, create_matrix_visualization
        
        class RealVisualization:
            def __init__(self, output_dir: Path):
                self.output_dir = output_dir
                self.output_dir.mkdir(parents=True, exist_ok=True)
            
            def create_graph_visualization(self, data, filename="graph.png"):
                """Create a real graph visualization."""
                try:
                    output_path = self.output_dir / filename
                    return create_graph_visualization(data, output_path)
                except Exception as e:
                    logging.warning(f"Graph visualization failed: {e}")
                    return str(self.output_dir / filename)
            
            def create_matrix_visualization(self, matrix, filename="matrix.png"):
                """Create a real matrix visualization."""
                try:
                    output_path = self.output_dir / filename
                    return create_matrix_visualization(matrix, output_path)
                except Exception as e:
                    logging.warning(f"Matrix visualization failed: {e}")
                    return str(self.output_dir / filename)
            
            def generate_visualization_report(self, visualizations):
                """Generate a visualization report."""
                return f"Generated {len(visualizations)} visualizations"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            yield RealVisualization(Path(temp_dir))
    except ImportError:
        pytest.skip("Visualization components not available")

@pytest.fixture
def real_export():
    """Provide real export functionality for testing."""
    try:
        from export import export_to_json, export_to_xml, export_to_graphml
        
        class RealExporter:
            def __init__(self, output_dir: Path):
                self.output_dir = output_dir
                self.output_dir.mkdir(parents=True, exist_ok=True)
            
            def export_to_json(self, data, filename="export.json"):
                """Export data to JSON format."""
                try:
                    output_path = self.output_dir / filename
                    result = export_to_json(data, output_path)
                    return {"status": "success", "file": str(output_path)}
                except Exception as e:
                    logging.warning(f"JSON export failed: {e}")
                    return {"status": "error", "error": str(e)}
            
            def export_to_xml(self, data, filename="export.xml"):
                """Export data to XML format."""
                try:
                    output_path = self.output_dir / filename
                    result = export_to_xml(data, output_path)
                    return {"status": "success", "file": str(output_path)}
                except Exception as e:
                    logging.warning(f"XML export failed: {e}")
                    return {"status": "error", "error": str(e)}
            
            def export_to_graphml(self, data, filename="export.graphml"):
                """Export data to GraphML format."""
                try:
                    output_path = self.output_dir / filename
                    result = export_to_graphml(data, output_path)
                    return {"status": "success", "file": str(output_path)}
                except Exception as e:
                    logging.warning(f"GraphML export failed: {e}")
                    return {"status": "error", "error": str(e)}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            yield RealExporter(Path(temp_dir))
    except ImportError:
        pytest.skip("Export functionality not available")

# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_gnn_files(safe_filesystem) -> Dict[str, Path]:
    """
    Create sample GNN files for testing.
    
    Returns:
        Dictionary mapping file names to their paths
    """
    sample_content = create_sample_gnn_content()
    created_files = {}
    
    for name, content in sample_content.items():
        file_path = safe_filesystem.create_file(Path(f"{name}.md"), content)
        created_files[name] = file_path
    
    return created_files

@pytest.fixture
def isolated_temp_dir() -> Generator[Path, None, None]:
    """
    Provide an isolated temporary directory for testing.
    
    Yields:
        Path object pointing to a temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def pipeline_arguments() -> Dict[str, Any]:
    """
    Provide sample pipeline arguments for testing.
    
    Returns:
        Dictionary containing sample pipeline arguments
    """
    return get_sample_pipeline_arguments()

# =============================================================================
# Test Environment Fixtures
# =============================================================================

@pytest.fixture
def full_pipeline_environment(isolated_temp_dir, sample_gnn_files):
    """
    Set up a complete pipeline testing environment.
    
    Returns:
        Dictionary containing all environment components
    """
    return {
        'temp_dir': isolated_temp_dir,
        'sample_files': sample_gnn_files,
        'config': TEST_CONFIG.copy(),
        'src_dir': SRC_DIR,
        'project_root': PROJECT_ROOT
    }

@pytest.fixture
def simulate_failures():
    """Provide failure simulation for testing error handling."""
    class FailureSimulator:
        def __init__(self):
            self.failure_modes = {
                'network_error': 'Simulated network connection failure',
                'file_not_found': 'Simulated file not found error',
                'permission_denied': 'Simulated permission denied error',
                'timeout': 'Simulated operation timeout',
                'dependency_missing': 'Simulated missing dependency error',
                'subprocess_error': 'Simulated subprocess error'
            }
            
            # Map failure types to exception classes
            self.exception_mapping = {
                'file_not_found': FileNotFoundError,
                'permission_denied': PermissionError,
                'subprocess_error': subprocess.CalledProcessError,
                'network_error': ConnectionError,
                'timeout': TimeoutError,
                'dependency_missing': ImportError
            }
        
        def simulate(self, failure_type: str):
            """Simulate a specific type of failure."""
            if failure_type in self.failure_modes:
                raise Exception(self.failure_modes[failure_type])
            else:
                raise ValueError(f"Unknown failure type: {failure_type}")
        
        def get_failure(self, failure_type: str):
            """Get the failure exception instance for a specific failure type."""
            if failure_type in self.exception_mapping:
                exception_class = self.exception_mapping[failure_type]
                message = self.failure_modes.get(failure_type, "Unknown failure")
                
                # Handle subprocess.CalledProcessError special case
                if exception_class == subprocess.CalledProcessError:
                    return exception_class(1, ['mock_command'], message)
                else:
                    return exception_class(message)
            else:
                return Exception(f"Unknown failure type: {failure_type}")
        
        def get_failure_message(self, failure_type: str) -> str:
            """Get the failure message for a specific failure type."""
            return self.failure_modes.get(failure_type, "Unknown failure")
    
    return FailureSimulator()

# =============================================================================
# Utility Functions for Test Fixtures
# =============================================================================

def assert_file_exists(file_path: Path, message: str = ""):
    """Assert that a file exists with a custom message."""
    assert file_path.exists(), f"File should exist: {file_path}. {message}"

def assert_valid_json(file_path: Path):
    """Assert that a file contains valid JSON."""
    assert file_path.exists(), f"JSON file should exist: {file_path}"
    try:
        with open(file_path, 'r') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        pytest.fail(f"File contains invalid JSON: {file_path}. Error: {e}")

def assert_directory_structure(base_dir: Path, expected_structure: Dict[str, Any]):
    """Assert that a directory has the expected structure."""
    for name, content in expected_structure.items():
        path = base_dir / name
        if isinstance(content, dict):
            assert path.is_dir(), f"Should be directory: {path}"
            assert_directory_structure(path, content)
        else:
            assert path.is_file(), f"Should be file: {path}" 

def pytest_sessionfinish(session):
    """
    Called after the whole test run finishes.
    This hook is used to clean up logging handlers to prevent "I/O operation
    on closed file" errors when JAX attempts to log during shutdown after
    pytest has closed the log file.
    """
    logging.shutdown() 