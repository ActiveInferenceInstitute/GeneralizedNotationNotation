"""
Test configuration and fixtures for GNN Processing Pipeline.

This module provides pytest fixtures and configuration for testing the GNN pipeline.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
import tempfile
import json
from unittest.mock import Mock

# Import pytest
import pytest

# Import utilities from utils.test_utils to avoid circular imports
from utils.test_utils import (
    SRC_DIR, 
    PROJECT_ROOT, 
    TEST_CONFIG,
    create_sample_gnn_content,
    is_safe_mode,
    setup_test_environment,
    cleanup_test_environment,
    validate_test_environment
)

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
    "pipeline": "Pipeline infrastructure tests",
    "recovery": "Pipeline recovery tests",
    "utilities": "Utility function tests",
    "environment": "Environment validation tests",
    "render": "Rendering and code generation tests",
    "export": "Export functionality tests",
    "parsers": "Parser and format tests",
    "main_orchestrator": "Main orchestrator tests",
    "type_checking": "Type checking tests",
    "mcp": "Model Context Protocol tests",
    "sapf": "SAPF audio generation tests",
    "visualization": "Visualization tests"
}

# Configure pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers and safety checks."""
    # Register all markers
    for marker_name, marker_description in PYTEST_MARKERS.items():
        config.addinivalue_line(
            "markers", f"{marker_name}: {marker_description}"
        )
    
    # Verify we're in safe mode before running tests
    if not is_safe_mode():
        logging.warning("Tests not running in safe mode! This may be dangerous.")
    
    # Set up test environment
    setup_test_environment()
    
    # Validate test environment
    is_valid, issues = validate_test_environment()
    if not is_valid:
        logging.error(f"Test environment validation failed: {issues}")

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
    return TEST_CONFIG

@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    return PROJECT_ROOT

@pytest.fixture(scope="session")
def src_dir() -> Path:
    """
    Get the source directory.
    
    Returns:
        Path to the source directory
    """
    return SRC_DIR

@pytest.fixture(scope="session")
def test_dir() -> Path:
    """
    Get the test directory.
    
    Returns:
        Path to the test directory
    """
    return SRC_DIR / "tests"

# =============================================================================
# Test utilities fixtures
# =============================================================================

@pytest.fixture
def safe_filesystem():
    """
    Provide safe filesystem operations for tests.
    
    This fixture creates temporary directories and files for testing
    without modifying the actual filesystem.
    """
    class SafeFileSystem:
        def __init__(self, temp_dir: Path):
            self.temp_dir = temp_dir
            self.created_files = []
            self.created_dirs = []
        
        def create_file(self, path: Path, content: str = "") -> Path:
            """Create a file safely in the temporary directory."""
            full_path = self.temp_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            self.created_files.append(full_path)
            return full_path
        
        def create_dir(self, path: Path) -> Path:
            """Create a directory safely in the temporary directory."""
            full_path = self.temp_dir / path
            full_path.mkdir(parents=True, exist_ok=True)
            self.created_dirs.append(full_path)
            return full_path
        
        def cleanup(self):
            """Clean up created files and directories."""
            for file in self.created_files:
                if file.exists():
                    file.unlink()
            for dir_path in reversed(self.created_dirs):  # Reverse to delete deepest first
                if dir_path.exists():
                    try:
                        dir_path.rmdir()
                    except OSError:
                        # Directory not empty - skip
                        pass
    
    # Create temporary directory for the filesystem
    temp_dir = Path(tempfile.mkdtemp(prefix="gnn_test_"))
    fs = SafeFileSystem(temp_dir)
    
    yield fs
    
    fs.cleanup()

@pytest.fixture
def mock_subprocess():
    """Provide a mock subprocess for tests."""
    def safe_run(cmd, **kwargs):
        """Run a command safely in tests (mock implementation)."""
        try:
            class SafeResult:
                def __init__(self):
                    self.returncode = 0
                    self.stdout = f"Mock stdout for: {cmd}"
                    self.stderr = ""
            
            return SafeResult()
        except Exception as exc:
            error_msg = str(exc)
            class FailResult:
                def __init__(self):
                    self.returncode = 1
                    self.stdout = ""
                    self.stderr = error_msg
            
            return FailResult()
    
    return safe_run

@pytest.fixture
def mock_imports():
    """Provide safe module imports for tests."""
    def safe_import(module_name: str, fallback=None):
        """Safely import a module, returning fallback if it fails."""
        try:
            return __import__(module_name)
        except ImportError:
            return fallback
    
    return safe_import

@pytest.fixture
def mock_logger():
    """
    Provide a mock logger for tests.
    
    This fixture captures logs instead of outputting them.
    """
    class MockLogger:
        def __init__(self):
            self.logs = []
            self.errors = []
            self.warnings = []
            self.infos = []
        
        def reset_mocks(self):
            """Reset log history."""
            self.logs = []
            self.errors = []
            self.warnings = []
            self.infos = []
        
        def error(self, msg, *args, **kwargs):
            """Log error message."""
            self.errors.append(msg)
            self.logs.append(f"ERROR: {msg}")
        
        def warning(self, msg, *args, **kwargs):
            """Log warning message."""
            self.warnings.append(msg)
            self.logs.append(f"WARNING: {msg}")
        
        def info(self, msg, *args, **kwargs):
            """Log info message."""
            self.infos.append(msg)
            self.logs.append(f"INFO: {msg}")
    
    return MockLogger()

# =============================================================================
# Test data fixtures
# =============================================================================

@pytest.fixture
def sample_gnn_files(safe_filesystem) -> Dict[str, Path]:
    """
    Create sample GNN files for testing.
    
    This fixture creates temporary GNN files for testing
    without modifying the actual filesystem.
    """
    gnn_content = create_sample_gnn_content()
    
    # Create files
    files = {}
    for name, content in gnn_content.items():
        file_path = safe_filesystem.create_file(Path(f"{name}.md"), content)
        files[name] = file_path
    
    return files

@pytest.fixture
def isolated_temp_dir() -> Generator[Path, None, None]:
    """
    Create an isolated temporary directory for tests.
    
    This fixture creates a temporary directory for test use
    and cleans it up afterward.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="gnn_test_"))
    yield temp_dir
    
    # Clean up (remove all files and directories)
    if temp_dir.exists():
        for path in temp_dir.glob('**/*'):
            if path.is_file():
                path.unlink()
        
        # Try to remove empty directories
        for path in sorted([p for p in temp_dir.glob('**/*') if p.is_dir()], key=lambda p: len(str(p)), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass
        
        # Try to remove the temp directory itself
        try:
            temp_dir.rmdir()
        except OSError:
            pass

@pytest.fixture
def pipeline_arguments() -> Dict[str, Any]:
    """
    Provide standard pipeline arguments for testing.
    
    This fixture creates a set of standard arguments
    for pipeline scripts.
    """
    return {
        "target_dir": str(PROJECT_ROOT / "input" / "gnn_files"),
        "output_dir": str(PROJECT_ROOT / "output"),
        "verbose": True,
        "strict": False,
        "force": False,
        "config": str(PROJECT_ROOT / "input" / "config.yaml")
    }

@pytest.fixture
def comprehensive_test_data(isolated_temp_dir) -> Dict[str, Any]:
    """
    Create comprehensive test data for tests.
    
    This fixture creates a complete set of test data
    including files, directories, and pipeline outputs.
    """
    # Create directory structure
    input_dir = isolated_temp_dir / "input"
    output_dir = isolated_temp_dir / "output"
    gnn_dir = input_dir / "gnn_files"
    
    for dir_path in [input_dir, output_dir, gnn_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create GNN files
    gnn_content = create_sample_gnn_content()
    gnn_files = {}
    for name, content in gnn_content.items():
        file_path = gnn_dir / f"{name}.md"
        with open(file_path, 'w') as f:
            f.write(content)
        gnn_files[name] = file_path
    
    # Create output directories and sample files
    output_dirs = [
        "setup_artifacts", "gnn_processing_step", "type_check", 
        "validation", "gnn_exports", "visualization"
    ]
    
    for dir_name in output_dirs:
        dir_path = output_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create a sample JSON file
        with open(dir_path / "results.json", 'w') as f:
            json.dump({"status": "success", "step": dir_name}, f)
    
    # Create pipeline arguments
    args = {
        "target_dir": str(input_dir),
        "output_dir": str(output_dir),
        "verbose": True,
        "strict": False
    }
    
    return {
        "temp_dir": isolated_temp_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "gnn_dir": gnn_dir,
        "gnn_files": gnn_files,
        "output_dirs": {name: output_dir / name for name in output_dirs},
        "args": args
    } 