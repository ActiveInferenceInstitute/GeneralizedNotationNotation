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

def pytest_unconfigure(config):
    """Clean up test environment after all tests complete."""
    pass

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
    """
    return {
        "test_mode": True,
        "safe_mode": True,
        "temp_dir": tempfile.mkdtemp(),
        "max_test_duration": 300,
        "memory_limit_mb": 1024
    }

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

@pytest.fixture(scope="session")
def src_dir() -> Path:
    """Get the src directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Get the test directory."""
    return Path(__file__).parent

# =============================================================================
# Function-level fixtures (run once per test function)
# =============================================================================

@pytest.fixture
def safe_filesystem():
    """Provide a safe filesystem for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    
    class SafeFileSystem:
        def __init__(self, temp_dir: Path):
            self.temp_dir = temp_dir
            self.created_files = []
            self.created_dirs = []
        
        def create_file(self, path: Path, content: str = "") -> Path:
            full_path = self.temp_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            self.created_files.append(full_path)
            return full_path
        
        def create_dir(self, path: Path) -> Path:
            full_path = self.temp_dir / path
            full_path.mkdir(parents=True, exist_ok=True)
            self.created_dirs.append(full_path)
            return full_path
        
        def cleanup(self):
            for file_path in self.created_files:
                if file_path.exists():
                    file_path.unlink()
            for dir_path in reversed(self.created_dirs):
                if dir_path.exists():
                    try:
                        dir_path.rmdir()
                    except OSError:
                        pass
            if self.temp_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(self.temp_dir)
                except OSError:
                    pass
    
    fs = SafeFileSystem(temp_dir)
    yield fs
    fs.cleanup()

@pytest.fixture
def mock_subprocess():
    """Provide a mock subprocess for testing."""
    def safe_run(cmd, **kwargs):
        class SafeResult:
            def __init__(self):
                self.returncode = 0
                self.stdout = "Mock output"
                self.stderr = ""
        
        return SafeResult()
    
    return safe_run

@pytest.fixture
def mock_imports():
    """Provide safe import mocking."""
    def safe_import(module_name: str, fallback=None):
        try:
            return __import__(module_name)
        except ImportError:
            return fallback or Mock()
    
    return safe_import

@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    class MockLogger:
        def __init__(self):
            self.messages = []
            self.errors = []
            self.warnings = []
            self.info_messages = []
        
        def reset_mocks(self):
            self.messages.clear()
            self.errors.clear()
            self.warnings.clear()
            self.info_messages.clear()
        
        def error(self, msg, *args, **kwargs):
            self.errors.append(msg)
            self.messages.append(("ERROR", msg))
        
        def warning(self, msg, *args, **kwargs):
            self.warnings.append(msg)
            self.messages.append(("WARNING", msg))
        
        def info(self, msg, *args, **kwargs):
            self.info_messages.append(msg)
            self.messages.append(("INFO", msg))
        
        def debug(self, msg, *args, **kwargs):
            self.messages.append(("DEBUG", msg))
        
        def critical(self, msg, *args, **kwargs):
            self.messages.append(("CRITICAL", msg))
    
    return MockLogger()

@pytest.fixture
def sample_gnn_files(safe_filesystem) -> Dict[str, Path]:
    """Provide sample GNN files for testing."""
    files = {}
    
    # Create a simple test GNN file
    content = """
# Test GNN Model

## Model Name
test_model

## State Space
s[3,1,type=int]

## Connections
s -> o

## Parameters
A = [[0.5, 0.3, 0.2]]
"""
    
    files["simple"] = safe_filesystem.create_file("simple.gnn", content)
    return files

@pytest.fixture
def isolated_temp_dir() -> Generator[Path, None, None]:
    """Provide an isolated temporary directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    
    # Cleanup
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except OSError:
        pass

@pytest.fixture
def pipeline_arguments() -> Dict[str, Any]:
    """Provide pipeline arguments for testing."""
    return {
        "target_dir": Path("input/gnn_files"),
        "output_dir": Path("output"),
        "verbose": True,
        "recursive": True,
        "only_steps": None,
        "skip_steps": None
    }

@pytest.fixture
def comprehensive_test_data(isolated_temp_dir) -> Dict[str, Any]:
    """Provide comprehensive test data."""
    return {
        "temp_dir": isolated_temp_dir,
        "gnn_files": {
            "simple": isolated_temp_dir / "simple.gnn",
            "complex": isolated_temp_dir / "complex.gnn"
        },
        "output_dir": isolated_temp_dir / "output",
        "config": {
            "test_mode": True,
            "safe_mode": True
        }
    } 