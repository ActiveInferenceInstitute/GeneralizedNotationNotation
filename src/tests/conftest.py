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

# Import utilities from test_utils to avoid circular imports
from .test_utils import (
    SRC_DIR, 
    PROJECT_ROOT, 
    TEST_DIR,
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
    "utilities": "Utility function tests",
    "environment": "Environment validation tests",
    "render": "Rendering and code generation tests",
    "export": "Export functionality tests",
    "parsers": "Parser and format tests"
}

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
    return TEST_DIR

@pytest.fixture(autouse=True, scope="session")
def patch_jax_cleanup():
    """
    Patch JAX cleanup to prevent test errors.
    
    This fixture prevents JAX from accessing GPU resources during tests
    which could cause errors or slowdowns.
    """
    # Create fake JAX module if not available
    if "jax" not in sys.modules:
        class FakeJaxConfig:
            def update(self, *args, **kwargs):
                pass
        
        class FakeJax:
            config = FakeJaxConfig()
        
        sys.modules["jax"] = FakeJax()
    
    # Patch JAX config
    try:
        import jax
        jax.config.update("jax_disable_jit", True)
        jax.config.update("jax_platform_name", "cpu")
    except (ImportError, AttributeError):
        # JAX not available or patching failed - continue without it
        pass

# =============================================================================
# Test utilities fixtures
# =============================================================================

@pytest.fixture
def safe_subprocess():
    """
    Provide safe subprocess execution for tests.
    
    This fixture prevents actual subprocess execution during tests
    and instead returns mock results.
    """
    def safe_run(cmd, **kwargs):
        """Run a command safely in tests (mock implementation)."""
        try:
            class SafeResult:
                def __init__(self):
                    self.returncode = 0
                    self.stdout = f"Mock stdout for: {cmd}"
                    self.stderr = ""
            
            return SafeResult()
        except Exception as e:
            class FailResult:
                def __init__(self):
                    self.returncode = 1
                    self.stdout = ""
                    self.stderr = str(e)
            
            return FailResult()
    
    return safe_run

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
def real_imports():
    """
    Provide safe module imports for tests.
    
    This fixture attempts real imports but returns None if they fail.
    """
    def safe_import(module_name: str, fallback=None):
        """Safely import a module, returning fallback if it fails."""
        try:
            return __import__(module_name)
        except ImportError:
            return fallback
    
    return safe_import

@pytest.fixture
def mock_subprocess(safe_subprocess):
    """Provide a mock subprocess for tests."""
    return safe_subprocess

@pytest.fixture
def mock_filesystem(safe_filesystem):
    """Provide a mock filesystem for tests."""
    return safe_filesystem

@pytest.fixture
def mock_imports(real_imports):
    """Provide mock imports for tests."""
    return real_imports

@pytest.fixture
def mock_dangerous_operations():
    """
    Provide mock implementations of dangerous operations.
    
    This fixture prevents tests from performing dangerous operations
    such as system calls or network requests.
    """
    class MockDangerousOps:
        def __init__(self):
            self.operations = {
                "system_call": lambda *args, **kwargs: {"status": "mocked", "args": args},
                "network_request": lambda *args, **kwargs: {"status": "mocked", "url": args[0] if args else None},
                "file_deletion": lambda *args, **kwargs: {"status": "mocked", "path": args[0] if args else None},
                "database_query": lambda *args, **kwargs: {"status": "mocked", "query": args[0] if args else None}
            }
            self.call_history = []
        
        def __getitem__(self, key):
            """Get the operation by name and record its access."""
            if key in self.operations:
                self.call_history.append({"operation": key, "time": "now"})
                return self.operations[key]
            raise KeyError(f"Operation {key} not mocked")
        
        def reset_mocks(self):
            """Reset call history."""
            self.call_history = []
    
    return MockDangerousOps()

@pytest.fixture
def mock_llm_provider():
    """
    Provide a mock LLM provider for tests.
    
    This fixture prevents tests from making actual LLM API calls.
    """
    class MockLLMProvider:
        def __init__(self):
            self.responses = {
                "default": "This is a mock LLM response.",
                "analyze": "The model appears to be well-structured with proper state spaces.",
                "describe": "This is an Active Inference model with states X, Y and Z.",
                "explain": "The model works by propagating beliefs through the generative model."
            }
            self.call_history = []
        
        def reset_mocks(self):
            """Reset call history."""
            self.call_history = []
        
        def complete(self, prompt, *args, **kwargs):
            """Mock completion endpoint."""
            self.call_history.append({"prompt": prompt, "args": args, "kwargs": kwargs})
            
            # Return appropriate mock response based on prompt
            for key, response in self.responses.items():
                if key in prompt.lower():
                    return response
            
            return self.responses["default"]
    
    return MockLLMProvider()

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

@pytest.fixture
def capture_logs():
    """
    Capture logs during test execution.
    
    This fixture sets up a log handler that captures logs.
    """
    class LogCapture:
        def __init__(self):
            self.handler = None
            self.logs = []
            self.setup()
        
        def setup(self):
            """Set up log capturing."""
            import logging
            
            class ListHandler(logging.Handler):
                def __init__(self, log_list):
                    super().__init__()
                    self.log_list = log_list
                
                def emit(self, record):
                    self.log_list.append(self.format(record))
            
            self.handler = ListHandler(self.logs)
            logging.getLogger().addHandler(self.handler)
        
        def get_logs(self) -> str:
            """Get all logs as a string."""
            return "\n".join(self.logs)
        
        def get_log_lines(self) -> List[str]:
            """Get all logs as a list of lines."""
            return self.logs
        
        def clear(self):
            """Clear captured logs."""
            self.logs = []
        
        def cleanup(self):
            """Remove log handler."""
            if self.handler:
                logging.getLogger().removeHandler(self.handler)
    
    capture = LogCapture()
    yield capture
    capture.cleanup()

@pytest.fixture
def test_logger():
    """
    Provide a test logger.
    
    This fixture sets up a logger for test use.
    """
    logger = logging.getLogger("test")
    
    # Create a new handler for this test
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Set log level
    logger.setLevel(logging.INFO)
    
    yield logger
    
    # Clean up
    logger.removeHandler(handler)

# =============================================================================
# GNN-related fixtures
# =============================================================================

@pytest.fixture
def real_gnn_parser():
    """
    Provide a real GNN parser if available.
    
    This fixture attempts to use the real GNN parser if available,
    otherwise it provides a mock implementation.
    """
    try:
        from gnn.parsers import parse_gnn_file, parse_gnn_directory
        
        class RealGNNParser:
            def parse_file(self, file_path: Path):
                """Parse a GNN file."""
                try:
                    return parse_gnn_file(file_path)
                except Exception as e:
                    return {
                        "error": str(e),
                        "file_path": str(file_path),
                        "status": "error"
                    }
            
            def parse_directory(self, dir_path: Path):
                """Parse a directory of GNN files."""
                try:
                    return parse_gnn_directory(dir_path)
                except Exception as e:
                    return {
                        "error": str(e),
                        "directory_path": str(dir_path),
                        "status": "error",
                        "files_processed": 0
                    }
        
        return RealGNNParser()
    except ImportError:
        # Return mock parser
        class MockGNNParser:
            def parse_file(self, file_path: Path):
                return {"mock": True, "file": str(file_path)}
            
            def parse_directory(self, dir_path: Path):
                return {"mock": True, "directory": str(dir_path)}
        
        return MockGNNParser()

@pytest.fixture
def real_type_checker():
    """
    Provide a real GNN type checker if available.
    
    This fixture attempts to use the real GNN type checker if available,
    otherwise it provides a mock implementation.
    """
    try:
        from type_checker import check_file, check_directory
        
        class RealTypeChecker:
            def check_file(self, file_path: Path):
                """Type check a GNN file."""
                try:
                    return check_file(file_path)
                except Exception as e:
                    return {
                        "error": str(e),
                        "file_path": str(file_path),
                        "status": "error"
                    }
            
            def check_directory(self, dir_path: Path):
                """Type check a directory of GNN files."""
                try:
                    return check_directory(dir_path)
                except Exception as e:
                    return {
                        "error": str(e),
                        "directory_path": str(dir_path),
                        "status": "error"
                    }
            
            def generate_report(self, results):
                """Generate a report from type checking results."""
                return {
                    "total_files": len(results) if isinstance(results, list) else 1,
                    "valid_files": sum(1 for r in results if r.get("status") == "valid") if isinstance(results, list) else (1 if results.get("status") == "valid" else 0),
                    "invalid_files": sum(1 for r in results if r.get("status") != "valid") if isinstance(results, list) else (0 if results.get("status") == "valid" else 1)
                }
        
        return RealTypeChecker()
    except ImportError:
        # Return mock type checker
        class MockTypeChecker:
            def check_file(self, file_path: Path):
                return {"mock": True, "file": str(file_path), "status": "valid"}
            
            def check_directory(self, dir_path: Path):
                return [{"mock": True, "file": f"{dir_path}/file{i}.md", "status": "valid"} for i in range(3)]
            
            def generate_report(self, results):
                return {"mock": True, "total_files": 3, "valid_files": 3}
        
        return MockTypeChecker()

@pytest.fixture
def real_visualization():
    """
    Provide a real visualization module if available.
    
    This fixture attempts to use the real visualization module if available,
    otherwise it provides a mock implementation.
    """
    try:
        from visualization import generate_matrix_visualization, generate_graph_visualization
        
        class RealVisualization:
            def __init__(self, output_dir: Path):
                self.output_dir = output_dir
            
            def create_graph_visualization(self, data, filename="graph.png"):
                """Create a graph visualization."""
                try:
                    output_path = self.output_dir / filename
                    result = generate_graph_visualization(data, output_path)
                    return {"success": True, "path": str(output_path), "result": result}
                except Exception as e:
                    return {"error": str(e), "success": False}
            
            def create_matrix_visualization(self, matrix, filename="matrix.png"):
                """Create a matrix visualization."""
                try:
                    output_path = self.output_dir / filename
                    result = generate_matrix_visualization(matrix, output_path)
                    return {"success": True, "path": str(output_path), "result": result}
                except Exception as e:
                    return {"error": str(e), "success": False}
            
            def generate_visualization_report(self, visualizations):
                """Generate a report of visualizations."""
                return {
                    "total_visualizations": len(visualizations),
                    "successful": sum(1 for v in visualizations if v.get("success", False)),
                    "failed": sum(1 for v in visualizations if not v.get("success", False))
                }
        
        return RealVisualization
    except ImportError:
        # Return mock visualization
        class MockVisualization:
            def __init__(self, output_dir: Path):
                self.output_dir = output_dir
            
            def create_graph_visualization(self, data, filename="graph.png"):
                return {"mock": True, "output": str(self.output_dir / filename)}
            
            def create_matrix_visualization(self, matrix, filename="matrix.png"):
                return {"mock": True, "output": str(self.output_dir / filename)}
            
            def generate_visualization_report(self, visualizations):
                return {"mock": True, "total": len(visualizations)}
        
        return MockVisualization

@pytest.fixture
def real_export():
    """
    Provide a real export module if available.
    
    This fixture attempts to use the real export module if available,
    otherwise it provides a mock implementation.
    """
    try:
        from export import export_to_json, export_to_xml, export_to_graphml
        
        class RealExporter:
            def __init__(self, output_dir: Path):
                self.output_dir = output_dir
            
            def export_to_json(self, data, filename="export.json"):
                """Export data to JSON."""
                try:
                    output_path = self.output_dir / filename
                    export_to_json(data, output_path)
                    return {"success": True, "path": str(output_path)}
                except Exception as e:
                    return {"error": str(e), "success": False}
            
            def export_to_xml(self, data, filename="export.xml"):
                """Export data to XML."""
                try:
                    output_path = self.output_dir / filename
                    export_to_xml(data, output_path)
                    return {"success": True, "path": str(output_path)}
                except Exception as e:
                    return {"error": str(e), "success": False}
            
            def export_to_graphml(self, data, filename="export.graphml"):
                """Export data to GraphML."""
                try:
                    output_path = self.output_dir / filename
                    export_to_graphml(data, output_path)
                    return {"success": True, "path": str(output_path)}
                except Exception as e:
                    return {"error": str(e), "success": False}
        
        return RealExporter
    except ImportError:
        # Return mock exporter
        class MockExporter:
            def __init__(self, output_dir: Path):
                self.output_dir = output_dir
            
            def export_to_json(self, data, filename="export.json"):
                return {"mock": True, "output": str(self.output_dir / filename)}
            
            def export_to_xml(self, data, filename="export.xml"):
                return {"mock": True, "output": str(self.output_dir / filename)}
            
            def export_to_graphml(self, data, filename="export.graphml"):
                return {"mock": True, "output": str(self.output_dir / filename)}
        
        return MockExporter

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
def sample_pipeline_data() -> Dict[str, Any]:
    """
    Create sample pipeline data for testing.
    
    This fixture creates a sample pipeline data structure
    for use in tests.
    """
    return {
        "report_generation_time": "2023-01-01T00:00:00",
        "pipeline_output_directory": "/test/output",
        "steps": {
            "setup_artifacts": {
                "directory": "/test/output/setup_artifacts",
                "exists": True,
                "file_count": 5,
                "total_size_mb": 2.5,
                "file_types": {".json": {"count": 3, "total_size_mb": 1.5}, ".md": {"count": 2, "total_size_mb": 1.0}},
                "last_modified": "2023-01-01T00:00:00",
                "status": "success"
            },
            "gnn_processing_step": {
                "directory": "/test/output/gnn_processing_step",
                "exists": True,
                "file_count": 3,
                "total_size_mb": 1.8,
                "file_types": {".md": {"count": 2, "total_size_mb": 1.2}, ".json": {"count": 1, "total_size_mb": 0.6}},
                "last_modified": "2023-01-01T00:00:00",
                "status": "success"
            },
            "test_reports": {
                "directory": "/test/output/test_reports",
                "exists": False,
                "file_count": 0,
                "total_size_mb": 0.0,
                "file_types": {},
                "last_modified": None,
                "status": "missing"
            }
        },
        "summary": {
            "total_files_processed": 8,
            "total_size_mb": 4.3,
            "success_rate": 66.7
        }
    }

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

# =============================================================================
# MCP integration fixtures
# =============================================================================

@pytest.fixture
def mock_mcp_tools():
    """
    Provide mock MCP tools for testing.
    
    This fixture creates a mock MCP instance that can
    register and execute tools for testing.
    """
    class MockTool:
        def __init__(self, name, function, schema, description):
            self.name = name
            self.function = function
            self.schema = schema
            self.description = description
    
    class MockResource:
        def __init__(self, uri, function, description):
            self.uri = uri
            self.function = function
            self.description = description
    
    class MockMCPInstance:
        def __init__(self):
            self.tools = {}
            self.resources = {}
        
        def register_tool(self, name, function, schema, description):
            """Register a tool with the MCP."""
            self.tools[name] = MockTool(name, function, schema, description)
        
        def register_resource(self, uri, function, description):
            """Register a resource with the MCP."""
            self.resources[uri] = MockResource(uri, function, description)
        
        def execute_tool(self, name, **kwargs):
            """Execute a registered tool."""
            if name not in self.tools:
                return {"error": f"Tool '{name}' not found"}
            
            try:
                tool = self.tools[name]
                return tool.function(**kwargs)
            except Exception as e:
                return {"error": str(e)}
        
        def get_tool_list(self):
            """Get a list of registered tools."""
            return [{"name": name, "description": tool.description} for name, tool in self.tools.items()]
        
        def get_resource_list(self):
            """Get a list of registered resources."""
            return [{"uri": uri, "description": resource.description} for uri, resource in self.resources.items()]
    
    return MockMCPInstance()

@pytest.fixture
def full_pipeline_environment(isolated_temp_dir, sample_gnn_files):
    """
    Create a full pipeline environment for testing.
    
    This fixture creates a complete environment including
    GNN files, configuration, and output directories.
    """
    # Create directory structure
    input_dir = isolated_temp_dir / "input"
    output_dir = isolated_temp_dir / "output"
    config_dir = isolated_temp_dir / "config"
    
    for dir_path in [input_dir, output_dir, config_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create configuration file
    config = {
        "pipeline": {
            "steps": [
                {"name": "setup", "script": "1_setup.py", "dependencies": []},
                {"name": "tests", "script": "2_tests.py", "dependencies": ["setup"]},
                {"name": "gnn", "script": "3_gnn.py", "dependencies": ["setup"]},
                {"name": "type_checker", "script": "4_type_checker.py", "dependencies": ["gnn"]}
            ],
            "output_dir": str(output_dir),
            "verbose": True
        }
    }
    
    with open(config_dir / "pipeline_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Copy sample GNN files
    gnn_dir = input_dir / "gnn_files"
    gnn_dir.mkdir(parents=True, exist_ok=True)
    
    for name, file_path in sample_gnn_files.items():
        with open(file_path, 'r') as src:
            content = src.read()
            
        with open(gnn_dir / f"{name}.md", 'w') as dst:
            dst.write(content)
    
    return {
        "root_dir": isolated_temp_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "config_dir": config_dir,
        "gnn_dir": gnn_dir,
        "config_file": config_dir / "pipeline_config.json"
    }

@pytest.fixture
def simulate_failures():
    """
    Provide tools to simulate failures for testing.
    
    This fixture creates tools to simulate various types of
    failures in a controlled manner.
    """
    class FailureSimulator:
        def __init__(self):
            self.failures = {
                "timeout": self._simulate_timeout,
                "exception": self._simulate_exception,
                "io_error": self._simulate_io_error,
                "import_error": self._simulate_import_error,
                "syntax_error": self._simulate_syntax_error,
                "type_error": self._simulate_type_error,
                "value_error": self._simulate_value_error,
                "assertion_error": self._simulate_assertion_error,
                "memory_error": self._simulate_memory_error,
                "system_exit": self._simulate_system_exit
            }
        
        def simulate(self, failure_type: str):
            """Simulate a failure of the specified type."""
            if failure_type not in self.failures:
                raise ValueError(f"Unknown failure type: {failure_type}")
            
            failure = self.failures[failure_type]
            failure()
        
        def get_failure(self, failure_type: str):
            """Get a failure function without executing it."""
            if failure_type not in self.failures:
                raise ValueError(f"Unknown failure type: {failure_type}")
            
            return self.failures[failure_type]
        
        def get_failure_message(self, failure_type: str) -> str:
            """Get the message for a failure type."""
            messages = {
                "timeout": "Operation timed out",
                "exception": "An unexpected error occurred",
                "io_error": "I/O operation failed",
                "import_error": "Failed to import module",
                "syntax_error": "Invalid syntax",
                "type_error": "Invalid type",
                "value_error": "Invalid value",
                "assertion_error": "Assertion failed",
                "memory_error": "Out of memory",
                "system_exit": "System exit"
            }
            
            return messages.get(failure_type, "Unknown failure")
        
        def _simulate_timeout(self):
            """Simulate a timeout."""
            import time
            time.sleep(60)  # This should trigger a timeout
        
        def _simulate_exception(self):
            """Simulate a generic exception."""
            raise Exception("Simulated exception")
        
        def _simulate_io_error(self):
            """Simulate an I/O error."""
            raise IOError("Simulated I/O error")

        def _simulate_import_error(self):
            """Simulate an import error."""
            raise ImportError("Simulated import error")
        
        def _simulate_syntax_error(self):
            """Simulate a syntax error."""
            raise SyntaxError("Simulated syntax error")
        
        def _simulate_type_error(self):
            """Simulate a type error."""
            raise TypeError("Simulated type error")
        
        def _simulate_value_error(self):
            """Simulate a value error."""
            raise ValueError("Simulated value error")
        
        def _simulate_assertion_error(self):
            """Simulate an assertion error."""
            raise AssertionError("Simulated assertion error")
        
        def _simulate_memory_error(self):
            """Simulate a memory error."""
            raise MemoryError("Simulated memory error")
        
        def _simulate_system_exit(self):
            """Simulate a system exit."""
            sys.exit(1)
    
    return FailureSimulator() 