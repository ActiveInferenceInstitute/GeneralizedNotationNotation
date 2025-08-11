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

# Ensure top-level 'tests' alias exists for imports like 'from tests.conftest import *'
try:
    import types as _types
    _pkg = _types.ModuleType('tests')
    _pkg.__path__ = [str(Path(__file__).parent)]  # type: ignore[attr-defined]
    sys.modules.setdefault('tests', _pkg)
    sys.modules['tests.conftest'] = sys.modules.get(__name__)
except Exception:
    pass

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

# Ensure certain optional attributes exist for compatibility with patches used in tests
def pytest_sessionstart(session):
    """Session start hook to prepare environment quirks required by tests."""
    try:
        import numpy.typing as _npt  # type: ignore
        # Some tests patch numpy.typing._array_like; ensure attribute exists
        if not hasattr(_npt, "_array_like"):
            setattr(_npt, "_array_like", object())
    except Exception:
        # Best-effort; do not fail session startup
        pass

    # Prepare essential pipeline artifacts when running directly after setup
    try:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "output"
        # Expected artifacts used by functionality tests
        expected_paths = [
            output_dir / "gnn_processing_step/actinf_pomdp_agent/actinf_pomdp_agent_parsed.json",
            output_dir / "type_check/type_check_results.json",
            output_dir / "gnn_exports/actinf_pomdp_agent",
            output_dir / "visualization/actinf_pomdp_agent",
            output_dir / "execution_results/execution_results.json",
            output_dir / "audio_processing_step/audio_results",
        ]

        if not all(p.exists() for p in expected_paths):
            # Run a minimal set of pipeline steps to generate artifacts
            import subprocess, sys as _sys
            main_py = project_root / "src" / "main.py"
            cmd = [
                _sys.executable,
                str(main_py),
                "--only-steps",
                "3,5,7,8,12,15",
                "--target-dir",
                str(project_root / "input/gnn_files"),
                "--output-dir",
                str(output_dir),
                "--verbose",
            ]
            # Best-effort run; tests should still pass locally even if some steps warn
            subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, timeout=900)
    except Exception:
        # Do not fail test collection if preparation fails
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
def mock_filesystem():
    """Back-compat alias to safe_filesystem expected by some tests."""
    from typing import Generator as _Gen
    # Reuse the safe_filesystem fixture implementation
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
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
            if self.temp_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(self.temp_dir)
                except Exception:
                    pass
    fs = SafeFileSystem(temp_dir)
    try:
        yield fs
    finally:
        fs.cleanup()

@pytest.fixture
def full_pipeline_environment(tmp_path) -> Dict[str, Any]:
    """Provide a basic environment dict used by some integration tests."""
    return {
        "project_root": tmp_path,
        "input_dir": tmp_path / "input",
        "output_dir": tmp_path / "output",
        "temp_dir": tmp_path,
        "env": {"PYTHONUNBUFFERED": "1"},
    }

@pytest.fixture
def simulate_failures() -> Dict[str, Any]:
    """Fixture to simulate step failures in pipeline tests."""
    return {"simulate": True, "failed_steps": ["render", "execute"]}

@pytest.fixture
def capture_logs(caplog):
    """Alias fixture to expose pytest's caplog as capture_logs expected by some tests."""
    return caplog

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

# =============================================================================
# Additional fixtures expected by migrated tests
# =============================================================================

@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Temporary output directory for visualization tests."""
    directory = Path(tempfile.mkdtemp()) / "viz_output"
    directory.mkdir(parents=True, exist_ok=True)
    yield directory
    try:
        import shutil
        shutil.rmtree(directory.parent)
    except Exception:
        pass

def _write_sample_gnn_markdown(target: Path):
    """Write a sample GNN markdown with ontology annotations to target path."""
    content = """
# Active Inference Model

## ActInfOntologyAnnotation
s = HiddenState
s_prime = NextHiddenState
o = Observation
π = PolicyVector
u = Action
t = Time
A = LikelihoodMatrix
B = TransitionMatrix
C = LogPreferenceVector
D = PriorOverHiddenStates
E = Habit
F = VariationalFreeEnergy
G = ExpectedFreeEnergy

## Connections
s -> o
""".strip()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)

@pytest.fixture
def test_data_dir() -> Generator[Path, None, None]:
    """Directory containing sample GNN files for directory-based tests."""
    base = Path(tempfile.mkdtemp())
    sample = base / "samples" / "actinf_pomdp_agent.md"
    _write_sample_gnn_markdown(sample)
    yield sample.parent
    try:
        import shutil
        shutil.rmtree(base)
    except Exception:
        pass

@pytest.fixture
def sample_gnn_file() -> Generator[Path, None, None]:
    """Path to a sample GNN markdown file used in tests."""
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "actinf_pomdp_agent.md"
    _write_sample_gnn_markdown(path)
    yield path
    try:
        import shutil
        shutil.rmtree(tmp)
    except Exception:
        pass

@pytest.fixture
def sample_gnn_spec() -> Dict[str, Any]:
    """Minimal in-memory GNN spec object for render tests."""
    return {
        "name": "actinf_pomdp_agent",
        "states": ["s"],
        "observations": ["o"],
        "parameters": {"A": [[0.5, 0.5]]}
    }

@pytest.fixture
def mock_render_module():
    """Mock render module exposing render_gnn_spec(spec, target, outdir)."""
    m = Mock()
    # Default behavior can be overridden in tests
    def _render(spec, target, outdir):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        artifact_name = f"{target}_artifact.txt"
        (outdir / artifact_name).write_text("ok")
        return True, "Success", [artifact_name]
    m.render_gnn_spec.side_effect = _render
    return m

@pytest.fixture
def mock_mcp_tools():
    """Simple MCP tools registry used by MCP tests.

    Supports both positional and keyword-based registration styles used across modules
    (func/function), and basic resource registration expected by tests.
    """
    class _MCPTools:
        def __init__(self):
            self.tools: Dict[str, Any] = {}
            self.resources: Dict[str, Any] = {}

        # Accept both legacy (name, func, schema, description) and modern keyword style
        def register_tool(self, name: str, *args, **kwargs):
            # Parse inputs
            function = kwargs.get("function")
            schema = kwargs.get("schema")
            description = kwargs.get("description", "")

            if function is None:
                # Positional style: register_tool(name, func, schema=None, description="")
                if len(args) >= 1:
                    function = args[0]
                if len(args) >= 2 and schema is None:
                    schema = args[1]
                if len(args) >= 3 and not description:
                    description = args[2]

            self.tools[name] = {
                "function": function,
                "func": function,
                "schema": schema or {},
                "description": description,
            }

        def register_resource(self, pattern: str, handler, description: str = ""):
            self.resources[pattern] = {
                "handler": handler,
                "description": description,
            }

        def execute_tool(self, name: str, **kwargs):
            if name not in self.tools:
                return {"error": "tool_not_found", "name": name}
            func = self.tools[name].get("function") or self.tools[name].get("func")
            return func(**kwargs)

    return _MCPTools()

# Parser sample inputs used in tests expecting fixtures
@pytest.fixture
def sample_markdown() -> str:
    return (
        "# TestModel\n\n"
        "## ModelName\nTestModel\n\n"
        "## StateSpaceBlock\ns[2,1,type=int]\n\n"
        "## Connections\ns->o\n\n"
        "## InitialParameterization\nA={(1,0),(0,1)}\n"
    )

@pytest.fixture
def sample_scala() -> str:
    return (
        "object MyModel {\n"
        "  val a_m: Matrix(Fin 2, Fin 3) = ???\n"
        "  val b_f: Matrix(Fin 2, Fin 2) = ???\n"
        "  // EFE = G + F\n"
        "}\n"
    )