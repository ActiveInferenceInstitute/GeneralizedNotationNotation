"""
Test configuration and fixtures for the GNN Processing Pipeline.

Phase 7: conftest was reduced from 545 lines to a focused set of project-
specific fixtures. Removed:
  - pathlib._local PurePosixPath patch (unused, speculative)
  - safe_to_fail marker auto-apply (Phase 7.1)
  - Unused fixtures: full_pipeline_environment, simulate_failures,
    capture_logs, pipeline_arguments
  - RealRenderModule recovery fallback (in-tree imports always succeed)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

# Make "tests.*" an importable alias for the src/tests/ directory so that
# tests which do `from tests.conftest import X` continue to resolve.
import types as _types
_pkg = _types.ModuleType("tests")
_pkg.__path__ = [str(Path(__file__).parent)]  # type: ignore[attr-defined]
sys.modules.setdefault("tests", _pkg)
sys.modules["tests.conftest"] = sys.modules.get(__name__)


# -----------------------------------------------------------------------------
# Marker configuration
# -----------------------------------------------------------------------------

PYTEST_MARKERS: Dict[str, str] = {
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for component interactions",
    "performance": "Performance and resource usage tests",
    "slow": "Tests that take significant time to complete",
    "fast": "Quick tests for rapid feedback",
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
    "visualization": "Visualization tests",
}


def pytest_configure(config: Any) -> None:
    """Register project-specific pytest markers."""
    for name, description in PYTEST_MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_collection_modifyitems(config: Any, items: list) -> None:
    """Tag slow tests with the performance marker for dashboarding."""
    for item in items:
        if any(m.name == "slow" for m in item.iter_markers()):
            item.add_marker(pytest.mark.performance)


# -----------------------------------------------------------------------------
# Session-level fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Session-wide test configuration."""
    return {
        "test_mode": True,
        "safe_mode": True,
        "temp_dir": tempfile.mkdtemp(),
        "max_test_duration": 300,
        "memory_limit_mb": 1024,
    }


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def src_dir() -> Path:
    """Absolute path to the src/ directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Absolute path to the src/tests/ directory."""
    return Path(__file__).parent


# -----------------------------------------------------------------------------
# Filesystem fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def safe_filesystem() -> Generator[Any, None, None]:
    """A scratch filesystem under a fresh tempdir that cleans itself up."""
    temp_dir = Path(tempfile.mkdtemp())

    class SafeFileSystem:
        def __init__(self, base: Path) -> None:
            self.temp_dir = base
            self.created_files: list[Path] = []
            self.created_dirs: list[Path] = []

        def create_file(self, path: Any, content: str = "") -> Path:
            full = self.temp_dir / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)
            self.created_files.append(full)
            return full

        def create_dir(self, path: Any) -> Path:
            full = self.temp_dir / path
            full.mkdir(parents=True, exist_ok=True)
            self.created_dirs.append(full)
            return full

        def cleanup(self) -> None:
            import shutil
            if self.temp_dir.exists():
                try:
                    shutil.rmtree(self.temp_dir)
                except OSError:
                    pass

    fs = SafeFileSystem(temp_dir)
    yield fs
    fs.cleanup()


@pytest.fixture
def isolated_temp_dir() -> Generator[Path, None, None]:
    """A throwaway temp directory with automatic cleanup."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_directories(tmp_path: Path) -> Dict[str, Path]:
    """Standard input/output/temp directory set anchored under tmp_path."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    for d in (input_dir, output_dir, temp_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "temp_dir": temp_dir,
        "root": tmp_path,
    }


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Temporary output directory for visualization tests."""
    base = Path(tempfile.mkdtemp())
    directory = base / "viz_output"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        yield directory
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


# -----------------------------------------------------------------------------
# GNN sample content fixtures
# -----------------------------------------------------------------------------

_SAMPLE_GNN_CONTENT = """
# Test GNN Model

## ModelName
test_model

## StateSpaceBlock
s[3,1,type=int]
o[3,1,type=int]

## Connections
s -> o

## InitialParameterization
A = [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]
B = [[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]]
C = [0.0, 0.0, 1.0]
D = [0.34, 0.33, 0.33]
"""


@pytest.fixture
def sample_gnn_files(safe_filesystem) -> Dict[str, Path]:
    """Pair of on-disk GNN files sharing a minimal coherent POMDP schema."""
    files = {
        "simple": safe_filesystem.create_file("simple.gnn", _SAMPLE_GNN_CONTENT),
        "second": safe_filesystem.create_file(
            "second.gnn", _SAMPLE_GNN_CONTENT.replace("test_model", "second_model"),
        ),
    }
    return files


def _write_sample_gnn_markdown(target: Path) -> None:
    """Write a minimal GNN markdown with ontology annotations to ``target``."""
    content = (
        "# Active Inference Model\n\n"
        "## ActInfOntologyAnnotation\n"
        "s = HiddenState\n"
        "s_prime = NextHiddenState\n"
        "o = Observation\n"
        "π = PolicyVector\n"
        "u = Action\n"
        "t = Time\n"
        "A = LikelihoodMatrix\n"
        "B = TransitionMatrix\n"
        "C = LogPreferenceVector\n"
        "D = PriorOverHiddenStates\n"
        "E = Habit\n"
        "F = VariationalFreeEnergy\n"
        "G = ExpectedFreeEnergy\n\n"
        "## Connections\n"
        "s -> o\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


@pytest.fixture
def test_data_dir() -> Generator[Path, None, None]:
    """Directory containing a sample GNN file at samples/actinf_pomdp_agent.md."""
    base = Path(tempfile.mkdtemp())
    sample = base / "samples" / "actinf_pomdp_agent.md"
    _write_sample_gnn_markdown(sample)
    try:
        yield sample.parent
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def sample_gnn_file() -> Generator[Path, None, None]:
    """Path to a single on-disk sample GNN markdown file."""
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "actinf_pomdp_agent.md"
    _write_sample_gnn_markdown(path)
    try:
        yield path
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_gnn_spec() -> Dict[str, Any]:
    """In-memory GNN spec dict used by render tests."""
    return {
        "name": "actinf_pomdp_agent",
        "states": ["s"],
        "observations": ["o"],
        "parameters": {"A": [[0.5, 0.5]]},
    }


@pytest.fixture
def sample_markdown() -> str:
    """Minimal GNN markdown source for parser tests."""
    return (
        "# TestModel\n\n"
        "## ModelName\nTestModel\n\n"
        "## StateSpaceBlock\ns[2,1,type=int]\n\n"
        "## Connections\ns->o\n\n"
        "## InitialParameterization\nA={(1,0),(0,1)}\n"
    )


@pytest.fixture
def sample_scala() -> str:
    """Minimal Scala source snippet for parser tests."""
    return (
        "object MyModel {\n"
        "  val a_m: Matrix(Fin 2, Fin 3) = ???\n"
        "  val b_f: Matrix(Fin 2, Fin 2) = ???\n"
        "  // EFE = G + F\n"
        "}\n"
    )


@pytest.fixture
def comprehensive_test_data(isolated_temp_dir: Path) -> Dict[str, Any]:
    """Consolidated test-data bundle used by integration tests."""
    return {
        "temp_dir": isolated_temp_dir,
        "gnn_files": {
            "simple": isolated_temp_dir / "simple.gnn",
            "complex": isolated_temp_dir / "complex.gnn",
        },
        "output_dir": isolated_temp_dir / "output",
        "config": {"test_mode": True, "safe_mode": True},
    }


# -----------------------------------------------------------------------------
# Render + MCP test helpers
# -----------------------------------------------------------------------------

class _RealRenderModule:
    """Thin adapter used by render integration tests — delegates to
    ``render.processor.render_gnn_spec``. Phase 7: fallback chain removed."""

    def render_gnn_spec(self, spec: Any, target: str, outdir: Any) -> Any:
        from render.processor import render_gnn_spec
        return render_gnn_spec(spec, target, outdir)


@pytest.fixture
def test_render_module() -> _RealRenderModule:
    return _RealRenderModule()


class _MCPTools:
    """Lightweight in-memory MCP registry used by MCP wiring tests."""

    def __init__(self) -> None:
        self.tools: Dict[str, Any] = {}
        self.resources: Dict[str, Any] = {}

    def register_tool(self, name: str, *args: Any, **kwargs: Any) -> None:
        function = kwargs.get("function")
        schema = kwargs.get("schema")
        description = kwargs.get("description", "")
        if function is None and args:
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

    def register_resource(self, pattern: str, handler: Any, description: str = "") -> None:
        self.resources[pattern] = {"handler": handler, "description": description}

    def execute_tool(self, name: str, **kwargs: Any) -> Any:
        if name not in self.tools:
            return {"error": "tool_not_found", "name": name}
        func = self.tools[name].get("function") or self.tools[name].get("func")
        return func(**kwargs)


@pytest.fixture
def test_mcp_tools() -> _MCPTools:
    return _MCPTools()
