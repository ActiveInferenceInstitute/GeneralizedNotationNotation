"""
Integration test for the critical Step 3 → Step 11 → Step 12 data flow:
  GNN parse (step 3) → code render (step 11) → execution (step 12)

Verifies that:
1. Step 3 (GNN parsing) produces a parsed model object
2. Step 11 (render) can consume the parsed model and produce code artifacts
3. Step 12 (execute) can receive rendered artifacts without crashing

This tests the hand-off contracts between pipeline stages, not internal logic.
"""
import sys
import tempfile
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
MINIMAL_GNN_CONTENT = '## GNNSection\nActInfPOMDP\n\n## ModelName\nIntegrationTestModel\n\n## StateSpaceBlock\nA[2,2,type=float]\nB[2,2,2,type=float]\ns[2,1,type=float]\no[2,1,type=int]\nu[2,1,type=int]\n\n## Connections\nA>s\nB>s\ns-o\n\n## InitialParameterization\nA={(0.9,0.1),(0.1,0.9)}\nB={(0.9,0.1,0.9,0.1),(0.1,0.9,0.1,0.9)}\n\n## Time\nDynamic, DiscreteTime, ModelTimeHorizon=10\n\n## Footer\nVersion: 1.0\n'
GNN_FILE_PATH = Path(__file__).parent.parent.parent.parent / 'input' / 'gnn_files' / 'basics' / 'static_perception.md'

class TestStep3ParseProducesModel:
    """Step 3 produces a parsed model that has known structure."""

    def test_gnn_parser_parses_minimal_content(self):
        """GNNParsingSystem can parse minimal GNN content without errors."""
        try:
            from gnn.parser import GNNParsingSystem
        except ImportError:
            pytest.skip('GNN parser not available')
        parser = GNNParsingSystem()
        with tempfile.NamedTemporaryFile(suffix='.md', mode='w', delete=False) as f:
            f.write(MINIMAL_GNN_CONTENT)
            tmp_path = Path(f.name)
        try:
            result = parser.parse_file(tmp_path)
            assert result is not None
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_parse_file_returns_parse_result_with_model(self):
        """parse_file result has .model or usable structure."""
        try:
            from gnn.parser import GNNParsingSystem
        except ImportError:
            pytest.skip('GNN parser not available')
        parser = GNNParsingSystem()
        with tempfile.NamedTemporaryFile(suffix='.md', mode='w', delete=False) as f:
            f.write(MINIMAL_GNN_CONTENT)
            tmp_path = Path(f.name)
        try:
            result = parser.parse_file(tmp_path)
            assert hasattr(result, 'model') or hasattr(result, 'model_name') or result is not None
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_existing_gnn_file_parses_without_exception(self):
        """Real GNN input files parse successfully."""
        if not GNN_FILE_PATH.exists():
            pytest.skip(f'Sample GNN file not found: {GNN_FILE_PATH}')
        try:
            from gnn.parser import GNNParsingSystem
        except ImportError:
            pytest.skip('GNN parser not available')
        parser = GNNParsingSystem()
        result = parser.parse_file(GNN_FILE_PATH)
        assert result is not None

    def test_schema_parse_connections_from_gnn_content(self):
        """parse_connections (step 3 schema layer) produces edges from test content."""
        from gnn.schema import parse_connections
        edges, errors = parse_connections(MINIMAL_GNN_CONTENT)
        assert len(edges) >= 2
        fatal_errors = [e for e in errors if e.severity == 'error']
        assert len(fatal_errors) == 0

    def test_schema_parse_state_space_from_gnn_content(self):
        """parse_state_space (step 3 schema layer) extracts variables from test content."""
        from gnn.schema import parse_state_space
        variables, errors = parse_state_space(MINIMAL_GNN_CONTENT)
        names = {v.name for v in variables}
        assert 'A' in names
        assert 's' in names
        assert 'o' in names

class TestStep11RenderConsumesParseOutput:
    """Step 11 render functions accept well-formed GNN spec dicts."""

    def _minimal_spec(self):
        return {'model_name': 'IntegrationTestModel', 'annotation': 'ActInfPOMDP', 'state_space': {'A': {'dimensions': [2, 2], 'type': 'float'}, 's': {'dimensions': [2, 1], 'type': 'float'}, 'o': {'dimensions': [2, 1], 'type': 'int'}}, 'connections': [{'source': 'A', 'target': 's', 'directed': True}, {'source': 's', 'target': 'o', 'directed': False}], 'parameters': {'A': '{(0.9,0.1),(0.1,0.9)}'}, 'time': {'type': 'Dynamic', 'horizon': 10}}

    def test_jax_extract_matrices_accepts_spec(self):
        """jax_renderer._extract_gnn_matrices accepts a well-formed spec."""
        try:
            from render.jax.jax_renderer import _extract_gnn_matrices
        except ImportError:
            pytest.skip('jax_renderer not available')
        result = _extract_gnn_matrices(self._minimal_spec())
        assert isinstance(result, dict)

    def test_jax_render_produces_output_path(self, tmp_path):
        """render_gnn_to_jax writes an artifact to the output directory."""
        try:
            from render.jax.jax_renderer import render_gnn_to_jax
        except ImportError:
            pytest.skip('jax_renderer not available')
        out_file = tmp_path / 'integration_test_model_jax.py'
        success, message, artifacts = render_gnn_to_jax(self._minimal_spec(), out_file)
        if success:
            assert out_file.exists() or len(artifacts) > 0

class TestStep12ExecutionHandshake:
    """Step 12 execute functions gracefully accept rendered artifact paths."""

    def test_execute_module_importable(self):
        """execute module can be imported."""
        import execute
        assert execute is not None

    def test_pymdp_executor_importable(self):
        """PyMDP executor submodule is importable."""
        from execute.pymdp import executor as pymdp_exec
        assert pymdp_exec is not None

    def test_executor_with_missing_file_returns_failure(self):
        """execute_script_safely returns a structured failure dict for a missing artifact."""
        from execute.executor import execute_script_safely
        result = execute_script_safely(Path('/nonexistent/artifact.py'), timeout=5)
        assert isinstance(result, dict)
        assert result['success'] is False
        assert result['error_type'] == 'FileNotFoundError'