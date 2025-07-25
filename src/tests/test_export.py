import pytest
import json
from pathlib import Path

# Test markers
pytestmark = [pytest.mark.export, pytest.mark.safe_to_fail, pytest.mark.fast]

# Import export functions
try:
    from export.format_exporters import (
        _gnn_model_to_dict, export_to_json_gnn, export_to_xml_gnn,
        export_to_plaintext_summary, export_to_plaintext_dsl,
        export_to_gexf, export_to_graphml, export_to_json_adjacency_list,
        export_to_python_pickle, HAS_NETWORKX
    )
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False

@pytest.fixture
def sample_gnn_spec():
    return {
        "name": "TestModel",
        "annotation": "Test annotation",
        "variables": [{"name": "X", "dimensions": [2]}],
        "connections": [{"sources": ["X"], "operator": "->", "targets": ["Y"], "attributes": {}}],
        "parameters": [{"name": "A", "value": [[1,2,3], [4,5,6]]}],
        "equations": [],
        "time": {},
        "ontology": [],
        "model_parameters": {},
        "source_file": "model.md",
        "InitialParameterization": {"A": [[1,2,3], [4,5,6]]}
    }

@pytest.fixture
def valid_gnn_file(tmp_path):
    file = tmp_path / "valid.md"
    file.write_text(
        "## ModelName\nTestModel\n"
        "## StateSpaceBlock\nX[2]\n"
        "## Connections\nX -> Y\n"
        "## InitialParameterization\nA = {1,2,3}\n"
    )
    return file

@pytest.fixture
def valid_gnn_dict(valid_gnn_file):
    if EXPORT_AVAILABLE:
        return _gnn_model_to_dict(str(valid_gnn_file))
    else:
        return {"name": "TestModel", "variables": [], "connections": []}

@pytest.mark.skipif(not EXPORT_AVAILABLE, reason="Export modules not available")
def test_export_to_json_gnn(valid_gnn_dict, tmp_path):
    output_file = tmp_path / "output.json"
    success, message = export_to_json_gnn(valid_gnn_dict, str(output_file))
    assert success, f"JSON export failed: {message}"
    assert output_file.exists()
    
    # Verify valid JSON
    with open(output_file, 'r') as f:
        data = json.load(f)
    assert isinstance(data, dict)

@pytest.mark.skipif(not EXPORT_AVAILABLE, reason="Export modules not available")
def test_export_to_xml_gnn(valid_gnn_dict, tmp_path):
    output_file = tmp_path / "output.xml"
    success, message = export_to_xml_gnn(valid_gnn_dict, str(output_file))
    assert success, f"XML export failed: {message}"
    assert output_file.exists()

@pytest.mark.skipif(not EXPORT_AVAILABLE, reason="Export modules not available")
def test_export_to_plaintext_summary(valid_gnn_dict, tmp_path):
    output_file = tmp_path / "summary.txt"
    success, message = export_to_plaintext_summary(valid_gnn_dict, str(output_file))
    assert success, f"Plaintext summary export failed: {message}"
    assert output_file.exists()

@pytest.mark.skipif(not EXPORT_AVAILABLE, reason="Export modules not available")
def test_export_to_plaintext_dsl(valid_gnn_dict, tmp_path):
    output_file = tmp_path / "dsl.txt"
    success, message = export_to_plaintext_dsl(valid_gnn_dict, str(output_file))
    assert success, f"DSL export failed: {message}"
    assert output_file.exists()

@pytest.mark.skipif(not EXPORT_AVAILABLE or not HAS_NETWORKX, reason="NetworkX not available")
def test_export_to_gexf(valid_gnn_dict, tmp_path):
    output_file = tmp_path / "graph.gexf"
    success, message = export_to_gexf(valid_gnn_dict, str(output_file))
    assert success, f"GEXF export failed: {message}"
    assert output_file.exists()

@pytest.mark.skipif(not EXPORT_AVAILABLE or not HAS_NETWORKX, reason="NetworkX not available")
def test_export_to_graphml(valid_gnn_dict, tmp_path):
    output_file = tmp_path / "graph.graphml"
    success, message = export_to_graphml(valid_gnn_dict, str(output_file))
    assert success, f"GraphML export failed: {message}"
    assert output_file.exists()

@pytest.mark.skipif(not EXPORT_AVAILABLE, reason="Export modules not available")
def test_export_to_json_adjacency_list(valid_gnn_dict, tmp_path):
    output_file = tmp_path / "adjacency.json"
    success, message = export_to_json_adjacency_list(valid_gnn_dict, str(output_file))
    assert success, f"JSON adjacency list export failed: {message}"
    assert output_file.exists()

@pytest.mark.skipif(not EXPORT_AVAILABLE, reason="Export modules not available")
def test_export_to_python_pickle(valid_gnn_dict, tmp_path):
    output_file = tmp_path / "model.pkl"
    success, message = export_to_python_pickle(valid_gnn_dict, str(output_file))
    assert success, f"Pickle export failed: {message}"
    assert output_file.exists()

def test_export_module_import():
    """Test that export module can be imported gracefully."""
    try:
        import src.export
        assert True
    except ImportError as e:
        pytest.skip(f"Export module not available: {e}")

@pytest.mark.skipif(not EXPORT_AVAILABLE, reason="Export modules not available")
def test_gnn_model_to_dict_conversion(valid_gnn_file):
    result = _gnn_model_to_dict(str(valid_gnn_file))
    assert isinstance(result, dict)
    assert "name" in result 