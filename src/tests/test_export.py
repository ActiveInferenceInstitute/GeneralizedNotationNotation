import pytest

# Test markers
pytestmark = [pytest.mark.export, pytest.mark.safe_to_fail, pytest.mark.fast]

from src.render.render import render_gnn_spec

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

def test_render_to_pymdp(tmp_path, sample_gnn_spec):
    ok, msg, artifacts = render_gnn_spec(sample_gnn_spec, "pymdp", tmp_path)
    assert ok or "not available" in msg

def test_render_to_rxinfer_toml(tmp_path, sample_gnn_spec):
    ok, msg, artifacts = render_gnn_spec(sample_gnn_spec, "rxinfer_toml", tmp_path)
    assert ok or "not available" in msg

def test_render_to_discopy(tmp_path, sample_gnn_spec):
    ok, msg, artifacts = render_gnn_spec(sample_gnn_spec, "discopy", tmp_path)
    assert ok or "not available" in msg

def test_render_to_discopy_combined(tmp_path, sample_gnn_spec):
    ok, msg, artifacts = render_gnn_spec(sample_gnn_spec, "discopy_combined", tmp_path)
    assert ok or "not available" in msg

def test_render_to_activeinference_jl(tmp_path, sample_gnn_spec):
    ok, msg, artifacts = render_gnn_spec(sample_gnn_spec, "activeinference_jl", tmp_path)
    assert ok or "not available" in msg

def test_render_to_jax(tmp_path, sample_gnn_spec):
    ok, msg, artifacts = render_gnn_spec(sample_gnn_spec, "jax", tmp_path)
    assert ok or "not available" in msg

def test_render_to_jax_pomdp(tmp_path, sample_gnn_spec):
    ok, msg, artifacts = render_gnn_spec(sample_gnn_spec, "jax_pomdp", tmp_path)
    assert ok or "not available" in msg

def test_render_with_unsupported_targets(tmp_path, sample_gnn_spec):
    ok, msg, artifacts = render_gnn_spec(sample_gnn_spec, "unsupported_target", tmp_path)
    assert not ok
    assert "Unsupported" in msg

import pytest
from export.format_exporters import (
    _gnn_model_to_dict, export_to_json_gnn, export_to_xml_gnn,
    export_to_plaintext_summary, export_to_plaintext_dsl,
    export_to_gexf, export_to_graphml, export_to_json_adjacency_list,
    export_to_python_pickle, HAS_NETWORKX
)
from pathlib import Path
import json

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
    return _gnn_model_to_dict(str(valid_gnn_file))

@pytest.fixture
def corrupted_gnn_file(tmp_path):
    file = tmp_path / "corrupt.md"
    file.write_text("## ModelName\n## StateSpaceBlock\nX[2\n## Connections\nX -> Y\n## InitialParameterization\nA = {1,2,3")
    return file

def test_gnn_model_to_dict_parsing(valid_gnn_file):
    d = _gnn_model_to_dict(str(valid_gnn_file))
    assert isinstance(d, dict)
    assert "name" in d

def test_export_to_json_gnn(tmp_path, valid_gnn_dict):
    out = tmp_path / "model.json"
    export_to_json_gnn(valid_gnn_dict, out)
    assert out.exists()
    with open(out) as f:
        json.load(f)

def test_export_to_xml_gnn(tmp_path, valid_gnn_dict):
    out = tmp_path / "model.xml"
    export_to_xml_gnn(valid_gnn_dict, out)
    assert out.exists()
    assert out.read_text().startswith('<?xml')

def test_export_to_plaintext_summary(tmp_path, valid_gnn_dict):
    out = tmp_path / "summary.txt"
    export_to_plaintext_summary(valid_gnn_dict, out)
    assert out.exists()
    assert "TestModel" in out.read_text()

def test_export_to_plaintext_dsl(tmp_path, valid_gnn_dict):
    out = tmp_path / "dsl.txt"
    export_to_plaintext_dsl(valid_gnn_dict, out)
    assert out.exists()

def test_export_to_python_pickle(tmp_path, valid_gnn_dict):
    out = tmp_path / "model.pkl"
    export_to_python_pickle(valid_gnn_dict, out)
    assert out.exists()

@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not available")
def test_export_to_gexf(tmp_path, valid_gnn_dict):
    out = tmp_path / "model.gexf"
    export_to_gexf(valid_gnn_dict, out)
    assert out.exists()

@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not available")
def test_export_to_graphml(tmp_path, valid_gnn_dict):
    out = tmp_path / "model.graphml"
    export_to_graphml(valid_gnn_dict, out)
    assert out.exists()

@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not available")
def test_export_to_json_adjacency_list(tmp_path, valid_gnn_dict):
    out = tmp_path / "adjacency.json"
    export_to_json_adjacency_list(valid_gnn_dict, out)
    assert out.exists()
    with open(out) as f:
        json.load(f)

def test_export_with_corrupted_gnn_file(corrupted_gnn_file):
    # The parser is designed to handle corrupted files gracefully
    # It should return a dictionary with whatever it can parse, not raise an exception
    result = _gnn_model_to_dict(str(corrupted_gnn_file))
    assert isinstance(result, dict), "Parser should return a dictionary even for corrupted files"
    assert "name" in result, "Parser should extract model name even from corrupted files"
    # The corrupted sections should be present but may be incomplete
    assert "raw_sections" in result, "Parser should preserve raw sections" 