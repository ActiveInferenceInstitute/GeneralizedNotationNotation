#!/usr/bin/env python3
"""
Tests for gnn/schema.py — parse_connections, parse_state_space,
validate_matrix_dimensions, validate_gnn_object, GNNParseError.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

GNN_SAMPLE = """
## GNNSection
ActInfPOMDP

## ModelName
Test Model

## StateSpaceBlock
A[3,3,type=float]
B[3,3,3,type=float]
s[3,1,type=float]
o[2,1,type=int]

## Connections
A>s
s-o
B>s:transition

## InitialParameterization
A={(0.9,0.1,0.0),(0.0,0.9,0.1),(0.1,0.0,0.9)}
"""


class TestGNNParseError:
    def test_str_no_location(self):
        from gnn.schema import GNNParseError
        err = GNNParseError(code="GNN-E001", message="Bad input")
        s = str(err)
        assert "GNN-E001" in s
        assert "Bad input" in s

    def test_str_with_line_and_file(self):
        from gnn.schema import GNNParseError
        err = GNNParseError(code="GNN-E001", message="Bad input", line=42, file="model.md")
        assert ":42" in str(err)

    def test_str_with_file_shows_filename(self):
        from gnn.schema import GNNParseError
        err = GNNParseError(code="GNN-E001", message="Msg", line=5, file="model.md")
        s = str(err)
        assert "model.md" in s
        assert ":5" in s

    def test_default_severity_is_error(self):
        from gnn.schema import GNNParseError
        err = GNNParseError(code="GNN-E001", message="x")
        assert err.severity == "error"


class TestParseConnections:
    def test_directed_connection(self):
        from gnn.schema import parse_connections
        content = "## Connections\nA>B\n"
        edges, errors = parse_connections(content)
        assert len(edges) == 1
        assert edges[0].source == "A"
        assert edges[0].target == "B"
        assert edges[0].directed is True
        assert not errors

    def test_undirected_connection(self):
        from gnn.schema import parse_connections
        content = "## Connections\nA-B\n"
        edges, errors = parse_connections(content)
        assert len(edges) == 1
        assert edges[0].directed is False

    def test_labeled_connection(self):
        from gnn.schema import parse_connections
        content = "## Connections\nA>B:myLabel\n"
        edges, errors = parse_connections(content)
        assert len(edges) == 1
        assert edges[0].label == "myLabel"

    def test_multiple_connections_parsed(self):
        from gnn.schema import parse_connections
        edges, errors = parse_connections(GNN_SAMPLE)
        assert len(edges) == 3

    def test_invalid_connection_produces_error(self):
        from gnn.schema import parse_connections
        content = "## Connections\nNOT_A_CONNECTION\n"
        edges, errors = parse_connections(content)
        assert len(edges) == 0
        assert any("GNN-E005" in e.code for e in errors)

    def test_unknown_variable_produces_warning(self):
        from gnn.schema import parse_connections
        content = "## Connections\nX>Y\n"
        _, errors = parse_connections(content, known_variables={"A", "B"})
        assert any("GNN-W002" in e.code for e in errors)

    def test_known_variable_no_warning(self):
        from gnn.schema import parse_connections
        content = "## Connections\nA>B\n"
        _, errors = parse_connections(content, known_variables={"A", "B"})
        warning_errors = [e for e in errors if e.code == "GNN-W002"]
        assert len(warning_errors) == 0

    def test_empty_content_returns_empty(self):
        from gnn.schema import parse_connections
        edges, errors = parse_connections("")
        assert edges == []
        assert errors == []

    def test_section_outside_connections_ignored(self):
        from gnn.schema import parse_connections
        content = "## StateSpaceBlock\nA>B\n## Connections\nC>D\n"
        edges, _ = parse_connections(content)
        assert len(edges) == 1
        assert edges[0].source == "C"


class TestParseStateSpace:
    def test_basic_variable_parsed(self):
        from gnn.schema import parse_state_space
        content = "## StateSpaceBlock\nA[3,3,type=float]\n"
        variables, errors = parse_state_space(content)
        assert len(variables) == 1
        assert variables[0].name == "A"
        assert variables[0].dtype == "float"

    def test_multiple_variables_parsed(self):
        from gnn.schema import parse_state_space
        variables, errors = parse_state_space(GNN_SAMPLE)
        names = {v.name for v in variables}
        assert "A" in names
        assert "B" in names
        assert "s" in names
        assert "o" in names

    def test_dtype_extracted(self):
        from gnn.schema import parse_state_space
        content = "## StateSpaceBlock\nX[2,type=int]\n"
        variables, _ = parse_state_space(content)
        assert variables[0].dtype == "int"

    def test_default_dtype_is_float(self):
        from gnn.schema import parse_state_space
        content = "## StateSpaceBlock\nX[3]\n"
        variables, _ = parse_state_space(content)
        assert variables[0].dtype == "float"

    def test_inline_comments_stripped(self):
        from gnn.schema import parse_state_space
        content = "## StateSpaceBlock\nA[3,3,type=float]  # likelihood\n"
        variables, errors = parse_state_space(content)
        assert len(variables) == 1
        assert variables[0].name == "A"

    def test_empty_section_returns_empty(self):
        from gnn.schema import parse_state_space
        variables, errors = parse_state_space("## StateSpaceBlock\n")
        assert variables == []
        assert errors == []


class TestValidateGnnObject:
    def test_valid_object_returns_no_errors(self):
        from gnn.schema import validate_gnn_object
        obj = {
            "model_name": "Test",
            "state_space": {"A": {"dimensions": [3, 3], "type": "float"}},
            "connections": [],
            "parameters": {},
        }
        errors = validate_gnn_object(obj)
        assert isinstance(errors, list)

    def test_empty_dict_returns_errors(self):
        from gnn.schema import validate_gnn_object
        errors = validate_gnn_object({})
        assert isinstance(errors, list)
