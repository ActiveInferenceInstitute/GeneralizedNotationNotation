"""Tests for JSONGNNParser."""
import json
import pytest
from gnn.parsers.json_parser import JSONGNNParser
from gnn.parsers.common import VariableType, DataType, ConnectionType


@pytest.fixture
def parser():
    return JSONGNNParser()


def _json(obj: dict) -> str:
    return json.dumps(obj)


# ── parse_string ───────────────────────────────────────────────────────────

class TestParseString:
    def test_valid_minimal_json(self, parser):
        result = parser.parse_string(_json({"model_name": "M"}))
        assert result.success is True
        assert result.model.model_name == "M"

    def test_non_json_content_returns_failure(self, parser):
        result = parser.parse_string("## GNNSection\nActInfPOMDP")
        assert result.success is False
        assert result.errors  # at least one error message

    def test_invalid_json_returns_failure(self, parser):
        result = parser.parse_string("{bad json:")
        assert result.success is False
        assert any("JSON" in e for e in result.errors)

    def test_version_and_annotation(self, parser):
        content = _json({"model_name": "M", "version": "2.0", "annotation": "note"})
        result = parser.parse_string(content)
        assert result.success is True
        assert result.model.version == "2.0"
        assert result.model.annotation == "note"

    def test_leading_whitespace_stripped(self, parser):
        result = parser.parse_string("  " + _json({"model_name": "M"}))
        assert result.success is True


# ── variables ──────────────────────────────────────────────────────────────

class TestParseVariables:
    def test_basic_variable(self, parser):
        content = _json({
            "model_name": "M",
            "variables": [{"name": "s", "var_type": "hidden_state",
                           "data_type": "categorical", "dimensions": [3]}]
        })
        result = parser.parse_string(content)
        assert result.success is True
        assert len(result.model.variables) == 1
        v = result.model.variables[0]
        assert v.name == "s"
        assert v.var_type == VariableType.HIDDEN_STATE
        assert v.data_type == DataType.CATEGORICAL
        assert v.dimensions == [3]

    def test_malformed_variable_entry_skipped(self, parser):
        """An entry that raises during construction should be skipped; valid entries kept."""
        # Use an unknown var_type to trigger fallback, not an exception — parser handles this.
        # To actually trigger the except clause, pass a non-dict entry which will fail
        # on .get() call.
        content = _json({
            "model_name": "M",
            "variables": [
                "not-a-dict",            # malformed — will be skipped
                {"name": "s", "var_type": "hidden_state", "data_type": "categorical"}
            ]
        })
        result = parser.parse_string(content)
        # The malformed entry is skipped; the valid one is parsed
        assert any(v.name == "s" for v in result.model.variables)

    def test_unknown_var_type_defaults_to_hidden_state(self, parser):
        content = _json({
            "model_name": "M",
            "variables": [{"name": "x", "var_type": "nonexistent_type"}]
        })
        result = parser.parse_string(content)
        assert result.model.variables[0].var_type == VariableType.HIDDEN_STATE

    def test_unknown_data_type_defaults_to_continuous(self, parser):
        content = _json({
            "model_name": "M",
            "variables": [{"name": "x", "data_type": "nonsense"}]
        })
        result = parser.parse_string(content)
        assert result.model.variables[0].data_type == DataType.CONTINUOUS


# ── connections ────────────────────────────────────────────────────────────

class TestParseConnections:
    def test_directed_connection(self, parser):
        content = _json({
            "model_name": "M",
            "connections": [{"source_variables": ["s"],
                             "target_variables": ["o"],
                             "connection_type": "directed"}]
        })
        result = parser.parse_string(content)
        assert len(result.model.connections) == 1
        c = result.model.connections[0]
        assert c.source_variables == ["s"]
        assert c.target_variables == ["o"]
        assert c.connection_type == ConnectionType.DIRECTED

    def test_unknown_connection_type_defaults_to_directed(self, parser):
        content = _json({
            "model_name": "M",
            "connections": [{"source_variables": ["s"],
                             "target_variables": ["o"],
                             "connection_type": "weird_type"}]
        })
        result = parser.parse_string(content)
        assert result.model.connections[0].connection_type == ConnectionType.DIRECTED


# ── get_supported_extensions ───────────────────────────────────────────────

class TestGetSupportedExtensions:
    def test_returns_dot_json(self, parser):
        assert parser.get_supported_extensions() == ['.json']


# ── full round-trip ────────────────────────────────────────────────────────

class TestFullModel:
    def test_complete_model_parse(self, parser):
        payload = {
            "model_name": "FullModel",
            "version": "1.1",
            "annotation": "ActInfPOMDP",
            "variables": [
                {"name": "s", "var_type": "hidden_state", "data_type": "categorical",
                 "dimensions": [3]},
                {"name": "o", "var_type": "observation", "data_type": "categorical",
                 "dimensions": [2]},
            ],
            "connections": [
                {"source_variables": ["s"], "target_variables": ["o"],
                 "connection_type": "directed"}
            ],
            "parameters": [{"name": "alpha", "value": 0.5, "type_hint": "float"}],
            "time_specification": {"time_type": "Dynamic", "horizon": 10},
            "ontology_mappings": [{"variable_name": "s",
                                   "ontology_term": "HiddenState"}],
        }
        result = parser.parse_string(_json(payload))
        assert result.success is True
        m = result.model
        assert m.model_name == "FullModel"
        assert len(m.variables) == 2
        assert len(m.connections) == 1
        assert len(m.parameters) == 1
        assert m.time_specification.time_type == "Dynamic"
        assert m.time_specification.horizon == 10
        assert len(m.ontology_mappings) == 1
