"""Tests for BaseGNNSerializer shared utility methods."""
import json
import pytest
from gnn.parsers.base_serializer import BaseGNNSerializer
from gnn.parsers.common import (
    GNNInternalRepresentation,
    Variable, Connection, Parameter, OntologyMapping,
    TimeSpecification, VariableType, DataType, ConnectionType,
)


class _MinimalSerializer(BaseGNNSerializer):
    """Minimal concrete subclass for testing BaseGNNSerializer."""

    def serialize(self, model: GNNInternalRepresentation) -> str:
        return ""


def _make_model(**kwargs) -> GNNInternalRepresentation:
    defaults = {"model_name": "TestModel", "annotation": "ActInfPOMDP"}
    defaults.update(kwargs)
    return GNNInternalRepresentation(**defaults)


# ── _serialize_time_spec ───────────────────────────────────────────────────

class TestSerializeTimeSpec:
    def setup_method(self):
        self.s = _MinimalSerializer()

    def test_none_returns_empty_dict(self):
        assert self.s._serialize_time_spec(None) == {}

    def test_empty_string_returns_empty_dict(self):
        assert self.s._serialize_time_spec("") == {}

    def test_populated_time_spec(self):
        ts = TimeSpecification(time_type="Dynamic", discretization="DiscreteTime",
                               horizon=10, step_size=0.1)
        result = self.s._serialize_time_spec(ts)
        assert result["time_type"] == "Dynamic"
        assert result["discretization"] == "DiscreteTime"
        assert result["horizon"] == 10
        assert result["step_size"] == pytest.approx(0.1)

    def test_defaults_via_getattr(self):
        """Objects without explicit attrs fall back to defaults."""
        class Minimal:
            pass
        result = self.s._serialize_time_spec(Minimal())
        assert result["time_type"] == "Static"
        assert result["discretization"] is None


# ── _serialize_ontology_mappings ───────────────────────────────────────────

class TestSerializeOntologyMappings:
    def setup_method(self):
        self.s = _MinimalSerializer()

    def test_none_returns_empty_list(self):
        assert self.s._serialize_ontology_mappings(None) == []

    def test_empty_list_returns_empty_list(self):
        assert self.s._serialize_ontology_mappings([]) == []

    def test_mapping_with_dict_protocol(self):
        om = OntologyMapping(variable_name="s", ontology_term="HiddenState",
                             description="latent state")
        result = self.s._serialize_ontology_mappings([om])
        assert len(result) == 1
        assert result[0]["variable_name"] == "s"
        assert result[0]["ontology_term"] == "HiddenState"
        assert result[0]["description"] == "latent state"

    def test_non_dict_protocol_stringified(self):
        result = self.s._serialize_ontology_mappings(["plain-string"])
        assert result == ["plain-string"]


# ── _create_embedded_model_data ────────────────────────────────────────────

class TestCreateEmbeddedModelData:
    def setup_method(self):
        self.s = _MinimalSerializer()

    def test_empty_model(self):
        model = _make_model()
        data = self.s._create_embedded_model_data(model)
        assert data["model_name"] == "TestModel"
        assert data["variables"] == []
        assert data["connections"] == []
        assert data["parameters"] == []
        assert data["equations"] == []
        assert data["time_specification"] == {}
        assert data["ontology_mappings"] == []

    def test_model_with_variable(self):
        var = Variable(name="s", var_type=VariableType.HIDDEN_STATE,
                       data_type=DataType.CATEGORICAL, dimensions=[3])
        model = _make_model(variables=[var])
        data = self.s._create_embedded_model_data(model)
        assert len(data["variables"]) == 1
        v = data["variables"][0]
        assert v["name"] == "s"
        assert v["var_type"] == "hidden_state"
        assert v["data_type"] == "categorical"
        assert v["dimensions"] == [3]

    def test_model_with_connection(self):
        conn = Connection(source_variables=["s"], target_variables=["o"],
                          connection_type=ConnectionType.DIRECTED)
        model = _make_model(connections=[conn])
        data = self.s._create_embedded_model_data(model)
        assert len(data["connections"]) == 1
        c = data["connections"][0]
        assert c["source_variables"] == ["s"]
        assert c["target_variables"] == ["o"]
        assert c["connection_type"] == "directed"

    def test_model_with_parameter(self):
        param = Parameter(name="alpha", value=0.5, type_hint="float")
        model = _make_model(parameters=[param])
        data = self.s._create_embedded_model_data(model)
        assert len(data["parameters"]) == 1
        p = data["parameters"][0]
        assert p["name"] == "alpha"
        assert p["value"] == pytest.approx(0.5)
        assert p["type_hint"] == "float"

    def test_model_with_time_spec(self):
        ts = TimeSpecification(time_type="Dynamic", horizon=5)
        model = _make_model(time_specification=ts)
        data = self.s._create_embedded_model_data(model)
        assert data["time_specification"]["time_type"] == "Dynamic"
        assert data["time_specification"]["horizon"] == 5


# ── _add_embedded_model_data ───────────────────────────────────────────────

class TestAddEmbeddedModelData:
    def setup_method(self):
        self.s = _MinimalSerializer()

    def _roundtrip_data(self, content: str, model: GNNInternalRepresentation) -> dict:
        result = self.s._add_embedded_model_data(content, model)
        # The last non-blank line contains the embedded JSON.
        last_line = [l for l in result.splitlines() if l.strip()][-1]
        prefix = self.s._get_embedded_comment_prefix(self.s.format_name)
        suffix = self.s._get_embedded_comment_suffix(self.s.format_name)
        json_str = last_line[len(prefix):]
        if suffix:
            json_str = json_str[: -len(suffix)]
        return json.loads(json_str)

    def test_appends_to_content(self):
        model = _make_model()
        result = self.s._add_embedded_model_data("body", model)
        assert result.startswith("body")
        assert "MODEL_DATA:" in result

    def test_embedded_json_is_valid(self):
        model = _make_model()
        data = self._roundtrip_data("body", model)
        assert data["model_name"] == "TestModel"

    def test_format_name_affects_prefix(self):
        """Subclass with format_name 'python' gets # prefix."""
        class PythonSerializer(_MinimalSerializer):
            def __init__(self):
                super().__init__()
                self.format_name = "python"

        s = PythonSerializer()
        model = _make_model()
        result = s._add_embedded_model_data("", model)
        assert "# MODEL_DATA:" in result

    def test_format_name_affects_suffix(self):
        """Subclass with format_name 'xml' gets <!-- prefix and --> suffix."""
        class XmlSerializer(_MinimalSerializer):
            def __init__(self):
                super().__init__()
                self.format_name = "xml"

        s = XmlSerializer()
        model = _make_model()
        result = s._add_embedded_model_data("", model)
        assert "<!-- MODEL_DATA:" in result
        assert result.rstrip().endswith("-->")
