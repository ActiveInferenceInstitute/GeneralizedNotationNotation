"""Simple JSON/XML round-trip tests for GNN serializers.

Adopted from ``src/gnn/testing/simple_round_trip_test.py`` (a
never-collected print-only script returning bools with zero assertions).
Converted to real pytest tests: serialize a minimal model, parse it back,
and assert fidelity.
"""

import json
import xml.etree.ElementTree as ET

from gnn.parsers.common import GNNInternalRepresentation
from gnn.parsers.json_parser import JSONGNNParser
from gnn.parsers.json_serializer import JSONSerializer
from gnn.parsers.xml_parser import XMLGNNParser
from gnn.parsers.xml_serializer import XMLSerializer


def _minimal_model() -> GNNInternalRepresentation:
    model = GNNInternalRepresentation(
        model_name="Test Model", annotation="Test annotation"
    )
    model.variables = []
    model.connections = []
    model.parameters = []
    model.equations = []
    model.raw_sections = {"content": "test content"}
    return model


def test_json_round_trip(tmp_path) -> None:
    """Serialize to JSON and parse back without loss of core fields."""
    model = _minimal_model()

    serializer = JSONSerializer()
    json_content = serializer.serialize(model)
    assert json_content, "JSON serialization produced empty content"

    json_file = tmp_path / "round_trip.json"
    json_file.write_text(json_content, encoding="utf-8")

    parsed_result = JSONGNNParser().parse_file(str(json_file))
    assert parsed_result.success, f"JSON parsing failed: {parsed_result.errors}"

    parsed_model = parsed_result.model
    assert parsed_model.model_name == "Test Model"
    assert parsed_model.annotation == "Test annotation"


def test_xml_round_trip(tmp_path) -> None:
    """Serialize to XML and parse back without loss of core fields."""
    model = _minimal_model()

    serializer = XMLSerializer()
    xml_content = serializer.serialize(model)
    assert xml_content, "XML serialization produced empty content"

    # Serialized XML must be well-formed
    ET.fromstring(xml_content)

    xml_file = tmp_path / "round_trip.xml"
    xml_file.write_text(xml_content, encoding="utf-8")

    parsed_result = XMLGNNParser().parse_file(str(xml_file))
    assert parsed_result.success, f"XML parsing failed: {parsed_result.errors}"

    parsed_model = parsed_result.model
    assert parsed_model.model_name == "Test Model"
    assert parsed_model.annotation == "Test annotation"


def test_json_serialization_is_valid_json() -> None:
    """The JSON serializer output must be parseable as JSON."""
    json_content = JSONSerializer().serialize(_minimal_model())
    assert json_content, "JSON serialization produced empty content"
    payload = json.loads(json_content)
    assert isinstance(payload, dict)
