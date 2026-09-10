"""XML parser embedded-data tests.

Adopted and upgraded from ``src/gnn/testing/test_xml_parser_only.py`` (a
never-collected print-only script with zero assertions). The original
diagnostics are converted into real assertions: embedded ``MODEL_DATA``
payloads must parse and variable/parameter names must be unique.
"""

import json
from pathlib import Path
from typing import Any

from gnn.parsers.xml_parser import XMLGNNParser


def _minimal_embedded_data() -> dict[str, Any]:
    """Minimal embedded data mirroring the original script's fixture."""
    return {
        "model_name": "Test Model",
        "annotation": "Test annotation",
        "variables": [
            {
                "name": "A",
                "var_type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [3, 3],
            },
            {
                "name": "B",
                "var_type": "transition_matrix",
                "data_type": "float",
                "dimensions": [3, 3],
            },
        ],
        "connections": [
            {
                "source_variables": ["A"],
                "target_variables": ["B"],
                "connection_type": "directed",
            },
        ],
        "parameters": [
            {"name": "param1", "value": "value1"},
            {"name": "param2", "value": "value2"},
        ],
        "equations": [],
        "time_specification": None,
        "ontology_mappings": [],
    }


def _xml_with_embedded_data(minimal_data: dict[str, Any]) -> str:
    """Create XML content with an embedded MODEL_DATA payload."""
    return f"""<?xml version="1.0" ?>
<gnn_model name="Test Model" version="1.0">
  <metadata>
    <annotation>Test annotation</annotation>
  </metadata>
  <variables>
    <variable name="A" type="likelihood_matrix" data_type="float" dimensions="3,3"/>
    <variable name="B" type="transition_matrix" data_type="float" dimensions="3,3"/>
  </variables>
  <connections>
    <connection type="directed">
      <sources>A</sources>
      <targets>B</targets>
    </connection>
  </connections>
  <parameters>
    <parameter name="param1">value1</parameter>
    <parameter name="param2">value2</parameter>
  </parameters>
  <!-- MODEL_DATA: {json.dumps(minimal_data, separators=(",", ":"))} -->
</gnn_model>"""


def test_xml_embedded_data_parses(tmp_path: Path) -> None:
    """Embedded MODEL_DATA payloads must round-trip through the XML parser."""
    minimal_data = _minimal_embedded_data()
    xml_content = _xml_with_embedded_data(minimal_data)

    xml_file = tmp_path / "minimal.xml"
    xml_file.write_text(xml_content, encoding="utf-8")

    xml_parser = XMLGNNParser()
    parsed_result = xml_parser.parse_file(str(xml_file))

    assert parsed_result.success, f"Failed to parse XML: {parsed_result.errors}"
    model = parsed_result.model
    assert model.model_name == "Test Model"
    assert len(model.variables) == 2
    assert len(model.connections) == 1
    assert len(model.parameters) == 2


def test_xml_embedded_data_no_duplicate_names(tmp_path: Path) -> None:
    """Embedded-data parsing must not duplicate variable or parameter names."""
    minimal_data = _minimal_embedded_data()
    xml_content = _xml_with_embedded_data(minimal_data)

    xml_file = tmp_path / "minimal.xml"
    xml_file.write_text(xml_content, encoding="utf-8")

    xml_parser = XMLGNNParser()
    parsed_result = xml_parser.parse_file(str(xml_file))

    assert parsed_result.success, f"Failed to parse XML: {parsed_result.errors}"
    model = parsed_result.model

    var_names = [var.name for var in model.variables]
    param_names = [param.name for param in model.parameters]

    assert len(var_names) == len(set(var_names)), (
        f"DUPLICATE VARIABLES: {[n for n in var_names if var_names.count(n) > 1]}"
    )
    assert len(param_names) == len(set(param_names)), (
        f"DUPLICATE PARAMETERS: {[n for n in param_names if param_names.count(n) > 1]}"
    )
