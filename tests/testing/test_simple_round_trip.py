"""Simple round-trip tests for the GNN JSON and XML formats.

Adopted from ``src/gnn/testing/simple_round_trip_test.py`` (SC-42) and
upgraded from print-and-return-False probes into real pytest assertions: a
failed serialization or parse-back now fails the test instead of passing
vacuously. The reference markdown comes from the shared sample writer in
``tests.helpers.gnn_samples`` (the same fixture source ``conftest.py`` uses),
so no repo-layout path math is needed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from tests.helpers.gnn_samples import write_sample_gnn_markdown


def test_json_round_trip(tmp_path: Any) -> None:
    """Test JSON round-trip conversion with minimal dependencies."""
    sample = tmp_path / "actinf_pomdp_agent.md"
    write_sample_gnn_markdown(sample)
    markdown_content = sample.read_text(encoding="utf-8")

    from gnn.parsers.common import GNNInternalRepresentation
    from gnn.parsers.json_parser import JSONGNNParser
    from gnn.parsers.json_serializer import JSONSerializer

    model = GNNInternalRepresentation(
        model_name="Test Model", annotation="Test annotation"
    )

    # Add some basic content to the model
    model.variables = []
    model.connections = []
    model.parameters = []
    model.equations = []
    model.raw_sections = {"content": markdown_content}

    # Serialize to JSON
    serializer = JSONSerializer()
    json_content = serializer.serialize(model)
    assert json_content, "JSON serialization produced empty content"

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        tf.write(json_content)
        temp_file = Path(tf.name)

    # Parse back from JSON
    parser = JSONGNNParser()
    parsed_result = parser.parse_file(str(temp_file))
    assert parsed_result.success, f"JSON parsing failed: {parsed_result.errors}"

    # Clean up
    temp_file.unlink()


def test_xml_round_trip() -> None:
    """Test XML round-trip conversion with minimal dependencies."""
    from gnn.parsers.common import GNNInternalRepresentation
    from gnn.parsers.xml_parser import XMLGNNParser
    from gnn.parsers.xml_serializer import XMLSerializer

    model = GNNInternalRepresentation(
        model_name="Test Model", annotation="Test annotation"
    )

    # Add some basic content to the model
    model.variables = []
    model.connections = []
    model.parameters = []
    model.equations = []
    model.raw_sections = {"content": "test content"}

    # Serialize to XML
    serializer = XMLSerializer()
    xml_content = serializer.serialize(model)
    assert xml_content, "XML serialization produced empty content"

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as tf:
        tf.write(xml_content)
        temp_file = Path(tf.name)

    # Parse back from XML
    parser = XMLGNNParser()
    parsed_result = parser.parse_file(str(temp_file))
    assert parsed_result.success, f"XML parsing failed: {parsed_result.errors}"

    # Clean up
    temp_file.unlink()
