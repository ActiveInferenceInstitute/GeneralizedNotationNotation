"""Connection labels survive data interchange and embedded reconstruction."""

import pytest

from gnn.parsers.common import GNNFormat
from gnn.parsers.markdown_parser import MarkdownGNNParser
from gnn.parsers.system import PARSER_REGISTRY, SERIALIZER_REGISTRY


@pytest.mark.parametrize("fmt", list(SERIALIZER_REGISTRY))
@pytest.mark.parametrize("annotation", [None, "likelihood:α"])
def test_connection_annotation_roundtrip(
    fmt: GNNFormat, annotation: str | None
) -> None:
    suffix = f":{annotation}" if annotation else ""
    original = MarkdownGNNParser().parse_string(
        f"## ModelName\nM\n## StateSpaceBlock\ns[2]\no[2]\n## Connections\ns>o{suffix}\n"
    )
    assert original.success
    assert original.model.connections[0].annotation == annotation
    serialized = SERIALIZER_REGISTRY[fmt]().serialize(original.model)
    restored = PARSER_REGISTRY[fmt]().parse_string(serialized)
    assert restored.success, restored.errors
    assert restored.model.connections[0].annotation == annotation
    assert restored.model.connections[0].target_variables == ["o"]
