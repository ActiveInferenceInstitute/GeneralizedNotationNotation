"""Serialize preset filtering for Step 3 multi-format output."""

from __future__ import annotations

from gnn.multi_format_processor import _formats_for_serialize_preset
from gnn.parsers.common import GNNFormat


def test_minimal_preset_keeps_only_tooling_formats() -> None:
    supported = list(GNNFormat)
    filtered, label = _formats_for_serialize_preset("minimal", supported)
    assert label == "minimal"
    names = {f.value for f in filtered}
    assert names <= {"markdown", "json", "python"}
    assert names == {"markdown", "json", "python"}


def test_full_preset_preserves_supported_list() -> None:
    supported = list(GNNFormat)
    filtered, label = _formats_for_serialize_preset("full", supported)
    assert label == "full"
    assert filtered == supported


def test_unknown_preset_defaults_to_full_behavior() -> None:
    supported = [GNNFormat.JSON, GNNFormat.PYTHON]
    filtered, label = _formats_for_serialize_preset("bogus", supported)
    assert label == "full"
    assert filtered == supported
