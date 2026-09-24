#!/usr/bin/env python3
"""
Contract tests for GNNParser.parse_content degradation visibility (BC-02c/M-04).

An unknown ``format_hint`` is a caller error and raises. A known hint whose
parser fails or reports ``success=False`` recovers to markdown parsing, but
the returned ParsedGNN carries a visible ``metadata["parse_degraded"]``
marker so the degraded result is distinguishable from an honest parse. The
plain markdown path is never marked degraded.
"""

from __future__ import annotations

from typing import Any

import pytest

from gnn.schema_validator.syntax import GNNParser
from gnn.types import ParsedGNN


def test_unknown_format_hint_raises() -> None:
    """An unknown format hint is a caller error and must fail loud."""
    parser = GNNParser(enhanced_validation=True)
    with pytest.raises(ValueError, match="Unknown format_hint 'bogus'"):
        parser.parse_content("## ModelName\nM\n", format_hint="bogus")


def test_failed_format_parse_carries_visible_degraded_marker() -> None:
    """Content that fails its claimed format degrades visibly, not silently."""
    parser = GNNParser(enhanced_validation=True)
    parsed: ParsedGNN = parser.parse_content(
        "## ModelName\nNotJson\n", format_hint="json"
    )
    assert isinstance(parsed, ParsedGNN)
    degraded: Any = parsed.metadata.get("parse_degraded")
    assert isinstance(degraded, dict)
    assert degraded["requested_format"] == "json"
    assert degraded["fallback"] == "markdown"
    assert degraded["reason"]


def test_markdown_path_has_no_degraded_marker() -> None:
    """A plain markdown parse must not be marked degraded."""
    parser = GNNParser(enhanced_validation=False)
    parsed = parser.parse_content("## ModelName\nM\n")
    assert isinstance(parsed, ParsedGNN)
    assert "parse_degraded" not in parsed.metadata


def test_metadata_marker_is_additive_to_parsed_gnn_shape() -> None:
    """The degraded marker rides in metadata; no ParsedGNN field changed."""
    parser = GNNParser(enhanced_validation=True)
    degraded_parse = parser.parse_content("## ModelName\nNotJson\n", format_hint="yaml")
    clean_parse = parser.parse_content("## ModelName\nM\n")
    assert type(degraded_parse) is type(clean_parse) is ParsedGNN
    assert "parse_degraded" in degraded_parse.metadata
    assert "parse_degraded" not in clean_parse.metadata
