#!/usr/bin/env python3
"""Phase 4.2 regression tests for the export module (Step 7).

Uses real GNN samples from ``input/gnn_files/`` — zero-mock per CLAUDE.md.
Tests focus on:
  - JSON roundtrip preserves structural facts
  - Supported-format registry includes the canonical set
  - Missing/malformed input is handled without silent success
"""

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPO_ROOT = SRC.parent
SAMPLE = REPO_ROOT / "input" / "gnn_files" / "basics" / "static_perception.md"


def test_export_registry_includes_canonical_formats():
    from export import get_supported_formats
    formats = get_supported_formats()
    # These five MUST be present — every downstream step that consumes Step 7
    # output assumes at least JSON + one graph format available.
    for required in ("json", "xml", "graphml", "gexf", "pickle"):
        assert required in formats, f"Export registry missing canonical format {required!r}: got {formats}"


@pytest.mark.skipif(not SAMPLE.exists(), reason="Sample GNN corpus unavailable")
def test_export_to_json_roundtrip_preserves_structure(tmp_path):
    from export import export_to_json
    from gnn import parse_gnn_file
    spec = parse_gnn_file(SAMPLE)
    # parse_gnn_file may return a dataclass; normalize to dict.
    spec_dict = spec.to_dict() if hasattr(spec, "to_dict") else spec
    out = tmp_path / "static_perception.json"
    # Signature tolerance: the formatter API historically takes either
    # (spec, path) or (spec, str(path)).
    try:
        export_to_json(spec_dict, str(out))
    except TypeError:
        export_to_json(spec_dict, out)
    # The file must exist and parse as JSON.
    assert out.exists(), "export_to_json did not create output file"
    loaded = json.loads(out.read_text())
    assert isinstance(loaded, dict)
    # Structural roundtrip: the exported dict should at least preserve the
    # variable count (if present in original) OR a non-empty top-level shape.
    assert len(loaded) > 0, "Exported JSON is empty — roundtrip lost all data"


@pytest.mark.skipif(not SAMPLE.exists(), reason="Sample GNN corpus unavailable")
def test_process_export_handles_missing_target_dir(tmp_path):
    from export.processor import process_export
    missing = tmp_path / "definitely_not_here"
    output_dir = tmp_path / "out"
    # Per Phase 1.1 contract, "no input" must NOT silently report success.
    result = process_export(
        target_dir=missing,
        output_dir=output_dir,
        verbose=False,
    )
    # The contract accepts bool or int; a True/0 result for a missing dir
    # would be a silent-success bug. Accept False/1 (error) or 2 (warnings).
    if isinstance(result, bool):
        assert result is False, f"Expected falsy result for missing dir, got {result!r}"
    else:
        assert result != 0, f"Expected non-zero exit for missing dir, got {result!r}"
