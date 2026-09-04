"""Structured error contract: on_error modes, GNNExtractionError, parse provenance.

Contract under test (pinned decisions #3):
- on_error="lenient" (default): return the spec; failed parameter blocks are
  recorded in matrix_provenance[matrix] = {"source": "parse_error", ...} and
  adapter_notes — never silently dropped.
- on_error="raise": raise GNNExtractionError carrying code, message, line, section.
- on_error="collect": return (spec_or_None, errors) with structured errors.
- invalid on_error value: ValueError.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from gnn.pomdp_extractor import GNNExtractionError, extract_pomdp_from_file

REPO = Path(__file__).resolve().parents[3]

VALID_A = "A={\n  (0.9, 0.05, 0.05),\n  (0.05, 0.9, 0.05),\n  (0.05, 0.05, 0.9)\n}"
MALFORMED_A = (
    "A={\n  (0.9, 0.05, 0.05),\n  (0.05, 0.9, oops),\n  (0.05, 0.05, 0.9)\n}"
)


def _gnn_file(tmp_path: Path, a_block: str, name: str = "malformed_a.md") -> Path:
    """Write an actinf-shaped GNN file with the given A parameterization.

    The offending actinf B InitialParameterization comment is normalized to the
    canonical (rows=next, cols=previous) phrasing so a B-orientation
    contradiction cannot mask the A parse failure under test.
    """
    template = (REPO / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md").read_text()
    content = template.replace(
        "# B: 3 states x 3 previous states x 3 actions. Each action "
        "deterministically moves to a state. For each slice, rows are previous "
        "states, columns are next states. Each slice is a transition matrix "
        "corresponding to a different action selection.",
        "# B: 3 states x 3 previous states x 3 actions. For each slice, rows "
        "are next states, columns are previous states.",
    )
    assert "rows are next states, columns are previous states" in content, (
        "B comment normalization anchor missing"
    )
    content = content.replace(VALID_A, a_block)
    assert content != template or a_block == VALID_A, "A block template anchor missing"
    path = tmp_path / name
    path.write_text(content)
    return path


def _prov_entry(spec: Any, matrix: str) -> dict[str, Any]:
    provenance = spec.matrix_provenance or {}
    return cast(dict[str, Any], provenance.get(matrix, {}))


def test_lenient_mode_records_parse_error_in_provenance(tmp_path: Path) -> None:
    """Default mode keeps the spec and records the failure, never silently."""
    path = _gnn_file(tmp_path, MALFORMED_A)
    spec = extract_pomdp_from_file(path)  # on_error defaults to "lenient"
    assert spec is not None
    assert not isinstance(spec, tuple)
    entry = _prov_entry(spec, "A")
    assert entry.get("source") == "parse_error"
    assert entry.get("code"), f"parse_error provenance missing code: {entry}"
    assert entry.get("message"), f"parse_error provenance missing message: {entry}"
    assert spec.adapter_notes is not None
    assert any("A" in str(note) for note in spec.adapter_notes), (
        f"adapter_notes must mention the failed parameter A: {spec.adapter_notes}"
    )


def test_raise_mode_raises_gnn_extraction_error(tmp_path: Path) -> None:
    path = _gnn_file(tmp_path, MALFORMED_A)
    with pytest.raises(GNNExtractionError) as excinfo:
        extract_pomdp_from_file(path, on_error="raise")
    err = excinfo.value
    assert isinstance(err.code, str) and err.code, "error code required"
    assert isinstance(err.message, str) and "A" in err.message
    assert isinstance(err.line, int) and err.line >= 1
    assert isinstance(err.section, str) and err.section


def test_collect_mode_returns_spec_and_structured_errors(tmp_path: Path) -> None:
    path = _gnn_file(tmp_path, MALFORMED_A)
    result: Any = extract_pomdp_from_file(path, on_error="collect")
    assert isinstance(result, tuple) and len(result) == 2
    spec, errors = result
    assert isinstance(errors, list) and len(errors) >= 1
    first = errors[0]
    assert isinstance(first, GNNExtractionError)
    assert isinstance(first.code, str) and first.code
    assert isinstance(first.line, int) and first.line >= 1
    assert isinstance(first.section, str) and first.section
    # spec may be present alongside collected errors (contract: spec_or_None)
    assert spec is None or hasattr(spec, "matrix_provenance")


def test_invalid_on_error_value_raises_value_error(tmp_path: Path) -> None:
    path = _gnn_file(tmp_path, VALID_A)
    with pytest.raises(ValueError):
        extract_pomdp_from_file(path, on_error=cast(Any, "bogus"))


def test_valid_file_has_no_parse_error_entries(tmp_path: Path) -> None:
    """Control: a fully valid file must not gain parse_error provenance."""
    path = _gnn_file(tmp_path, VALID_A)
    spec = extract_pomdp_from_file(path)
    assert spec is not None
    assert not isinstance(spec, tuple)
    for matrix, entry in (spec.matrix_provenance or {}).items():
        if isinstance(entry, dict):
            assert entry.get("source") != "parse_error", (
                f"unexpected parse_error provenance for {matrix}: {entry}"
            )


def test_malformed_fixture_differs_from_valid_by_a_block_only(tmp_path: Path) -> None:
    """Fixture integrity: A is the only perturbation between the two files."""
    valid = _gnn_file(tmp_path, VALID_A, name="valid.md")
    broken = _gnn_file(tmp_path, MALFORMED_A, name="broken.md")
    valid_lines = valid.read_text().splitlines()
    broken_lines = broken.read_text().splitlines()
    assert len(valid_lines) == len(broken_lines)
    differing = [i for i, (a, b) in enumerate(zip(valid_lines, broken_lines)) if a != b]
    assert differing, "fixtures must differ"
    for i in differing:
        assert "oops" in broken_lines[i] or "0.9" in valid_lines[i]
