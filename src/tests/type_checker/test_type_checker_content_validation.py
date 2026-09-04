"""Pin behavior of the additive type-checker surface.

Covers: section-scoped extraction helpers (``checking.sections``), the typed
validation summary (``checking.summary``), the pure ``validate_content``
entry point, ``strict_mode`` constructor plumbing for B-orientation
contradictions, the real-newline fix in the Markdown summary, and the
``estimate_resources`` flag that writes resource-estimate reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from type_checker import GNNTypeChecker, summarize_type_check_results
from type_checker.checking.sections import (
    CANONICAL_GNN_SECTIONS,
    connection_group,
    detect_time_dynamics,
    extract_markdown_section,
    parse_resource_connections,
    section_presence,
)

# A spec whose StateSpaceBlock declaration comment claims the canonical
# B axis order (next_state, prev_state, action) while the
# InitialParameterization comment claims the old (rows=previous,
# columns=next) layout. This is a comment-vs-comment contradiction that
# ``_validate_b_orientation`` flags as [GNN-E002].
_B_CONTRADICTION = """## GNNSection
ActInfPOMDP

## ModelName
BOrientation

## StateSpaceBlock
B[2,2,2]  # B[next_state, prev_state, action]
s[2,1,type=float]

## InitialParameterization
# rows are previous states, columns are next states
B={
  ( (0.95, 0.05), (0.05, 0.95) ),
  ( (0.05, 0.95), (0.95, 0.05) )
}

## Connections
s-s

## Footer
BOrientation
"""

_VALID_MINIMAL = """## GNNSection
ActInfPOMDP

## ModelName
Minimal

## StateSpaceBlock
s[2,1,type=float]
o[2,1,type=int]

## Connections
s-o

## Time
Static

## Footer
Minimal
"""


# --- sections.extract_markdown_section ---------------------------------------


def test_extract_markdown_section_ignores_prose_outside_block() -> None:
    content = "preamble s>s noise\n## Connections\na>b\n## Footer\na-b in prose"
    assert extract_markdown_section(content, "Connections") == "a>b"


def test_extract_markdown_section_absent_returns_empty() -> None:
    assert extract_markdown_section("## Other\nx", "Connections") == ""


# --- sections.connection_group ---------------------------------------------


def test_connection_group_unwraps_parens_and_normalizes_pi() -> None:
    assert connection_group("(a, b, pi)") == ["a", "b", "π"]
    assert connection_group("s") == ["s"]


# --- sections.parse_resource_connections ------------------------------------


def test_parse_resource_connections_is_section_scoped() -> None:
    content = "## Connections\na>b\nc-d\n## Notes\nx>y should be ignored"
    edges, diagnostics = parse_resource_connections(content, {"a", "b", "c", "d"})
    assert {"source": "a", "target": "b", "type": "directed"} in edges
    assert {"source": "c", "target": "d", "type": "undirected"} in edges
    assert not any("x" in d for d in diagnostics)
    assert all("x>y" not in d for d in diagnostics)


def test_parse_resource_connections_flags_undeclared_variables() -> None:
    edges, diagnostics = parse_resource_connections("## Connections\na>z", {"a"})
    assert {"source": "a", "target": "z", "type": "directed"} in edges
    assert any("'z'" in d for d in diagnostics)


# --- sections.section_presence ---------------------------------------------


def test_section_presence_maps_canonical_sections() -> None:
    present = section_presence(_VALID_MINIMAL)
    assert set(present) == set(CANONICAL_GNN_SECTIONS)
    assert present["StateSpaceBlock"] is True
    assert present["GNNSection"] is True
    assert present["InitialParameterization"] is False


# --- sections.detect_time_dynamics ------------------------------------------


def test_detect_time_dynamics_reads_only_time_section() -> None:
    assert detect_time_dynamics("## Time\nDynamic\n") is True
    assert detect_time_dynamics("## Time\nStatic\n") is False
    # A stray "t"/"dynamic" in prose must not flip a static model.
    assert detect_time_dynamics("## Time\nStatic\n## Notes\ndynamic talk\n") is False


# --- summary.summarize_type_check_results -----------------------------------


def _results_envelope(files: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "processed_files": len(files),
        "success": all(f.get("valid", False) for f in files),
        "validation_results": files,
        "errors": [],
        "type_analysis": [],
    }


def test_summarize_counts_valid_invalid_and_warning_files() -> None:
    files = [
        {
            "valid": True,
            "errors": [],
            "warnings": [],
            "resource_estimation": {
                "complexity_tier": "small",
                "total_parameters": 4,
                "estimated_memory_bytes": 16,
            },
        },
        {
            "valid": True,
            "errors": [],
            "warnings": ["x"],
            "resource_estimation": {
                "complexity_tier": "medium",
                "total_parameters": 9,
                "estimated_memory_bytes": 36,
            },
        },
        {
            "valid": False,
            "errors": ["bad"],
            "warnings": [],
            "resource_estimation": {
                "complexity_tier": "large",
                "total_parameters": 16,
                "estimated_memory_bytes": 64,
            },
        },
    ]
    summary = summarize_type_check_results(_results_envelope(files))
    assert summary["files_processed"] == 3
    assert summary["valid_files"] == 2
    assert summary["invalid_files"] == 1
    assert summary["warning_files"] == 1
    assert summary["total_errors"] == 1
    assert summary["total_warnings"] == 1
    assert summary["complexity_tiers"] == {"small": 1, "medium": 1, "large": 1}
    assert summary["total_parameters"] == 29
    assert summary["total_estimated_memory_bytes"] == 116
    assert summary["success"] is False


def test_summarize_accepts_flat_file_map_shape() -> None:
    flat = {
        "a.gnn": {"valid": True, "errors": [], "warnings": []},
        "b.gnn": {"valid": False, "errors": ["e"], "warnings": []},
    }
    summary = summarize_type_check_results(flat)
    assert summary["files_processed"] == 2
    assert summary["valid_files"] == 1
    assert summary["invalid_files"] == 1


# --- GNNTypeChecker.validate_content ---------------------------------------


def test_validate_content_valid_minimal_spec() -> None:
    result = GNNTypeChecker().validate_content(
        _VALID_MINIMAL, source_name="minimal.gnn"
    )
    assert result["valid"] is True
    assert result["errors"] == []
    assert result["variable_count"] == 2
    assert result["connection_count"] == 1
    assert {v["name"] for v in result["variables"]} == {"s", "o"}
    assert result["connections"][0]["is_temporal"] is False
    assert result["sections"]["StateSpaceBlock"] is True
    assert result["time_dynamics"]["is_dynamic"] is False


def test_validate_content_unknown_type_is_error() -> None:
    bad = "## StateSpaceBlock\ns[2,1,type=notatype]\n"
    result = GNNTypeChecker().validate_content(bad, source_name="bad.gnn")
    assert result["valid"] is False
    assert any("Unknown type" in e for e in result["errors"])


def test_validate_content_duplicate_variable_is_error() -> None:
    dup = "## StateSpaceBlock\ns[2,1,type=float]\ns[3,1,type=float]\n"
    result = GNNTypeChecker().validate_content(dup, source_name="dup.gnn")
    assert result["valid"] is False
    assert any("Duplicate" in e for e in result["errors"])


# --- strict_mode plumbing ---------------------------------------------------


def test_strict_mode_constructor_promotes_b_contradiction_to_error() -> None:
    loose = GNNTypeChecker(strict_mode=False).validate_content(
        _B_CONTRADICTION, source_name="b.gnn"
    )
    strict = GNNTypeChecker(strict_mode=True).validate_content(
        _B_CONTRADICTION, source_name="b.gnn"
    )
    assert loose["valid"] is True
    assert any("[GNN-E002]" in w for w in loose["warnings"])
    assert strict["valid"] is False
    assert any("[GNN-E002]" in e for e in strict["errors"])


def test_validate_content_strict_override_beats_instance_default() -> None:
    # Instance default is loose, but an explicit strict=True wins.
    loose_instance = GNNTypeChecker(strict_mode=False)
    strict_result = loose_instance.validate_content(
        _B_CONTRADICTION, source_name="b.gnn", strict=True
    )
    assert strict_result["valid"] is False
    assert any("[GNN-E002]" in e for e in strict_result["errors"])


# --- summary markdown uses real newlines (regression) -----------------------


def test_generate_type_check_summary_uses_real_newlines() -> None:
    results = {
        "processed_files": 1,
        "success": True,
        "errors": [],
        "validation_results": [{"valid": True, "errors": [], "warnings": []}],
        "type_analysis": [{"total_variables": 2}],
        "visual_embeddings": ["![Mosaic](visualizations/mosaic.png)"],
    }
    summary = GNNTypeChecker()._generate_type_check_summary(results)
    # The embedding must sit on its own line — no literal backslash-n text.
    assert "![Mosaic](visualizations/mosaic.png)" in summary
    assert "\\n" not in summary
    assert "\n![Mosaic](visualizations/mosaic.png)\n" in summary


def test_generate_type_check_summary_no_visuals_message_real_newlines() -> None:
    results = {
        "processed_files": 0,
        "success": False,
        "errors": ["No GNN files found"],
        "validation_results": [],
        "type_analysis": [],
    }
    summary = GNNTypeChecker()._generate_type_check_summary(results)
    assert "*No visual summaries could be generated.*" in summary
    assert "\\n" not in summary
    assert "- No errors encountered" not in summary  # errors list is non-empty
    assert "- No GNN files found" in summary  # the actual error is rendered


def test_validate_gnn_files_estimate_resources_writes_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "probe.gnn").write_text(_VALID_MINIMAL, encoding="utf-8")
    out = tmp_path / "out"

    # Headless backend keeps the resource-estimation HTML render deterministic.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    # Keep the run deterministic and graphics-free for this contract test.
    monkeypatch.setattr(
        "type_checker.visualizer.generate_all_visualizations",
        lambda *_a, **_kw: [],
    )

    success = GNNTypeChecker().validate_gnn_files(
        tmp_path, out, estimate_resources=True
    )
    assert success is True
    assert (out / "type_check_results.json").exists()
    assert (out / "type_check_summary.json").exists()
    assert (out / "type_check_summary.md").exists()
    assert (out / "resource_estimates" / "resource_data.json").exists()
    assert (out / "resource_estimates" / "resource_report.md").exists()


def test_validate_gnn_files_writes_summary_json_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "probe.gnn").write_text(_VALID_MINIMAL, encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(
        "type_checker.visualizer.generate_all_visualizations",
        lambda *_a, **_kw: [],
    )
    GNNTypeChecker().validate_gnn_files(tmp_path, out)
    envelope = json.loads((out / "type_check_summary.json").read_text())
    assert envelope["files_processed"] == 1
    assert envelope["valid_files"] == 1
    assert envelope["invalid_files"] == 0


def test_validate_single_gnn_file_never_raises_on_content_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spec that trips validate_content must surface as an invalid dict,
    never propagate — the directory run treats it as recoverable exit-2."""
    spec = tmp_path / "probe.gnn"
    spec.write_text(_VALID_MINIMAL, encoding="utf-8")

    def _boom(content: str) -> Any:
        raise RuntimeError("synthetic parser failure")

    monkeypatch.setattr("type_checker.checking.core.estimate_file_resources", _boom)
    result = GNNTypeChecker().validate_single_gnn_file(spec)
    assert result["valid"] is False
    assert any("synthetic parser failure" in e for e in result["errors"])

    # The directory run treats the invalid-but-processable file as
    # recoverable (exit 2), not hard failure (exit 1).
    monkeypatch.setattr(
        "type_checker.visualizer.generate_all_visualizations",
        lambda *_a, **_kw: [],
    )
    out = tmp_path / "out"
    success = GNNTypeChecker().validate_gnn_files(tmp_path, out)
    assert success == 2


def test_validate_gnn_files_no_files_is_warning_exit_2(tmp_path: Path) -> None:
    """Phase 1.1 widened contract: "no GNN files found" is a warning
    (exit 2), not a hard error (exit 1) — matching Steps 12/16 and the
    render step. See doc/gnn/testing/SPEC.md."""
    out = tmp_path / "out"
    success = GNNTypeChecker().validate_gnn_files(tmp_path, out)
    assert success == 2
    # Artifacts still land so downstream consumers find a summary.
    receipt = json.loads((out / "type_check_results.json").read_text())
    assert receipt["processed_files"] == 0
    assert "No GNN files found" in receipt["errors"]
    envelope = json.loads((out / "type_check_summary.json").read_text())
    assert envelope["files_processed"] == 0


def test_validate_content_reports_model_type_and_granular_complexity() -> None:
    """validate_content exposes model_type (Time-section classification) and
    the granular complexity metrics the per-file report renders."""
    result = GNNTypeChecker().validate_content(_VALID_MINIMAL)
    assert result["model_type"] == "Static"
    complexity = result["model_complexity"]
    for key in (
        "state_space_complexity",
        "graph_density",
        "cyclic_complexity",
        "temporal_complexity",
        "equation_complexity",
        "overall_complexity",
    ):
        assert key in complexity


def test_per_file_markdown_report_renders_model_type_and_complexity() -> None:
    """The CLI per-file report renders the enriched metadata instead of
    placeholders ('Unknown' model type, zero-only complexity)."""
    from type_checker.output_utils import per_file_markdown_report

    result = GNNTypeChecker().validate_content(_VALID_MINIMAL)
    report = per_file_markdown_report("minimal.gnn", {"is_valid": True, **result})
    assert "Model Type**: Static" in report
    assert "Graph Density**:" in report
    assert "Equation Complexity**:" in report


def test_classify_time_spec_agrees_with_detect_time_dynamics() -> None:
    """classify_time_spec and detect_time_dynamics share one marker set, so
    model_type and time_dynamics.is_dynamic can never contradict each other
    (a "continuous-time" spec must classify Dynamic, not Static)."""
    from type_checker.checking.sections import (
        classify_time_spec,
        detect_time_dynamics,
    )

    assert classify_time_spec("## Time\ncontinuous-time\n") == "Dynamic"
    assert classify_time_spec("## Time\nHierarchical\n") == "Hierarchical"
    assert classify_time_spec("## Time\nStatic\n") == "Static"
    assert classify_time_spec("no time section") == "Static"

    for content in (
        "## Time\nDynamic\n",
        "## Time\ncontinuous-time\n",
        "## Time\ntime-varying\n",
        "## Time\nStatic\n",
        "## ModelName\nX\n",
    ):
        classified = classify_time_spec(content)
        assert (classified == "Dynamic") == detect_time_dynamics(content), content
        assert (classified != "Static") >= detect_time_dynamics(content), content
