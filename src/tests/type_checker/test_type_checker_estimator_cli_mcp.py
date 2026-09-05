"""Pin behavior of the resource estimator, CLI, and MCP integration.

Covers: the ``## Time`` section classification (replacing the old
``"Dynamic" if 't' in content`` bug), registered-extension discovery in
the estimator, the CLI end-to-end path that previously crashed with
``KeyError: 'is_valid'``, and MCP strict-mode passthrough.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from type_checker.estimation.estimator import GNNResourceEstimator
from type_checker.mcp import validate_single_gnn_file_mcp

_STATIC = """## GNNSection
ActInfPOMDP

## ModelName
TimeProbe

## StateSpaceBlock
s[2,1,type=float]

## Time
Static
"""

_DYNAMIC = """## GNNSection
ActInfPOMDP

## ModelName
TimeProbe

## StateSpaceBlock
s[2,1,type=float]

## Time
Dynamic
"""

_HIER = """## GNNSection
ActInfPOMDP

## ModelName
TimeProbe

## StateSpaceBlock
s[2,1,type=float]

## Time
Hierarchical
"""

# A spec whose StateSpaceBlock declaration comment claims canonical B
# axis order while the InitialParameterization comment claims the old
# layout — a [GNN-E002] contradiction.
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


# --- estimator time-spec classification ------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [(_STATIC, "Static"), (_DYNAMIC, "Dynamic"), (_HIER, "Hierarchical")],
)
def test_estimator_classifies_time_from_section(
    tmp_path: Path, content: str, expected: str
) -> None:
    path = tmp_path / "probe.gnn"
    path.write_text(content, encoding="utf-8")
    result = GNNResourceEstimator().estimate_from_file(str(path))
    assert result["model_info"]["time_spec"] == expected


def test_estimator_time_spec_ignores_stray_t_in_prose(tmp_path: Path) -> None:
    # The old ``"Dynamic" if 't' in content`` test was true for any spec
    # containing the letter "t"; the section-scoped classifier must not
    # regress to that.
    content = "## StateSpaceBlock\ns[2,1,type=float]\n## Time\nStatic\n## Notes\ntalk\n"
    path = tmp_path / "probe.gnn"
    path.write_text(content, encoding="utf-8")
    result = GNNResourceEstimator().estimate_from_file(str(path))
    assert result["model_info"]["time_spec"] == "Static"


# --- estimator registered-extension discovery ------------------------------


def test_estimator_directory_estimates_gnn_extension(tmp_path: Path) -> None:
    (tmp_path / "probe.gnn").write_text(_STATIC, encoding="utf-8")
    results = GNNResourceEstimator().estimate_from_directory(str(tmp_path))
    assert len(results) == 1
    only = next(iter(results))
    assert only.endswith("probe.gnn")
    assert "error" not in results[only]


def test_estimator_directory_ignores_pickle_specs(tmp_path: Path) -> None:
    (tmp_path / "probe.gnn").write_text(_STATIC, encoding="utf-8")
    (tmp_path / "data.pickle").write_text("not a spec", encoding="utf-8")
    results = GNNResourceEstimator().estimate_from_directory(str(tmp_path))
    assert all(not k.endswith(".pickle") for k in results)
    assert len(results) == 1


# --- CLI end-to-end (previously crashed with KeyError: 'is_valid') ---------


def test_cli_valid_file_writes_reports_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from type_checker.cli import main

    spec = tmp_path / "probe.gnn"
    spec.write_text(_STATIC, encoding="utf-8")
    out = tmp_path / "out" / "type_check"

    monkeypatch.setenv("GNN_TYPE_CHECKER_ALLOW_ANY_OUTPUT_DIR", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")

    exit_code = main([str(spec), "--output-dir", str(out)])
    assert exit_code == 0

    # Per-file markdown report renders without KeyError and shows VALID.
    md = (out / "reports" / "probe_type_check.md").read_text(encoding="utf-8")
    assert "Type Check Report: probe" in md
    assert "VALID" in md

    # CSV artifacts carry structured data (variables_table, section matrix).
    variables_csv = (out / "artifacts" / "variables_table.csv").read_text(
        encoding="utf-8"
    )
    assert "probe.gnn" in variables_csv
    assert "s" in variables_csv

    presence = (out / "artifacts" / "section_presence_matrix.csv").read_text(
        encoding="utf-8"
    )
    assert "StateSpaceBlock" in presence
    assert "probe.gnn" in presence


def test_cli_rejects_forbidden_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from type_checker.cli import main

    spec = tmp_path / "probe.gnn"
    spec.write_text(_STATIC, encoding="utf-8")
    # ``output`` is a forbidden base dir; the CLI must refuse with exit 1.
    monkeypatch.delenv("GNN_TYPE_CHECKER_ALLOW_ANY_OUTPUT_DIR", raising=False)
    exit_code = main([str(spec), "--output-dir", "output"])
    assert exit_code == 1


# --- MCP strict-mode passthrough -------------------------------------------


def test_mcp_single_file_strict_passthrough(tmp_path: Path) -> None:
    path = tmp_path / "strict.gnn"
    path.write_text(_B_CONTRADICTION, encoding="utf-8")

    loose = validate_single_gnn_file_mcp(str(path), strict=False)
    strict = validate_single_gnn_file_mcp(str(path), strict=True)

    assert loose["success"] is True
    assert loose["validation_result"]["valid"] is True
    assert strict["success"] is False
    assert strict["validation_result"]["valid"] is False
    assert any("[GNN-E002]" in e for e in strict["validation_result"]["errors"])


def test_mcp_single_file_valid_spec(tmp_path: Path) -> None:
    path = tmp_path / "ok.gnn"
    path.write_text(_STATIC, encoding="utf-8")
    result = validate_single_gnn_file_mcp(str(path))
    assert result["success"] is True
    assert result["validation_result"]["valid"] is True
