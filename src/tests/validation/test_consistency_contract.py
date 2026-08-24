"""Regression tests for canonical consistency and best-effort receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import pytest

import validation
from validation.consistency_checker import ConsistencyChecker, check_consistency

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_real_markdown_gnn_runs_reference_checks() -> None:
    model_path = (
        REPOSITORY_ROOT / "input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md"
    )

    result = check_consistency(model_path)
    references = result["checks"]["reference_consistency"]

    assert result["recovery"] is False
    assert references["invalid_references"] == []
    assert references["isolated_blocks"] == ["t"]
    assert references["circular_references"] == []


def test_structured_model_reports_unknown_connection_reference() -> None:
    result = check_consistency(
        {
            "variables": [
                {"name": "state", "dimensions": [2]},
                {"name": "observation", "dimensions": [2]},
            ],
            "connections": [
                {
                    "source_variables": ["missing"],
                    "target_variables": ["observation"],
                }
            ],
        }
    )
    references = result["checks"]["reference_consistency"]

    assert result["recovery"] is False
    assert result["consistent"] is False
    assert references["invalid_references"] == [(0, "From", "missing")]
    assert references["isolated_blocks"] == ["state"]
    assert result["consistency_score"] < 1.0


def test_raw_sections_dictionary_uses_canonical_gnn_syntax() -> None:
    result = check_consistency(
        {
            "raw_sections": {
                "StateSpaceBlock": "state[2,type=float]\nobs[2,type=int]",
                "Connections": "state>obs",
            }
        }
    )

    assert result["consistent"] is True
    assert result["warnings"] == []
    assert result["checks"]["reference_consistency"]["invalid_references"] == []


def test_markdown_groups_unicode_and_temporal_identifiers_are_cross_referenced() -> (
    None
):
    content = """## StateSpaceBlock
s_t[3,type=float]
u_t[2,type=int]
B_t[3,3,2,type=float]
s_t+1[3,type=float]
γ[1,type=float]

## Connections
(s_t, u_t)>B_t
B_t>s_t+1
γ>B_t
"""

    result = ConsistencyChecker().check(content)
    references = result["checks"]["reference_consistency"]

    assert result["is_consistent"] is True
    assert references["invalid_references"] == []
    assert references["isolated_blocks"] == []


def test_malformed_structured_model_returns_all_diagnostics() -> None:
    result = check_consistency({"variables": ["not-a-mapping"], "connections": [42]})

    assert result["recovery"] is False
    assert result["consistent"] is False
    assert result["checks"]["structural_integrity"]["warnings"] == [
        "Variable 0 must be a mapping",
        "Connection 0 must be a mapping",
        "No valid StateSpaceBlock declarations found",
    ]


def test_cycle_members_exclude_nodes_that_only_lead_into_cycle() -> None:
    result = check_consistency(
        {
            "variables": [{"name": name} for name in ("alpha", "beta", "tail")],
            "connections": [
                {"source": "alpha", "target": "beta"},
                {"source": "beta", "target": "alpha"},
                {"source": "tail", "target": "alpha"},
            ],
        }
    )

    assert result["checks"]["reference_consistency"]["circular_references"] == [
        "alpha",
        "beta",
    ]


def test_direct_checker_rejects_non_string_content() -> None:
    with pytest.raises(TypeError, match="content must be a string"):
        ConsistencyChecker().check(None)  # type: ignore[arg-type]


def test_process_validation_persists_stage_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_output = tmp_path / "run"
    gnn_output = base_output / "3_gnn_output"
    validation_output = base_output / "6_validation_output"
    gnn_output.mkdir(parents=True)
    parsed_model = tmp_path / "parsed.json"
    parsed_model.write_text("{}", encoding="utf-8")
    (gnn_output / "gnn_processing_results.json").write_text(
        json.dumps(
            {
                "processed_files": [
                    {
                        "file_name": "model.gnn",
                        "file_path": "model.gnn",
                        "parse_success": True,
                        "parsed_model_file": str(parsed_model),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fail_semantic_validation(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("semantic probe failed")

    monkeypatch.setattr(
        validation, "process_semantic_validation", fail_semantic_validation
    )
    monkeypatch.setattr(
        validation,
        "profile_performance",
        lambda *_args, **_kwargs: {"performance_score": 1.0, "recovery": False},
    )
    monkeypatch.setattr(
        validation,
        "check_consistency",
        lambda *_args, **_kwargs: {"consistency_score": 1.0, "recovery": False},
    )

    success = validation.process_validation(tmp_path / "models", validation_output)
    receipt = json.loads(
        (validation_output / "validation_results.json").read_text(encoding="utf-8")
    )
    file_result = receipt["files_validated"][0]

    assert success is False
    assert file_result["success"] is False
    assert file_result["errors"] == ["semantic probe failed"]
    assert file_result["validations"]["semantic"] == {
        "status": "error",
        "error": "semantic probe failed",
        "recovery": True,
    }
    assert receipt["summary"]["failed_validations"] == 1
