"""Regression coverage for real and degenerate type-checker inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from type_checker.checking.core import GNNTypeChecker, estimate_file_resources

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_real_gnn_resource_estimate_is_section_scoped() -> None:
    model_path = REPOSITORY_ROOT / "input/gnn_files/basics/static_perception.md"
    result = estimate_file_resources(model_path.read_text(encoding="utf-8"))

    assert result["variables"] == 7
    assert result["connections"] == 6
    assert result["total_parameters"] == 21
    assert result["estimated_memory_bytes"] == 84
    assert result["diagnostics"] == []


def test_real_gnn_type_check_ignores_prose_and_accepts_unicode_name() -> None:
    model_path = REPOSITORY_ROOT / "input/gnn_files/discrete/simple_mdp.md"

    result = GNNTypeChecker().validate_single_gnn_file(model_path)

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["resource_estimation"]["variables"] == 11
    assert result["resource_estimation"]["connections"] == 10


@pytest.mark.parametrize(
    ("relative_path", "expected_variables", "expected_connections"),
    [
        ("discrete/time_varying_dynamics.md", 8, 7),
        ("precision/precision_weighted.md", 15, 16),
        ("structured/factorized_posterior.md", 13, 14),
    ],
)
def test_real_gnn_extended_identifiers_and_connection_groups(
    relative_path: str,
    expected_variables: int,
    expected_connections: int,
) -> None:
    model_path = REPOSITORY_ROOT / "input/gnn_files" / relative_path

    result = GNNTypeChecker().validate_single_gnn_file(model_path)
    estimate = result["resource_estimation"]

    assert result["valid"] is True
    assert estimate["variables"] == expected_variables
    assert estimate["connections"] == expected_connections
    assert estimate["diagnostics"] == []


def test_empty_spec_is_total_but_not_type_valid(tmp_path: Path) -> None:
    model_path = tmp_path / "empty.gnn"
    model_path.write_text("", encoding="utf-8")

    estimate = estimate_file_resources("")
    validation = GNNTypeChecker().validate_single_gnn_file(model_path)

    assert estimate["estimated_memory_bytes"] == 0
    assert estimate["total_parameters"] == 0
    assert estimate["variables"] == 0
    assert estimate["connections"] == 0
    assert estimate["diagnostics"] == ["Missing StateSpaceBlock section"]
    assert validation["valid"] is False
    assert "Missing StateSpaceBlock section" in validation["errors"]


def test_malformed_dimensions_use_safe_estimates_and_fail_type_check(
    tmp_path: Path,
) -> None:
    content = """## StateSpaceBlock
A[-4,unknown,type=float]
s[2,1,type=float]

## Connections
A>s
"""
    model_path = tmp_path / "malformed.gnn"
    model_path.write_text(content, encoding="utf-8")

    estimate = estimate_file_resources(content)
    validation = GNNTypeChecker().validate_single_gnn_file(model_path)

    assert estimate["estimated_memory_bytes"] >= 0
    assert estimate["total_parameters"] >= 0
    assert any("non-positive dimension -4" in item for item in estimate["diagnostics"])
    assert any(
        "unresolved dimension 'unknown'" in item for item in estimate["diagnostics"]
    )
    assert validation["valid"] is False
    assert any("non-positive dimension -4" in item for item in validation["errors"])
    assert any(
        "unresolved dimension 'unknown'" in item for item in validation["errors"]
    )


def test_directory_result_fails_when_a_discovered_model_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Avoid exercising graphical output in this contract-focused regression.
    monkeypatch.setattr(
        "type_checker.visualizer.generate_all_visualizations",
        lambda *_args, **_kwargs: [],
    )
    (tmp_path / "invalid.gnn").write_text("", encoding="utf-8")
    output_dir = tmp_path / "output"

    success = GNNTypeChecker().validate_gnn_files(tmp_path, output_dir)
    receipt = json.loads(
        (output_dir / "type_check_results.json").read_text(encoding="utf-8")
    )

    # An invalid-but-processable file is a recoverable outcome: the run
    # completes and reports the invalid file, signalling warning-continuation
    # (exit 2) rather than a hard directory failure, so the pipeline can
    # recover and continue (see test_error_recovery_and_continuation).
    assert success == 2
    assert receipt["validation_results"][0]["valid"] is False
