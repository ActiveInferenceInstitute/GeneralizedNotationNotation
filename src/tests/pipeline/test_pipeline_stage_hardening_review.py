"""Coverage tests for the maintained pipeline hardening review artifact."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REVIEW_PATH = REPO_ROOT / "doc" / "pipeline" / "pipeline_stage_hardening_review.md"


def test_pipeline_stage_hardening_review_covers_all_steps() -> None:
    text = REVIEW_PATH.read_text(encoding="utf-8")

    assert "Pipeline Stage Hardening Review" in text
    assert "Last reviewed" in text

    documented_steps = {
        int(match.group(1))
        for match in re.finditer(r"^\|\s*(\d+)\s*\|", text, flags=re.MULTILINE)
    }
    assert documented_steps == set(range(25))


def test_pipeline_stage_hardening_review_locks_gridworld_contract() -> None:
    text = REVIEW_PATH.read_text(encoding="utf-8")

    assert (
        'uv run python src/main.py --only-steps "3,5,8,11,12,16" '
        "--target-dir input/gnn_files/pomdp_gridworld "
        '--frameworks "pymdp,rxinfer,activeinference_jl" --verbose'
    ) in text
    for schema in (
        "pymdp_simulation_v1",
        "rxinfer_simulation_v1",
        "activeinference_jl_simulation_v1",
        "gridworld_analysis_manifest_v1",
    ):
        assert schema in text

    for artifact in (
        "simulation_results.json",
        "gridworld_analysis_manifest.json",
        "PNG",
        "GIF",
    ):
        assert artifact in text


def test_pipeline_stage_hardening_review_lists_required_gates() -> None:
    text = REVIEW_PATH.read_text(encoding="utf-8")

    required_commands = (
        "uv run --extra dev ruff format --check src scripts",
        "uv run --extra dev ruff check src scripts",
        "uv run --extra dev mypy src --show-error-codes",
        "uv run --extra dev bandit -r src -c pyproject.toml -q",
        "uv run --extra dev python scripts/check_repo_terminology.py --strict",
        "uv run --extra dev python scripts/check_maintained_doc_terms.py --strict",
        "uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write",
        "uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict",
        "uv run --extra dev python -m pytest src/tests/pipeline/test_pomdp_gridworld_cross_framework.py -q --tb=short",
        "uv run --extra dev python -m pytest --collect-only src/tests/ -q --tb=no",
        "uv run --extra dev python -m pytest src/tests/ -q --tb=no",
    )
    for command in required_commands:
        assert command in text
