#!/usr/bin/env python3
"""Phase 4.2 regression tests for intelligent_analysis (Step 24, hard import).

Zero-mock per CLAUDE.md — uses realistic pipeline-summary shapes as fixtures
rather than MagicMock.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# --- validate_pipeline_summary ------------------------------------------

def test_validate_pipeline_summary_accepts_minimal_valid_shape():
    from intelligent_analysis import validate_pipeline_summary
    summary = {"steps": [{"name": "3_gnn", "status": "success"}]}
    # Either True or False — but must not raise.
    result = validate_pipeline_summary(summary)
    assert isinstance(result, bool)


def test_validate_pipeline_summary_rejects_non_dict():
    from intelligent_analysis import validate_pipeline_summary
    result = validate_pipeline_summary("not-a-dict")  # type: ignore[arg-type]
    assert result is False


# --- health scoring ------------------------------------------------------

def test_calculate_pipeline_health_score_perfect_run_is_high():
    from intelligent_analysis.analyzer import calculate_pipeline_health_score
    summary = {
        "steps": [
            {"name": f"step_{i}", "status": "success", "duration_s": 1.0}
            for i in range(25)
        ],
        "total_steps": 25,
        "successful_steps": 25,
        "failed_steps": 0,
    }
    score = calculate_pipeline_health_score(summary)
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0
    # A perfect run should score at least 80/100 by any reasonable heuristic.
    assert score >= 80.0, f"Perfect pipeline scored low: {score}"


def test_calculate_pipeline_health_score_handles_empty_summary():
    """Regression: empty summary must not raise — it gets a baseline score."""
    from intelligent_analysis.analyzer import calculate_pipeline_health_score
    score = calculate_pipeline_health_score({})
    assert isinstance(score, float)


# --- classify_failure_severity ------------------------------------------

def test_classify_failure_severity_returns_recognized_level():
    from intelligent_analysis.analyzer import classify_failure_severity
    severity = classify_failure_severity({
        "name": "3_gnn",
        "status": "failed",
        "error": "Failed to parse input file",
    })
    assert isinstance(severity, str)
    # Accept the full taxonomy used by intelligent_analysis (including "minor"
    # and "major", which the module uses in its triage rubric).
    allowed = {
        "critical", "high", "medium", "low",
        "info", "warning", "error",
        "minor", "major",
    }
    assert severity.lower() in allowed, (
        f"Unknown severity {severity!r} — if this is a new level, add it to the allowed set."
    )


# --- detect_performance_patterns + generate_optimization_suggestions ---

def test_detect_performance_patterns_returns_list():
    from intelligent_analysis.analyzer import detect_performance_patterns
    summary = {
        "steps": [
            {"name": "11_render", "duration_s": 30.0, "status": "success"},
            {"name": "12_execute", "duration_s": 120.0, "status": "success"},
        ]
    }
    patterns = detect_performance_patterns(summary)
    assert isinstance(patterns, list)


def test_generate_optimization_suggestions_returns_list():
    from intelligent_analysis.analyzer import generate_optimization_suggestions
    result = generate_optimization_suggestions({"steps": []})
    assert isinstance(result, list)


# --- remediation --------------------------------------------------------

def test_suggest_fix_accepts_violation_object():
    """suggest_fix expects a violation object exposing .framework and .field
    (a simple namespace-style object is enough for testing — no mock needed)."""
    from types import SimpleNamespace
    from intelligent_analysis.remediation import suggest_fix
    violation = SimpleNamespace(framework="pymdp", field="matrix_dims", message="mismatch")
    result = suggest_fix(violation)
    # Either a RemediationPlan (when the lookup finds one) or None.
    assert result is None or result.__class__.__name__ == "RemediationPlan"


def test_suggest_fix_raises_on_dict_violation():
    """suggest_fix has a documented contract: the argument must be a
    violation object, not a dict. Passing a dict should raise AttributeError
    — we test this so that future API changes toward dict acceptance are
    captured as an intentional change, not a silent regression."""
    import pytest as _pytest
    from intelligent_analysis.remediation import suggest_fix
    with _pytest.raises(AttributeError):
        suggest_fix({"type": "unknown", "message": "x"})  # type: ignore[arg-type]
