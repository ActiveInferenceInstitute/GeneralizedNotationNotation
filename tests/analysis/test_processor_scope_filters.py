"""Pins for the Step-16 scope-filter helpers (``analysis.processor``)."""

from __future__ import annotations

import logging

import pytest

from gnn.analysis.processor import (
    _filter_execution_summary,
    _normalize_generate_animations,
    _scope_from_execution_summary,
)


def _logger() -> logging.Logger:
    return logging.getLogger("w2probe-processor")


def _summary() -> dict:
    return {
        "requested_frameworks": ["PyMDP", "JAX"],
        "execution_details": [
            {
                "framework": "pymdp",
                "success": True,
                "model_name": "Simple MDP Agent",
                "structured_result_file": "out/pymdp/results.json",
            },
            {
                "framework": "discopy",
                "success": False,
                "skipped": True,
                "model_name": "Circuit Model",
            },
        ],
        "framework_status": {"pymdp": "ok", "discopy": "skipped"},
    }


# --- _normalize_generate_animations ------------------------------------------


def test_canonical_true_and_false_are_returned() -> None:
    assert _normalize_generate_animations({"generate_animations": True}, _logger()) is True
    assert (
        _normalize_generate_animations({"generate_animations": False}, _logger()) is False
    )


def test_string_flags_are_coerced() -> None:
    assert (
        _normalize_generate_animations({"generate_animations": "false"}, _logger())
        is False
    )
    assert _normalize_generate_animations({"generate_animations": 1}, _logger()) is True


def test_conflicting_legacy_flag_raises() -> None:
    with pytest.raises(ValueError, match="Ambiguous animation flags"):
        _normalize_generate_animations(
            {"generate_animations": True, "no_animations": True}, _logger()
        )


def test_legacy_only_flag_is_inverted() -> None:
    assert _normalize_generate_animations({"no_animations": True}, _logger()) is False


def test_default_is_animations_enabled() -> None:
    assert _normalize_generate_animations({}, _logger()) is True


def test_requested_frameworks_normalize_and_win_when_present() -> None:
    # Requested frameworks seed the scope; frameworks with a successful
    # result pointer narrow it (pymdp succeeded, discopy was skipped).
    scope = _scope_from_execution_summary(_summary())

    assert scope["frameworks"] == {"pymdp"}
    assert scope["models"] is not None and "Simple MDP Agent" in scope["models"]


def test_requested_frameworks_survive_when_no_details_succeed() -> None:
    summary = {
        "requested_frameworks": ["PyMDP", "JAX"],
        "execution_details": [
            {"framework": "pymdp", "success": False, "skipped": True}
        ],
    }

    scope = _scope_from_execution_summary(summary)

    # With no successful result pointers, the requested list stands.
    assert scope["frameworks"] == {"pymdp", "jax"}


# --- _filter_execution_summary -----------------------------------------------




def test_successful_details_restrict_frameworks() -> None:
    summary = {
        "execution_details": [
            {
                "framework": "pymdp",
                "success": True,
                "structured_result_file": "results.json",
            },
            {"framework": "jax", "success": False},
        ]
    }

    scope = _scope_from_execution_summary(summary)

    # Only frameworks with successful result pointers remain in scope.
    assert scope["frameworks"] == {"pymdp"}


def test_empty_summary_yields_none_scope() -> None:
    scope = _scope_from_execution_summary({})

    assert scope == {"frameworks": None, "models": None}


def test_target_model_names_seed_the_model_scope() -> None:
    scope = _scope_from_execution_summary({}, target_model_names={"seeded_model"})

    assert scope["models"] == {"seeded_model"}
    assert scope["frameworks"] is None


def test_framework_directory_in_script_path_seeds_model() -> None:
    summary = {
        "execution_details": [
            {
                "framework": "pymdp",
                "success": True,
                "output_file": "x",
                "script_path": "output/12_execute_output/actinf_pomdp_agent/pymdp/run.py",
            }
        ]
    }

    scope = _scope_from_execution_summary(summary)

    assert "actinf_pomdp_agent" in (scope["models"] or set())


# --- _filter_execution_summary ------------------------------------------------


def test_filter_without_allowlist_returns_original_object() -> None:
    summary = _summary()

    assert _filter_execution_summary(summary, None) is summary


def test_filter_keeps_only_allowed_frameworks() -> None:
    summary = _summary()

    filtered = _filter_execution_summary(summary, {"pymdp"})

    details = filtered["execution_details"]
    assert [d["framework"] for d in details] == ["pymdp"]
    assert set(filtered["framework_status"]) == {"pymdp"}
    # The input summary is deep-copied, not mutated.
    assert len(summary["execution_details"]) == 2
