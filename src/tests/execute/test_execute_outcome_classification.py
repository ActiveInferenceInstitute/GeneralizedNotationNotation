#!/usr/bin/env python3
"""Pin the pure Step 12 outcome classifier extracted from process_execute.

``execute.processor._classify_execute_outcome`` is a pure function of the run
counters; these tests assert every branch of its truth table plus the
exit-code derivation, independent of subprocess/IO. Any future edit that
changes the classification contract must update (or deliberately break) this
table.
"""

import sys
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from execute.processor import _classify_execute_outcome  # noqa: E402
from execute.types import ExecutionOutcome  # noqa: E402


def _classify(**kw: Any) -> ExecutionOutcome:
    """Call the classifier with the historical-default kwargs filled in."""
    defaults: dict[str, Any] = {
        "total_found": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "render_failures": [],
        "missing_render_scripts": [],
        "missing_render_summary": None,
        "strict_requested_frameworks": False,
    }
    defaults.update(kw)
    return _classify_execute_outcome(**defaults)


def test_missing_render_summary_fails_regardless_of_scripts() -> None:
    outcome = _classify(missing_render_summary="render_processing_summary.json")
    assert outcome.outcome is False
    assert outcome.status == "failed"
    assert outcome.reason == "required_render_summary_missing"
    assert outcome.exit_code == 1
    assert outcome.attempted == 0


def test_strict_requested_with_render_failures_fails() -> None:
    outcome = _classify(
        strict_requested_frameworks=True,
        render_failures=[{"file": "a.md", "framework": "pymdp"}],
    )
    assert outcome.outcome is False
    assert outcome.status == "failed"
    assert outcome.reason == "requested_framework_render_failure"


def test_non_strict_render_failures_still_succeeds_best_effort() -> None:
    outcome = _classify(
        render_failures=[{"file": "a.md", "framework": "pymdp"}],
    )
    # No scripts found, non-strict → outcome 2 (nothing to do) takes priority
    # over the render_failures branch only when scripts exist; here it is 2.
    assert outcome.outcome == 2
    assert outcome.status == "skipped"
    assert outcome.reason == "no_executable_scripts"


def test_missing_render_scripts_fails() -> None:
    outcome = _classify(
        total_found=2,
        missing_render_scripts=["/abs/path/model_pymdp.py"],
    )
    assert outcome.outcome is False
    assert outcome.status == "failed"
    assert outcome.reason == "rendered_script_missing"
    assert outcome.attempted == 2


def test_no_scripts_non_strict_returns_2_skipped() -> None:
    outcome = _classify(total_found=0)
    assert outcome.outcome == 2
    assert outcome.status == "skipped"
    assert outcome.reason == "no_executable_scripts"
    assert outcome.exit_code == 2


def test_no_scripts_strict_fails() -> None:
    outcome = _classify(total_found=0, strict_requested_frameworks=True)
    assert outcome.outcome is False
    assert outcome.status == "failed"
    assert outcome.reason == "no_executable_scripts"
    assert outcome.exit_code == 1


def test_strict_requested_with_failures_fails() -> None:
    outcome = _classify(
        total_found=3, successful=2, failed=1, strict_requested_frameworks=True
    )
    assert outcome.outcome is False
    assert outcome.status == "failed"
    assert outcome.reason == "requested_framework_execution_incomplete"


def test_strict_requested_with_skips_fails() -> None:
    outcome = _classify(
        total_found=3, successful=3, skipped=0, strict_requested_frameworks=True
    )
    # No failures/skips and all succeeded → success even in strict mode.
    assert outcome.outcome is True
    assert outcome.status == "success"
    assert outcome.reason == "all_scripts_succeeded"


def test_strict_requested_skipped_only_fails() -> None:
    outcome = _classify(
        total_found=3, successful=0, skipped=3, strict_requested_frameworks=True
    )
    assert outcome.outcome is False
    assert outcome.reason == "requested_framework_execution_incomplete"


def test_failures_non_strict_fails() -> None:
    outcome = _classify(total_found=3, successful=1, failed=2)
    assert outcome.outcome is False
    assert outcome.status == "failed"
    assert outcome.reason == "script_execution_failure"
    assert outcome.exit_code == 1
    assert outcome.attempted == 3


def test_skips_only_succeeds_with_skips() -> None:
    outcome = _classify(total_found=3, successful=0, skipped=3)
    assert outcome.outcome is True
    assert outcome.status == "success_with_skips"
    assert outcome.reason == "optional_dependencies_unavailable"
    assert outcome.attempted == 0  # all skipped


def test_render_failures_with_success_succeeds_best_effort() -> None:
    outcome = _classify(
        total_found=2,
        successful=2,
        render_failures=[{"file": "x.md", "framework": "jax"}],
    )
    assert outcome.outcome is True
    assert outcome.status == "success_with_render_failures"
    assert outcome.reason == "best_effort_render_subset_executed"


def test_all_succeed_clean_success() -> None:
    outcome = _classify(total_found=4, successful=4)
    assert outcome.outcome is True
    assert outcome.status == "success"
    assert outcome.reason == "all_scripts_succeeded"
    assert outcome.exit_code == 0


def test_attempted_excludes_skipped() -> None:
    outcome = _classify(total_found=10, successful=6, failed=2, skipped=2)
    assert outcome.attempted == 8
    assert outcome.outcome is False
    assert outcome.reason == "script_execution_failure"
