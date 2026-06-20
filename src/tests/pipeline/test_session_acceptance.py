#!/usr/bin/env python3
"""Tests for pipeline/session_acceptance.py — resumable acceptance (real objects only).

Every ``runner`` injected below is a real deterministic callable that writes
real on-disk pipeline-summary artifacts and returns a real
``subprocess.CompletedProcess`` — no test-double framework is used anywhere. The
families exercised are real entries from the maintained model-family manifest,
whose representative fixtures exist on disk; only the pipeline *execution* is
replaced by the injected runner so no real pipeline runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.run_session import UnitStatus, load_session  # noqa: E402
from pipeline.session_acceptance import run_session_acceptance  # noqa: E402

MANIFEST = Path("input/model_family_manifest.json")
PRESENT_STEPS = (3, 5, 6, 11, 12, 15, 16, 23)
STEP_ARTIFACTS = {
    3: (
        "3_gnn_output/gnn_processing_summary.json",
        "3_gnn_output/gnn_processing_results.json",
    ),
    5: ("5_type_checker_output/type_check_results.json",),
    6: (
        "6_validation_output/validation_summary.json",
        "6_validation_output/validation_results.json",
    ),
    11: ("11_render_output/render_processing_summary.json",),
    12: ("12_execute_output/summaries/execution_summary.json",),
    15: ("15_audio_output/audio_results.json",),
    16: ("16_analysis_output/analysis_results.json",),
    23: ("23_report_output/report_processing_summary.json",),
}


def _write_passing_summary(output_dir: Path) -> None:
    """Write the real artifacts + pipeline summary a successful run produces."""
    for step in PRESENT_STEPS:
        for relative in STEP_ARTIFACTS[step]:
            path = output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"step": step, "success": True}), encoding="utf-8"
            )
    summary_dir = output_dir / "00_pipeline_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    steps = [
        {"script_name": f"{step}_step.py", "status": "SUCCESS"}
        for step in PRESENT_STEPS
    ]
    (summary_dir / "pipeline_execution_summary.json").write_text(
        json.dumps({"overall_status": "SUCCESS", "steps": steps}),
        encoding="utf-8",
    )


def _passing_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    output_dir = Path(command[command.index("--output-dir") + 1])
    _write_passing_summary(output_dir)
    return subprocess.CompletedProcess(
        args=list(command), returncode=0, stdout="pipeline ok", stderr=""
    )


def _failing_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    # No pipeline summary written + non-zero return => the family ledger fails,
    # but the orchestration returns (does not raise) so the session is resumable.
    return subprocess.CompletedProcess(
        args=list(command), returncode=1, stdout="", stderr="boom"
    )


def test_three_family_run_checkpoints_and_completes(tmp_path: Path) -> None:
    session_path = tmp_path / "session.json"
    output_dir = tmp_path / "out"

    result = run_session_acceptance(
        MANIFEST,
        output_dir,
        session_path,
        family_names=["basics", "discrete", "continuous"],
        runner=_passing_runner,
    )

    # Session file exists on disk and reads back via load_session.
    assert session_path.exists()
    persisted = load_session(session_path)
    assert [u.unit_id for u in persisted.units] == [
        "basics",
        "discrete",
        "continuous",
    ]
    assert all(u.status == UnitStatus.DONE for u in persisted.units)

    status = result["status"]
    assert status["done"] is True
    assert status["total"] == 3
    assert status["completed"] == 3
    assert set(result["ledgers"]) == {"basics", "discrete", "continuous"}
    for ledger in result["ledgers"].values():
        assert ledger["status"] == "passed"


def test_resume_completes_only_remaining_failed_family(tmp_path: Path) -> None:
    session_path = tmp_path / "session.json"
    output_dir = tmp_path / "out"

    # First runner: the 2nd family ("discrete") fails; others pass.
    first_calls: list[str] = []

    def first_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        # --output-dir is <out>/<family>/pipeline_output, so the family name is
        # the parent directory of the pipeline output dir.
        output_dir_arg = Path(command[command.index("--output-dir") + 1])
        family = output_dir_arg.parent.name
        first_calls.append(family)
        if family == "discrete":
            return _failing_runner(command)
        return _passing_runner(command)

    first = run_session_acceptance(
        MANIFEST,
        output_dir,
        session_path,
        family_names=["basics", "discrete", "continuous"],
        runner=first_runner,
    )

    # First pass processed all three; discrete did not reach DONE.
    assert set(first_calls) == {"basics", "discrete", "continuous"}
    persisted = load_session(session_path)
    by_status = {u.unit_id: u.status for u in persisted.units}
    assert by_status["basics"] == UnitStatus.DONE
    assert by_status["continuous"] == UnitStatus.DONE
    assert by_status["discrete"] == UnitStatus.FAILED
    assert first["status"]["done"] is False

    # Second runner: now everything succeeds. Assert it is invoked ONLY for the
    # remaining family ("discrete").
    second_calls: list[str] = []

    def second_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        output_dir_arg = Path(command[command.index("--output-dir") + 1])
        second_calls.append(output_dir_arg.parent.name)
        return _passing_runner(command)

    second = run_session_acceptance(
        MANIFEST,
        output_dir,
        session_path,
        family_names=["basics", "discrete", "continuous"],
        runner=second_runner,
        resume=True,
    )

    assert second_calls == ["discrete"]
    assert set(second["ledgers"]) == {"discrete"}
    final = load_session(session_path)
    assert all(u.status == UnitStatus.DONE for u in final.units)
    assert second["status"]["done"] is True


def test_unknown_family_raises_keyerror(tmp_path: Path) -> None:
    session_path = tmp_path / "session.json"
    output_dir = tmp_path / "out"

    with pytest.raises(KeyError):
        run_session_acceptance(
            MANIFEST,
            output_dir,
            session_path,
            family_names=["basics", "does-not-exist"],
            runner=_passing_runner,
        )


def test_resume_with_family_not_in_session_raises(tmp_path: Path) -> None:
    """NEGATIVE: resuming with a family the persisted session never knew about must
    FAIL LOUDLY, not silently skip it and report done=True having run nothing."""
    session_path = tmp_path / "session.json"
    output_dir = tmp_path / "out"

    # Build & complete a session over just {basics, discrete}.
    run_session_acceptance(
        MANIFEST,
        output_dir,
        session_path,
        family_names=["basics", "discrete"],
        runner=_passing_runner,
    )
    assert load_session(session_path).units  # session persisted

    # Resuming with a DIFFERENT family must raise rather than no-op to done=True.
    with pytest.raises(ValueError, match="not in the persisted session"):
        run_session_acceptance(
            MANIFEST,
            output_dir,
            session_path,
            family_names=["continuous"],
            runner=_passing_runner,
            resume=True,
        )
