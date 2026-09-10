"""Wiring tests for ``src/gnn/main.py`` composition decisions (W2-D5/D7).

The real orchestrator runs with a faked ``execute_pipeline_step`` (and, for
matrix fan-out, a faked ``execute_command_streaming``) so every test asserts a
real main.py decision — step selection fallbacks, the ``skip_llm`` CLI
auto-inject, the ``--autonomous`` branch, serial/parallel loops, the publish
gate, crash receipts, ``GNN_RUN_ID`` scoping, and ``testing_matrix`` fan-out —
without spawning subprocesses.

Replaces the deleted ``test_pipeline_overall.py`` hasattr-façade checks and
the dict-replay step-numbering test with behavior assertions.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest

import gnn.main as main_mod
from gnn.main import parse_step_list_strict
from gnn.utils.pipeline_arguments import PipelineArguments

pytestmark = [pytest.mark.pipeline]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(tmp_path: Path, **overrides: Any) -> PipelineArguments:
    """Build override args pointed at an isolated tmp layout."""
    kwargs: dict[str, Any] = {
        "target_dir": tmp_path / "input",
        "output_dir": tmp_path / "output",
    }
    kwargs.update(overrides)
    return PipelineArguments(**kwargs)


def _seed_target(tmp_path: Path) -> Path:
    """Create a target dir with one model file so run-hash verification passes."""
    target = tmp_path / "input"
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.md").write_text("# Model\n## StateSpace\n", encoding="utf-8")
    return target


class ExecutorRecorder:
    """Fake execute_pipeline_step recording calls and returning canned results."""

    def __init__(self, fail_on: set[str] | None = None, raise_on: set[str] | None = None):
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.fail_on = fail_on or set()
        self.raise_on = raise_on or set()

    def __call__(
        self,
        script_name: str,
        args: Any,
        logger: Any,
        *,
        run_id: str | None = None,
        pipeline_config: Any = None,
    ) -> dict[str, Any]:
        with self.lock:
            self.calls.append(
                {
                    "script_name": script_name,
                    "run_id": run_id,
                    "pipeline_config": pipeline_config,
                }
            )
        if script_name in self.raise_on:
            raise RuntimeError(f"boom in {script_name}")
        if script_name in self.fail_on:
            return {"status": "FAILED", "exit_code": 1, "stdout": "", "stderr": ""}
        return {"status": "SUCCESS", "exit_code": 0, "stdout": "", "stderr": ""}


def _summary_path(output_dir: Path) -> Path:
    return output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"


# ---------------------------------------------------------------------------
# W2-D5: config-only preflight wiring
# ---------------------------------------------------------------------------


def test_config_skip_steps_unknown_value_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config ``skip_steps: ["abc"]`` aborts the run before any step runs."""
    monkeypatch.chdir(tmp_path)
    recorder = ExecutorRecorder()
    monkeypatch.setattr(main_mod, "execute_pipeline_step", recorder)

    rc = main_mod.main(
        override_args=_make_args(tmp_path),
        override_config={"pipeline": {"skip_steps": ["abc"]}},
    )

    assert rc == 1
    assert recorder.calls == []
    receipt = _summary_path(tmp_path / "output")
    assert receipt.is_file()
    assert "Invalid pipeline.skip_steps" in receipt.read_text(encoding="utf-8")


def test_config_bad_llm_timeout_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    recorder = ExecutorRecorder()
    monkeypatch.setattr(main_mod, "execute_pipeline_step", recorder)

    rc = main_mod.main(
        override_args=_make_args(tmp_path),
        override_config={"llm": {"timeout_seconds": -5}},
    )

    assert rc == 1
    assert recorder.calls == []


def test_config_free_run_executes_steps_with_sequential_numbering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config-free run reaches the (faked) executor and numbers steps by
    execution order — the real behavior the dict-replay test faked."""
    monkeypatch.chdir(tmp_path)
    _seed_target(tmp_path)
    recorder = ExecutorRecorder()
    monkeypatch.setattr(main_mod, "execute_pipeline_step", recorder)

    rc = main_mod.main(
        override_args=_make_args(tmp_path),
        override_config={"pipeline": {}},
    )

    assert rc == 0
    assert len(recorder.calls) == 25
    assert recorder.calls[0]["script_name"] == "0_template.py"
    assert recorder.calls[-1]["script_name"] == "24_intelligent_analysis.py"

    summary = json.loads(_summary_path(tmp_path / "output").read_text("utf-8"))
    assert summary["overall_status"] == "SUCCESS"
    assert [step["step_number"] for step in summary["steps"]] == list(
        range(1, len(summary["steps"]) + 1)
    )


def test_resolve_steps_skip_sets_merge_fully_consumed_raises() -> None:
    """CLI and config skip sets merge (union); a selection fully consumed by
    skips fails fast under the W2-D5 no-empty-selection contract."""
    args = _make_args(Path("/tmp"), skip_steps="3")
    settings = {"only_steps": "3,5", "skip_steps": [5]}
    with pytest.raises(ValueError, match="no executable steps"):
        main_mod._resolve_steps_to_execute(args, settings, logging.getLogger("t"))


def test_resolve_steps_skip_sets_partial_consumption_runs() -> None:
    """A partially-consumed selection still yields its executable steps."""
    args = _make_args(Path("/tmp"), skip_steps="3")
    settings = {"only_steps": "1,3"}
    steps = main_mod._resolve_steps_to_execute(args, settings, logging.getLogger("t"))
    assert [name for name, _ in steps] == ["1_setup.py"]


# ---------------------------------------------------------------------------
# W2-D7: composition wiring
# ---------------------------------------------------------------------------


def test_resolve_steps_config_only_steps_fallback() -> None:
    """Config ``only_steps`` applies when the CLI value is unset."""
    args = _make_args(Path("/tmp"))  # only_steps unset
    settings = {"only_steps": "3,5"}
    steps = main_mod._resolve_steps_to_execute(args, settings, logging.getLogger("t"))
    assert [name for name, _ in steps] == ["3_gnn.py", "5_type_checker.py"]


def test_resolve_steps_config_skip_steps_applies() -> None:
    args = _make_args(Path("/tmp"))
    settings = {"only_steps": "3,5", "skip_steps": [5]}
    steps = main_mod._resolve_steps_to_execute(args, settings, logging.getLogger("t"))
    assert [name for name, _ in steps] == ["3_gnn.py"]


def test_parse_step_list_strict_rejects_bad_tokens() -> None:
    with pytest.raises(ValueError, match="Invalid step number"):
        parse_step_list_strict("3,abc")


def test_build_main_args_auto_injects_skip_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--skip-llm`` appends step 13 to the CLI skip list."""
    monkeypatch.setattr("sys.argv", ["gnn", "--skip-llm"])
    args, _ = main_mod._build_main_args(None)
    assert args.skip_llm is True
    assert args.skip_steps is not None
    assert "13" in args.skip_steps.split(",")


def test_build_main_args_skip_llm_does_not_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["gnn", "--skip-llm", "--skip-steps", "13"])
    args, _ = main_mod._build_main_args(None)
    skip_raw = args.skip_steps
    assert skip_raw is not None
    assert skip_raw.split(",").count("13") == 1


def test_autonomous_branch_skips_step_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--autonomous`` runs the proposal loop and never executes steps."""
    monkeypatch.chdir(tmp_path)
    recorder = ExecutorRecorder()
    monkeypatch.setattr(main_mod, "execute_pipeline_step", recorder)

    def fake_loop(target_dir: Any, output_dir: Any) -> dict[str, Any]:
        return {"candidate_count": 1}

    monkeypatch.setattr(
        "gnn.pipeline.autonomous.run_autonomous_proposal_loop", fake_loop
    )

    rc = main_mod.main(
        override_args=_make_args(tmp_path, autonomous=True),
        override_config={"pipeline": {}},
    )

    assert rc == 0
    assert recorder.calls == []


def test_serial_loop_records_every_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_target(tmp_path)
    recorder = ExecutorRecorder()
    monkeypatch.setattr(main_mod, "execute_pipeline_step", recorder)

    rc = main_mod.main(
        override_args=_make_args(tmp_path, only_steps="3,7"),
        override_config={"pipeline": {}},
    )

    assert rc == 0
    recorded = [call["script_name"] for call in recorder.calls]
    assert recorded == ["3_gnn.py", "7_export.py"]


def test_parallel_loop_records_every_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parallel tier loop records results through the same recorder."""
    monkeypatch.chdir(tmp_path)
    _seed_target(tmp_path)
    recorder = ExecutorRecorder()
    monkeypatch.setattr(main_mod, "execute_pipeline_step", recorder)

    rc = main_mod.main(
        override_args=_make_args(tmp_path, only_steps="3,7", parallel=True),
        override_config={"pipeline": {}},
    )

    assert rc == 0
    assert sorted(call["script_name"] for call in recorder.calls) == [
        "3_gnn.py",
        "7_export.py",
    ]


def test_publish_gate_failure_saves_minimal_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publish-gate (run identity) failure fails the run with a receipt."""
    monkeypatch.chdir(tmp_path)
    _seed_target(tmp_path)
    recorder = ExecutorRecorder()
    monkeypatch.setattr(main_mod, "execute_pipeline_step", recorder)
    monkeypatch.setattr(
        "gnn.pipeline.hasher.verify_indexed_run", lambda entry: ["input drift"]
    )

    rc = main_mod.main(
        override_args=_make_args(tmp_path),
        override_config={"pipeline": {}},
    )

    assert rc == 1
    assert recorder.calls, "steps must have run before the gate"
    receipt = _summary_path(tmp_path / "output")
    assert receipt.is_file()
    assert "input drift" in receipt.read_text(encoding="utf-8")


def test_mid_run_crash_writes_failed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An executor crash outside a step still produces a FAILED receipt."""
    monkeypatch.chdir(tmp_path)
    _seed_target(tmp_path)
    recorder = ExecutorRecorder(raise_on={"3_gnn.py"})
    monkeypatch.setattr(main_mod, "execute_pipeline_step", recorder)

    rc = main_mod.main(
        override_args=_make_args(tmp_path),
        override_config={"pipeline": {}},
    )

    assert rc == 1
    # Serial loop: 0_template, 1_setup, 2_tests run clean; the fake records
    # the call, then raises on 3_gnn.py (the 4th executed step — 1_setup and
    # 2_tests pull in their dependencies). Calls == steps attempted.
    assert len(recorder.calls) == 4
    assert recorder.calls[0]["script_name"] == "0_template.py"
    assert recorder.calls[-1]["script_name"] == "3_gnn.py"
    summary = json.loads(_summary_path(tmp_path / "output").read_text("utf-8"))
    assert summary["overall_status"] == "FAILED"
    assert summary["run_id"]  # minted uuid, not None
    assert [s["script_name"] for s in summary["steps"]] == [
        "0_template.py",
        "1_setup.py",
        "2_tests.py",
    ]  # the crashing 3_gnn.py is never recorded as a step result


def test_gnn_run_id_env_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incoming GNN_RUN_ID is authoritative and restored after the run."""
    monkeypatch.chdir(tmp_path)
    _seed_target(tmp_path)
    recorder = ExecutorRecorder()
    monkeypatch.setattr(main_mod, "execute_pipeline_step", recorder)
    monkeypatch.setenv("GNN_RUN_ID", "fixed-run-id")

    rc = main_mod.main(
        override_args=_make_args(tmp_path),
        override_config={"pipeline": {}},
    )

    assert rc == 0
    assert all(call["run_id"] == "fixed-run-id" for call in recorder.calls)
    assert main_mod.os.environ.get("GNN_RUN_ID") == "fixed-run-id"


def test_gnn_run_id_fresh_call_generates_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_target(tmp_path)
    recorder = ExecutorRecorder()
    monkeypatch.setattr(main_mod, "execute_pipeline_step", recorder)
    monkeypatch.delenv("GNN_RUN_ID", raising=False)

    rc = main_mod.main(
        override_args=_make_args(tmp_path),
        override_config={"pipeline": {}},
    )

    assert rc == 0
    first_run_id = recorder.calls[0]["run_id"]
    assert first_run_id  # a uuid was minted
    assert len(recorder.calls) == 25
    assert "GNN_RUN_ID" not in main_mod.os.environ


# ---------------------------------------------------------------------------
# testing_matrix wiring (execute_pipeline_step, faked command runner)
# ---------------------------------------------------------------------------


def test_testing_matrix_global_steps_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A globally-disabled step returns SKIPPED without spawning a command."""
    from gnn.main import execute_pipeline_step

    monkeypatch.chdir(tmp_path)
    ran: list[list[str]] = []

    def fake_streaming(cmd: Any, **kwargs: Any) -> dict[str, Any]:
        ran.append(list(cmd))
        return {"stdout": "", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(
        "gnn.utils.execution_utils.execute_command_streaming", fake_streaming
    )

    result = execute_pipeline_step(
        "0_template.py",
        _make_args(tmp_path),
        logging.getLogger("t"),
        pipeline_config={"testing_matrix": {"enabled": True, "global_steps": {"0_template": False}}},
    )

    assert result["status"] == "SKIPPED"
    assert result["exit_code"] == 0
    assert "global_steps" in result["stdout"]
    assert ran == []


def test_testing_matrix_folder_fanout_runs_per_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Folder fan-out runs one child command per matching target folder."""
    from gnn.main import execute_pipeline_step

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "input"
    (target / "discrete").mkdir(parents=True)
    (target / "continuous").mkdir(parents=True)

    ran: list[list[str]] = []

    def fake_streaming(cmd: Any, **kwargs: Any) -> dict[str, Any]:
        ran.append(list(cmd))
        return {"stdout": "ok", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(
        "gnn.utils.execution_utils.execute_command_streaming", fake_streaming
    )

    result = execute_pipeline_step(
        "3_gnn.py",
        _make_args(tmp_path),
        logging.getLogger("t"),
        pipeline_config={
            "testing_matrix": {"enabled": True, "default_steps": [3]},
        },
    )

    assert result["exit_code"] == 0
    assert result["status"] == "SUCCESS"
    assert len(ran) == 2  # one child command per matching folder
    # Every child invocation targets this step's script.
    for cmd in ran:
        assert "3_gnn.py" in " ".join(cmd)
