"""main() wiring contracts: config preflight gate, execution loops, and early returns.

All tests are offline and monkeypatched — no pipeline step subprocess is ever
spawned. The tmp cwd has no ``input/config.yaml`` unless a test writes one,
which also exercises the tolerated missing-config path on every run.
"""

import json
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import gnn.main as orchestrator
from gnn.utils.arguments.pipeline_arguments import PipelineArguments


def _receipt(args: PipelineArguments) -> dict[str, Any]:
    return json.loads(
        (
            args.output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"
        ).read_text()
    )


def _success_step_recorder(calls: list[str], threads: list[str] | None = None) -> Any:
    """Return an execute_pipeline_step double that records calls and succeeds."""

    def fake_execute_pipeline_step(
        script_name: str,
        args: PipelineArguments,
        logger: Any,
        *,
        run_id: str | None = None,
        pipeline_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        calls.append(script_name)
        if threads is not None:
            threads.append(threading.current_thread().name)
        return {
            "status": "SUCCESS",
            "stdout": f"ran {script_name}",
            "stderr": "",
            "exit_code": 0,
            "retry_count": 0,
            "prerequisite_check": True,
            "dependency_warnings": [],
            "recoverable": False,
            "memory_usage_mb": 0.0,
            "peak_memory_mb": 0.0,
            "memory_delta_mb": 0.0,
        }

    return fake_execute_pipeline_step


@pytest.fixture
def wired_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PipelineArguments:
    """Isolated cwd/run dirs; no config file, so the preflight gate is tolerated."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "model.md").write_text("# Model\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GNN_RUN_ID", raising=False)
    return PipelineArguments(target_dir=input_dir, output_dir=tmp_path / "output")


def test_invalid_config_skip_steps_fails_before_step_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "model.md").write_text("# Model\n")
    (input_dir / "config.yaml").write_text('pipeline:\n  skip_steps: ["abc"]\n')
    args = PipelineArguments(target_dir=input_dir, output_dir=tmp_path / "output")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GNN_RUN_ID", raising=False)
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator, "execute_pipeline_step", _success_step_recorder(calls)
    )

    assert orchestrator.main(args) == 1

    # Failed at the config gate: no step ran, startup receipt is persisted.
    assert calls == []
    receipt = _receipt(args)
    assert receipt["overall_status"] == "FAILED"
    assert receipt["steps"] == []
    assert "Preflight config validation failed" in receipt["error"]
    assert "pipeline.skip_steps" in receipt["error"]
    assert "'abc'" in receipt["error"]


def test_valid_config_passes_gate_and_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "model.md").write_text("# Model\n")
    (input_dir / "config.yaml").write_text("pipeline:\n  skip_steps: [15]\n")
    args = PipelineArguments(
        target_dir=input_dir, output_dir=tmp_path / "output", only_steps="0"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GNN_RUN_ID", raising=False)
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator, "execute_pipeline_step", _success_step_recorder(calls)
    )

    assert orchestrator.main(args) == 0

    assert calls == ["0_template.py"]
    receipt = _receipt(args)
    assert receipt["overall_status"] == "SUCCESS"
    assert receipt["performance_summary"]["successful_steps"] == 1


def test_skip_llm_flag_auto_injects_step_13_skip(
    wired_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--target-dir",
            str(wired_run.target_dir),
            "--output-dir",
            str(wired_run.output_dir),
            "--skip-llm",
        ],
    )
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator, "execute_pipeline_step", _success_step_recorder(calls)
    )

    assert orchestrator.main() == 0

    assert len(calls) == 24  # canonical 25 steps minus the injected 13 skip
    assert "13_llm.py" not in calls
    assert _receipt(wired_run)["arguments"]["skip_steps"] == "13"


def test_serial_execution_order_and_result_aggregation(
    wired_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    wired_run.only_steps = "0,1"
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator, "execute_pipeline_step", _success_step_recorder(calls)
    )

    assert orchestrator.main(wired_run) == 0

    assert calls == ["0_template.py", "1_setup.py"]
    receipt = _receipt(wired_run)
    assert [step["script_name"] for step in receipt["steps"]] == [
        "0_template.py",
        "1_setup.py",
    ]
    assert all(step["status"] == "SUCCESS" for step in receipt["steps"])
    perf = receipt["performance_summary"]
    assert perf["successful_steps"] == 2
    assert perf["failed_steps"] == 0
    assert receipt["overall_status"] == "SUCCESS"


def test_parallel_execution_same_tier_and_result_aggregation(
    wired_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    wired_run.parallel = True
    wired_run.only_steps = "0,1"  # steps 0 and 1 share a dependency tier
    calls: list[str] = []
    threads: list[str] = []
    monkeypatch.setattr(
        orchestrator,
        "execute_pipeline_step",
        _success_step_recorder(calls, threads),
    )

    assert orchestrator.main(wired_run) == 0

    assert set(calls) == {"0_template.py", "1_setup.py"}
    assert any("ThreadPoolExecutor" in name for name in threads)
    # Tier results are recorded in resolved tier order and fully aggregated.
    receipt = _receipt(wired_run)
    assert [step["script_name"] for step in receipt["steps"]] == [
        "0_template.py",
        "1_setup.py",
    ]
    assert receipt["performance_summary"]["successful_steps"] == 2
    assert receipt["overall_status"] == "SUCCESS"


def test_publish_gate_failure_saves_minimal_summary(
    wired_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    wired_run.only_steps = "0"
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator, "execute_pipeline_step", _success_step_recorder(calls)
    )
    saved: list[tuple[Path, Exception]] = []
    real_save = orchestrator._save_minimal_pipeline_summary

    def spy(
        summary_path: Path,
        pipeline_summary: dict[str, Any],
        error: Exception,
        logger: Any,
    ) -> None:
        saved.append((summary_path, error))
        real_save(summary_path, pipeline_summary, error, logger)

    monkeypatch.setattr(orchestrator, "_save_minimal_pipeline_summary", spy)
    monkeypatch.setattr(
        "gnn.pipeline.hasher.verify_indexed_run",
        lambda *args, **kwargs: ["file set drift"],
    )

    assert orchestrator.main(wired_run) == 1

    assert calls == ["0_template.py"]
    assert len(saved) == 1
    assert "Run identity changed before publication" in str(saved[0][1])
    receipt = _receipt(wired_run)
    assert receipt["overall_status"] == "FAILED"
    assert receipt["steps_count"] == 1
    assert "file set drift" in receipt["error"]


def test_autonomous_proposal_loop_bypasses_step_execution(
    wired_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_calls: list[tuple[Path, Path]] = []

    def fake_proposal_loop(target_dir: Path, output_dir: Path) -> dict[str, int]:
        loop_calls.append((target_dir, output_dir))
        return {"candidate_count": 2}

    monkeypatch.setattr(
        "gnn.pipeline.autonomous.run_autonomous_proposal_loop", fake_proposal_loop
    )
    wired_run.autonomous = True
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator, "execute_pipeline_step", _success_step_recorder(calls)
    )

    assert orchestrator.main(wired_run) == 0

    assert loop_calls == [(wired_run.target_dir, wired_run.output_dir)]
    assert calls == []  # the proposal loop replaces step execution entirely
