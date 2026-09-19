"""Live run-session + durable-stream wiring contracts for the main.py composition.

All tests are offline and monkeypatched — no pipeline step subprocess is ever
spawned. The doubles create the conventional step output directories so the
honest-provenance branch of ``record_step_result`` (existing dir -> artifact
ref) is exercised against real on-disk state.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest

import gnn.main as orchestrator
from gnn.pipeline.run_manifest import read_trace, trace_integrity, verify_run_manifests
from gnn.pipeline.run_session import UnitStatus, load_session, resume_plan
from gnn.pipeline.run_session_wiring import (
    close_run_session,
    open_run_session,
    record_step_result,
    run_session_path,
)
from gnn.utils.arguments.pipeline_arguments import PipelineArguments


def _receipt(args: PipelineArguments) -> dict[str, Any]:
    return json.loads(
        (
            args.output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"
        ).read_text()
    )


def _success_step_recorder(calls: list[str], threads: list[str] | None = None) -> Any:
    """Return an execute_pipeline_step double that records calls, succeeds,
    and materializes the step's conventional output directory."""

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
        step_dir = args.output_dir / f"{script_name.split('.')[0]}_output"
        step_dir.mkdir(parents=True, exist_ok=True)
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


def test_pipeline_run_produces_durable_session_record(
    wired_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    wired_run.only_steps = "0,1"
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator, "execute_pipeline_step", _success_step_recorder(calls)
    )

    assert orchestrator.main(wired_run) == 0

    assert calls == ["0_template.py", "1_setup.py"]
    session = load_session(
        wired_run.output_dir / "00_pipeline_summary" / "run_session.json"
    )
    assert session.schema_version == "1.0"
    assert session.run_hash
    assert [wu.unit_id for wu in session.units] == ["0_template.py", "1_setup.py"]
    assert all(wu.status == UnitStatus.DONE for wu in session.units)
    assert [wu.steps for wu in session.units] == [[0], [1]]
    # Honest provenance: both step output dirs exist, so both are referenced.
    assert session.units[0].artifact_refs == ["0_template_output"]
    assert session.units[1].artifact_refs == ["1_setup_output"]

    # Re-open over the same output dir: a fresh session cleanly replaces the
    # old one (units match the new selection; no checkpoint residue left).
    wired_run.only_steps = "1"
    calls.clear()
    assert orchestrator.main(wired_run) == 0

    reloaded = load_session(
        wired_run.output_dir / "00_pipeline_summary" / "run_session.json"
    )
    assert [wu.unit_id for wu in reloaded.units] == ["1_setup.py"]
    assert reloaded.units[0].status == UnitStatus.DONE
    summary_dir = wired_run.output_dir / "00_pipeline_summary"
    residue = [
        name
        for name in (p.name for p in summary_dir.iterdir())
        if ".tmp" in name or ".part" in name
    ]
    assert residue == []


def test_pipeline_run_emits_durable_run_manifest(
    wired_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    wired_run.only_steps = "0,1"
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator, "execute_pipeline_step", _success_step_recorder(calls)
    )

    assert orchestrator.main(wired_run) == 0

    manifest_dir = wired_run.output_dir / "v3_run_manifest"
    index_path = manifest_dir / "index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text())
    assert verify_run_manifests(manifest_dir, wired_run.output_dir) == []
    trace = read_trace(manifest_dir / index["trace_file"])
    assert trace_integrity(trace) == []
    assert len(trace.events) == len(calls) == 2


def test_failed_step_marks_unit_failed(
    wired_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    wired_run.only_steps = "0,1"
    calls: list[str] = []

    def fake_execute_pipeline_step(
        script_name: str,
        args: PipelineArguments,
        logger: Any,
        *,
        run_id: str | None = None,
        pipeline_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        calls.append(script_name)
        if script_name == "0_template.py":
            return {
                "status": "FAILED",
                "stdout": "",
                "stderr": "boom",
                "exit_code": 1,
                "retry_count": 0,
                "prerequisite_check": True,
                "dependency_warnings": [],
                "recoverable": False,
                "memory_usage_mb": 0.0,
                "peak_memory_mb": 0.0,
                "memory_delta_mb": 0.0,
                "error": "step 0 exploded",
            }
        return _success_step_recorder([])(script_name, args, logger)

    monkeypatch.setattr(
        orchestrator, "execute_pipeline_step", fake_execute_pipeline_step
    )

    assert orchestrator.main(wired_run) == 1

    session = load_session(
        wired_run.output_dir / "00_pipeline_summary" / "run_session.json"
    )
    by_id = {wu.unit_id: wu for wu in session.units}
    assert by_id["0_template.py"].status == UnitStatus.FAILED
    assert by_id["0_template.py"].error == "step 0 exploded"
    assert by_id["1_setup.py"].status == UnitStatus.DONE
    assert resume_plan(session) == ["0_template.py"]
    assert _receipt(wired_run)["overall_status"] == "FAILED"


def test_parallel_run_updates_session(
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
    session = load_session(
        wired_run.output_dir / "00_pipeline_summary" / "run_session.json"
    )
    assert [wu.unit_id for wu in session.units] == ["0_template.py", "1_setup.py"]
    assert all(wu.status == UnitStatus.DONE for wu in session.units)
    assert _receipt(wired_run)["overall_status"] == "SUCCESS"


def test_record_step_result_status_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GNN_RUN_ID", raising=False)
    args = PipelineArguments(target_dir=tmp_path, output_dir=tmp_path / "out")
    summary: dict[str, Any] = {"run_id": "unit-test-run"}
    session = open_run_session(
        args, [("0_template.py", "t"), ("1_setup.py", "s")], summary
    )
    assert session.session_id == "unit-test-run"

    warned = record_step_result(
        session, "0_template.py", {"status": "SUCCESS_WITH_WARNINGS"}, tmp_path / "out"
    )
    unit = next(wu for wu in warned.units if wu.unit_id == "0_template.py")
    assert unit.status == UnitStatus.DONE
    assert unit.error == ""

    failed = record_step_result(
        warned, "1_setup.py", {"status": "FAILED", "error": "kaput"}, tmp_path / "out"
    )
    unit = next(wu for wu in failed.units if wu.unit_id == "1_setup.py")
    assert unit.status == UnitStatus.FAILED
    assert unit.error == "kaput"


def test_close_run_session_survives_emission_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GNN_RUN_ID", raising=False)

    def boom(output_dir: Any) -> dict[str, Any]:
        raise RuntimeError("emission exploded")

    monkeypatch.setattr("gnn.pipeline.run_manifest.emit_run_manifests", boom)
    args = PipelineArguments(target_dir=tmp_path, output_dir=tmp_path / "out")
    session = open_run_session(args, [("0_template.py", "t")], {"run_hash": "abc123"})
    session_path = run_session_path(args.output_dir)
    assert session_path.exists()

    warnings: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warnings.append(record.getMessage())

    logger = logging.getLogger("test_close_run_session_survives")
    logger.setLevel(logging.INFO)
    handler = _ListHandler()
    logger.addHandler(handler)
    try:
        returned = close_run_session(session, args.output_dir, logger)
    finally:
        logger.removeHandler(handler)

    assert returned is session
    assert any("emission exploded" in w for w in warnings)
    # The session still checkpoints (idempotent final state) and does not raise.
    reloaded = load_session(session_path)
    assert [wu.unit_id for wu in reloaded.units] == ["0_template.py"]
