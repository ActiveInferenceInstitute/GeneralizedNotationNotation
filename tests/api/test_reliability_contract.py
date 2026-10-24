"""API lifecycle regressions with real subprocesses and FastAPI requests."""

import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from gnn.api import processor
from gnn.api.models import RunRequest
from gnn.api.pipeline_runner import PIPELINE_SUMMARY, read_pipeline_summary


@pytest.mark.parametrize("root", [None, [], "invalid", 1, {"steps": {}}])
def test_summary_rejects_malformed_root(tmp_path: Path, root: object) -> None:
    summary = tmp_path / PIPELINE_SUMMARY
    summary.parent.mkdir()
    summary.write_text(json.dumps(root))
    assert read_pipeline_summary(tmp_path) is None


@pytest.mark.parametrize("state", ["queued", "running"])
def test_delete_active_run_cancels_and_removes_record(
    state: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Deleting an active run cancels it through its token and pops it."""
    from gnn.execute.subprocess_envelope import CancelToken

    module = importlib.import_module("gnn.api.app")
    monkeypatch.setenv("GNN_RATE_LIMIT", "0")
    monkeypatch.delenv("GNN_API_KEY", raising=False)
    entry: dict = {"status": state, "request": {"output_dir": str(tmp_path / "out")}}

    class _CancelMarksTerminal(CancelToken):
        """Token whose cancel mirrors the executor's prompt terminal write."""

        def __init__(self, target: dict) -> None:
            super().__init__()
            self._target = target

        def cancel(self, reason: str | None = None) -> None:
            super().cancel(reason)
            self._target["status"] = "cancelled"
            self._target["completed_at"] = "2026-01-01T00:00:00"

    token = _CancelMarksTerminal(entry) if state == "running" else CancelToken()
    entry["cancel_token"] = token
    store = {"abc123": entry}
    app = module.create_app(runs_store=store)
    response = TestClient(app).delete("/api/v1/runs/abc123")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["data"]["cancelled"] is True
    assert "abc123" not in store


@pytest.mark.parametrize("exit_code", [0, 1, 2])
@pytest.mark.parametrize("strict", [False, True])
def test_both_api_surfaces_share_exit_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int, strict: bool
) -> None:
    module = importlib.import_module("gnn.api.app")
    command = [sys.executable, "-c", f"raise SystemExit({exit_code})"]
    monkeypatch.setattr(processor, "build_pipeline_command", lambda *a, **kw: command)
    monkeypatch.setattr(module, "build_pipeline_command", lambda *a, **kw: command)
    job_id = processor.create_job(target_dir=".", strict=strict)
    processor._JOBS[job_id]["output_dir"] = str(tmp_path)
    store = {"run": {"status": "queued", "events": [], "errors": []}}
    try:
        asyncio.run(processor.execute_job_async(job_id))
        expected = (
            "completed"
            if exit_code == 0 or (exit_code == 2 and not strict)
            else "failed"
        )
        job = processor.get_job(job_id)
        assert job is not None
        assert job["status"] == expected
        request = RunRequest(output_dir=str(tmp_path), strict=strict)
        asyncio.run(module._execute_pipeline("run", request, store))
        expected = (
            "completed"
            if exit_code == 0 or (exit_code == 2 and not strict)
            else "failed"
        )
        job = processor.get_job(job_id)
        assert job is not None
        assert job["status"] == expected
        assert store["run"]["status"] == expected
        job = processor.get_job(job_id)
        assert job is not None
        assert job["exit_code"] == exit_code
        assert store["run"]["exit_code"] == exit_code
    finally:
        processor._JOBS.pop(job_id, None)


@pytest.mark.parametrize("value", [True, "3", 3.0])
def test_request_step_numbers_reject_coercion(value: object) -> None:
    from gnn.api.models import ProcessRequest

    with pytest.raises(ValueError):
        ProcessRequest(steps=[cast("int", value)])
    with pytest.raises(ValueError):
        RunRequest(skip_steps=[cast("int", value)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("file_workers", None),
        ("file_workers", True),
        ("step_workers", False),
        ("file_workers", 0),
        ("step_workers", -1),
        ("file_workers", 257),
        ("step_workers", 257),
        ("file_workers", "2"),
        ("step_workers", 2.0),
    ],
)
def test_worker_limits_reject_coercion_and_bounds_before_job_creation(
    field: str, value: object
) -> None:
    from gnn.api.mcp import gnn_submit_job_mcp
    from gnn.api.models import ProcessRequest, ToolRequest
    from gnn.api.pipeline_runner import build_pipeline_command

    settings = {field: value}
    before = set(processor._JOBS)
    for model in (ProcessRequest, RunRequest, ToolRequest):
        with pytest.raises(ValueError, match="integer between 1 and 256"):
            model(**settings)
    with pytest.raises(ValueError, match="integer between 1 and 256"):
        build_pipeline_command("not-admitted", "unused", **settings)
    with pytest.raises(ValueError, match="integer between 1 and 256"):
        processor.create_job("not-admitted", **settings)
    result = gnn_submit_job_mcp("not-admitted", **settings)
    assert result["status"] == "error"
    assert "integer between 1 and 256" in result["message"]
    assert set(processor._JOBS) == before


def test_worker_limit_defaults_and_argv_preserve_requested_limits() -> None:
    from gnn.api.models import ProcessRequest, ToolRequest
    from gnn.api.pipeline_runner import build_pipeline_command

    for model in (ProcessRequest, RunRequest, ToolRequest):
        assert model().file_workers == 1
        assert model().step_workers is None
        assert model(file_workers=256, step_workers=256).file_workers == 256
        properties = model.model_json_schema()["properties"]
        assert properties["file_workers"]["minimum"] == 1
        assert properties["file_workers"]["maximum"] == 256
    default = build_pipeline_command("input", "output")
    assert default[default.index("--file-workers") + 1] == "1"
    assert "--step-workers" not in default
    configured = build_pipeline_command(
        "input", "output", file_workers=3, step_workers=2, parallel=True
    )
    assert configured[configured.index("--file-workers") + 1] == "3"
    assert configured[configured.index("--step-workers") + 1] == "2"
    assert "--parallel" in configured


def test_summary_ignores_previous_invocation(tmp_path: Path) -> None:
    import time

    path = tmp_path / PIPELINE_SUMMARY
    path.parent.mkdir()
    path.write_text(json.dumps({"steps": [{"status": "SUCCESS", "step_num": 3}]}))
    started = time.time_ns()
    assert read_pipeline_summary(tmp_path, not_before_ns=started) is None
    path.write_text(json.dumps({"steps": [{"status": "FAILED", "step_num": 3}]}))
    summary = read_pipeline_summary(tmp_path, not_before_ns=started)
    assert summary is not None
    assert summary[0]["status"] == "FAILED"


def test_summary_rejects_another_current_job(tmp_path: Path) -> None:
    path = tmp_path / PIPELINE_SUMMARY
    path.parent.mkdir()
    path.write_text(
        json.dumps({"run_id": "other-job", "steps": [{"status": "SUCCESS"}]})
    )
    assert read_pipeline_summary(tmp_path, expected_run_id="this-job") is None
    assert read_pipeline_summary(tmp_path, expected_run_id="other-job") == [
        {"status": "SUCCESS"}
    ]
