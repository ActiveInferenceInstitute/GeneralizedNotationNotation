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
def test_delete_active_run_conflicts(
    state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("gnn.api.app")
    monkeypatch.setenv("GNN_RATE_LIMIT", "0")
    monkeypatch.delenv("GNN_API_KEY", raising=False)
    store = {"abc123": {"status": state}}
    app = module.create_app(runs_store=store)
    response = TestClient(app).delete("/api/v1/runs/abc123")
    assert response.status_code == 409
    assert response.json()["status"] == "error"
    assert "abc123" in store


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
