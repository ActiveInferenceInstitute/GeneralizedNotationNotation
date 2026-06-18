"""Tests for the GNN API endpoints."""

from pathlib import Path
from typing import Any

import pytest

from api.app import FASTAPI_AVAILABLE, create_app


def test_api_health_endpoint() -> Any:
    """Test the health check endpoint."""
    from fastapi.testclient import TestClient

    assert FASTAPI_AVAILABLE is True
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "renderers" in data
    assert "version" in data


def test_api_list_runs_empty() -> Any:
    """Test listing runs when none have been submitted."""
    from fastapi.testclient import TestClient

    # We need to clear the global _runs dict if we want a clean state,
    # but since it's module-level in app.py, isolations is tricky without reloading.
    # For baseline, we just check it returns a dict.
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_api_submit_run_invalid_payload() -> Any:
    """Test submitting a run with invalid JSON."""
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    response = client.post("/api/v1/run", json={"skip_steps": ["not_an_int"]})
    # FastAPI returns 422 Unprocessable Entity for schema validation errors
    assert response.status_code == 422


def test_api_submit_run_success() -> Any:
    """Test successful run submission."""
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    import api.app as api_app
    import pipeline.hasher

    orig_hasher = getattr(pipeline.hasher, "compute_run_hash", None)
    orig_execute = getattr(api_app, "_execute_pipeline", None)

    async def _complete_pipeline_immediately(*args: Any, **kwargs: Any) -> None:
        return None

    # Replace the reference directly for this endpoint wiring test.
    pipeline.hasher.compute_run_hash = lambda *args, **kwargs: "test_hash_123"
    api_app._execute_pipeline = _complete_pipeline_immediately

    try:
        response = client.post(
            "/api/v1/run",
            json={
                "target_dir": "input/gnn_files",
                "output_dir": "output",
                "skip_steps": [13],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["run_hash"] == "test_hash_123"
        assert data["status"] == "queued"
    finally:
        if orig_hasher:
            pipeline.hasher.compute_run_hash = orig_hasher
        if orig_execute:
            api_app._execute_pipeline = orig_execute


def test_api_submit_run_rejects_output_outside_repo() -> None:
    """The run API should validate output_dir as well as target_dir."""
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/v1/run",
        json={"target_dir": ".", "output_dir": "../outside-repo-output"},
    )

    assert response.status_code == 400
    assert "Output directory" in response.json()["detail"]


def test_api_submit_run_stores_normalized_output_dir(monkeypatch: Any) -> None:
    """Queued run metadata should preserve the caller-selected output dir."""
    from fastapi.testclient import TestClient

    import api.app as api_app
    import pipeline.hasher

    app = create_app()
    client = TestClient(app)
    run_hash = "output_dir_contract_hash"
    output_dir = "output/api_contract_test"

    orig_hasher = getattr(pipeline.hasher, "compute_run_hash", None)
    orig_execute = getattr(api_app, "_execute_pipeline", None)
    api_app._runs.pop(run_hash, None)

    async def _complete_pipeline_immediately(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pipeline.hasher, "compute_run_hash", lambda *a, **k: run_hash)
    monkeypatch.setattr(api_app, "_execute_pipeline", _complete_pipeline_immediately)

    try:
        response = client.post(
            "/api/v1/run",
            json={"target_dir": ".", "output_dir": output_dir},
        )

        assert response.status_code == 200
        stored_output = api_app._runs[run_hash]["request"]["output_dir"]
        assert Path(stored_output).resolve() == (
            Path(__file__).resolve().parents[3] / output_dir
        ).resolve()
    finally:
        api_app._runs.pop(run_hash, None)
        if orig_hasher:
            pipeline.hasher.compute_run_hash = orig_hasher
        if orig_execute:
            api_app._execute_pipeline = orig_execute


@pytest.mark.asyncio
async def test_job_processor_uses_requested_output_dir(monkeypatch: Any) -> None:
    """Async job execution should pass the job's output_dir to src/main.py."""
    from api import processor as job_mgr

    captured_cmd: list[str] = []

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def fake_create_subprocess_exec(*cmd: str, **kwargs: Any) -> FakeProcess:
        captured_cmd.extend(cmd)
        return FakeProcess()

    monkeypatch.setattr(
        job_mgr.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    requested_output = "output/job_contract_test"
    job_id = job_mgr.create_job(target_dir=".", output_dir=requested_output, steps=[3])
    try:
        await job_mgr.execute_job_async(job_id)
        output_arg = captured_cmd[captured_cmd.index("--output-dir") + 1]
        assert Path(output_arg).resolve() == (
            Path(__file__).resolve().parents[3] / requested_output
        ).resolve()
    finally:
        job_mgr._JOBS.pop(job_id, None)


def test_api_availability_flag() -> Any:
    """Verify that the availability flag matches reality."""
    try:
        import fastapi

        expected = True
    except ImportError:
        expected = False

    assert FASTAPI_AVAILABLE == expected
