#!/usr/bin/env python3
"""Regression coverage for the canonical FastAPI response contract."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api import processor as job_mgr
from api.responses import install_exception_handlers
from api.server import app as job_app


def _assert_envelope(payload: Any, *, status: str) -> dict[str, Any]:
    """Assert the exact top-level API contract and return the payload."""
    assert set(payload) == {"status", "data", "error", "meta"}
    assert isinstance(payload, dict)
    assert payload["status"] == status
    assert isinstance(payload["data"], dict)
    assert isinstance(payload["meta"], dict)
    assert "timestamp" in payload["meta"]
    if status == "success":
        assert payload["error"] is None
    else:
        assert set(payload["error"]) == {"code", "message", "details"}
    return payload


@pytest.fixture(autouse=True)
def disable_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep response-contract checks independent from global limiter history."""
    monkeypatch.setenv("GNN_RATE_LIMIT", "0")


@pytest.mark.parametrize("path", ["/api/v1/health", "/api/v1/jobs", "/api/v1/tools"])
def test_job_api_success_endpoints_use_envelope(path: str) -> None:
    response = TestClient(job_app).get(path)
    assert response.status_code == 200
    _assert_envelope(response.json(), status="success")


def test_job_api_framework_errors_use_envelope() -> None:
    client = TestClient(job_app)

    missing = client.get("/api/v1/jobs/not-a-job")
    assert missing.status_code == 404
    payload = _assert_envelope(missing.json(), status="error")
    assert payload["error"]["code"] == "not_found"

    invalid_limit = client.get("/api/v1/jobs?limit=0")
    assert invalid_limit.status_code == 422
    payload = _assert_envelope(invalid_limit.json(), status="error")
    assert payload["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "payload",
    [
        {"steps": [-1]},
        {"steps": [25]},
        {"steps": [3, 3]},
        {"steps": [3], "skip_steps": [3]},
        {"unexpected": True},
    ],
)
def test_job_api_rejects_invalid_process_requests(payload: dict[str, Any]) -> None:
    response = TestClient(job_app).post("/api/v1/process", json=payload)
    assert response.status_code == 422
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "validation_error"


def test_tool_endpoint_rejects_silently_ignored_kwargs() -> None:
    response = TestClient(job_app).post(
        "/api/v1/tools/3",
        json={"target_dir": ".", "kwargs": {"unsupported": True}},
    )
    assert response.status_code == 422
    envelope = _assert_envelope(response.json(), status="error")
    assert "not supported" in str(envelope["error"]["details"])


def test_processor_rejects_invalid_steps_before_registering_job() -> None:
    before = set(job_mgr._JOBS)
    with pytest.raises(ValueError, match="between 0 and 24"):
        job_mgr.create_job(target_dir=".", steps=[99])
    assert set(job_mgr._JOBS) == before


def test_processor_rejects_non_integer_steps_explicitly() -> None:
    with pytest.raises(ValueError, match="integers between 0 and 24"):
        job_mgr.create_job(target_dir=".", steps=[[3]])  # type: ignore[list-item]


def test_mcp_path_validation_rejects_blank_target() -> None:
    from api.mcp import gnn_submit_job_mcp

    result = gnn_submit_job_mcp("   ")
    assert result["status"] == "error"
    assert "must not be empty" in result["message"]


def test_nonstandard_http_error_code_still_uses_envelope() -> None:
    test_app = FastAPI()
    install_exception_handlers(test_app)

    @test_app.get("/custom")
    async def custom_error() -> None:
        raise HTTPException(status_code=499, detail={"reason": "custom"})

    response = TestClient(test_app).get("/custom")
    assert response.status_code == 499
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "http_error"
    assert envelope["error"]["message"] == "HTTP error"


def test_unexpected_exception_is_sanitized_and_enveloped() -> None:
    test_app = FastAPI()
    install_exception_handlers(test_app)

    @test_app.get("/explode")
    async def explode() -> None:
        raise RuntimeError("sensitive implementation detail")

    response = TestClient(test_app, raise_server_exceptions=False).get("/explode")
    assert response.status_code == 500
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "internal_error"
    assert "sensitive" not in envelope["error"]["message"]
