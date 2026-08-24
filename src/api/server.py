#!/usr/bin/env python3
"""
GNN Pipeline FastAPI Server.

Provides REST endpoints for pipeline job management and tool invocation.
Optional API-key authentication is available through ``GNN_API_KEY``.

Run with:
    python -m api.server
    # or:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from datetime import datetime
from typing import Annotated, Any

logger = logging.getLogger(__name__)

try:
    import uvicorn
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as e:
    raise ImportError(
        "FastAPI and uvicorn are required for the GNN API server. "
        "Install with: uv sync --extra api"
    ) from e

from api import processor as job_mgr
from api.auth import api_key_middleware, require_secure_bind
from api.models import (
    HealthResponse,
    JobResponse,
    JobStatus,
    JobStatusResponse,
    ProcessRequest,
    ToolInfo,
    ToolRequest,
    ToolsResponse,
)
from api.path_utils import PathValidationError, resolve_repo_path
from api.rate_limit import rate_limit_middleware
from api.responses import APIEnvelope, install_exception_handlers, success_envelope

# Application metadata
app = FastAPI(
    title="GNN Pipeline API",
    description=(
        "REST interface for the Generalized Notation Notation (GNN) processing pipeline. "
        "Submit jobs, poll status, and invoke individual pipeline steps."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for local browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API-key auth: active only when GNN_API_KEY is set (see api/auth.py).
app.middleware("http")(api_key_middleware)

# Per-client rate limiting: active unless GNN_RATE_LIMIT=0 (see api/rate_limit.py).
# Registered after auth so it runs outermost, protecting the API even when
# authentication is disabled (e.g. localhost research use).
app.middleware("http")(rate_limit_middleware)
install_exception_handlers(app)


@app.get("/api/v1/health", response_model=APIEnvelope, tags=["Meta"])
async def health_check() -> APIEnvelope:
    """Check API health and get basic system info."""
    jobs = job_mgr.list_jobs()
    active = sum(1 for j in jobs if j.get("status") in ("pending", "running"))
    health = HealthResponse(
        status="healthy",
        version="1.0.0",
        pipeline_steps=len(job_mgr.PIPELINE_STEPS),
        active_jobs=active,
        timestamp=datetime.now(),
    )
    return success_envelope(health.model_dump(mode="json"), endpoint="health")


@app.post("/api/v1/process", response_model=APIEnvelope, tags=["Jobs"])
async def submit_process_job(
    request: ProcessRequest, background_tasks: BackgroundTasks
) -> APIEnvelope:
    """
    Submit a GNN pipeline processing job.

    Accepts a target directory and optional step selection.
    Returns a job ID for polling with GET /api/v1/jobs/{job_id}.
    """
    try:
        target_path = resolve_repo_path(
            request.target_dir,
            purpose="Target directory",
            must_exist=True,
        )
        output_path = resolve_repo_path(
            request.output_dir,
            purpose="Output directory",
            create=True,
        )
    except PathValidationError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    try:
        job_id = job_mgr.create_job(
            target_dir=str(target_path),
            output_dir=str(output_path),
            steps=request.steps,
            skip_steps=request.skip_steps,
            verbose=request.verbose,
            strict=request.strict,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    # Launch async execution in background
    background_tasks.add_task(job_mgr.execute_job_async, job_id)

    job = job_mgr.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=500, detail=f"Job {job_id} was not registered")
    response = JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.fromisoformat(job["created_at"]),
        steps_requested=request.steps,
        message=f"Job {job_id} queued. Poll GET /api/v1/jobs/{job_id} for status.",
    )
    return success_envelope(
        response.model_dump(mode="json"), endpoint="process", job_id=job_id
    )


@app.get("/api/v1/jobs/{job_id}", response_model=APIEnvelope, tags=["Jobs"])
async def get_job_status(job_id: str) -> APIEnvelope:
    """Poll the status of a submitted pipeline job."""
    job = job_mgr.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    def _dt(s: Any) -> Any:
        """Handle dt for internal callers."""
        return datetime.fromisoformat(s) if s else None

    response = JobStatusResponse(
        job_id=job["job_id"],
        status=JobStatus(job["status"]),
        created_at=_dt(job["created_at"]),
        started_at=_dt(job.get("started_at")),
        completed_at=_dt(job.get("completed_at")),
        progress_step=job.get("progress_step"),
        steps_completed=job.get("steps_completed", []),
        steps_failed=job.get("steps_failed", []),
        exit_code=job.get("exit_code"),
        error_message=job.get("error_message"),
        output_dir=job.get("output_dir"),
    )
    return success_envelope(
        response.model_dump(mode="json"), endpoint="job_status", job_id=job_id
    )


@app.delete("/api/v1/jobs/{job_id}", response_model=APIEnvelope, tags=["Jobs"])
async def cancel_job(job_id: str) -> APIEnvelope:
    """Cancel a pending or running job."""
    success = job_mgr.cancel_job(job_id)
    if not success:
        job = job_mgr.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is already in terminal state: {job['status']}",
        )
    return success_envelope(
        {"message": f"Job {job_id} cancelled"},
        endpoint="cancel_job",
        job_id=job_id,
    )


@app.get("/api/v1/jobs", response_model=APIEnvelope, tags=["Jobs"])
async def list_jobs(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> APIEnvelope:
    """List recent pipeline jobs."""
    jobs = job_mgr.list_jobs(limit=limit)
    return success_envelope(
        {"jobs": jobs, "total": len(jobs)}, endpoint="list_jobs", limit=limit
    )


@app.get("/api/v1/tools", response_model=APIEnvelope, tags=["Tools"])
async def list_tools() -> APIEnvelope:
    """List all available pipeline steps/tools."""
    tools = [ToolInfo(**t) for t in job_mgr.get_pipeline_tools()]
    response = ToolsResponse(tools=tools, total=len(tools))
    return success_envelope(response.model_dump(mode="json"), endpoint="list_tools")


@app.post("/api/v1/tools/{step}", response_model=APIEnvelope, tags=["Tools"])
async def invoke_tool(
    step: int, request: ToolRequest, background_tasks: BackgroundTasks
) -> APIEnvelope:
    """
    Invoke a single pipeline step as a job.

    Equivalent to submitting a process request with steps=[step].
    """
    if step not in job_mgr.PIPELINE_STEPS:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline step: {step}")

    try:
        target_path = resolve_repo_path(
            request.target_dir,
            purpose="Target directory",
            must_exist=True,
        )
        output_path = resolve_repo_path(
            request.output_dir,
            purpose="Output directory",
            create=True,
        )
    except PathValidationError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    try:
        job_id = job_mgr.create_job(
            target_dir=str(target_path),
            output_dir=str(output_path),
            steps=[step],
            verbose=request.verbose,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    background_tasks.add_task(job_mgr.execute_job_async, job_id)

    job = job_mgr.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=500, detail=f"Job {job_id} was not registered")
    step_name = job_mgr.PIPELINE_STEPS[step][0]
    response = JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.fromisoformat(job["created_at"]),
        steps_requested=[step],
        message=f"Step {step} ({step_name}) queued as job {job_id}",
    )
    return success_envelope(
        response.model_dump(mode="json"), endpoint="invoke_tool", job_id=job_id
    )


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> Any:
    """Start the API server."""
    if not require_secure_bind(host):
        raise RuntimeError(
            f"Refusing to bind API server to non-loopback address {host!r} "
            "without authentication. Set GNN_API_KEY to enable API-key auth, "
            "or GNN_ALLOW_INSECURE_BIND=1 to explicitly accept the risk."
        )
    if host not in ("127.0.0.1", "localhost"):
        logger.warning(
            "Binding to non-loopback address %s with no authentication — "
            "ensure network-level access control is in place",
            host,
        )
    uvicorn.run("api.server:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GNN Pipeline API Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload", action="store_true", help="Auto-reload on code changes"
    )
    args = parser.parse_args()
    run_server(args.host, args.port, args.reload)
