#!/usr/bin/env python3
"""
GNN Pipeline FastAPI Server.

Provides REST endpoints for pipeline job management and tool invocation.
No authentication — designed for local research use.

Run with:
    python -m api.server
    # or:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI and uvicorn are required for the GNN API server. "
        "Install with: uv sync --extra api"
    )

from api.models import (
    ProcessRequest, ToolRequest, JobResponse, JobStatusResponse,
    JobStatus, ToolsResponse, ToolInfo, HealthResponse
)
from api import processor as job_mgr

# Application metadata
app = FastAPI(
    title="GNN Pipeline API",
    description=(
        "REST interface for the Generalized Notation Notation (GNN) processing pipeline. "
        "Submit jobs, poll status, and invoke individual pipeline steps. "
        "No authentication required — research tool for local use."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for local browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Meta"])
async def health_check():
    """Check API health and get basic system info."""
    jobs = job_mgr.list_jobs()
    active = sum(1 for j in jobs if j.get("status") in ("pending", "running"))
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        pipeline_steps=len(job_mgr.PIPELINE_STEPS),
        active_jobs=active,
        timestamp=datetime.now()
    )


@app.post("/api/v1/process", response_model=JobResponse, tags=["Jobs"])
async def submit_process_job(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Submit a GNN pipeline processing job.

    Accepts a target directory and optional step selection.
    Returns a job ID for polling with GET /api/v1/jobs/{job_id}.
    """
    # Validate target directory exists
    target_path = Path(request.target_dir)
    if not target_path.exists():
        # Try relative to repo root
        repo_root = Path(__file__).parent.parent.parent
        target_path = repo_root / request.target_dir

    if not target_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Target directory not found: {request.target_dir}"
        )

    job_id = job_mgr.create_job(
        target_dir=str(target_path),
        steps=request.steps,
        skip_steps=request.skip_steps,
        verbose=request.verbose,
        strict=request.strict
    )

    # Launch async execution in background
    background_tasks.add_task(job_mgr.execute_job_async, job_id)

    job = job_mgr.get_job(job_id)
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.fromisoformat(job["created_at"]),
        steps_requested=request.steps,
        message=f"Job {job_id} queued. Poll GET /api/v1/jobs/{job_id} for status."
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Poll the status of a submitted pipeline job."""
    job = job_mgr.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    def _dt(s):
        return datetime.fromisoformat(s) if s else None

    return JobStatusResponse(
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
        output_dir=job.get("output_dir")
    )


@app.delete("/api/v1/jobs/{job_id}", tags=["Jobs"])
async def cancel_job(job_id: str):
    """Cancel a pending or running job."""
    success = job_mgr.cancel_job(job_id)
    if not success:
        job = job_mgr.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is already in terminal state: {job['status']}"
        )
    return {"message": f"Job {job_id} cancelled"}


@app.get("/api/v1/jobs", tags=["Jobs"])
async def list_jobs(limit: int = 20):
    """List recent pipeline jobs."""
    jobs = job_mgr.list_jobs(limit=limit)
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/api/v1/tools", response_model=ToolsResponse, tags=["Tools"])
async def list_tools():
    """List all available pipeline steps/tools."""
    tools = [ToolInfo(**t) for t in job_mgr.get_pipeline_tools()]
    return ToolsResponse(tools=tools, total=len(tools))


@app.post("/api/v1/tools/{step}", response_model=JobResponse, tags=["Tools"])
async def invoke_tool(step: int, request: ToolRequest, background_tasks: BackgroundTasks):
    """
    Invoke a single pipeline step as a job.

    Equivalent to submitting a process request with steps=[step].
    """
    if step not in job_mgr.PIPELINE_STEPS:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline step: {step}")

    target_path = Path(request.target_dir)
    if not target_path.exists():
        repo_root = Path(__file__).parent.parent.parent
        target_path = repo_root / request.target_dir

    if not target_path.exists():
        raise HTTPException(status_code=400, detail=f"Target directory not found: {request.target_dir}")

    job_id = job_mgr.create_job(
        target_dir=str(target_path),
        steps=[step],
        verbose=request.verbose
    )

    background_tasks.add_task(job_mgr.execute_job_async, job_id)

    step_name = job_mgr.PIPELINE_STEPS[step][0]
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now(),
        steps_requested=[step],
        message=f"Step {step} ({step_name}) queued as job {job_id}"
    )


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server."""
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GNN Pipeline API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()
    run_server(args.host, args.port, args.reload)
