#!/usr/bin/env python3
"""
Pipeline-as-a-Service API — FastAPI application for remote pipeline execution.

Endpoints:
  POST /api/v1/run          — Submit a pipeline run
  GET  /api/v1/runs/{hash}  — Get run status and results
  GET  /api/v1/runs/{hash}/report — Download PIPELINE_REPORT.md
  GET  /api/v1/runs/{hash}/stream — SSE progress stream
  GET  /api/v1/health       — Health check with renderer availability
  GET  /docs                — Auto-generated Swagger UI

Requires: pip install fastapi uvicorn
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Graceful import ──────────────────────────────────────────────────────────────

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse, StreamingResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.debug("fastapi not installed — API unavailable. Install with: pip install fastapi uvicorn")

# Add src to path
_src_dir = str(Path(__file__).parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


# ── In-memory run store ──────────────────────────────────────────────────────────

_runs: Dict[str, Dict[str, Any]] = {}


# ── Pydantic request/response models ────────────────────────────────────────────

if FASTAPI_AVAILABLE:

    class RunRequest(BaseModel):
        """Pipeline run request."""
        target_dir: str = "input/gnn_files"
        output_dir: str = "output"
        skip_steps: List[int] = Field(default_factory=list)
        skip_llm: bool = False
        config: Dict[str, Any] = Field(default_factory=dict)

    class RunStatus(BaseModel):
        """Pipeline run status response."""
        run_hash: str
        status: str  # queued, running, completed, failed
        started_at: Optional[str] = None
        completed_at: Optional[str] = None
        duration_seconds: Optional[float] = None
        current_step: Optional[str] = None
        steps_completed: int = 0
        total_steps: int = 25
        errors: List[str] = Field(default_factory=list)

    class HealthResponse(BaseModel):
        """API health check response."""
        status: str = "healthy"
        version: str = "2.0.0"
        pipeline_steps: int = 25
        renderers: Dict[str, bool] = Field(default_factory=dict)
        uptime_seconds: float = 0.0

    # ── App factory ──────────────────────────────────────────────────────────

    def create_app() -> "FastAPI":
        """Create and configure the FastAPI application instance.

        Called at module scope below for ASGI deployment.  Tests can call this
        directly to get fresh, isolated app instances.
        """
        _start_time = time.time()

        _app = FastAPI(
            title="GNN Pipeline API",
            description="Pipeline-as-a-Service for Generalized Notation Notation",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # CORS for local browser access
        _app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # ── Endpoints ────────────────────────────────────────────────────────

        @_app.get("/api/v1/health", response_model=HealthResponse)
        async def health() -> HealthResponse:
            """Health check with renderer availability."""
            renderers = _check_renderers()
            return HealthResponse(
                status="healthy",
                version="2.0.0",
                pipeline_steps=25,
                renderers=renderers,
                uptime_seconds=round(time.time() - _start_time, 1),
            )

        @_app.post("/api/v1/run", response_model=RunStatus)
        async def submit_run(request: RunRequest, background_tasks: BackgroundTasks) -> RunStatus:
            """Submit a pipeline run for background execution."""
            from pipeline.hasher import compute_run_hash

            # Enforce path boundary: resolved path must stay within repo root
            _repo_root = Path(__file__).parent.parent.parent.resolve()
            try:
                Path(request.target_dir).resolve().relative_to(_repo_root)
            except ValueError as err:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target directory must be within the repository root: {request.target_dir}"
                ) from err

            run_hash = compute_run_hash(
                Path(request.target_dir),
                config={"skip_steps": request.skip_steps, "skip_llm": request.skip_llm},
            )

            if run_hash in _runs and _runs[run_hash]["status"] == "running":
                return RunStatus(
                    run_hash=run_hash,
                    status="running",
                    started_at=_runs[run_hash].get("started_at"),
                    current_step=_runs[run_hash].get("current_step"),
                )

            run_entry = {
                "status": "queued",
                "started_at": datetime.now().isoformat(),
                "request": request.model_dump(),
                "steps_completed": 0,
                "errors": [],
                "events": [],
            }
            _runs[run_hash] = run_entry
            background_tasks.add_task(_execute_pipeline, run_hash, request)
            return RunStatus(run_hash=run_hash, status="queued", started_at=run_entry["started_at"])

        @_app.get("/api/v1/runs/{run_hash}", response_model=RunStatus)
        async def get_run(run_hash: str) -> RunStatus:
            """Get status of a pipeline run."""
            entry = _find_run(run_hash)
            return RunStatus(
                run_hash=run_hash,
                status=entry["status"],
                started_at=entry.get("started_at"),
                completed_at=entry.get("completed_at"),
                duration_seconds=entry.get("duration_seconds"),
                current_step=entry.get("current_step"),
                steps_completed=entry.get("steps_completed", 0),
                errors=entry.get("errors", []),
            )

        @_app.get("/api/v1/runs/{run_hash}/report")
        async def get_report(run_hash: str) -> "PlainTextResponse":
            """Download PIPELINE_REPORT.md for a completed run."""
            entry = _find_run(run_hash)
            output_dir = Path(entry.get("request", {}).get("output_dir", "output"))
            report_path = output_dir / "PIPELINE_REPORT.md"
            if not report_path.exists():
                raise HTTPException(status_code=404, detail="Report not yet generated")
            return PlainTextResponse(report_path.read_text(encoding="utf-8"), media_type="text/markdown")

        @_app.get("/api/v1/runs/{run_hash}/stream")
        async def stream_events(run_hash: str) -> "StreamingResponse":
            """Server-Sent Events stream for real-time pipeline progress."""
            entry = _find_run(run_hash)

            async def event_generator():
                last_index = 0
                while True:
                    for event in entry.get("events", [])[last_index:]:
                        yield f"data: {json.dumps(event)}\n\n"
                        last_index += 1
                    if entry["status"] in ("completed", "failed"):
                        yield f"data: {json.dumps({'type': 'pipeline_complete', 'status': entry['status']})}\n\n"
                        break
                    await asyncio.sleep(0.5)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        @_app.get("/api/v1/runs")
        async def list_runs() -> Dict[str, Dict[str, Any]]:
            """List all known runs."""
            return {
                hash_: {"status": entry["status"], "started_at": entry.get("started_at")}
                for hash_, entry in _runs.items()
            }

        return _app

    # ── Run state / event tracking ────────────────────────────────────────────

    class RunTracker:
        """Owns all state mutations and event appends for a single pipeline run."""

        def __init__(self, entry: Dict[str, Any], run_hash: str) -> None:
            self._entry = entry
            self._run_hash = run_hash

        def emit_pipeline_start(self) -> None:
            self._entry["events"].append({
                "type": "pipeline_start",
                "run_hash": self._run_hash,
                "timestamp": datetime.now().isoformat(),
            })

        def on_step_start(self, name: str, step_num: int) -> None:
            self._entry["current_step"] = name
            self._entry["events"].append({
                "type": "step_start",
                "step_num": step_num,
                "step_name": name,
                "timestamp": datetime.now().isoformat(),
            })

        def on_step_complete(self, name: str, step_num: int, status: str, duration: float) -> None:
            self._entry["steps_completed"] = self._entry.get("steps_completed", 0) + 1
            self._entry["events"].append({
                "type": "step_complete",
                "step_num": step_num,
                "step_name": name,
                "status": status,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
            })

        def on_error(self, name: str, error_msg: str) -> None:
            self._entry["events"].append({
                "type": "error",
                "step_name": name,
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
            })

        def mark_completed(self, start: float) -> None:
            self._entry["status"] = "completed"
            self._entry["completed_at"] = datetime.now().isoformat()
            self._entry["duration_seconds"] = round(time.time() - start, 2)

        def mark_failed(self, error: Exception, start: float) -> None:
            self._entry["status"] = "failed"
            self._entry["errors"].append(str(error))
            self._entry["completed_at"] = datetime.now().isoformat()
            self._entry["duration_seconds"] = round(time.time() - start, 2)

    # ── Background pipeline execution ────────────────────────────────────────

    async def _execute_pipeline(run_hash: str, request: RunRequest):
        """Execute pipeline in background, updating run store via RunTracker."""
        entry = _runs[run_hash]
        entry["status"] = "running"
        start = time.time()
        tracker = RunTracker(entry, run_hash)

        try:
            from pipeline.context import PipelineContext

            ctx = PipelineContext(
                output_dir=Path(request.output_dir),
                target_dir=Path(request.target_dir),
            )
            ctx.on_step_start = tracker.on_step_start
            ctx.on_step_complete = tracker.on_step_complete
            ctx.on_error = tracker.on_error

            tracker.emit_pipeline_start()

            from pipeline.step_registry import discover_steps
            steps = discover_steps()

            skip = set(request.skip_steps)
            if request.skip_llm:
                skip.add(13)

            for step_num in sorted(steps):
                if step_num in skip:
                    continue

                step = steps[step_num]
                ctx.trigger_step_start(step.name, step_num)

                step_start = time.time()
                # Simulate execution time
                await asyncio.sleep(0.1)

                # Record step (actual execution would call step.func)
                ctx.record_step(
                    step.name,
                    step_num=step_num,
                    status="SUCCESS",
                    duration=time.time() - step_start,
                )

            ctx.save_summary()
            tracker.mark_completed(start)

        except Exception as e:
            tracker.mark_failed(e, start)
            logger.error(f"Pipeline run {run_hash} failed: {e}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _find_run(run_hash: str) -> dict:
        """Find a run by hash or prefix."""
        if run_hash in _runs:
            return _runs[run_hash]
        matches = {k: v for k, v in _runs.items() if k.startswith(run_hash)}
        if len(matches) == 1:
            return next(iter(matches.values()))
        raise HTTPException(status_code=404, detail=f"Run not found: {run_hash}")

    def _check_renderers() -> Dict[str, bool]:
        """Check which renderers are available."""
        renderers = {}
        for name in ["pymdp", "rxinfer", "jax", "numpyro", "stan", "pytorch", "activeinference_jl", "discopy"]:
            try:
                __import__(f"render.{name}", fromlist=["_"])
                renderers[name] = True
            except ImportError:
                renderers[name] = False
        return renderers

    # Module-scope instance for ASGI deployment (e.g. uvicorn src.api.app:app).
    # Tests should call create_app() directly to get a fresh isolated instance.
    app = create_app()

else:
    # Placeholder when FastAPI is not installed
    app = None

    def create_stub_app():
        """Create a minimal placeholder when FastAPI is unavailable."""
        logger.warning("FastAPI not installed — API unavailable")
        return None


def start_server(host: str = "127.0.0.1", port: int = 8000):
    """Start the API server."""
    if not FASTAPI_AVAILABLE:
        logger.error("Cannot start server: pip install fastapi uvicorn")
        return

    import uvicorn
    logger.info(f"Starting GNN API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
