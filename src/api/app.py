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

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

FASTAPI_AVAILABLE = True

# Add src to path
_src_dir = str(Path(__file__).parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from api.auth import api_key_middleware, require_secure_bind
from api.path_utils import PathValidationError, resolve_repo_path  # noqa: E402,I001
from api.rate_limit import rate_limit_middleware
from api.responses import APIEnvelope, install_exception_handlers, success_envelope

# ── In-memory run store ──────────────────────────────────────────────────────────

_runs: Dict[str, Dict[str, Any]] = {}


# ── Pydantic request/response models ────────────────────────────────────────────

if FASTAPI_AVAILABLE:

    class RunRequest(BaseModel):
        """Pipeline run request."""

        target_dir: str = Field(default="input/gnn_files", min_length=1)
        output_dir: str = Field(default="output", min_length=1)
        skip_steps: List[int] = Field(default_factory=list)
        skip_llm: bool = False
        config: Dict[str, Any] = Field(
            default_factory=dict,
            description="Reserved for future run configuration; currently must be empty",
        )

        model_config = ConfigDict(extra="forbid")

        @field_validator("skip_steps")
        @classmethod
        def validate_skip_steps(cls, values: List[int]) -> List[int]:
            """Require unique pipeline step numbers in the supported range."""
            invalid = sorted({step for step in values if step < 0 or step > 24})
            if invalid:
                raise ValueError(f"Pipeline steps must be between 0 and 24: {invalid}")
            if len(values) != len(set(values)):
                raise ValueError("skip_steps must not contain duplicates")
            return values

        @field_validator("config")
        @classmethod
        def reject_unsupported_config(cls, value: Dict[str, Any]) -> Dict[str, Any]:
            """Reject configuration that the background runner cannot honor."""
            if value:
                raise ValueError("Custom run config is not supported by this endpoint")
            return value

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

        # Optional API-key auth: active only when GNN_API_KEY is set.
        _app.middleware("http")(api_key_middleware)

        # Per-client rate limiting: active unless GNN_RATE_LIMIT=0.
        # Registered after auth so it runs outermost, protecting the API even
        # when authentication is disabled (e.g. localhost research use).
        _app.middleware("http")(rate_limit_middleware)
        install_exception_handlers(_app)

        # ── Endpoints ────────────────────────────────────────────────────────

        @_app.get("/api/v1/health", response_model=APIEnvelope)
        async def health() -> APIEnvelope:
            """Health check with renderer availability."""
            renderers = _check_renderers()
            response = HealthResponse(
                status="healthy",
                version="2.0.0",
                pipeline_steps=25,
                renderers=renderers,
                uptime_seconds=round(time.time() - _start_time, 1),
            )
            return success_envelope(response.model_dump(mode="json"), endpoint="health")

        @_app.post("/api/v1/run", response_model=APIEnvelope)
        async def submit_run(
            request: RunRequest, background_tasks: BackgroundTasks
        ) -> APIEnvelope:
            """Submit a pipeline run for background execution."""
            from pipeline.hasher import compute_run_hash

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
                raise HTTPException(
                    status_code=400,
                    detail=str(err),
                ) from err

            run_hash = compute_run_hash(
                target_path,
                config={"skip_steps": request.skip_steps, "skip_llm": request.skip_llm},
            )
            normalized_request = request.model_copy(
                update={
                    "target_dir": str(target_path),
                    "output_dir": str(output_path),
                }
            )

            if run_hash in _runs and _runs[run_hash]["status"] == "running":
                response = RunStatus(
                    run_hash=run_hash,
                    status="running",
                    started_at=_runs[run_hash].get("started_at"),
                    current_step=_runs[run_hash].get("current_step"),
                )
                return success_envelope(
                    response.model_dump(mode="json"),
                    endpoint="submit_run",
                    run_hash=run_hash,
                    deduplicated=True,
                )

            run_entry: dict[str, Any] = {
                "status": "queued",
                "started_at": datetime.now().isoformat(),
                "request": normalized_request.model_dump(),
                "steps_completed": 0,
                "total_steps": 25
                - len(set(request.skip_steps) | ({13} if request.skip_llm else set())),
                "errors": [],
                "events": [],
            }
            _runs[run_hash] = run_entry
            background_tasks.add_task(_execute_pipeline, run_hash, normalized_request)
            response = RunStatus(
                run_hash=run_hash, status="queued", started_at=run_entry["started_at"]
            )
            return success_envelope(
                response.model_dump(mode="json"),
                endpoint="submit_run",
                run_hash=run_hash,
                deduplicated=False,
            )

        @_app.get("/api/v1/runs/{run_hash}", response_model=APIEnvelope)
        async def get_run(run_hash: str) -> APIEnvelope:
            """Get status of a pipeline run."""
            entry = _find_run(run_hash)
            response = RunStatus(
                run_hash=run_hash,
                status=entry["status"],
                started_at=entry.get("started_at"),
                completed_at=entry.get("completed_at"),
                duration_seconds=entry.get("duration_seconds"),
                current_step=entry.get("current_step"),
                steps_completed=entry.get("steps_completed", 0),
                total_steps=entry.get("total_steps", 25),
                errors=entry.get("errors", []),
            )
            return success_envelope(
                response.model_dump(mode="json"),
                endpoint="get_run",
                run_hash=run_hash,
            )

        @_app.get("/api/v1/runs/{run_hash}/report")
        async def get_report(run_hash: str) -> "PlainTextResponse":
            """Download PIPELINE_REPORT.md for a completed run."""
            entry = _find_run(run_hash)
            output_dir = Path(entry.get("request", {}).get("output_dir", "output"))
            report_path = output_dir / "PIPELINE_REPORT.md"
            if not report_path.exists():
                raise HTTPException(status_code=404, detail="Report not yet generated")
            return PlainTextResponse(
                report_path.read_text(encoding="utf-8"), media_type="text/markdown"
            )

        @_app.get("/api/v1/runs/{run_hash}/stream")
        async def stream_events(run_hash: str) -> "StreamingResponse":
            """Server-Sent Events stream for real-time pipeline progress."""
            entry = _find_run(run_hash)

            async def event_generator() -> Any:
                """Provide event generator behavior."""
                last_index = 0
                while True:
                    for event in entry.get("events", [])[last_index:]:
                        payload = success_envelope(event, endpoint="stream_events")
                        yield f"data: {payload.model_dump_json()}\n\n"
                        last_index += 1
                    if entry["status"] in ("completed", "failed"):
                        payload = success_envelope(
                            {
                                "type": "pipeline_complete",
                                "run_status": entry["status"],
                            },
                            endpoint="stream_events",
                        )
                        yield f"data: {payload.model_dump_json()}\n\n"
                        break
                    await asyncio.sleep(0.5)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        @_app.get("/api/v1/runs", response_model=APIEnvelope)
        async def list_runs() -> APIEnvelope:
            """List all known runs."""
            runs = {
                hash_: {
                    "status": entry["status"],
                    "started_at": entry.get("started_at"),
                }
                for hash_, entry in _runs.items()
            }
            return success_envelope(
                {"runs": runs, "total": len(runs)}, endpoint="list_runs"
            )

        return _app

    # ── Run state / event tracking ────────────────────────────────────────────

    class RunTracker:
        """Owns all state mutations and event appends for a single pipeline run."""

        def __init__(self, entry: Dict[str, Any], run_hash: str) -> None:
            """Initialize the instance."""
            self._entry = entry
            self._run_hash = run_hash

        def emit_pipeline_start(self) -> None:
            """Emit pipeline start."""
            self._entry["events"].append(
                {
                    "type": "pipeline_start",
                    "run_hash": self._run_hash,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        def on_step_start(self, name: str, step_num: int) -> None:
            """Provide on step start behavior."""
            self._entry["current_step"] = name
            self._entry["events"].append(
                {
                    "type": "step_start",
                    "step_num": step_num,
                    "step_name": name,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        def on_step_complete(
            self, name: str, step_num: int, status: str, duration: float
        ) -> None:
            """Provide on step complete behavior."""
            self._entry["steps_completed"] = self._entry.get("steps_completed", 0) + 1
            self._entry["events"].append(
                {
                    "type": "step_complete",
                    "step_num": step_num,
                    "step_name": name,
                    "status": status,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        def on_error(self, name: str, error_msg: str) -> None:
            """Provide on error behavior."""
            self._entry["events"].append(
                {
                    "type": "error",
                    "step_name": name,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        def mark_completed(self, start: float) -> None:
            """Mark completed."""
            self._entry["status"] = "completed"
            self._entry["completed_at"] = datetime.now().isoformat()
            self._entry["duration_seconds"] = round(time.time() - start, 2)

        def mark_failed(self, error: Exception, start: float) -> None:
            """Mark failed."""
            self._entry["status"] = "failed"
            self._entry["errors"].append(str(error))
            self._entry["completed_at"] = datetime.now().isoformat()
            self._entry["duration_seconds"] = round(time.time() - start, 2)

    # ── Background pipeline execution ────────────────────────────────────────

    async def _execute_pipeline(run_hash: str, request: RunRequest) -> Any:
        """Execute the real pipeline orchestrator in a worker thread."""
        entry = _runs[run_hash]
        entry["status"] = "running"
        start = time.time()
        tracker = RunTracker(entry, run_hash)
        tracker.emit_pipeline_start()

        try:
            skipped_steps = set(request.skip_steps)
            if request.skip_llm:
                skipped_steps.add(13)
            repo_root = Path(__file__).resolve().parents[2]
            command = [
                sys.executable,
                str(repo_root / "src" / "main.py"),
                "--target-dir",
                request.target_dir,
                "--output-dir",
                request.output_dir,
            ]
            if skipped_steps:
                command.extend(
                    [
                        "--skip-steps",
                        ",".join(str(step) for step in sorted(skipped_steps)),
                    ]
                )
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(repo_root),
            )
            entry["process_id"] = getattr(process, "pid", None)
            _stdout, stderr = await process.communicate()
            if process.returncode is None:
                raise RuntimeError("Pipeline process ended without an exit code")
            exit_code = int(process.returncode)
            entry["exit_code"] = exit_code
            _load_pipeline_summary_events(entry, tracker, Path(request.output_dir))
            entry["current_step"] = None

            if exit_code in (0, 2):
                tracker.mark_completed(start)
                if exit_code == 2:
                    entry["events"].append(
                        {
                            "type": "pipeline_warning",
                            "message": "Pipeline completed with warnings",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
            else:
                stderr_tail = stderr.decode("utf-8", errors="replace")[-1000:]
                if stderr_tail:
                    logger.error(
                        "Pipeline run %s exited with code %d: %s",
                        run_hash,
                        exit_code,
                        stderr_tail,
                    )
                tracker.mark_failed(
                    RuntimeError(f"Pipeline exited with code {exit_code}"), start
                )

        except Exception as e:
            tracker.mark_failed(RuntimeError("Pipeline execution failed"), start)
            logger.exception("Pipeline run %s failed: %s", run_hash, e)

    def _load_pipeline_summary_events(
        entry: Dict[str, Any], tracker: RunTracker, output_dir: Path
    ) -> None:
        """Load completed step events from the canonical pipeline summary."""
        summary_path = (
            output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"
        )
        if not summary_path.is_file():
            logger.warning("Pipeline summary not found after API run: %s", summary_path)
            return
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read pipeline summary %s: %s", summary_path, exc)
            return

        steps = summary.get("steps", [])
        if not isinstance(steps, list):
            logger.warning("Pipeline summary has non-list steps: %s", summary_path)
            return
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            script_name = str(step.get("script_name") or step.get("name") or "step")
            raw_step_num = step.get("step_num")
            if raw_step_num is None:
                prefix = script_name.split("_", 1)[0]
                raw_step_num = prefix if prefix.isdigit() else index
            try:
                step_num = int(raw_step_num)
            except (TypeError, ValueError):
                step_num = index
            status = str(step.get("status", "UNKNOWN"))
            try:
                duration = float(step.get("duration_seconds", 0.0) or 0.0)
            except (TypeError, ValueError):
                duration = 0.0
            tracker.on_step_start(script_name, step_num)
            tracker.on_step_complete(script_name, step_num, status, duration)
        entry["total_steps"] = len(steps)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _find_run(run_hash: str) -> dict:
        """Find a run by hash or prefix."""
        if run_hash in _runs:
            return _runs[run_hash]
        matches = {k: v for k, v in _runs.items() if k.startswith(run_hash)}
        if len(matches) == 1:
            return next(iter(matches.values()))
        if len(matches) > 1:
            raise HTTPException(
                status_code=409,
                detail=f"Run hash prefix is ambiguous: {run_hash}",
            )
        raise HTTPException(status_code=404, detail=f"Run not found: {run_hash}")

    def _check_renderers() -> Dict[str, bool]:
        """Check which renderers are available."""
        renderers: dict[Any, Any] = {}
        for name in [
            "pymdp",
            "rxinfer",
            "jax",
            "numpyro",
            "stan",
            "pytorch",
            "activeinference_jl",
            "discopy",
        ]:
            try:
                __import__(f"render.{name}", fromlist=["_"])
                renderers[name] = True
            except Exception as exc:
                logger.debug("Renderer %s is unavailable: %s", name, exc)
                renderers[name] = False
        return renderers

    # Module-scope instance for ASGI deployment (e.g. uvicorn src.api.app:app).
    # Tests should call create_app() directly to get a fresh isolated instance.
    app: FastAPI | None = create_app()

else:
    app = None

    def create_app() -> "FastAPI":
        """Report that the API server requires FastAPI."""
        raise RuntimeError("FastAPI is required to create the API application")


def start_server(host: str = "127.0.0.1", port: int = 8000) -> Any:
    """Start the API server."""
    if not FASTAPI_AVAILABLE:
        logger.error("Cannot start server: pip install fastapi uvicorn")
        return

    if not require_secure_bind(host):
        raise RuntimeError(
            f"Refusing to bind API server to non-loopback address {host!r} "
            "without authentication. Set GNN_API_KEY to enable API-key auth, "
            "or GNN_ALLOW_INSECURE_BIND=1 to explicitly accept the risk."
        )

    import uvicorn

    if app is None:
        raise RuntimeError("API application was not initialized")
    logger.info(f"Starting GNN API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
