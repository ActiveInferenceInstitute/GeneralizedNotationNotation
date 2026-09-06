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
import os
import sys
import time
import uuid
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

from gnn.api.auth import api_key_middleware, require_secure_bind
from gnn.api.models import RunHealthResponse, RunRequest, RunStatus  # noqa: E402,I001
from gnn.api.path_utils import PathValidationError, resolve_repo_path  # noqa: E402,I001
from gnn.api.pipeline_runner import (  # noqa: E402,I001
    build_pipeline_command,
    normalize_summary_steps,
    pipeline_exit_succeeded,
    read_pipeline_summary,
)
from gnn.api.rate_limit import rate_limit_middleware
from gnn.api.responses import APIEnvelope, install_exception_handlers, success_envelope

# ── In-memory run store ──────────────────────────────────────────────────────────

_runs: Dict[str, Dict[str, Any]] = {}


if FASTAPI_AVAILABLE:
    # ── App factory ──────────────────────────────────────────────────────────

    def create_app(
        runs_store: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "FastAPI":
        """Create and configure the FastAPI application instance.

        Called at module scope below for ASGI deployment.  Tests can call this
        directly to get fresh, isolated app instances.
        """
        _start_time = time.time()
        runs = runs_store if runs_store is not None else _runs

        _app = FastAPI(
            title="GNN Pipeline API",
            description="Pipeline-as-a-Service for Generalized Notation Notation",
            version="3.2.0",
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
            response = RunHealthResponse(
                status="healthy",
                version="3.2.0",
                pipeline_steps=25,
                renderers=_renderer_availability(),
                uptime_seconds=round(time.time() - _start_time, 1),
            )
            return success_envelope(response.model_dump(mode="json"), endpoint="health")

        @_app.post("/api/v1/run", response_model=APIEnvelope)
        async def submit_run(
            request: RunRequest, background_tasks: BackgroundTasks
        ) -> APIEnvelope:
            """Submit a pipeline run for background execution."""
            from gnn.pipeline.hasher import compute_run_hash

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
                config={
                    "skip_steps": sorted(
                        set(request.skip_steps) | ({13} if request.skip_llm else set())
                    ),
                    "strict": request.strict,
                    "output_dir": str(output_path),
                },
            )
            normalized_request = request.model_copy(
                update={
                    "target_dir": str(target_path),
                    "output_dir": str(output_path),
                }
            )

            if run_hash in runs and runs[run_hash]["status"] in {"queued", "running"}:
                response = RunStatus(
                    run_hash=run_hash,
                    status=runs[run_hash]["status"],
                    started_at=runs[run_hash].get("started_at"),
                    current_step=runs[run_hash].get("current_step"),
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
            runs[run_hash] = run_entry
            background_tasks.add_task(
                _execute_pipeline, run_hash, normalized_request, runs
            )
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
            entry = runs[_find_run_key(run_hash, runs)]
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
            entry = runs[_find_run_key(run_hash, runs)]
            output_dir = Path(entry.get("request", {}).get("output_dir", "output"))
            report_path = output_dir / "PIPELINE_REPORT.md"
            if not report_path.exists():
                raise HTTPException(status_code=404, detail="Report not yet generated")
            return PlainTextResponse(
                report_path.read_text(encoding="utf-8"), media_type="text/markdown"
            )

        @_app.delete("/api/v1/runs/{run_hash}", response_model=APIEnvelope)
        async def delete_run(run_hash: str) -> APIEnvelope:
            """Remove a run record from the in-memory store (housekeeping).

            Additive endpoint for clearing completed/failed runs from the
            in-memory store without restarting the server. Unknown hashes
            return 404; ambiguous prefixes return 409.
            """
            key = _find_run_key(run_hash, runs)
            if runs[key]["status"] in {"queued", "running"}:
                raise HTTPException(
                    status_code=409, detail="Active runs cannot be deleted"
                )
            removed = runs.pop(key, None)
            return success_envelope(
                {"deleted": key, "existed": removed is not None},
                endpoint="delete_run",
                run_hash=key,
            )

        @_app.get("/api/v1/runs/{run_hash}/stream")
        async def stream_events(run_hash: str) -> "StreamingResponse":
            """Server-Sent Events stream for real-time pipeline progress."""
            entry = runs[_find_run_key(run_hash, runs)]

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
            summary = {
                hash_: {
                    "status": entry["status"],
                    "started_at": entry.get("started_at"),
                }
                for hash_, entry in runs.items()
            }
            return success_envelope(
                {"runs": summary, "total": len(summary)}, endpoint="list_runs"
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

    async def _execute_pipeline(
        run_hash: str,
        request: RunRequest,
        runs_store: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Any:
        """Execute the real pipeline orchestrator in a worker thread.

        ``runs_store`` defaults to the module-level ``_runs`` store; the
        ``create_app`` factory passes its own injected store so background
        runs land in the same dict the endpoints read.
        """
        runs = runs_store if runs_store is not None else _runs
        entry = runs[run_hash]
        entry["status"] = "running"
        start = time.time()
        tracker = RunTracker(entry, run_hash)
        tracker.emit_pipeline_start()

        try:
            skipped_steps = set(request.skip_steps)
            if request.skip_llm:
                skipped_steps.add(13)
            repo_root = Path(__file__).resolve().parents[3]
            command = build_pipeline_command(
                request.target_dir,
                request.output_dir,
                skip_steps=sorted(skipped_steps),
                strict=request.strict,
                repo_root=repo_root,
            )
            invocation_id = uuid.uuid4().hex
            entry["run_id"] = invocation_id
            invocation_start_ns = time.time_ns()
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(repo_root),
                env={**os.environ, "GNN_RUN_ID": invocation_id},
            )
            entry["process_id"] = getattr(process, "pid", None)
            _stdout, stderr = await process.communicate()
            if process.returncode is None:
                raise RuntimeError("Pipeline process ended without an exit code")
            exit_code = int(process.returncode)
            entry["exit_code"] = exit_code
            _load_pipeline_summary_events(
                entry,
                tracker,
                Path(request.output_dir),
                not_before_ns=invocation_start_ns,
                expected_run_id=invocation_id,
            )
            entry["current_step"] = None

            if pipeline_exit_succeeded(exit_code, strict=request.strict):
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
        entry: Dict[str, Any],
        tracker: RunTracker,
        output_dir: Path,
        *,
        not_before_ns: Optional[int] = None,
        expected_run_id: Optional[str] = None,
    ) -> None:
        """Load completed step events from the canonical pipeline summary."""
        steps = read_pipeline_summary(
            output_dir, not_before_ns=not_before_ns, expected_run_id=expected_run_id
        )
        if steps is None:
            return
        for outcome in normalize_summary_steps(steps):
            tracker.on_step_start(outcome.script_name, outcome.step_num)
            tracker.on_step_complete(
                outcome.script_name,
                outcome.step_num,
                outcome.status,
                outcome.duration_seconds,
            )
        entry["total_steps"] = len(steps)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _find_run_key(run_hash: str, runs: Dict[str, Dict[str, Any]]) -> str:
        """Return the store key for a run hash or an unambiguous prefix.

        Raises ``HTTPException`` (404 not found / 409 ambiguous prefix) so the
        endpoints can resolve and look up in one step.
        """
        if run_hash in runs:
            return run_hash
        matches = [key for key in runs if key.startswith(run_hash)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise HTTPException(
                status_code=409,
                detail=f"Run hash prefix is ambiguous: {run_hash}",
            )
        raise HTTPException(status_code=404, detail=f"Run not found: {run_hash}")

    def _renderer_availability() -> Dict[str, bool]:
        """Report renderer availability from the canonical ``render.health`` registry.

        Falls back to an empty map (the endpoint stays healthy) when the
        canonical registry cannot be imported, so a missing optional
        dependency never turns the health probe into a hard failure.
        """
        try:
            from gnn.render.health import check_renderers
        except Exception as exc:
            logger.debug("Canonical renderer health check unavailable: %s", exc)
            return {}
        try:
            statuses = check_renderers()
        except Exception as exc:
            logger.debug("Renderer health check failed: %s", exc)
            return {}
        return {name: status.available for name, status in statuses.items()}

    # Module-scope instance for ASGI deployment (e.g. uvicorn src.api.app:app).
    # Tests should call create_app() directly to get a fresh isolated instance.
    app: FastAPI | None = create_app()

else:
    app = None

    def create_app(
        runs_store: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "FastAPI":
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
