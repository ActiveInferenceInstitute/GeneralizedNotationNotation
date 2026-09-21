#!/usr/bin/env python3
"""
GNN API Job Manager — in-memory job tracking and async pipeline execution.

Manages job lifecycle: create → execute → poll → result.
Uses asyncio for non-blocking pipeline execution.
No database dependency — jobs are stored in memory (lost on restart).
"""

import asyncio
import logging
import os
import re
import signal
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from gnn.api.models import validate_step_numbers as _shared_validate_step_numbers
from gnn.api.path_utils import resolve_repo_path
from gnn.api.pipeline_runner import (
    build_pipeline_command,
    normalize_summary_steps,
    pipeline_exit_succeeded,
    read_pipeline_summary,
    summarize_step_progress,
)
from gnn.pipeline.step_registry import STEPS

# In-memory job store (cleared on restart — research tool, not production service)
_JOBS: Dict[str, dict[str, Any]] = {}


def _validate_step_numbers(
    values: Optional[List[int]], *, field_name: str
) -> Optional[List[int]]:
    """Validate an optional list of unique pipeline steps.

    Delegates to the shared contract validator in ``api.models`` so the job
    manager and the request models can never disagree on step semantics.
    """
    return _shared_validate_step_numbers(values, field_name=field_name)


def create_job(
    target_dir: str,
    output_dir: Optional[str] = None,
    steps: Optional[List[int]] = None,
    skip_steps: Optional[List[int]] = None,
    verbose: bool = False,
    strict: bool = False,
) -> str:
    """
    Create a new pipeline job and return its ID.

    Args:
        target_dir: Directory containing GNN files
        output_dir: Directory where pipeline outputs should be written
        steps: Specific steps to run (None = all)
        skip_steps: Steps to skip
        verbose: Enable verbose output
        strict: Treat warnings as errors

    Returns:
        Unique job ID string
    """
    steps = _validate_step_numbers(steps, field_name="steps")
    skip_steps = _validate_step_numbers(skip_steps, field_name="skip_steps")
    overlap = set(steps or ()) & set(skip_steps or ())
    if overlap:
        raise ValueError(f"steps and skip_steps must not overlap: {sorted(overlap)}")

    target_path = resolve_repo_path(
        target_dir,
        purpose="Target directory",
        must_exist=True,
    )
    output_path = resolve_repo_path(
        output_dir or "output",
        purpose="Output directory",
        create=True,
    )

    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "target_dir": str(target_path),
        "steps": steps,
        "skip_steps": skip_steps,
        "verbose": verbose,
        "strict": strict,
        "progress_step": None,
        "steps_completed": [],
        "steps_failed": [],
        "exit_code": None,
        "error_message": None,
        "output_dir": str(output_path),
        "process": None,  # subprocess handle (not serializable, stripped in get_job)
    }
    logger.info(
        f"Created job {job_id} for target={target_path}, output={output_path}, steps={steps}"
    )
    return job_id


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    """
    Retrieve job status by ID.

    Returns a serializable dict (subprocess handle is stripped).
    """
    job = _JOBS.get(job_id)
    if job is None:
        return None

    # Return copy without non-serializable fields
    serializable = {k: v for k, v in job.items() if k != "process"}
    return serializable


def _terminate_process_tree(proc: Any, job_id: str) -> None:
    """Terminate a job subprocess together with its process group.

    Jobs spawn the pipeline in its own session (``start_new_session``), so a
    cancelled job can signal the whole group: rendered scripts that shell out
    to grandchildren would otherwise survive a direct-child terminate. This
    mirrors the process-group kill pattern in
    ``gnn.utils.pipeline_orchestration.execution_utils`` and falls back to
    the direct child on non-POSIX platforms or when the group is gone.
    """
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            logger.info(f"Signalled process group of job {job_id}")
        else:
            proc.terminate()
            logger.info(f"Terminated subprocess for job {job_id}")
    except (ProcessLookupError, PermissionError, AttributeError) as exc:
        try:
            proc.terminate()
            logger.info(f"Terminated subprocess (direct child) for job {job_id}: {exc}")
        except Exception as terminate_error:
            logger.warning(
                f"Could not terminate process for job {job_id}: {terminate_error}"
            )
    except Exception as exc:
        logger.warning(f"Could not terminate process for job {job_id}: {exc}")


def cancel_job(job_id: str) -> bool:
    """
    Cancel a running or pending job.

    Returns True if cancelled, False if job not found or already terminal.
    """
    job = _JOBS.get(job_id)
    if job is None:
        return False

    if job["status"] in ("completed", "failed", "cancelled"):
        return False

    # Terminate the subprocess tree if running
    proc = job.get("process")
    if proc is not None:
        _terminate_process_tree(proc, job_id)

    job["status"] = "cancelled"
    job["completed_at"] = datetime.now().isoformat()
    return True


def list_jobs(limit: int = 50) -> List[dict[str, Any]]:
    """List recent jobs (most recent first)."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    jobs = [get_job(jid) for jid in list(_JOBS.keys())[-limit:]]
    return [j for j in jobs if j is not None]


async def execute_job_async(job_id: str) -> None:
    """
    Execute a pipeline job asynchronously.

    Runs `uv run python src/gnn/main.py` with appropriate arguments in a subprocess.
    Updates job status as execution progresses.

    This coroutine is meant to be launched with asyncio.create_task().
    """
    job = _JOBS.get(job_id)
    if job is None:
        logger.error(f"Cannot execute unknown job: {job_id}")
        return

    if job["status"] in ("completed", "failed", "cancelled"):
        # A cancel raced us before execution started; a cancelled job must
        # never launch its pipeline.
        logger.info(f"Skipping execution of job {job_id}: already {job['status']}")
        return

    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat()
    logger.info(f"Starting job {job_id}")

    # Build the real orchestrator command via the shared pure builder so the
    # job surface and the run surface can never drift on argv shape.
    repo_root = Path(__file__).parent.parent.parent
    output_dir = Path(job.get("output_dir") or (repo_root / "output"))
    job["output_dir"] = str(output_dir)

    cmd = build_pipeline_command(
        str(job["target_dir"]),
        str(output_dir),
        only_steps=job.get("steps") or None,
        skip_steps=job.get("skip_steps") or None,
        verbose=bool(job.get("verbose")),
        strict=bool(job.get("strict")),
        repo_root=repo_root,
    )

    try:
        invocation_start_ns = time.time_ns()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(repo_root),
            env={**os.environ, "GNN_RUN_ID": job_id},
            # Own session/group so a cancel can signal the whole process
            # tree (mirrors utils.pipeline_orchestration.execution_utils).
            start_new_session=os.name == "posix",
        )
        job["process"] = proc

        stdout, stderr = await proc.communicate()

        # A cancel can race this coroutine: cancel_job terminates the
        # subprocess and writes the terminal 'cancelled' state while
        # communicate() is still waiting, and a terminated process exits
        # nonzero — which would otherwise re-report the job as 'failed'
        # with a stderr error message. Snapshot the flag before any state
        # write; there are no awaits below, so the check and the guarded
        # writes are atomic on the single loop thread that mutates jobs.
        was_cancelled = job["status"] == "cancelled"

        job["exit_code"] = proc.returncode
        if not was_cancelled:
            job["completed_at"] = datetime.now().isoformat()

        # Populate per-step progress from the canonical summary the pipeline
        # writes; the single subprocess gives us no live per-step view, so
        # progress is observable only once the run has finished.
        progress = summarize_step_progress(
            normalize_summary_steps(
                read_pipeline_summary(
                    output_dir,
                    not_before_ns=invocation_start_ns,
                    expected_run_id=job_id,
                )
                or []
            )
        )
        job["steps_completed"] = progress["steps_completed"]
        job["steps_failed"] = progress["steps_failed"]

        if was_cancelled:
            # Keep cancel_job's terminal write: 'cancelled' with no
            # fabricated error message, whatever partial output exists.
            logger.info(f"Job {job_id} was cancelled; keeping cancelled state")
        elif pipeline_exit_succeeded(proc.returncode, strict=bool(job.get("strict"))):
            job["status"] = "completed"
            logger.info(f"Job {job_id} completed successfully")
        else:
            job["status"] = "failed"
            # Capture a sanitized tail of stderr for the error message. Raw
            # stderr leaks internal paths, library versions, and stack traces;
            # redact the repository root and other absolute paths first.
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
            job["error_message"] = _sanitize_stderr(stderr_text, repo_root)
            logger.error(f"Job {job_id} failed with exit code {proc.returncode}")

    except Exception as e:
        if job["status"] == "cancelled":
            # cancel_job won the race during the exception; keep its write.
            logger.info(f"Job {job_id} was cancelled; ignoring exception: {e}")
            return
        job["status"] = "failed"
        job["error_message"] = str(e)
        job["completed_at"] = datetime.now().isoformat()
        logger.error(f"Job {job_id} raised exception: {e}")
    finally:
        job["process"] = None


def _sanitize_stderr(stderr_text: str, repo_root: Path) -> str:
    """Redact internal paths from a stderr tail before exposing it to clients.

    Keeps the diagnostic value of the tail (the last 500 chars) while removing
    the repository root and other absolute filesystem paths that would disclose
    host layout to an API caller.
    """
    tail = stderr_text[-500:] if len(stderr_text) > 500 else stderr_text
    tail = tail.replace(str(repo_root), "<repo>")
    # Common absolute path prefixes (home/usr/tmp/var/etc.) become <path>.
    tail = re.sub(r"(?:/[A-Za-z0-9_.-]+){2,}(?:/[^\s\"']*)?", "<path>", tail)
    return tail


# Pipeline step registry for the /tools endpoint — derived once from the
# canonical ``pipeline.step_registry.STEPS`` so this module is never the
# authority on which steps exist (single source of truth).
def _derive_pipeline_steps() -> Dict[int, tuple[str, str]]:
    """Map step number → (name, description) from the canonical registry."""
    registry: Dict[int, tuple[str, str]] = {}
    for step in STEPS:
        num_text, _, name = step.script_stem.partition("_")
        registry[int(num_text)] = (name, step.description)
    return registry


PIPELINE_STEPS: Dict[int, tuple[str, str]] = _derive_pipeline_steps()


def get_pipeline_tools() -> List[dict[str, Any]]:
    """Return list of available pipeline tools."""
    return [
        {
            "step_number": step,
            "name": name,
            "description": desc,
            "script": f"src/gnn/{step}_{name}.py",
        }
        for step, (name, desc) in PIPELINE_STEPS.items()
    ]
