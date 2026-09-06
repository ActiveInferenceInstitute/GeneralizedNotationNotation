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
_JOBS: Dict[str, dict] = {}


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


def get_job(job_id: str) -> Optional[dict]:
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

    # Terminate subprocess if running
    proc = job.get("process")
    if proc is not None:
        try:
            proc.terminate()
            logger.info(f"Terminated subprocess for job {job_id}")
        except Exception as e:
            logger.warning(f"Could not terminate process for job {job_id}: {e}")

    job["status"] = "cancelled"
    job["completed_at"] = datetime.now().isoformat()
    return True


def list_jobs(limit: int = 50) -> List[dict]:
    """List recent jobs (most recent first)."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    jobs = [get_job(jid) for jid in list(_JOBS.keys())[-limit:]]
    return [j for j in jobs if j is not None]


async def execute_job_async(job_id: str) -> None:
    """
    Execute a pipeline job asynchronously.

    Runs `python src/main.py` with appropriate arguments in a subprocess.
    Updates job status as execution progresses.

    This coroutine is meant to be launched with asyncio.create_task().
    """
    job = _JOBS.get(job_id)
    if job is None:
        logger.error(f"Cannot execute unknown job: {job_id}")
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
        )
        job["process"] = proc

        stdout, stderr = await proc.communicate()

        job["exit_code"] = proc.returncode
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

        if pipeline_exit_succeeded(proc.returncode, strict=bool(job.get("strict"))):
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


def get_pipeline_tools() -> List[dict]:
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
