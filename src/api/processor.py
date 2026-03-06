#!/usr/bin/env python3
"""
GNN API Job Manager — in-memory job tracking and async pipeline execution.

Manages job lifecycle: create → execute → poll → result.
Uses asyncio for non-blocking pipeline execution.
No database dependency — jobs are stored in memory (lost on restart).
"""

import asyncio
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# In-memory job store (cleared on restart — research tool, not production service)
_JOBS: Dict[str, dict] = {}


def create_job(
    target_dir: str,
    steps: Optional[List[int]] = None,
    skip_steps: Optional[List[int]] = None,
    verbose: bool = False,
    strict: bool = False
) -> str:
    """
    Create a new pipeline job and return its ID.

    Args:
        target_dir: Directory containing GNN files
        steps: Specific steps to run (None = all)
        skip_steps: Steps to skip
        verbose: Enable verbose output
        strict: Treat warnings as errors

    Returns:
        Unique job ID string
    """
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "target_dir": target_dir,
        "steps": steps,
        "skip_steps": skip_steps,
        "verbose": verbose,
        "strict": strict,
        "progress_step": None,
        "steps_completed": [],
        "steps_failed": [],
        "exit_code": None,
        "error_message": None,
        "output_dir": None,
        "process": None,  # subprocess handle (not serializable, stripped in get_job)
    }
    logger.info(f"Created job {job_id} for target={target_dir}, steps={steps}")
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

    # Build command
    repo_root = Path(__file__).parent.parent.parent
    main_script = repo_root / "src" / "main.py"

    cmd = [sys.executable, str(main_script)]
    cmd += ["--target-dir", str(job["target_dir"])]

    output_dir = repo_root / "output"
    cmd += ["--output-dir", str(output_dir)]
    job["output_dir"] = str(output_dir)

    if job.get("steps"):
        cmd += ["--only-steps", ",".join(str(s) for s in job["steps"])]

    if job.get("skip_steps"):
        cmd += ["--skip-steps", ",".join(str(s) for s in job["skip_steps"])]

    if job.get("verbose"):
        cmd.append("--verbose")

    if job.get("strict"):
        cmd.append("--strict")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(repo_root)
        )
        job["process"] = proc

        stdout, stderr = await proc.communicate()

        job["exit_code"] = proc.returncode
        job["completed_at"] = datetime.now().isoformat()

        if proc.returncode == 0:
            job["status"] = "completed"
            logger.info(f"Job {job_id} completed successfully")
        else:
            job["status"] = "failed"
            # Capture tail of stderr for error message
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
            job["error_message"] = stderr_text[-500:] if len(stderr_text) > 500 else stderr_text
            logger.error(f"Job {job_id} failed with exit code {proc.returncode}")

    except Exception as e:
        job["status"] = "failed"
        job["error_message"] = str(e)
        job["completed_at"] = datetime.now().isoformat()
        logger.error(f"Job {job_id} raised exception: {e}")
    finally:
        job["process"] = None


# Pipeline step registry for /tools endpoint
PIPELINE_STEPS = {
    0: ("template", "Template initialization and output directory setup"),
    1: ("setup", "Environment setup and dependency validation"),
    2: ("tests", "Test suite execution"),
    3: ("gnn", "GNN file parsing and validation"),
    4: ("model_registry", "Model registry management"),
    5: ("type_checker", "Type checking and dimension validation"),
    6: ("validation", "Schema and semantic validation"),
    7: ("export", "Multi-format export (JSON, YAML, XML)"),
    8: ("visualization", "Graph visualization"),
    9: ("advanced_viz", "Advanced visualization with Altair/Plotly"),
    10: ("ontology", "Ontology annotation processing"),
    11: ("render", "Code generation for all frameworks"),
    12: ("execute", "Simulation execution"),
    13: ("llm", "LLM-powered analysis"),
    14: ("ml_integration", "Machine learning model integration"),
    15: ("audio", "Audio/SAPF processing"),
    16: ("analysis", "Statistical analysis"),
    17: ("integration", "Cross-framework integration"),
    18: ("security", "Security validation"),
    19: ("research", "Research hypothesis generation"),
    20: ("website", "Static website generation"),
    21: ("mcp", "MCP tool registration"),
    22: ("gui", "GUI interface"),
    23: ("report", "Report generation"),
    24: ("intelligent_analysis", "Intelligent pipeline analysis"),
}


def get_pipeline_tools() -> List[dict]:
    """Return list of available pipeline tools."""
    return [
        {
            "step_number": step,
            "name": name,
            "description": desc,
            "script": f"src/{step}_{name}.py"
        }
        for step, (name, desc) in PIPELINE_STEPS.items()
    ]
