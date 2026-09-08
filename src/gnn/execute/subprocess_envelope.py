#!/usr/bin/env python3
"""
Uniform subprocess execution envelope for GNN execution backends.

Every GNN execution path (script runners, MCP executors, tool
probes) converts ``subprocess.run`` outcomes into the same structured
envelope so callers never have to differentiate between a timeout, an
``OSError``, and a non-zero exit code.

Extracted from ``execute.executor`` (the canonical
``execute_script_safely`` envelope) into a leaf module so the
per-framework runners can share one implementation without importing the
executor module (which imports the runner modules at module scope).
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Sentinel exit codes shared across GNN execution surfaces (this envelope,
# the step-12 processor, per-framework runners). A sentinel alone is
# ambiguous — always pair it with ``error_type`` when interpreting failures.
NEVER_STARTED = -1  # the process never ran: OSError or caller-side timeout
INTERNAL_ERROR = -2  # the harness itself failed while orchestrating the run
UNKNOWN_STATE = -3  # no execution record exists at all

__all__ = [
    "run_subprocess_envelope",
    "NEVER_STARTED",
    "INTERNAL_ERROR",
    "UNKNOWN_STATE",
]


def run_subprocess_envelope(
    command: List[str],
    *,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    cwd: Optional[Union[str, Path]] = None,
    input: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ``command`` via ``subprocess.run`` and return a structured envelope.

    Args:
        command:        Argument vector (no shell).
        timeout:        Wall-clock timeout in seconds (``None`` = unbounded).
        cwd:            Working directory for the subprocess.
        env:            Environment variable overrides, merged over
                        ``os.environ`` (``None`` inherits the parent env).
        capture_output: If True, capture stdout/stderr; otherwise stream to
                        the parent process.
        input:          Text piped to the child's stdin (implies a stdin
                        pipe; ``None`` leaves stdin attached to the parent).

    Returns:
        Dict with keys:
            - ``success`` (bool): True iff the process exited with code 0.
            - ``return_code`` (int): Exit code (``NEVER_STARTED`` if never
              started).
            - ``stdout`` (str): Captured stdout (empty when not capturing).
              On timeout, the child's partial stdout survives (kill + drain).
            - ``stderr`` (str): Captured stderr (empty when not capturing).
              On timeout, the child's partial stderr survives (kill + drain).
            - ``duration_seconds`` (float): Wall-clock execution time.
            - ``error`` (str, optional): Populated on failure.
            - ``error_type`` (str, optional): Exception class name on failure.
    """
    envelope: Dict[str, Any] = {
        "success": False,
        "return_code": NEVER_STARTED,
        "stdout": "",
        "stderr": "",
        "duration_seconds": 0.0,
    }

    merged_env: Optional[Dict[str, str]] = None
    if env is not None:
        merged_env = dict(os.environ)
        merged_env.update(env)

    def _as_text(value: Any) -> str:
        """Normalize stream captures to text (subprocess mixes str/bytes)."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    start = time.time()
    try:
        # subprocess.run is C-optimized for the common success path and
        # internally kills + drains on timeout, populating the
        # TimeoutExpired exception's stdout/stderr with the partial output.
        completed = subprocess.run(  # nosec B603 — argument vector, no shell
            command,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=merged_env,
            input=input,
            check=False,
        )
        envelope["return_code"] = completed.returncode
        envelope["success"] = completed.returncode == 0
        envelope["stdout"] = _as_text(completed.stdout)
        envelope["stderr"] = _as_text(completed.stderr)
    except subprocess.TimeoutExpired as exc:
        envelope["error"] = f"Execution timed out after {timeout}s"
        envelope["error_type"] = "TimeoutExpired"
        # subprocess.run kills and drains on timeout; the exception carries
        # the partial output (str under text=True, bytes on some interpreters).
        envelope["stdout"] = _as_text(exc.stdout) if capture_output else ""
        envelope["stderr"] = _as_text(exc.stderr) if capture_output else ""
    except Exception as exc:  # noqa: BLE001 — convert any failure to envelope
        envelope["error"] = str(exc)
        envelope["error_type"] = type(exc).__name__
    finally:
        envelope["duration_seconds"] = time.time() - start

    return envelope
