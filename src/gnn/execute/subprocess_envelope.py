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

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Sentinel exit codes shared across GNN execution surfaces (this envelope,
# the step-12 processor, per-framework runners). A sentinel alone is
# ambiguous — always pair it with ``error_type`` when interpreting failures.
NEVER_STARTED = -1  # the process never ran: OSError or caller-side timeout
INTERNAL_ERROR = -2  # the harness itself failed while orchestrating the run
UNKNOWN_STATE = -3  # no execution record exists at all


def _as_text(value: Any) -> str:
    """Normalize stream captures to text (subprocess mixes str/bytes)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


__all__ = [
    "run_subprocess_envelope",
    "NEVER_STARTED",
    "INTERNAL_ERROR",
    "UNKNOWN_STATE",
]


def _sandbox_prefix_and_mode() -> tuple[List[str], str, Optional[str]]:
    """Return ``(prefix, effective_mode, blocked_reason)`` from ``GNN_SANDBOX``.

    Mirrors the Step-12 semantics (``execute.processor._sandbox_mode`` /
    ``_sandbox_command_prefix``): ``off`` runs unsandboxed, ``prefer``
    falls back to unsandboxed execution (with a warning) when no backend
    exists, and ``require`` without a backend yields a non-None
    ``blocked_reason`` so the caller can refuse to execute.
    """
    from gnn.execute.sandbox import _resolve_mode, detect_sandbox

    mode = _resolve_mode(None)
    if mode == "off":
        return [], mode, None
    spec = detect_sandbox()
    if spec is None:
        if mode == "require":
            return (
                [],
                mode,
                (
                    "GNN_SANDBOX=require but no sandbox backend "
                    "(firejail/bwrap/nsjail) is installed"
                ),
            )
        logger.warning(
            "GNN_SANDBOX=%s but no sandbox backend found; running unsandboxed",
            mode,
        )
        return [], mode, None
    return list(spec.prefix), mode, None


def run_subprocess_envelope(
    command: List[str],
    *,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    cwd: Optional[Union[str, Path]] = None,
    input: Optional[str] = None,
    sandbox: bool = True,
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
        sandbox:        Apply the ``GNN_SANDBOX`` env-prefix pattern
                        (Step-12 semantics). ``True`` prefixes the command
                        with the configured sandbox backend's argument
                        vector; ``False`` runs unsandboxed unconditionally.

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
            - ``sandbox_mode`` (str): Effective ``GNN_SANDBOX`` mode
              (``"off"`` when ``sandbox=False``).
            - ``sandboxed`` (bool): True iff the command was actually
              wrapped with a sandbox backend prefix.
            - ``error`` (str, optional): Populated on failure.
            - ``error_type`` (str, optional): Exception class name on
              failure; ``"SandboxUnavailable"`` when ``GNN_SANDBOX=require``
              found no backend and the run was refused.

    When ``sandbox`` is True and ``GNN_SANDBOX`` is ``off`` (the default),
    a ``sandbox_disabled_receipt`` warning is emitted but the command runs
    unsandboxed. When ``sandbox`` is False, the caller is asserting that it
    manages sandboxing itself — the same receipt is emitted and the env
    prefix is never applied.
    """
    envelope: Dict[str, Any] = {
        "success": False,
        "return_code": NEVER_STARTED,
        "stdout": "",
        "stderr": "",
        "duration_seconds": 0.0,
        "sandbox_mode": "off",
        "sandboxed": False,
    }

    subject = command[0] if command else "<empty-command>"
    if not sandbox:
        logger.warning(
            "sandbox_disabled_receipt: %s executed WITHOUT a sandbox "
            "(sandbox=False). The command runs with operator privileges; "
            "set GNN_SANDBOX=prefer/require and sandbox=True for isolation.",
            subject,
            extra={
                "event": "sandbox_disabled_receipt",
                "sandbox_mode": "off",
                "command": subject,
            },
        )
    else:
        sandbox_prefix, mode, blocked = _sandbox_prefix_and_mode()
        envelope["sandbox_mode"] = mode
        if blocked is not None:
            # Mirrors the Step-12 processor: refuse to run unsandboxed.
            envelope["error"] = blocked
            envelope["error_type"] = "SandboxUnavailable"
            logger.error(blocked)
            return envelope
        if sandbox_prefix:
            envelope["sandboxed"] = True
            logger.info(
                "sandbox_active_receipt: %s will run under sandbox "
                "isolation (GNN_SANDBOX=%s)",
                subject,
                mode,
                extra={
                    "event": "sandbox_active_receipt",
                    "sandbox_mode": mode,
                },
            )
            command = [*sandbox_prefix, *command]
        else:
            logger.warning(
                "sandbox_disabled_receipt: %s executed WITHOUT a sandbox "
                "(GNN_SANDBOX=off, the default). The command runs with "
                "operator privileges; set GNN_SANDBOX=prefer/require for "
                "isolation.",
                subject,
                extra={
                    "event": "sandbox_disabled_receipt",
                    "sandbox_mode": mode,
                    "command": subject,
                },
            )

    merged_env: Optional[Dict[str, str]] = None
    if env is not None:
        merged_env = dict(os.environ)
        merged_env.update(env)

    start = time.time()
    try:
        # subprocess.run is C-optimized for the common success path and
        # internally kills + drains on timeout, populating the
        # TimeoutExpired exception's stdout/stderr with the partial output.
        completed = subprocess.run(  # nosec B603 — argument vector, no shell
            command,
            capture_output=capture_output,
            text=False,
            timeout=timeout,
            cwd=cwd,
            env=merged_env,
            input=input.encode("utf-8") if input is not None else None,
            check=False,
        )
        envelope["return_code"] = completed.returncode
        envelope["success"] = completed.returncode == 0
        # text=False returns bytes (faster than text=True which wraps in
        # TextIOWrapper); decode inline for the common success path.
        _stdout = completed.stdout
        _stderr = completed.stderr
        envelope["stdout"] = (
            _stdout.decode("utf-8", "replace")
            if isinstance(_stdout, bytes)
            else (_stdout or "")
        )
        envelope["stderr"] = (
            _stderr.decode("utf-8", "replace")
            if isinstance(_stderr, bytes)
            else (_stderr or "")
        )
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
