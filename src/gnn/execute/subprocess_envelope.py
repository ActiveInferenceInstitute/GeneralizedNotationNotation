#!/usr/bin/env python3
"""
Uniform subprocess execution envelope for GNN execution backends.

Every GNN execution path (script runners, MCP executors, tool
probes) converts ``subprocess`` outcomes into the same structured
envelope so callers never have to differentiate between a timeout, an
``OSError``, a cancellation, and a non-zero exit code.

Timeout and cancellation semantics: the child is spawned as the leader
of a fresh process group (``start_new_session`` on POSIX,
``CREATE_NEW_PROCESS_GROUP`` on Windows) and the poll loop watches a
:class:`CancelToken` plus the resolved deadline in 0.25s slices. On
timeout, cancellation, or KeyboardInterrupt the WHOLE group is killed
and the pipes drained, so shell/tree grandchildren (e.g. Julia worker
processes) die with the child instead of being orphaned. Under a
sandbox backend prefix (firejail/bwrap/nsjail) the kill targets the
wrapper's process group — the sandboxed target dies with its wrapper.
A ``timeout=None`` run is still bounded: ``_resolve_timeout`` applies
``DEFAULT_TIMEOUT_SECONDS``, overridable per process via
``GNN_EXECUTE_DEFAULT_TIMEOUT``.

Extracted from ``execute.executor`` (the canonical
``execute_script_safely`` envelope) into a leaf module so the
per-framework runners can share one implementation without importing the
executor module (which imports the runner modules at module scope).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
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

# Default wall-clock bound applied when a caller passes ``timeout=None``.
# Overridable per process via ``GNN_EXECUTE_DEFAULT_TIMEOUT`` (positive int).
DEFAULT_TIMEOUT_SECONDS: int = 3600

# Cancellation/deadline poll cadence; never check faster than this.
_POLL_INTERVAL_SECONDS = 0.25

# Invalid ``GNN_EXECUTE_DEFAULT_TIMEOUT`` values that already warned (warn-once).
_INVALID_DEFAULT_TIMEOUTS_WARNED: set[str] = set()


class CancelToken:
    """Thread-safe cooperative cancellation flag for in-flight subprocess runs.

    Pass a token to :func:`run_subprocess_envelope` (or the executor helpers
    built on it) and call :meth:`cancel` from any thread: the in-flight run's
    poll loop observes the flag within one poll slice (<=0.25s), group-kills
    the child, drains the streams, and reports ``error_type="Cancelled"``
    without raising. ``cancel()`` is idempotent; the first ``reason`` wins.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: Optional[str] = None

    def cancel(self, reason: Optional[str] = None) -> None:
        """Request cancellation (idempotent); the first ``reason`` wins."""
        if not self._event.is_set():
            self._reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been called from any thread."""
        return self._event.is_set()

    @property
    def reason(self) -> Optional[str]:
        """Reason recorded by the first :meth:`cancel` call (may be None)."""
        return self._reason


def _warn_invalid_default_timeout(raw: str) -> None:
    """Warn once per distinct invalid ``GNN_EXECUTE_DEFAULT_TIMEOUT`` value."""
    if raw in _INVALID_DEFAULT_TIMEOUTS_WARNED:
        return
    _INVALID_DEFAULT_TIMEOUTS_WARNED.add(raw)
    logger.warning(
        "GNN_EXECUTE_DEFAULT_TIMEOUT=%r is not a positive int; using "
        "DEFAULT_TIMEOUT_SECONDS=%s instead",
        raw,
        DEFAULT_TIMEOUT_SECONDS,
    )


def _resolve_timeout(timeout: Optional[int]) -> int:
    """Return the effective wall-clock timeout for an envelope run.

    An explicit ``timeout`` passes through untouched. ``None`` resolves to
    ``GNN_EXECUTE_DEFAULT_TIMEOUT`` when that env var parses to a positive
    int; an unset, invalid, or non-positive value warns once and falls back
    to :data:`DEFAULT_TIMEOUT_SECONDS`.
    """
    if timeout is not None:
        return timeout
    raw = os.environ.get("GNN_EXECUTE_DEFAULT_TIMEOUT")
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        resolved = int(raw)
    except ValueError:
        _warn_invalid_default_timeout(raw)
        return DEFAULT_TIMEOUT_SECONDS
    if resolved <= 0:
        _warn_invalid_default_timeout(raw)
        return DEFAULT_TIMEOUT_SECONDS
    return resolved


def _as_text(value: Any) -> str:
    """Normalize stream captures to text (subprocess mixes str/bytes)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


__all__ = [
    "run_subprocess_envelope",
    "CancelToken",
    "DEFAULT_TIMEOUT_SECONDS",
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


def _spawn_kwargs() -> Dict[str, Any]:
    """Popen kwargs making the child the leader of a fresh process group."""
    if hasattr(os, "setsid"):
        return {"start_new_session": True}
    # CREATE_NEW_PROCESS_GROUP exists only on Windows; getattr keeps the
    # attribute access runtime- and mypy-safe on POSIX checkouts.
    windows_process_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return {"creationflags": windows_process_group}


def _kill_process_group(proc: subprocess.Popen[Any]) -> None:
    """SIGKILL the child's whole process group (POSIX) or the child itself.

    The child is spawned as a process-group leader (``_spawn_kwargs``), so
    the group kill reaps the whole spawned tree — shells, grandchildren,
    sandbox wrappers and their targets — instead of orphaning them.
    """
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass  # tree already gone or unreapable; fall back to proc.kill()
    try:
        proc.kill()
    except OSError:
        pass


def _mark_cancelled(envelope: Dict[str, Any], reason: Optional[str]) -> None:
    """Stamp a run-cancelled envelope (partial streams are the caller's job)."""
    envelope["cancelled"] = True
    envelope["success"] = False
    envelope["error"] = (
        "Execution cancelled" if reason is None else f"Execution cancelled: {reason}"
    )
    envelope["error_type"] = "Cancelled"
    envelope["return_code"] = NEVER_STARTED


def run_subprocess_envelope(
    command: List[str],
    *,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    cwd: Optional[Union[str, Path]] = None,
    input: Optional[str] = None,
    sandbox: bool = True,
    cancel_token: Optional[CancelToken] = None,
) -> Dict[str, Any]:
    """Run ``command`` in a fresh process group and return a structured envelope.

    Args:
        command:        Argument vector (no shell).
        timeout:        Wall-clock timeout in seconds. ``None`` resolves via
                        ``_resolve_timeout``: ``GNN_EXECUTE_DEFAULT_TIMEOUT``
                        when it parses to a positive int, else
                        ``DEFAULT_TIMEOUT_SECONDS`` — runs are never
                        unbounded.
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
        cancel_token:   Optional cooperative :class:`CancelToken`; checked
                        before spawning and every poll slice. A cancel
                        (pre-spawn or mid-flight) group-kills the child and
                        reports ``error_type="Cancelled"`` without raising.

    Returns:
        Dict with keys:
            - ``success`` (bool): True iff the process exited with code 0.
            - ``return_code`` (int): Exit code (``NEVER_STARTED`` if never
              started — OSError, timeout, or cancellation).
            - ``stdout`` (str): Captured stdout (empty when not capturing).
              On timeout/cancel the child's partial stdout survives
              (group-kill + drain).
            - ``stderr`` (str): Captured stderr (empty when not capturing).
              On timeout/cancel the child's partial stderr survives
              (group-kill + drain).
            - ``duration_seconds`` (float): Wall-clock execution time.
            - ``cancelled`` (bool): Always present; True iff the run was
              cancelled via ``cancel_token`` (pre-spawn or mid-flight).
            - ``sandbox_mode`` (str): Effective ``GNN_SANDBOX`` mode
              (``"off"`` when ``sandbox=False``).
            - ``sandboxed`` (bool): True iff the command was actually
              wrapped with a sandbox backend prefix.
            - ``error`` (str, optional): Populated on failure.
            - ``error_type`` (str, optional): Exception class name on
              failure; ``"SandboxUnavailable"`` when ``GNN_SANDBOX=require``
              found no backend and the run was refused;
              ``"TimeoutExpired"`` on timeout; ``"Cancelled"`` when the
              ``cancel_token`` fired.

    The child runs as the leader of a fresh process group
    (``start_new_session`` on POSIX, ``CREATE_NEW_PROCESS_GROUP`` on
    Windows); on timeout, cancellation, or KeyboardInterrupt the WHOLE
    group is killed and the pipes drained, so shell/tree grandchildren die
    with the child instead of being orphaned. Under a sandbox backend
    prefix the kill targets the wrapper's process group (the sandboxed
    target dies with its wrapper). ``KeyboardInterrupt`` group-kills and
    re-raises, matching ``subprocess.run``.

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
        "cancelled": False,
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
    resolved_timeout = _resolve_timeout(timeout)
    deadline = time.monotonic() + resolved_timeout
    # communicate() accepts the stdin payload on the FIRST call only — every
    # retry (timeout/cancel slices and the post-kill drain) must pass None.
    # Re-passing input after TimeoutExpired raises ValueError ("Cannot send
    # input after starting communication"); partial output accumulates across
    # communicate calls internally and is returned by the final drain.
    stdin_payload: Optional[bytes] = (
        input.encode("utf-8") if input is not None else None
    )
    stdout_bytes: Optional[bytes] = None
    stderr_bytes: Optional[bytes] = None
    timed_out = False
    proc: Optional[subprocess.Popen[Any]] = None
    try:
        if cancel_token is not None and cancel_token.cancelled:
            # Pre-spawn cancel: identical envelope, no process is spawned.
            _mark_cancelled(envelope, cancel_token.reason)
            return envelope

        proc = subprocess.Popen(  # nosec B603 — argument vector, no shell
            command,
            stdin=subprocess.PIPE if stdin_payload is not None else None,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            cwd=cwd,
            env=merged_env,
            **_spawn_kwargs(),
        )
        while True:
            try:
                stdout_bytes, stderr_bytes = proc.communicate(
                    stdin_payload, timeout=_POLL_INTERVAL_SECONDS
                )
                stdin_payload = None
                envelope["return_code"] = proc.returncode
                envelope["success"] = proc.returncode == 0
                break  # process exited; final streams collected
            except subprocess.TimeoutExpired:
                stdin_payload = None  # input was sent on the first call only
                if cancel_token is not None and cancel_token.cancelled:
                    _kill_process_group(proc)
                    stdout_bytes, stderr_bytes = proc.communicate()
                    _mark_cancelled(envelope, cancel_token.reason)
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    _kill_process_group(proc)
                    stdout_bytes, stderr_bytes = proc.communicate()
                    break
    except KeyboardInterrupt:
        # Match subprocess.run's re-raise, but take the whole group with us.
        if proc is not None:
            _kill_process_group(proc)
            try:
                proc.communicate()
            except Exception:  # noqa: BLE001 — best-effort drain before raise
                pass
        raise
    except Exception as exc:  # noqa: BLE001 — convert any failure to envelope
        envelope["error"] = str(exc)
        envelope["error_type"] = type(exc).__name__
    finally:
        envelope["duration_seconds"] = time.time() - start

    if timed_out:
        envelope["error"] = f"Execution timed out after {resolved_timeout}s"
        envelope["error_type"] = "TimeoutExpired"
        envelope["return_code"] = NEVER_STARTED
    envelope["stdout"] = _as_text(stdout_bytes) if capture_output else ""
    envelope["stderr"] = _as_text(stderr_bytes) if capture_output else ""

    return envelope
