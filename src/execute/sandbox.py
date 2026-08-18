"""Opt-in sandbox for rendered-script execution (Step 12).

GNN renders untrusted text specifications into executable Python/Julia scripts.
By default those scripts run **unsandboxed** with the operator's privileges; the
documented contract is trusted, local, single-user input (see
``SECURITY.md`` and ``RED_TEAM_REVIEW.md``). This module adds a safe, opt-in
wrapper that runs rendered scripts under an available sandbox/namespace tool and
degrades **loudly** (never silently) when no such tool is present.

Modes (selected via ``GNN_SANDBOX`` or an explicit argument):

- ``off`` (default): behave exactly as before — no wrapping. Preserves the
  existing trusted-local contract.
- ``prefer``: wrap when a sandbox binary is found; otherwise run unsandboxed
  and emit a warning.
- ``require``: wrap when a sandbox binary is found; otherwise **refuse to run**
  the script and return an error envelope.

Supported sandbox backends, in preference order (first found wins):

- ``firejail``: ``firejail --noprofile --net=none --private --caps.drop=all``
- ``bwrap``  (bubblewrap): ``bwrap --unshare-all --die-with-parent --ro-bind / / ...``
- ``nsjail``: ``nsjail -Mo --disable-proc --net none``

The wrapper is intentionally conservative: it only changes the *command prefix*,
leaving the executor's own timeout/cwd/env handling untouched.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Valid ``GNN_SANDBOX`` modes.
SANDBOX_MODES = ("off", "prefer", "require")
_DEFAULT_MODE = "off"


@dataclass(frozen=True)
class SandboxSpec:
    """One recognized sandbox backend and the prefix it contributes."""

    binary: str
    prefix: tuple[str, ...]

    def available(self) -> bool:
        """Whether the backend binary is on PATH."""
        return shutil.which(self.binary) is not None


#: Preference-ordered backends. ``bwrap`` builds a read-only bind of the root
#: filesystem so a fixed ``/`` exists inside the namespace; ``--dev /dev`` and
#: ``--proc /proc`` keep standard runtime mounts available without network.
_SANDBOX_BACKENDS: tuple[SandboxSpec, ...] = (
    SandboxSpec(
        "firejail",
        ("firejail", "--noprofile", "--net=none", "--private", "--caps.drop=all"),
    ),
    SandboxSpec(
        "bwrap",
        (
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
        ),
    ),
    SandboxSpec(
        "nsjail",
        ("nsjail", "-Mo", "--disable-proc", "--net", "none"),
    ),
)


def _resolve_mode(mode: Optional[str]) -> str:
    """Resolve the effective sandbox mode from an explicit arg or the env var."""
    if mode is None:
        mode = os.environ.get("GNN_SANDBOX", _DEFAULT_MODE)
    mode = (mode or _DEFAULT_MODE).strip().lower()
    if mode not in SANDBOX_MODES:
        logger.warning(
            "Unknown GNN_SANDBOX mode %r; falling back to %r",
            mode,
            _DEFAULT_MODE,
        )
        return _DEFAULT_MODE
    return mode


def detect_sandbox() -> Optional[SandboxSpec]:
    """Return the first available sandbox backend, or ``None`` if none exist."""
    for spec in _SANDBOX_BACKENDS:
        if spec.available():
            return spec
    return None


def wrap_command(command: List[str], spec: SandboxSpec) -> List[str]:
    """Return ``command`` prefixed with the sandbox backend's isolation flags."""
    return list(spec.prefix) + list(command)


def run_sandboxed(
    command: List[str],
    *,
    timeout: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ``command`` under a sandbox according to ``mode``.

    Returns a uniform envelope regardless of sandbox availability, so callers
    never have to branch on the failure mode:

    - ``sandboxed`` (bool): whether the command was actually wrapped.
    - ``sandbox`` (str|None): backend binary name, or ``None``.
    - ``mode`` (str): the effective mode used.
    - ``blocked`` (bool): True only when ``require`` could not find a backend.
    - ``success`` (bool), ``return_code`` (int), ``stdout``/``stderr`` (str).
    """
    effective = _resolve_mode(mode)
    spec = detect_sandbox() if effective != "off" else None

    envelope: Dict[str, Any] = {
        "sandboxed": False,
        "sandbox": None,
        "mode": effective,
        "blocked": False,
        "success": False,
        "return_code": -1,
        "stdout": "",
        "stderr": "",
    }

    if effective == "require" and spec is None:
        envelope["blocked"] = True
        envelope["error"] = (
            "GNN_SANDBOX=require but no sandbox backend (firejail/bwrap/nsjail) "
            "is installed; refusing to execute rendered script unsandboxed."
        )
        logger.error(envelope["error"])
        return envelope

    if spec is None:
        if effective == "prefer":
            logger.warning(
                "GNN_SANDBOX=prefer but no sandbox backend found; "
                "running rendered script unsandboxed."
            )
        final_command = list(command)
    else:
        final_command = wrap_command(command, spec)
        envelope["sandboxed"] = True
        envelope["sandbox"] = spec.binary

    try:
        completed = subprocess.run(  # nosec B603 — command list, no shell
            final_command,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            check=False,
        )
        envelope["return_code"] = completed.returncode
        envelope["success"] = completed.returncode == 0
        envelope["stdout"] = completed.stdout or ""
        envelope["stderr"] = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        envelope["error"] = f"Sandboxed execution timed out after {timeout}s"
        envelope["error_type"] = "TimeoutExpired"
        envelope["stdout"] = (exc.stdout or "") if capture_output else ""
        envelope["stderr"] = (exc.stderr or "") if capture_output else ""
    except Exception as exc:  # noqa: BLE001 — normalize every failure mode
        envelope["error"] = str(exc)
        envelope["error_type"] = type(exc).__name__
        envelope["stderr"] = str(exc)

    return envelope


# Convenience re-export so callers can reason about the module's surface without
# importing private helpers.
__all__ = [
    "SANDBOX_MODES",
    "SandboxSpec",
    "detect_sandbox",
    "wrap_command",
    "run_sandboxed",
]
