#!/usr/bin/env python3
"""
Shared pre-execution security gate for GNN rendered-script execution.

RED_TEAM_REVIEW V-01/V-06: rendered code must be scanned BEFORE it runs.
Historically only the Step 12 processor path (``execute.processor``) applied
this gate; the ``GNNExecutor`` dispatch path behind the MCP tools
(``execute_gnn_model_mcp`` → ``execute_simulation_from_gnn``) ran scripts
with no scan at all. This module extracts the gate into ONE helper so both
call sites share identical semantics:

- Default: scan via ``gnn.security.processor.scan_script_for_execution`` and
  block on any finding at or above the configured severity.
- ``GNN_ALLOW_UNSAFE_EXEC=1``: explicit operator opt-out, honored at every
  call site (the ONLY way past a failed import).
- Import failure of the security module: hard-block (fail closed), never
  silently skip. A missing scanner must not become a silent bypass.
Lean dispatch (``GNNExecutor`` → ``execute_gnn_model(..., "lean")``) executes
``.md`` documents through the fep-lean bridge, so a ``.md`` destined for lean
is gate-checked too: its fenced ``python``/``julia`` blocks are scanned with
the same rendered-script verdict machinery (prose and GNN-notation fences are
data and are not scanned).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

#: Environment escape hatch: set to "1" to bypass the pre-execution security
#: gate (trusted-local research use only; see SECURITY.md).
GNN_ALLOW_UNSAFE_EXEC_ENV = "GNN_ALLOW_UNSAFE_EXEC"


# Markdown documents dispatched to the Lean verifier (``fep-lean bridge
# verify-document``) are not inert data: their fenced code blocks are
# executable-shaped content. Only these fence languages are scanned; prose and
# GNN-notation fences are data.
_SCANNED_FENCE_SUFFIX: dict[str, str] = {
    "python": ".py",
    "py": ".py",
    "julia": ".jl",
    "jl": ".jl",
}
_FENCE_RE = re.compile(
    r"^[ \t]*```[ \t]*([A-Za-z0-9_+-]+)[^\n]*\n(.*?)(?:^[ \t]*```|\Z)",
    re.MULTILINE | re.DOTALL,
)
__all__ = ["check_script_allowed", "allow_unsafe_exec", "GNN_ALLOW_UNSAFE_EXEC_ENV"]


def allow_unsafe_exec() -> bool:
    """Whether the operator has explicitly opted out of the pre-exec gate."""
    return os.environ.get(GNN_ALLOW_UNSAFE_EXEC_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def check_script_allowed(script_path: Path) -> Dict[str, Any]:
    """Scan ``script_path`` and return a uniform gate verdict.

    Returns a dict with keys:

    - ``ok`` (bool): True iff execution may proceed.
    - ``overridden`` (bool): True when ``GNN_ALLOW_UNSAFE_EXEC`` bypassed
      the scan.
    - ``blocked`` (list): scanner finding dicts (empty when allowed or
      unavailable).
    - ``reason`` (str): human-readable detail for a block (or the override).
    - ``error_type`` (str, on block): always ``"SecurityGateBlocked"``.

    The gate fails CLOSED: if the security module cannot be imported, the
    verdict blocks execution (log at ERROR). The only bypass is the explicit
    ``GNN_ALLOW_UNSAFE_EXEC`` environment opt-out.
    """
    if allow_unsafe_exec():
        return {
            "ok": True,
            "overridden": True,
            "blocked": [],
            "reason": "GNN_ALLOW_UNSAFE_EXEC set; pre-exec security gate bypassed",
        }

    try:
        from gnn.security.processor import scan_script_for_execution
    except ImportError as exc:
        logger.error(
            "Pre-exec security gate unavailable "
            "(gnn.security.processor import failed: %s); hard-blocking execution",
            exc,
        )
        return {
            "ok": False,
            "overridden": False,
            "blocked": [],
            "error_type": "SecurityGateBlocked",
            "reason": (
                "Security scanner module unavailable; refusing to execute "
                f"({exc}). Set GNN_ALLOW_UNSAFE_EXEC=1 to override."
            ),
        }
    if script_path.suffix.lower() == ".md":
        return _scan_lean_document(script_path, scan_script_for_execution)

    verdict = scan_script_for_execution(script_path)
    if verdict.get("ok", True):
        return {"ok": True, "overridden": False, "blocked": [], "reason": ""}

    blocked = verdict.get("blocked", [])
    detail = "; ".join(
        f"{b.get('vulnerability_type', 'unknown')}@{b.get('line', '?')}"
        for b in blocked[:5]
    )
    return {
        "ok": False,
        "overridden": False,
        "blocked": blocked,
        "error_type": "SecurityGateBlocked",
        "reason": detail,
    }


def _scan_lean_document(
    document_path: Path, scan_script_for_execution: Any
) -> Dict[str, Any]:
    """Scan a ``.md`` document destined for Lean (fep-lean bridge) execution.

    Lean dispatch executes markdown documents, so the same gate must apply.
    GNN documents are prose plus fenced notation, which the scanner cannot
    parse as code — only fenced ``python``/``julia`` blocks are executable-
    shaped, so each of those is written to a temp file with its real suffix
    and scanned with the exact ``.py``/``.jl`` verdict machinery. An unparseable
    Python fence fails closed (same rule as a rendered script); a Julia fence
    degrades to the advisory sweep exactly like a rendered ``.jl``.

    Unreadable documents fail closed.
    """
    try:
        content = document_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.error("Pre-exec security gate could not read %s: %s", document_path, exc)
        return {
            "ok": False,
            "overridden": False,
            "blocked": [],
            "error_type": "SecurityGateBlocked",
            "reason": f"Lean document unreadable, refusing to execute ({exc})",
        }

    blocked: list[dict[str, Any]] = []
    for match in _FENCE_RE.finditer(content):
        suffix = _SCANNED_FENCE_SUFFIX.get(match.group(1).lower())
        if suffix is None:
            continue
        block = match.group(2)
        with tempfile.NamedTemporaryFile(
            "w", suffix=suffix, delete=False, encoding="utf-8"
        ) as handle:
            handle.write(block)
            temp_path = Path(handle.name)
        try:
            verdict = scan_script_for_execution(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
        if not verdict.get("ok", True):
            blocked.extend(verdict.get("blocked", []))

    if blocked:
        return {
            "ok": False,
            "overridden": False,
            "blocked": blocked,
            "error_type": "SecurityGateBlocked",
            "reason": "; ".join(
                f"{b.get('vulnerability_type', 'unknown')}@{b.get('line', '?')}"
                for b in blocked[:5]
            ),
        }
    return {"ok": True, "overridden": False, "blocked": [], "reason": ""}
