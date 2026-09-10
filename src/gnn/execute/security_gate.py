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
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

#: Environment escape hatch: set to "1" to bypass the pre-execution security
#: gate (trusted-local research use only; see SECURITY.md).
GNN_ALLOW_UNSAFE_EXEC_ENV = "GNN_ALLOW_UNSAFE_EXEC"

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
