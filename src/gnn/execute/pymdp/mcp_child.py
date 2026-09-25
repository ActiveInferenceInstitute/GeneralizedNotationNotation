#!/usr/bin/env python3
"""Child-process driver for MCP-mediated PyMDP simulations.

The MCP ``execute_pymdp_simulation`` tool runs this module in a fresh
process through the shared subprocess envelope
(``gnn.execute.subprocess_envelope.run_subprocess_envelope``), so an
MCP-transported simulation carries the same gating as every other
execution backend: the ``GNN_SANDBOX`` prefix semantics, a bounded
wall clock, process-group kill on timeout, and structured failure
receipts instead of an unbounded in-process run.

Protocol:
    argv[1]  GNN model file path (already validated by the MCP tool).
    argv[2]  Output directory for simulation artifacts.

The driver writes exactly one JSON line to stdout:

    {"success": <bool>, "results": <dict>}

when the PyMDP runner completed (the runner's own verdict carried in
``success``), or ``{"success": false, "error": "<message>"}`` when the
driver failed to produce a runner verdict at all. Exit code 0 means the
driver completed and reported; exit code 1 means no report could be
produced. Diagnostics and library logging go to stderr only, keeping
stdout reserved for the machine-readable report.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _report(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def main(argv: List[str]) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    if len(argv) != 2:
        _report(
            {
                "success": False,
                "error": (
                    "expected exactly <gnn_file> <output_dir>, got "
                    f"{len(argv)} argument(s)"
                ),
            }
        )
        return 1
    gnn_file = Path(argv[0])
    output_dir = Path(argv[1])
    try:
        from .execute_pymdp import execute_from_gnn_file

        success, results = execute_from_gnn_file(
            gnn_file, output_dir, correlation_id="mcp"
        )
    except Exception as exc:  # noqa: BLE001 - the driver reports, never propagates
        logger.error("PyMDP child driver failed: %s", exc, exc_info=True)
        _report({"success": False, "error": str(exc)})
        return 1
    results_payload: Dict[str, Any] = (
        results if isinstance(results, dict) else {"result": results}
    )
    _report({"success": bool(success), "results": results_payload})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
