#!/usr/bin/env python3
"""Line-count band audit for the GNN pipeline (SCOPE-2026-09-23 M-13 / M-01).

Walks ``src/gnn/**/*.py`` and emits a deterministic table (path, lines,
bytes) for every file above the reporting band: >1200 physical lines OR
>40960 bytes (40 KiB). Exit status:

- default: exit 1 when any file exceeds the hard band (>1200 lines OR
  >49152 bytes / 48 KiB), exit 0 otherwise
- ``--soft``: print the table but always exit 0 (advisory sweep)

Physical line count uses ``bytes.splitlines()`` (a final unterminated line
still counts); byte counts are raw file sizes. Output is sorted by lines
descending, then bytes descending, then path — fully deterministic: file
reads only, no network, no subprocess.

Standalone tool, intentionally not wired into the gate runners (justfile /
local-gates.yml / ci.yml hardcode their gate lists, and the tree currently
held 17 raw .py files over 1200 lines at the 2026-09-24 census sweep (HEAD 9fb81279e: 11 tracked M-01 + 6 extras), so a wired hard gate would be red
on landing). Run it directly; ``--json`` gives machine-readable output for
other tooling:

    uv run --no-sync python scripts/audit_line_bands.py
    uv run --no-sync python scripts/audit_line_bands.py --soft
    uv run --no-sync python scripts/audit_line_bands.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOT = REPO_ROOT / "src" / "gnn"

REPORT_LINES = 1200
REPORT_BYTES = 40 * 1024
HARD_LINES = 1200
HARD_BYTES = 48 * 1024


def _measure_file(path: Path, base: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(base)),
        "lines": len(data.splitlines()),
        "bytes": len(data),
    }


def audit(root: Path, base: Path | None = None) -> list[dict[str, Any]]:
    """Return over-band rows (path/lines/bytes), lines descending.

    Paths are reported relative to ``base`` (default: the repo root, so
    the default scan reports ``src/gnn/...``); tests pass a tmp_path base.
    """
    base = REPO_ROOT if base is None else base
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        if not path.is_file():
            continue
        row = _measure_file(path, base)
        if row["lines"] > REPORT_LINES or row["bytes"] > REPORT_BYTES:
            rows.append(row)
    rows.sort(key=lambda r: (-r["lines"], -r["bytes"], r["path"]))
    return rows


def over_hard_band(row: dict[str, Any]) -> bool:
    return bool(row["lines"] > HARD_LINES or row["bytes"] > HARD_BYTES)


def render_text(rows: list[dict[str, Any]]) -> str:
    headers = ("path", "lines", "bytes")
    if not rows:
        return "No files above the reporting band (>1200 lines or >40960 bytes).\n"
    path_w = max(len(headers[0]), *(len(r["path"]) for r in rows))
    lines_w = max(len(headers[1]), *(len(str(r["lines"])) for r in rows))
    bytes_w = max(len(headers[2]), *(len(str(r["bytes"])) for r in rows))
    out = [f"{headers[0]:<{path_w}}  {headers[1]:>{lines_w}}  {headers[2]:>{bytes_w}}"]
    out.append(f"{'-' * path_w}  {'-' * lines_w}  {'-' * bytes_w}")
    for r in rows:
        out.append(
            f"{r['path']:<{path_w}}  {r['lines']:>{lines_w}}  {r['bytes']:>{bytes_w}}"
        )
    out.append("")
    out.append(
        f"{len(rows)} file(s) above the reporting band "
        f"(> {REPORT_LINES} lines or > {REPORT_BYTES} bytes)."
    )
    return "\n".join(out) + "\n"


def render_json(rows: list[dict[str, Any]]) -> str:
    payload = {
        "band": {
            "report_lines": REPORT_LINES,
            "report_bytes": REPORT_BYTES,
            "hard_lines": HARD_LINES,
            "hard_bytes": HARD_BYTES,
        },
        "over_hard_count": sum(1 for r in rows if over_hard_band(r)),
        "files": [{**r, "over_hard": over_hard_band(r)} for r in rows],
    }
    return json.dumps(payload, indent=2) + "\n"


def main(
    argv: list[str] | None = None,
    scan_root: Path | None = None,
    base: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Audit src/gnn for line-count band regressions (M-13)."
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        help="report over-band files but always exit 0",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of a text table",
    )
    args = parser.parse_args(argv)

    root = scan_root if scan_root is not None else SCAN_ROOT
    base = REPO_ROOT if base is None else base
    rows = audit(root, base=base)

    print(render_json(rows) if args.json else render_text(rows), end="")

    if args.soft:
        return 0
    return 1 if any(over_hard_band(r) for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
