#!/usr/bin/env python3
"""Run resumable, session-wrapped GNN model-family acceptance.

Thin CLI over :func:`pipeline.session_acceptance.run_session_acceptance`. The
acceptance run is checkpointed to a session manifest after every family, so an
interrupted run can be resumed with ``--resume``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.session_acceptance import run_session_acceptance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("input/model_family_manifest.json"),
        help="Path to the model-family manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-family acceptance ledger artifacts",
    )
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Path to the resumable session manifest JSON",
    )
    parser.add_argument(
        "--families",
        default="",
        help="Comma-separated family names to run; defaults to all families",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on family failure")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing session manifest instead of starting fresh",
    )
    args = parser.parse_args(argv)

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    try:
        result = run_session_acceptance(
            args.manifest,
            args.output_dir,
            args.session,
            family_names=families or None,
            strict=args.strict,
            resume=args.resume,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    status = result["status"]
    print(json.dumps(status, indent=2))
    if args.strict and not status.get("done"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
