#!/usr/bin/env python3
"""
Thin CLI: emit durable v3 run manifests for a COMPLETED pipeline run.

Walks a finished run's output directory, builds a content-addressed
:class:`StreamManifest` per produced JSON artifact, reconstructs an
:class:`ExecutionTrace` from the run summary (or step dirs), writes them plus an
index JSON, prints the summary, and re-verifies the result. This script reads
on-disk data only — it NEVER executes the pipeline or any container.

Exits non-zero if the trace integrity check fails or any emitted manifest fails
re-validation (e.g. a tampered artifact).

Usage:
    python scripts/emit_run_manifest.py output/ --out output/v3_run_manifest
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.run_manifest import (  # noqa: E402
    emit_run_manifests,
    verify_run_manifests,
)


def main() -> int:
    """Parse args, emit + verify manifests, print the summary, return exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Emit durable v3 run manifests + execution trace for a completed "
            "pipeline run output directory."
        )
    )
    parser.add_argument(
        "run_output_dir",
        help="Output directory of a completed pipeline run (e.g. output/).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Destination directory for emitted artifacts "
            "(default: <run_output_dir>/v3_run_manifest)."
        ),
    )
    args = parser.parse_args()

    summary = emit_run_manifests(args.run_output_dir, manifest_out=args.out)

    print(f"manifest_dir:        {summary['manifest_dir']}")
    print(f"stream_count:        {summary['stream_count']}")
    print(f"trace_event_count:   {summary['trace_event_count']}")
    print(f"trace_integrity_ok:  {summary['trace_integrity_ok']}")

    if not summary["trace_integrity_ok"]:
        print("ERROR: trace integrity check failed.", file=sys.stderr)
        return 1

    problems = verify_run_manifests(summary["manifest_dir"], args.run_output_dir)
    if problems:
        print(f"Re-validation found {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("Re-validation: clean (0 problems).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
