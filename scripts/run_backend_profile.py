#!/usr/bin/env python3
"""Measure lightweight backend work; use Python 3.15 for profiling.tracing."""

import argparse
import json
from pathlib import Path

from gnn.processing.profiling import profile_directory


def main(argv: list[str] | None = None) -> int:
    """Delegate profiling and emit one machine-readable result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=Path("input/gnn_files"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/backend_profile")
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, help="Explicit file workers (1..256)")
    parser.add_argument("--executor", choices=("process", "thread"))
    args = parser.parse_args(argv)
    try:
        report = profile_directory(
            args.target_dir,
            args.output_dir,
            repeats=args.repeats,
            workers=args.workers,
            executor=args.executor,
        )
    except KeyboardInterrupt:
        print(json.dumps({"status": "cancelled", "error": "Profiling interrupted"}))
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        )
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
