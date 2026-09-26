#!/usr/bin/env python3
"""Run the empirical complexity benchmark harness over the fixed corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gnn.analysis.complexity.benchmark import run_complexity_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("input/gnn_files"),
        help="Directory containing the fixed GNN corpus models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/cross_framework"),
        help="Directory for benchmark artifacts and the two receipts",
    )
    parser.add_argument(
        "--frameworks",
        default="all",
        help="Comma-separated frameworks to benchmark (or 'all'/'lite')",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Benchmark repeats K per rendered script",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose step logging")
    args = parser.parse_args(argv)

    try:
        ledger = run_complexity_benchmark(
            args.corpus_dir,
            args.output_dir,
            frameworks=args.frameworks,
            repeats=args.repeats,
            verbose=args.verbose,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        f"Complexity benchmark {ledger['status']}: {len(ledger['rows'])} rows over "
        f"{len(ledger['models'])} models; {ledger['benchmark_receipt']} + "
        f"{ledger['calibration_receipt']}"
    )
    return 0 if str(ledger["status"]).startswith("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
