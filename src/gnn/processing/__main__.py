"""Headless processing CLI with a single JSON result on stdout."""

import argparse
import json
from pathlib import Path

from gnn.processing import process_gnn_directory


def main(argv: list[str] | None = None) -> int:
    """Process authored inputs and return zero only on complete success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=Path("input/gnn_files"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args(argv)
    result = process_gnn_directory(
        args.target_dir, args.output_dir, parallel=args.parallel
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
