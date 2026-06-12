#!/usr/bin/env python3
"""Run strict semantic fidelity checks for maintained model families."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.semantic_fidelity import run_semantic_fidelity_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("input/model_family_manifest.json"),
        help="Path to the model-family manifest",
    )
    parser.add_argument(
        "--families",
        default="",
        help="Comma-separated family names to run; defaults to all families",
    )
    parser.add_argument(
        "--formats",
        default="json",
        help="Comma-separated serializer/parser formats to check",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for semantic fidelity artifacts",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on mismatch")
    args = parser.parse_args(argv)

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    try:
        ledger = run_semantic_fidelity_gate(
            args.manifest,
            args.output_dir,
            family_names=families,
            formats=formats,
            strict=args.strict,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"Semantic fidelity {ledger['status']}: {args.output_dir}")
    return 0 if ledger["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
