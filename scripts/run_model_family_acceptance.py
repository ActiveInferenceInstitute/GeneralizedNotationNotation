#!/usr/bin/env python3
"""Run manifest-driven GNN model-family acceptance checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.model_family_acceptance import run_model_family_acceptance


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
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for acceptance ledger artifacts",
    )
    parser.add_argument(
        "--only-steps",
        default="3,5,6,11,12,15,16,23",
        help="Pipeline steps to run for each family; use empty string for full pipeline",
    )
    parser.add_argument(
        "--frameworks",
        default="",
        help="Override manifest renderer/executor frameworks for evidence-step runs",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on family failure")
    args = parser.parse_args(argv)

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    try:
        ledger = run_model_family_acceptance(
            args.manifest,
            args.output_dir,
            family_names=families,
            only_steps=args.only_steps or None,
            frameworks=args.frameworks or None,
            strict=args.strict,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"Model-family acceptance {ledger['status']}: {args.output_dir}")
    return 0 if ledger["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
