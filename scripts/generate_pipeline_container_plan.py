#!/usr/bin/env python3
"""
Thin CLI: generate a hardened container plan for running the GNN pipeline.

Reads the pipeline config YAML, builds a digest-pinned, hardened container plan
via :func:`pipeline.pipeline_container_plan.plan_for_pipeline`, runs the static
security review, writes the serialized plan to ``--out`` (or stdout), and prints
the findings. Exits non-zero if the security review returns any finding — a
generated plan must be clean. This script NEVER executes the pipeline or any
container.

Usage:
    python scripts/generate_pipeline_container_plan.py \
        --config input/config.yaml --image <digest-pinned> --out plan.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.container_plan import serialize_plan  # noqa: E402
from pipeline.pipeline_container_plan import (  # noqa: E402
    plan_for_pipeline,
    review_pipeline_plan,
)


def main() -> int:
    """Parse args, generate + review the plan, write it, return an exit code."""
    parser = argparse.ArgumentParser(
        description="Generate a hardened container plan for the GNN pipeline."
    )
    parser.add_argument(
        "--config",
        default="input/config.yaml",
        help="Path to the pipeline config YAML (default: input/config.yaml).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Digest-pinned image (default: documented placeholder digest).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write the serialized plan JSON (default: stdout).",
    )
    args = parser.parse_args()

    plan = plan_for_pipeline(args.config, image=args.image)
    findings = review_pipeline_plan(plan)

    serialized = serialize_plan(plan)
    if args.out:
        Path(args.out).write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote plan to {args.out}")
    else:
        print(serialized)

    if findings:
        print(f"Security review found {len(findings)} issue(s):", file=sys.stderr)
        for f in findings:
            print(f"  [{f.severity}] {f.code} ({f.spec_name}): {f.message}", file=sys.stderr)
        return 1

    print("Security review: clean (0 findings).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
