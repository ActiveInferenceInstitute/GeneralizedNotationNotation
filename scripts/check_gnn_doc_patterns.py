#!/usr/bin/env python3
"""
Spot-check markdown under doc/ and src/gnn/ for known-stale GNN documentation patterns.

Does not fail the build by default; run in CI with --strict to exit non-zero on hits.

Usage:
  uv run python scripts/check_gnn_doc_patterns.py
  uv run python scripts/check_gnn_doc_patterns.py --strict
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (regex, description) — tune as docs evolve
PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"from gnn import GNNParser\b"), "obsolete import (use exports from src/gnn/__init__.py)"),
    (re.compile(r"from gnn\.parser import GNNParser\b"), "GNNParser not exported from gnn.parser"),
    (re.compile(r"from gnn\.parsing import"), "gnn.parsing is not the public package layout"),
    (re.compile(r"parsers/serializers\.py"), "serializers live in parsers/*_serializer.py"),
    (re.compile(r"lark_parser\.py"), "stale filename unless reintroduced"),
    (re.compile(r"- Python 3\.12\+"), "use Python >= 3.11 to match pyproject.toml in SPEC stubs"),
    # Stale doc paths (canonical: doc/gnn/reference/gnn_file_structure_doc.md; punctuation: src/gnn/documentation/punctuation.md)
    (re.compile(r"doc/gnn/gnn_file_structure\.md"), "use doc/gnn/reference/gnn_file_structure_doc.md"),
    (re.compile(r"doc/gnn/gnn_punctuation\.md"), "use src/gnn/documentation/punctuation.md or doc/gnn/gnn_syntax.md"),
    (re.compile(r"src/gnn/gnn_file_structure\.md"), "use src/gnn/documentation/file_structure.md"),
    (re.compile(r"src/gnn/gnn_punctuation\.md"), "use src/gnn/documentation/punctuation.md"),
]


def scan(paths: list[Path]) -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    for base in paths:
        if not base.exists():
            continue
        for md in sorted(base.rglob("*.md")):
            if "node_modules" in md.parts or ".git" in md.parts:
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                for pat, desc in PATTERNS:
                    if pat.search(line):
                        violations.append((md, i, line.strip(), desc))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any pattern matches.",
    )
    args = parser.parse_args()

    targets = [ROOT / "doc", ROOT / "src" / "gnn"]
    violations = scan(targets)

    if not violations:
        print("check_gnn_doc_patterns: no banned patterns in doc/ and src/gnn/ (markdown).")
        return 0

    print(f"check_gnn_doc_patterns: {len(violations)} match(es):\n")
    for path, line_no, line, desc in violations:
        rel = path.relative_to(ROOT)
        print(f"  {rel}:{line_no}: {desc}")
        print(f"    {line[:200]}")

    if args.strict:
        return 1
    print("\n(non-strict: exit 0; use --strict to fail)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
