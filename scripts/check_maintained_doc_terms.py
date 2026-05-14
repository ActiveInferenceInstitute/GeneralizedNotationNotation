#!/usr/bin/env python3
"""
Check maintained Markdown for stale compatibility, placeholder, and PyMDP terms.

Generated outputs and archive documentation are intentionally skipped. The check
is non-failing by default; pass ``--strict`` for CI/local enforcement.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ROOT_MARKDOWN = {"README.md", "AGENTS.md", "DOCS.md", "ARCHITECTURE.md", "CLAUDE.md", "SPEC.md", "SKILL.md"}
MAINTAINED_DIRS = (".agent_rules", ".github", "doc", "input", "scripts", "src")
SKIP_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "output",
    "archive",
}


@dataclass(frozen=True)
class Finding:
    path: Path
    line_no: int
    line: str
    description: str


PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bFallbackAgent\b"), "stale PyMDP execution agent name"),
    (re.compile(r"Fallback Circuit Breakers", re.IGNORECASE), "unsupported feature claim"),
    (re.compile(r"\bsimple_simulation\.py\b"), "removed PyMDP execution module"),
    (re.compile(r"\bpymdp_converter\.py\b"), "removed PyMDP render module"),
    (re.compile(r"\bpymdp>=0\.0\.7\b"), "old PyMDP package constraint"),
    (re.compile(r"\bcompatibility shim\b", re.IGNORECASE), "legacy compatibility wording"),
    (re.compile(r"\bthin shim\b", re.IGNORECASE), "legacy shim wording"),
    (re.compile(r"\bcompatibility alias\b", re.IGNORECASE), "legacy alias wording"),
    (re.compile(r"\bbackwards? compatible\b", re.IGNORECASE), "backwards-compatibility claim"),
    (re.compile(r"\bbackwards? compatibility\b", re.IGNORECASE), "backwards-compatibility claim"),
    (re.compile(r"\bNotImplementedError\b"), "placeholder exception in maintained docs"),
    (re.compile(r"\bnot implemented\b", re.IGNORECASE), "placeholder implementation wording"),
]


def _is_generated_path(rel: Path) -> bool:
    parts = rel.parts
    if any(part.startswith("activeinference_outputs_") for part in parts):
        return True
    if rel.name.startswith("round_trip_report_") and rel.suffix == ".md":
        return True
    if any(part.endswith("_outputs") or "_outputs_" in part for part in parts):
        return True
    return "pomdp_gridworld_outputs" in parts


def _should_skip(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    return any(part in SKIP_PARTS for part in rel.parts) or _is_generated_path(rel)


def iter_maintained_markdown(root: Path = ROOT) -> list[Path]:
    files: set[Path] = set()

    for name in ROOT_MARKDOWN:
        path = root / name
        if path.exists() and path.suffix == ".md":
            files.add(path)

    for dirname in MAINTAINED_DIRS:
        base = root / dirname
        if not base.exists():
            continue
        for path in base.rglob("*.md"):
            if not _should_skip(path, root):
                files.add(path)

    return sorted(files)


def scan_files(files: list[Path], root: Path = ROOT) -> list[Finding]:
    findings: list[Finding] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern, description in PATTERNS:
                if pattern.search(line):
                    rel = path.relative_to(root) if path.is_relative_to(root) else path
                    findings.append(Finding(rel, line_no, line.strip(), description))
    return findings


def scan(root: Path = ROOT) -> list[Finding]:
    return scan_files(iter_maintained_markdown(root), root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any finding is present.")
    args = parser.parse_args()

    findings = scan(ROOT)
    if not findings:
        print("check_maintained_doc_terms: no stale maintained-doc terminology found.")
        return 0

    print(f"check_maintained_doc_terms: {len(findings)} finding(s):\n")
    for finding in findings:
        print(f"  {finding.path}:{finding.line_no}: {finding.description}")
        print(f"    {finding.line[:200]}")

    if args.strict:
        return 1
    print("\n(non-strict: exit 0; use --strict to fail)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
