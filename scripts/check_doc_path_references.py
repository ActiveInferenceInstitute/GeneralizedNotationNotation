#!/usr/bin/env python3
"""Fail when maintained docs cite known-nonexistent ``src/**.py`` modules.

Companion to the TO-DO "Stale singular module paths" sweep: a handful of
pre-restructuring module paths no longer exist, yet maintained documentation
still cites them, planting authoritative-looking wrong imports. This gate
tracks exactly that registered residue class — no other citation-validity
concerns (schematic module names such as ``src/X_module.py``, or unrelated
stale paths, are intentionally out of scope here).

Mechanics: ``RESIDUE_PATHS`` are verified-nonexistent files. Every citation
of one in a maintained Markdown file (root ``README.md``, ``docs/**/*.md``,
``src/gnn/**/*.md`` minus the historical exclusions shared with
``scripts/audit_validate_surface.py``) is counted. The count is capped at
``REGISTERED_RESIDUE_COUNT`` (the state at sweep registration, also recorded
in TO-DO); exceeding the cap means a NEW stale citation appeared and fails
the gate. When a path is restored on disk, its citations become valid and
stop counting. When the TO-DO sweep fixes sites, lower the cap — 0 makes the
gate strict.

Usage: uv run --extra dev python scripts/check_doc_path_references.py
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITED_PATH_RE = re.compile(r"\bsrc/[\w./-]+\.py\b")

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "output",
    "build",
    "dist",
    "__pycache__",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    "site-packages",
    ".tox",
    ".eggs",
}
HISTORICAL_DOC_FILES = {"CHANGELOG.md", "VERSION_MAP.md"}
HISTORICAL_DOC_PREFIXES = (
    "docs/development/fleet-logs/",
    "docs/other/",
    "docs/sympy/",
)

# Verified nonexistent on disk at gate registration (2026-09-08). If one of
# these ever exists again, its citations are valid and stop counting.
RESIDUE_PATHS = (
    "src/gnn/parser.py",
    "src/gnn/schema.py",
    "src/gnn/schema_validator.py",
    "src/gnn/cross_format_validator.py",
)

# Citation count at sweep registration (see TO-DO). Exceeding this means a
# NEW stale citation appeared. Lower as the sweep fixes sites; 0 = strict.
REGISTERED_RESIDUE_COUNT = 21


def iter_doc_files() -> Iterator[Path]:
    readme = ROOT / "README.md"
    if readme.exists():
        yield readme
    for base in (ROOT / "docs", ROOT / "src" / "gnn"):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.md")):
            rel = path.relative_to(ROOT).as_posix()
            if path.name in HISTORICAL_DOC_FILES or rel.startswith(
                HISTORICAL_DOC_PREFIXES
            ):
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            yield path


def main() -> int:
    residue = {cited for cited in RESIDUE_PATHS if not (ROOT / cited).exists()}
    hits: list[tuple[str, int, str]] = []
    for path in iter_doc_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in CITED_PATH_RE.finditer(line):
                cited = match.group(0)
                if cited in residue:
                    hits.append((rel, line_no, cited))

    if len(hits) > REGISTERED_RESIDUE_COUNT:
        print(
            f"check_doc_path_references: {len(hits)} citations of known-"
            f"nonexistent paths exceed the registered residue count "
            f"({REGISTERED_RESIDUE_COUNT}). New stale citation(s):"
        )
        for rel, line_no, cited in hits[REGISTERED_RESIDUE_COUNT:]:
            print(f"  {rel}:{line_no}: {cited}")
        print("Cite the real module path (see the TO-DO stale-path sweep).")
        return 1

    print(
        "check_doc_path_references: "
        f"{len(hits)}/{REGISTERED_RESIDUE_COUNT} registered stale citations "
        "pending the TO-DO sweep (cap not exceeded)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
