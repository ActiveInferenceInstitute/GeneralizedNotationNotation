#!/usr/bin/env python3
"""Shared utilities for scripts/ — reducing duplication across audit/check scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import FrozenSet, List, Optional, Set


def repo_root() -> Path:
    """Return the repository root (two levels up from scripts/lib/)."""
    return Path(__file__).resolve().parents[2]


# Common skip-path fragments used across audit scripts
DEFAULT_SKIP_PARTS: FrozenSet[str] = frozenset(
    {
        ".git",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "archive",
        "build",
        "dist",
        "node_modules",
        "output",
    }
)


# Common generated-output-path detectors
GENERATED_DIR_SUFFIXES = frozenset({"_outputs", "_output"})
GENERATED_DIR_PREFIXES = frozenset({"activeinference_outputs_"})


def is_generated_output(rel: Path) -> bool:
    """Return True when a relative path matches a generated-output pattern."""
    parts = rel.parts
    if not parts:
        return False
    if "pomdp_gridworld_outputs" in parts:
        return True
    for part in parts:
        # e.g. "activeinference_outputs_pymdp"
        if any(part.startswith(p) for p in GENERATED_DIR_PREFIXES):
            return True
        # e.g. "10_ontology_output"
        if any(part.endswith(s) for s in GENERATED_DIR_SUFFIXES):
            return True
    return False


def should_skip_path(
    path: Path, root: Path, extra_skip_parts: Optional[Set[str]] = None
) -> bool:
    """Return True when a path should be excluded from scanning.

    Args:
        path: The file path to check.
        root: Repository root for relative path resolution.
        extra_skip_parts: Additional path fragments to treat as skippable.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    skip_parts = DEFAULT_SKIP_PARTS
    if extra_skip_parts:
        skip_parts = skip_parts | frozenset(extra_skip_parts)
    if any(part in skip_parts for part in rel.parts):
        return True
    return is_generated_output(rel)


def add_strict_flag(parser: argparse.ArgumentParser) -> None:
    """Add a standard ``--strict`` flag to a script's argument parser."""
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any finding is present.",
    )


def exit_with_findings(count: int, strict: bool, prefix: str = "") -> int:
    """Print a finding count summary and return the appropriate exit code.

    Args:
        count: Number of findings.
        strict: Whether ``--strict`` was passed.
        prefix: Optional label for the summary line.

    Returns:
        0 when count == 0 or not strict; 1 when strict and count > 0.
    """
    label = f"{prefix}: " if prefix else ""
    if count == 0:
        print(f"{label}no findings.")
        return 0
    print(f"{label}{count} finding(s) — use --strict to fail.")
    return 1 if strict else 0
