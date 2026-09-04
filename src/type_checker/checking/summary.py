"""Typed aggregation of type-check validation results.

The type checker's directory run produces a per-file validation dict
plus a global results dict. Downstream consumers (the website step, the
model-family acceptance runner, and report tooling) want a small,
predictable summary they can rely on without re-deriving counts from the
raw per-file structure. :func:`summarize_type_check_results` is the pure,
typed entry point for that.

Kept free of I/O and side effects so it composes into reports, tests,
and MCP responses identically.
"""

from __future__ import annotations

from typing import Any, Mapping, TypedDict

__all__ = ["ValidationSummary", "summarize_type_check_results"]


class ValidationSummary(TypedDict):
    """Compact, serializable view of one type-checker directory run."""

    files_processed: int
    success: bool
    valid_files: int
    invalid_files: int
    warning_files: int
    total_errors: int
    total_warnings: int
    complexity_tiers: dict[str, int]
    total_parameters: int
    total_estimated_memory_bytes: int


def summarize_type_check_results(results: Mapping[str, Any]) -> ValidationSummary:
    """Aggregate a directory-run results dict into a :class:`ValidationSummary`.

    ``results`` is the structure built by
    :meth:`GNNTypeChecker.validate_gnn_files` — a mapping of file path to
    per-file validation data (when called with that shape) **or** the
    directory-run envelope dict that carries ``validation_results``. Both
    shapes are accepted so callers can pass whichever they happen to hold.
    """
    files = _validation_files(results)
    valid_files = 0
    invalid_files = 0
    warning_files = 0
    total_errors = 0
    total_warnings = 0
    complexity_tiers: dict[str, int] = {}
    total_parameters = 0
    total_estimated_memory_bytes = 0

    for entry in files:
        is_valid = bool(entry.get("valid", entry.get("is_valid", False)))
        errors = entry.get("errors", [])
        warnings = entry.get("warnings", [])
        total_errors += len(errors)
        total_warnings += len(warnings)
        if is_valid:
            valid_files += 1
        else:
            invalid_files += 1
        if is_valid and warnings:
            warning_files += 1

        resources = entry.get("resource_estimation") or {}
        tier = str(resources.get("complexity_tier") or "unknown")
        complexity_tiers[tier] = complexity_tiers.get(tier, 0) + 1
        total_parameters += int(resources.get("total_parameters", 0) or 0)
        total_estimated_memory_bytes += int(
            resources.get("estimated_memory_bytes", 0) or 0
        )

    return ValidationSummary(
        files_processed=len(files),
        success=invalid_files == 0,
        valid_files=valid_files,
        invalid_files=invalid_files,
        warning_files=warning_files,
        total_errors=total_errors,
        total_warnings=total_warnings,
        complexity_tiers=complexity_tiers,
        total_parameters=total_parameters,
        total_estimated_memory_bytes=total_estimated_memory_bytes,
    )


def _validation_files(results: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Normalize the two accepted result shapes into a flat file list."""
    if "validation_results" in results and isinstance(
        results["validation_results"], list
    ):
        return list(results["validation_results"])
    if all(isinstance(value, Mapping) for value in results.values()):
        return list(results.values())
    return []
