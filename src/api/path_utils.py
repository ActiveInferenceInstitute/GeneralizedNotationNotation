#!/usr/bin/env python3
"""Shared path validation for local-only API execution."""

from pathlib import Path
from typing import Union


class PathValidationError(ValueError):
    """Raised when an API path violates repository-local execution policy."""


def get_repo_root() -> Path:
    """Return the repository root for API path boundary checks."""
    return Path(__file__).parent.parent.parent.resolve()


def resolve_repo_path(
    path_value: Union[str, Path],
    *,
    purpose: str,
    must_exist: bool = False,
    must_be_dir: bool = True,
    create: bool = False,
) -> Path:
    """Resolve a caller-provided path and enforce repository-local boundaries."""
    raw = Path(path_value).expanduser()
    if not str(raw):
        raise PathValidationError(f"{purpose} path must not be empty")

    repo_root = get_repo_root()
    candidate = raw if raw.is_absolute() else repo_root / raw
    resolved = candidate.resolve(strict=False)

    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise PathValidationError(
            f"{purpose} must be within the repository root: {path_value}"
        ) from exc

    if must_exist and not resolved.exists():
        raise PathValidationError(f"{purpose} not found: {path_value}")

    if must_be_dir and resolved.exists() and not resolved.is_dir():
        raise PathValidationError(f"{purpose} must be a directory: {path_value}")

    if create:
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PathValidationError(
                f"{purpose} could not be created: {path_value}: {exc}"
            ) from exc

    return resolved
