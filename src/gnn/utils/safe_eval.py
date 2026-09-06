"""Bounded ``ast.literal_eval`` for untrusted GNN parameter strings.

``ast.literal_eval`` is safe against code execution but not against
resource-exhaustion DoS: a deeply nested structure (``[[[[...]]]]``) or a very
long literal can still consume unbounded time and memory. This wrapper enforces
explicit length and nesting-depth limits *before* evaluation and raises
``ValueError`` otherwise, so callers can fall back to their manual parsers the
same way they already do for ``ValueError``/``SyntaxError``.
"""

from __future__ import annotations

import ast
from typing import Any

#: Default maximum accepted literal length, in characters.
DEFAULT_MAX_LEN = 10_000

#: Default maximum bracket nesting depth.
DEFAULT_MAX_DEPTH = 10

#: Larger length bound for matrix/tensor literals. GNN scaling-study
#: fixtures legitimately reach ~2.6M characters (the N=64 B tensor), so the
#: generic ``DEFAULT_MAX_LEN`` scalar bound must not gate them. Depth stays
#: bounded by ``DEFAULT_MAX_DEPTH`` — matrices are only 3 levels deep.
MATRIX_MAX_LEN = 8 * 1024 * 1024  # 8 MiB


def _nesting_depth(text: str) -> int:
    """Return the approximate maximum bracket nesting depth of ``text``."""
    depth = 0
    max_depth = 0
    for ch in text:
        if ch in "[({":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch in "])}":
            depth -= 1
    return max_depth


def safe_literal_eval(
    value: Any,
    *,
    max_len: int = DEFAULT_MAX_LEN,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Any:
    """Evaluate ``value`` as a Python literal, bounded against DoS.

    Args:
        value: The string to evaluate. Non-strings are returned unchanged so
            callers can pass through already-parsed values.
        max_len: Maximum accepted character length.
        max_depth: Maximum bracket nesting depth.

    Returns:
        The evaluated literal.

    Raises:
        ValueError: If ``value`` exceeds the length or depth limits.
        SyntaxError: If ``value`` is not a valid Python literal.
    """
    if not isinstance(value, str):
        return value
    if len(value) > max_len:
        raise ValueError(f"literal exceeds {max_len} characters; refusing to evaluate")
    if _nesting_depth(value) > max_depth:
        raise ValueError(
            f"literal exceeds nesting depth {max_depth}; refusing to evaluate"
        )
    return ast.literal_eval(value)


__all__ = [
    "DEFAULT_MAX_LEN",
    "DEFAULT_MAX_DEPTH",
    "MATRIX_MAX_LEN",
    "safe_literal_eval",
]
