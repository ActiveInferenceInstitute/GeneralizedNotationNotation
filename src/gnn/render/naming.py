#!/usr/bin/env python3
"""Shared output-naming and atomic-write helpers for the render module.

Single source of truth for the filesystem-safe output stem used by
``render.processor`` and ``render.pomdp_processor`` (previously duplicated
verbatim in both), and for the temp-file-plus-``os.replace`` write pattern
used by framework renderers (previously inlined per renderer).
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

_STEM_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")

#: Output stems longer than this are truncated (keeps artifact names bounded).
MAX_STEM_LENGTH = 120


def safe_output_stem(value: Any, fallback: str = "model") -> str:
    """Return a filesystem-safe stem for an output artifact name.

    Non-alphanumeric characters (except ``_``, ``-``, ``.``) collapse to
    ``_``; leading/trailing ``.``/``_`` are stripped; the result is capped at
    120 characters. Empty results fall back to *fallback*.

    Args:
        value: Raw stem candidate (usually a model name or filename stem).
        fallback: Replacement when sanitization yields an empty string.

    Returns:
        Sanitized stem, at most 120 characters long.
    """
    stem = _STEM_SANITIZE_RE.sub("_", str(value)).strip("._")
    if not stem:
        return fallback
    return stem[:MAX_STEM_LENGTH]


def atomic_write_text(path: Path | str, content: str) -> Path:
    """Write *content* to *path* atomically.

    The temp file is created in the target's parent directory (same
    filesystem) and moved into place with :func:`os.replace`, so readers
    never observe a partially written artifact.

    Args:
        path: Destination file path (parent created if missing).
        content: Text to write.

    Returns:
        The resolved destination path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        handle.write(content)
        temp_name = handle.name
    os.replace(temp_name, str(destination))
    return destination


__all__ = ["MAX_STEM_LENGTH", "atomic_write_text", "safe_output_stem"]
