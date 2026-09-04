#!/usr/bin/env python3
"""Shared atomic-file primitives for the pipeline module.

Every durable artifact in ``pipeline`` (stream manifests, execution traces,
run-session checkpoints, run-manifest indexes, history indexes) must never be
observed half-written. The canonical recipe is: write to a unique temporary
file in the *same directory* as the destination (so the rename cannot cross a
filesystem boundary), then ``os.replace`` it into place — atomic on POSIX.

This module is the single implementation of that recipe. It is intentionally
dependency-free (stdlib only) so any ``pipeline`` submodule can import it
without cycles.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["atomic_write_bytes", "atomic_write_text"]


def atomic_write_text(path: Path | str, text: str) -> Path:
    """Atomically write ``text`` to ``path`` and return the resolved path.

    Creates the parent directory when missing. An interrupted write (exception
    or crash before the replace) leaves any prior file at ``path`` intact and
    removes the temporary file.
    """
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    """Atomically write ``data`` to ``path`` and return the resolved path.

    Same guarantees as :func:`atomic_write_text` for binary payloads.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent), prefix=f"{dest.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
    except BaseException:
        # Clean up the temp file if anything went wrong before the replace.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return dest
