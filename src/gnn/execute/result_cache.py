#!/usr/bin/env python3
"""
Execution Result Cache Module

Content-addressed on-disk caching for subprocess execution envelopes.
Eliminates redundant re-execution of unchanged scripts when the cache is
explicitly enabled (``GNN_EXEC_CACHE=1`` or an ``enabled=True`` instance).

Mirrors LLMCache (src/gnn/llm/cache.py): sha256 NUL-joined key payload, JSON entries
under ``<cache_dir>/<key>.json``, corrupt entry == miss, Lock-guarded counters,
explicit ``invalidate() -> int``.

Cache key: sha256(interpreter + script hash + args + cwd + env + capture)
Storage: output/12_execute_output/.cache/<key>.json (gitignored)
"""

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union, cast

logger = logging.getLogger(__name__)

_CACHE_ENABLED_ENV = "GNN_EXEC_CACHE"
_CACHE_ENABLED_TRUTHY = {"1", "true", "yes", "on"}


class ExecutionResultCache:
    """Content-addressed on-disk cache for subprocess execution envelopes."""

    def __init__(
        self, cache_dir: Optional[Path] = None, *, enabled: Optional[bool] = None
    ) -> None:
        """
        Initialize the instance.

        Args:
            cache_dir: Explicit directory for cached envelopes. When None,
                uses ``output/12_execute_output/.cache`` relative to CWD
                (LLMCache style; the ``output/[0-9]*_*_output/`` trees are
                gitignored, so the cache never dirties tracked files).
            enabled: Explicit on/off switch. When None, the cache is enabled
                iff ``GNN_EXEC_CACHE`` is truthy (evaluated at access time).
        """
        self.cache_dir = (
            cache_dir
            if cache_dir is not None
            else Path("output/12_execute_output/.cache")
        )
        self._explicit_enabled = enabled
        self._stats: Dict[str, int] = {"hits": 0, "misses": 0, "writes": 0}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """Whether the cache is active: explicit flag wins, else env opt-in."""
        if self._explicit_enabled is not None:
            return self._explicit_enabled
        return (
            os.environ.get(_CACHE_ENABLED_ENV, "").strip().lower()
            in _CACHE_ENABLED_TRUTHY
        )

    @staticmethod
    def make_key(
        *,
        interpreter: str,
        script_bytes: bytes,
        args: Sequence[str],
        cwd: Optional[str],
        env_overrides: Optional[Mapping[str, str]],
        capture_output: bool,
    ) -> str:
        """Generate a deterministic cache key from the execution inputs.

        Timeout is deliberately NOT part of the key: a different wall-clock
        budget must not invalidate a cached result for identical inputs.
        """
        env_payload = (
            " ".join(f"{k}={v}" for k, v in sorted(env_overrides.items()))
            if env_overrides
            else ""
        )
        payload = "\x00".join(
            (
                interpreter,
                hashlib.sha256(script_bytes).hexdigest(),
                " ".join(args),
                cwd or "",
                env_payload,
                "1" if capture_output else "0",
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Return the file path for a given cache key."""
        return self.cache_dir / f"{key}.json"

    def lookup(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Look up a cached execution envelope.

        Args:
            key: Cache key from ``make_key``.

        Returns:
            The cached envelope (json round-trip → fresh dict, no aliasing
            with the stored copy), or None on miss. A missing, corrupt, or
            unreadable entry is a miss.
        """
        cache_file = self._cache_path(key)

        with self._lock:
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        envelope = json.load(f)
                    self._stats["hits"] += 1
                    logger.debug(f"Cache hit: {key[:12]}…")
                    return cast("dict[str, Any]", envelope)
                except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                    logger.debug("Corrupt cache entry, treating as miss: %s", e)

            self._stats["misses"] += 1
            logger.debug(f"Cache miss: {key[:12]}…")
            return None

    def store(self, key: str, envelope: Dict[str, Any]) -> bool:
        """
        Store a successful execution envelope.

        Args:
            key: Cache key from ``make_key``.
            envelope: Execution envelope; only envelopes with
                ``envelope["success"] is True`` are stored (failures,
                timeouts, and cancels are never cached).

        Returns:
            True when stored, False when skipped or the write failed.
        """
        if not envelope.get("success"):
            return False
        # json round-trip: store an isolated copy and fail fast on
        # non-serializable payloads before any file is touched.
        entry = json.loads(json.dumps(envelope))
        cache_file = self._cache_path(key)

        with self._lock:
            try:
                # Lazy mkdir: a default-off cache must not create directories.
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(entry, f)
                self._stats["writes"] += 1
                logger.debug(f"Cache write: {key[:12]}…")
                return True
            except OSError as e:
                logger.warning(f"Failed to write cache entry {key[:12]}…: {e}")
                return False

    def invalidate(self) -> int:
        """
        Invalidate all cached entries.

        Returns:
            Number of entries invalidated.
        """
        count = 0
        with self._lock:
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
                count += 1

        if count:
            logger.info(f"Cache invalidated: {count} entries")
        return count

    @property
    def stats(self) -> Dict[str, int]:
        """Cache hit/miss/write statistics (computed atomically)."""
        with self._lock:
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "writes": self._stats["writes"],
            }


def cache_key_for_script(
    script_path: Union[str, Path],
    *,
    interpreter: str,
    args: Sequence[str] = (),
    cwd: Optional[str] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
    capture_output: bool = True,
) -> str:
    """
    Build the execution cache key for a script on disk.

    Hashes the script bytes; an unreadable or missing file falls back to
    hashing the resolved path string.
    """
    script = Path(script_path)
    try:
        script_bytes = script.read_bytes()
    except OSError:
        script_bytes = str(script.resolve()).encode("utf-8")
    return ExecutionResultCache.make_key(
        interpreter=interpreter,
        script_bytes=script_bytes,
        args=args,
        cwd=cwd,
        env_overrides=env_overrides,
        capture_output=capture_output,
    )
