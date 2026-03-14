#!/usr/bin/env python3
"""
Incremental Parse Cache — Hash-based section re-parsing avoidance.

Hashes each `## Section` independently and only re-parses changed sections.
Stores cached parse results in a caller-supplied directory.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ParseCache:
    """
    Section-level parse cache for GNN files.

    Each section is hashed independently. On subsequent parses,
    unchanged sections return cached results instantly.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(".parse_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stats = {"hits": 0, "misses": 0}

    def get_section(self, file_path: str, section_name: str, section_content: str) -> Optional[Dict[str, Any]]:
        """
        Look up cached parse result for a section.

        Args:
            file_path: Source file path.
            section_name: Section header (e.g., "StateSpaceBlock").
            section_content: Raw section content string.

        Returns:
            Cached parse dict if hit, None if miss.
        """
        key = self._cache_key(file_path, section_name, section_content)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                self._stats["hits"] += 1
                logger.debug(f"Cache hit: {section_name} ({key[:8]})")
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Corrupted cache entry, treating as miss: %s", e)

        self._stats["misses"] += 1
        return None

    def set_section(self, file_path: str, section_name: str, section_content: str, result: Dict[str, Any]) -> None:
        """
        Store parse result for a section.

        Args:
            file_path: Source file path.
            section_name: Section header.
            section_content: Raw section content.
            result: Parse result dict to cache.
        """
        key = self._cache_key(file_path, section_name, section_content)
        cache_file = self.cache_dir / f"{key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(result, f)
            logger.debug(f"Cache write: {section_name} ({key[:8]})")
        except OSError as e:
            logger.warning(f"Cache write failed: {e}")

    def invalidate(self, file_path: Optional[str] = None) -> int:
        """
        Invalidate cached entries.

        Args:
            file_path: If given, only invalidate entries for this file.
                       If None, clear entire cache.

        Returns:
            Number of entries invalidated.
        """
        count = 0
        if file_path:
            prefix = hashlib.sha256(file_path.encode()).hexdigest()[:8]
            for f in self.cache_dir.glob(f"{prefix}_*.json"):
                f.unlink()
                count += 1
        else:
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
                count += 1

        if count:
            logger.info(f"🗑️ Cache invalidated: {count} entries")
        return count

    @property
    def stats(self) -> Dict[str, Any]:
        """Cache hit/miss statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        ratio = self._stats["hits"] / total if total > 0 else 0
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_ratio": round(ratio, 3),
        }

    def _cache_key(self, file_path: str, section_name: str, section_content: str) -> str:
        """Generate unique cache key for a file+section+content combination."""
        prefix = hashlib.sha256(file_path.encode()).hexdigest()[:8]
        content_hash = hashlib.sha256(section_content.encode()).hexdigest()[:12]
        return f"{prefix}_{section_name}_{content_hash}"
