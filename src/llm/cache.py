#!/usr/bin/env python3
"""
LLM Response Cache Module

Content-hash caching for LLM responses. Eliminates redundant API calls
when running the pipeline on unchanged GNN files.

Cache key: sha256(file_content + model_name + prompt_template)
Storage: output/13_llm_output/.cache/<hash>.json
"""

import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class LLMCache:
    """Content-addressed cache for LLM prompt responses."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cached responses.
                       Defaults to output/13_llm_output/.cache/
        """
        self.cache_dir = cache_dir or Path("output/13_llm_output/.cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @staticmethod
    def _make_key(file_content: str, model_name: str, prompt_template: str) -> str:
        """Generate a deterministic cache key from content + model + prompt."""
        payload = f"{file_content}\x00{model_name}\x00{prompt_template}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Return the file path for a given cache key."""
        return self.cache_dir / f"{key}.json"

    def get(
        self, file_content: str, model_name: str, prompt_template: str
    ) -> Optional[str]:
        """
        Look up a cached response.

        Args:
            file_content: Raw GNN file content.
            model_name: Ollama / provider model name.
            prompt_template: The prompt text sent to the LLM.

        Returns:
            Cached response string, or None on miss.
        """
        key = self._make_key(file_content, model_name, prompt_template)
        path = self._cache_path(key)

        if path.exists():
            try:
                with open(path, "r") as f:
                    entry = json.load(f)
                self.hits += 1
                logger.debug(f"Cache HIT: {key[:12]}…")
                return entry.get("response")
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(f"Corrupt cache entry {key[:12]}…: {exc}")
                path.unlink(missing_ok=True)

        self.misses += 1
        logger.debug(f"Cache MISS: {key[:12]}…")
        return None

    def put(
        self,
        file_content: str,
        model_name: str,
        prompt_template: str,
        response: str,
    ) -> None:
        """
        Store a response in the cache.

        Args:
            file_content: Raw GNN file content.
            model_name: Ollama / provider model name.
            prompt_template: The prompt text sent to the LLM.
            response: The LLM response to cache.
        """
        key = self._make_key(file_content, model_name, prompt_template)
        path = self._cache_path(key)

        entry = {
            "key": key,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "content_length": len(file_content),
            "prompt_length": len(prompt_template),
            "response": response,
        }

        try:
            with open(path, "w") as f:
                json.dump(entry, f, indent=2)
            self.writes += 1
            logger.debug(f"Cache WRITE: {key[:12]}…")
        except OSError as exc:
            logger.warning(f"Failed to write cache entry {key[:12]}…: {exc}")

    def summary(self) -> dict:
        """Return cache statistics for logging."""
        total = self.hits + self.misses
        ratio = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_ratio_pct": round(ratio, 1),
            "cache_dir": str(self.cache_dir),
            "entries_on_disk": sum(1 for _ in self.cache_dir.glob("*.json")),
        }

    def clear(self) -> int:
        """Remove all cached entries. Returns count of entries removed."""
        count = 0
        for p in self.cache_dir.glob("*.json"):
            p.unlink()
            count += 1
        logger.info(f"Cache cleared: {count} entries removed")
        return count
