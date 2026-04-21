#!/usr/bin/env python3
"""Phase 2.2 regression: LLMCache respects explicit/base output dir overrides.

Before Phase 2.2 the cache directory was hardcoded relative to CWD, making
multi-workspace runs conflict on-disk. The fix accepts an explicit
``cache_dir`` OR a ``base_output_dir`` that resolves through
``pipeline.config.get_output_dir_for_script``.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm.cache import LLMCache  # noqa: E402


def test_llm_cache_respects_explicit_cache_dir(tmp_path):
    target = tmp_path / "my_cache"
    cache = LLMCache(cache_dir=target)
    assert cache.cache_dir == target
    assert target.exists()


def test_llm_cache_resolves_via_base_output_dir(tmp_path):
    base = tmp_path / "pipeline_out"
    cache = LLMCache(base_output_dir=base)
    # Cache dir must be inside base (never leak to CWD-relative path).
    assert base in cache.cache_dir.parents or cache.cache_dir == base / "13_llm_output" / ".cache"
    assert cache.cache_dir.exists()


def test_llm_cache_explicit_overrides_base(tmp_path):
    """When both are provided, cache_dir wins — explicit is always stronger."""
    explicit = tmp_path / "explicit"
    base = tmp_path / "base"
    cache = LLMCache(cache_dir=explicit, base_output_dir=base)
    assert cache.cache_dir == explicit


def test_llm_cache_roundtrip(tmp_path):
    """A put/get cycle works with any of the resolution paths."""
    cache = LLMCache(cache_dir=tmp_path / "cache")
    cache.put("hello", "modelX", "promptA", "response-bytes")
    assert cache.get("hello", "modelX", "promptA") == "response-bytes"
    assert cache.hits >= 1
    assert cache.writes >= 1


def test_llm_cache_miss_returns_none(tmp_path):
    cache = LLMCache(cache_dir=tmp_path / "cache")
    assert cache.get("never_stored", "m", "p") is None
    assert cache.misses >= 1
