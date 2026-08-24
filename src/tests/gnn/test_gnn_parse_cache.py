#!/usr/bin/env python3
"""
Regression tests for gnn.parse_cache — incremental section parse cache.

Pins hit/miss semantics, invalidation (per-file and whole-cache), statistics,
and thread-safety of the shared counters.
"""

import sys
import threading
from pathlib import Path
from typing import Any, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.parse_cache import ParseCache  # noqa: E402


class TestParseCacheHitMiss:
    """Definition of hit/miss behaviour."""

    @pytest.mark.unit
    def test_miss_then_store_then_hit(self, tmp_path: Path) -> None:
        cache = ParseCache(cache_dir=tmp_path / "cache")
        # First lookup for unchanged content is a miss.
        assert cache.get_section("a.gnn", "StateSpace", "x[1]") is None
        cache.set_section("a.gnn", "StateSpace", "x[1]", {"vars": ["x"]})
        # Second lookup of identical content is a hit with stored data.
        got = cache.get_section("a.gnn", "StateSpace", "x[1]")
        assert got == {"vars": ["x"]}
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1

    @pytest.mark.unit
    def test_changed_content_is_a_miss(self, tmp_path: Path) -> None:
        cache = ParseCache(cache_dir=tmp_path / "cache")
        cache.set_section("a.gnn", "State", "x[1]", {"n": 1})
        assert (
            cache.get_section("a.gnn", "State", "x[1],[x[2]") is None
        )
        assert cache.stats["misses"] == 1

    @pytest.mark.unit
    def test_different_file_same_section_is_miss(self, tmp_path: Path) -> None:
        cache = ParseCache(cache_dir=tmp_path / "cache")
        cache.set_section("a.gnn", "State", "x[1]", {"n": 1})
        assert cache.get_section("b.gnn", "State", "x[1]") is None
        assert cache.get_section("a.gnn", "State", "x[1]") is not None

    @pytest.mark.unit
    def test_hit_ratio(self, tmp_path: Path) -> None:
        cache = ParseCache(cache_dir=tmp_path / "cache")
        cache.set_section("a.gnn", "S", "x", {"v": 1})
        cache.get_section("a.gnn", "S", "x")  # hit
        cache.get_section("a.gnn", "S", "y")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_ratio"] == 0.5


class TestParseCacheInvalidation:
    """File-scoped and whole-cache invalidation."""

    @pytest.mark.unit
    def test_invalidate_single_file(self, tmp_path: Path) -> None:
        cache = ParseCache(cache_dir=tmp_path / "cache")
        cache.set_section("a.gnn", "S", "x", {"v": 1})
        cache.set_section("b.gnn", "S", "x", {"v": 2})
        count = cache.invalidate(file_path="a.gnn")
        assert count >= 1
        # b.gnn entry survives.
        assert cache.get_section("b.gnn", "S", "x") is not None
        # a.gnn entry is gone.
        cache.get_section("a.gnn", "S", "x")
        assert cache.stats["misses"] == 1

    @pytest.mark.unit
    def test_invalidate_calls_reset_lookup(self, tmp_path: Path) -> None:
        cache = ParseCache(cache_dir=tmp_path / "cache")
        cache.set_section("a.gnn", "S", "x", {"v": 1})
        assert cache.get_section("a.gnn", "S", "x") == {"v": 1}
        cache.invalidate()
        assert cache.get_section("a.gnn", "S", "x") is None

    @pytest.mark.unit
    def test_invalidate_writes_new_file_after_clear(self, tmp_path: Path) -> None:
        cache = ParseCache(cache_dir=tmp_path / "cache")
        cache.set_section("a.gnn", "S", "new", {"v": 9})
        cache.get_section("a.gnn", "S", "new")
        cache.invalidate()
        # Storing again must still work after a full clear.
        cache.set_section("a.gnn", "S", "new", {"v": 10})
        assert cache.get_section("a.gnn", "S", "new") == {"v": 10}


class TestParseCacheThreadSafety:
    """Concurrent access must not lose counter updates or corrupt data."""

    @pytest.mark.unit
    def test_concurrent_hits_and_misses_are_atomic(self, tmp_path: Path) -> None:
        cache = ParseCache(cache_dir=tmp_path / "cache")
        threads = 8
        keys_per_thread = 20

        errors: List[Exception] = []

        def worker(worker_id: int) -> None:
            try:
                for k in range(keys_per_thread):
                    section = f"W{worker_id}-{k}"
                    # Fresh key -> miss, then store, then hit.
                    assert cache.get_section("t.gnn", section, f"c{k}") is None
                    cache.set_section(
                        "t.gnn", section, f"c{k}", {"id": worker_id, "k": k}
                    )
                    got = cache.get_section("t.gnn", section, f"c{k}")
                    assert got is not None
                    assert got["id"] == worker_id
            except Exception as e:  # pragma: no cover - error reporting
                errors.append(e)

        threads_ = [
            threading.Thread(target=worker, args=(i,)) for i in range(threads)
        ]
        for t in threads_:
            t.start()
        for t in threads_:
            t.join()

        assert errors == []
        # Each key produces exactly one miss and one hit.
        assert cache.stats["misses"] == threads * keys_per_thread
        assert cache.stats["hits"] == threads * keys_per_thread
        assert (
            cache.stats["hits"] + cache.stats["misses"]
            == 2 * threads * keys_per_thread
        )