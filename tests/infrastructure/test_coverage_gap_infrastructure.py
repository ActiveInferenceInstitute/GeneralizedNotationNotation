#!/usr/bin/env python3
"""
Infrastructure Coverage Gap Tests

Addresses modules that historically had 0% coverage: timeout_manager,
visualization_optimizer. The ``utils/recovery.py`` fallback was removed in
Phase 6 as dead code — ``setup_step_logging`` is covered in place via
``utils/logging/logging_utils``. ``utils/simulation_utils.py`` was removed as
dead code (R4 residue); its only importer was this file.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from gnn.utils.observability.visualization_optimizer import (
    DataSampler,
    VisualizationCache,
    VisualizationOptimizer,
)
from gnn.utils.runtime_safety.timeout_manager import (
    LLMTimeoutManager,
    ProcessTimeoutManager,
    TimeoutConfig,
    TimeoutManager,
)


# 1. Tests for utils/timeout_manager.py
class TestTimeoutManager:
    def test_sync_timeout_success(self) -> Any:
        manager = TimeoutManager()
        config = TimeoutConfig(base_timeout=1.0, max_retries=0)

        def fast_func(x: Any) -> Any:
            return x * 2

        with manager.sync_timeout("test_sync", config, fast_func, 5) as result:
            assert result.success is True
            assert result.result == 10

    def test_async_timeout_success(self) -> Any:
        """Exercise async_timeout without pytest-asyncio (core env may omit dev extras)."""

        async def _run() -> None:
            manager = TimeoutManager()
            config = TimeoutConfig(base_timeout=1.0, max_retries=0)

            async def fast_async(x: int) -> int:
                return x * 2

            async with manager.async_timeout(
                "test_async", config, fast_async, 5
            ) as result:
                assert result.success is True
                assert result.result == 10

        asyncio.run(_run())

    def test_llm_timeout_manager(self) -> Any:
        manager = LLMTimeoutManager()
        assert manager.default_config.base_timeout == 60.0

    def test_process_timeout_manager(self) -> Any:
        manager = ProcessTimeoutManager()
        assert manager.default_config.base_timeout == 120.0


# 5. Tests for utils/visualization_optimizer.py
class TestVisualizationOptimizer:
    def test_visualization_cache(self, tmp_path: Any) -> Any:
        cache_dir = tmp_path / "cache"
        cache = VisualizationCache(cache_dir=cache_dir)
        key = cache.get_cache_key("content", {"p": 1})

        assert cache.is_cached(key) is False

        test_file = tmp_path / "viz.png"
        test_file.touch()
        cache.cache_visualization(key, [str(test_file)])

        assert cache.is_cached(key) is True
        assert cache.get_cached_files(key) == [str(test_file)]

    def test_data_sampler(self) -> Any:
        sampler = DataSampler(max_nodes=10)
        data: dict[str, Any] = {"nodes": [{"id": i} for i in range(20)]}

        assert sampler.should_sample(data) is True
        sampled = sampler.sample_data(data)
        assert len(sampled["nodes"]) == 10
        assert sampled["_sampling_applied"] is True

    def test_optimizer_batch(self, tmp_path: Any) -> Any:
        optimizer = VisualizationOptimizer(cache_dir=tmp_path / "cache")

        def sample_proc(file_path: Any, **kwargs: Any) -> Any:
            return {"success": True, "file": str(file_path)}

        files = [tmp_path / f"file_{i}.md" for i in range(3)]
        for f in files:
            f.touch()

        results = optimizer.optimize_batch_processing(files, tmp_path, sample_proc)
        assert len(results["processed_files"]) == 3
        assert results["optimization_stats"]["caching_enabled"] is True
