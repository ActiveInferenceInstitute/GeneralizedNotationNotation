#!/usr/bin/env python3
"""
Infrastructure Coverage Gap Tests

Addresses modules that historically had 0% coverage. The timeout_manager
gap remains covered here; the visualization_optimizer gap module was
removed as dead code (2026-09-21), so only timeout_manager coverage
remains. The ``utils/recovery.py`` fallback was removed in
Phase 6 as dead code — ``setup_step_logging`` is covered in place via
``utils/logging/logging_utils``. ``utils/simulation_utils.py`` was removed as
dead code (R4 residue); its only importer was this file.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

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
