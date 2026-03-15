#!/usr/bin/env python3
"""
Infrastructure Coverage Gap Tests
Addresses modules with 0% coverage: recovery, timeout_manager, simulation_monitor, 
simulation_utils, visualization_optimizer.
"""

import pytest
import asyncio
import time
import logging
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

# Import targets
from utils.recovery import RecoveryArgumentParser, setup_step_logging
from utils.timeout_manager import (
    TimeoutManager, TimeoutConfig, TimeoutStrategy, 
    LLMTimeoutManager, ProcessTimeoutManager, with_timeout, with_async_timeout
)
from utils.simulation_monitor import SimulationMonitor, track_simulation as global_track_simulation
from utils.simulation_utils import SimulationTracker, DiagramAnalyzer
from utils.visualization_optimizer import (
    VisualizationCache, DataSampler, ParallelVisualizationProcessor, VisualizationOptimizer
)

# 1. Tests for utils/recovery.py
class TestRecoveryUtils:
    def test_recovery_argument_parser(self):
        args = RecoveryArgumentParser.parse_step_arguments("test_step")
        assert args.step_name == "test_step"
        assert args.verbose is False
        assert isinstance(args.output_dir, Path)

    def test_setup_step_logging(self):
        logger = setup_step_logging("test_recovery", verbose=True)
        assert logger.level == logging.DEBUG
        logger = setup_step_logging("test_recovery_info", verbose=False)
        assert logger.level == logging.INFO

# 2. Tests for utils/timeout_manager.py
class TestTimeoutManager:
    def test_sync_timeout_success(self):
        manager = TimeoutManager()
        config = TimeoutConfig(base_timeout=1.0, max_retries=0)
        
        def fast_func(x): return x * 2
        
        with manager.sync_timeout("test_sync", config, fast_func, 5) as result:
            assert result.success is True
            assert result.result == 10

    @pytest.mark.asyncio
    async def test_async_timeout_success(self):
        manager = TimeoutManager()
        config = TimeoutConfig(base_timeout=1.0, max_retries=0)
        
        async def fast_async(x): return x * 2
        
        async with manager.async_timeout("test_async", config, fast_async, 5) as result:
            assert result.success is True
            assert result.result == 10

    def test_llm_timeout_manager(self):
        manager = LLMTimeoutManager()
        assert manager.default_config.base_timeout == 60.0

    def test_process_timeout_manager(self):
        manager = ProcessTimeoutManager()
        assert manager.default_config.base_timeout == 120.0

# 3. Tests for utils/simulation_monitor.py
class TestSimulationMonitor:
    def test_monitor_initialization(self, tmp_path):
        log_file = tmp_path / "sim.log"
        monitor = SimulationMonitor(log_file=log_file)
        assert monitor.log_file == log_file
        assert monitor.execution_data["total_attempted"] == 0

    def test_track_simulation_decorator(self, tmp_path):
        monitor = SimulationMonitor(log_file=tmp_path / "sim.log")
        
        @monitor.track_simulation("test_sim")
        def lucky_sim(x): return x + 1
        
        result = lucky_sim(10)
        assert result == 11
        assert monitor.execution_data["total_successful"] == 1
        assert "test_sim" in monitor.execution_data["simulations"]

    def test_monitor_data_collection(self, tmp_path):
        monitor = SimulationMonitor(log_file=tmp_path / "sim.log")
        assert monitor.monitor_data_collection([1, 2, 3], "test") is True
        assert monitor.monitor_data_collection([], "fail") is False

# 4. Tests for utils/simulation_utils.py
class TestSimulationUtils:
    def test_simulation_tracker(self, tmp_path):
        tracker = SimulationTracker("model_a", "pymdp", tmp_path)
        tracker.log_step(0, [1, 0], [0], [1], 1.0)
        assert len(tracker.data["traces"]["rewards"]) == 1
        
        tracker.calculate_summary_stats()
        assert tracker.data["summary_stats"]["total_reward"] == 1.0

    def test_diagram_analyzer(self, tmp_path):
        analyzer = DiagramAnalyzer("test_model", tmp_path)
        analyzer.log_diagram("D1", "A", "B", {"prop": 1})
        assert len(analyzer.analysis_data["diagrams"]) == 1
        
        report_path = analyzer.generate_diagram_report()
        assert report_path.exists()

# 5. Tests for utils/visualization_optimizer.py
class TestVisualizationOptimizer:
    def test_visualization_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = VisualizationCache(cache_dir=cache_dir)
        key = cache.get_cache_key("content", {"p": 1})
        
        assert cache.is_cached(key) is False
        
        dummy_file = tmp_path / "viz.png"
        dummy_file.touch()
        cache.cache_visualization(key, [str(dummy_file)])
        
        assert cache.is_cached(key) is True
        assert cache.get_cached_files(key) == [str(dummy_file)]

    def test_data_sampler(self):
        sampler = DataSampler(max_nodes=10)
        data = {"nodes": [{"id": i} for i in range(20)]}
        
        assert sampler.should_sample(data) is True
        sampled = sampler.sample_data(data)
        assert len(sampled["nodes"]) == 10
        assert sampled["_sampling_applied"] is True

    def test_optimizer_batch(self, tmp_path):
        optimizer = VisualizationOptimizer(cache_dir=tmp_path / "cache")
        
        def dummy_proc(file_path, **kwargs):
            return {"success": True, "file": str(file_path)}
            
        files = [tmp_path / f"file_{i}.md" for i in range(3)]
        for f in files: f.touch()
        
        results = optimizer.optimize_batch_processing(files, tmp_path, dummy_proc)
        assert len(results["processed_files"]) == 3
        assert results["optimization_stats"]["caching_enabled"] is True
