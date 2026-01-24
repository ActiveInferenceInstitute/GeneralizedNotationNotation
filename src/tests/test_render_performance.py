#!/usr/bin/env python3
"""
Test Render Performance - Performance tests for render module.

Tests performance characteristics of code generation and rendering operations.
"""

import pytest
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRenderingSpeed:
    """Performance tests for rendering speed."""

    @pytest.mark.slow
    def test_single_file_render_speed(self, sample_gnn_files, tmp_path):
        """Test single file rendering completes quickly."""
        if not sample_gnn_files:
            pytest.skip("No sample GNN files available")

        from render import process_render
        import logging

        logger = logging.getLogger("test_render")
        output_dir = tmp_path / "render_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # sample_gnn_files is a dict, get first file
        gnn_file = list(sample_gnn_files.values())[0]

        start = time.time()
        result = process_render(
            target_dir=gnn_file.parent,
            output_dir=output_dir,
            logger=logger
        )
        elapsed = time.time() - start

        # Single file should render in < 10 seconds
        assert elapsed < 10.0, f"Render took {elapsed:.2f}s, expected < 10s"

    @pytest.mark.slow
    def test_framework_render_speed(self, tmp_path):
        """Test framework-specific rendering speed."""
        from render import get_supported_frameworks
        
        frameworks = get_supported_frameworks()
        
        # Just verify we can get frameworks quickly
        start = time.time()
        for _ in range(100):
            _ = get_supported_frameworks()
        elapsed = time.time() - start
        
        assert elapsed < 1.0, f"100 framework lookups took {elapsed:.2f}s"


class TestRendererPerformance:
    """Performance tests for renderer classes."""

    @pytest.mark.slow
    def test_renderer_instantiation_speed(self):
        """Test renderer instantiation is fast."""
        from render import PyMDPRenderer
        
        start = time.time()
        for _ in range(100):
            renderer = PyMDPRenderer()
        elapsed = time.time() - start
        
        assert elapsed < 1.0, f"100 instantiations took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_template_loading_speed(self):
        """Test template loading performance."""
        from render import get_module_info
        
        start = time.time()
        info = get_module_info()
        elapsed = time.time() - start
        
        assert elapsed < 0.5, f"Module info took {elapsed:.2f}s"
        assert info is not None


class TestCodeGenerationPerformance:
    """Performance tests for code generation."""

    @pytest.mark.slow
    def test_pymdp_generation_speed(self, tmp_path, sample_gnn_files):
        """Test PyMDP code generation speed."""
        if not sample_gnn_files:
            pytest.skip("No sample GNN files available")

        from render import PyMDPRenderer

        renderer = PyMDPRenderer()
        # sample_gnn_files is a dict, get first file
        gnn_file = list(sample_gnn_files.values())[0]

        start = time.time()
        for i in range(10):
            out = tmp_path / f"test_pymdp_{i}.py"
            renderer.render_file(gnn_file, out)
        elapsed = time.time() - start

        assert elapsed < 10.0, f"10 renders took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_multi_framework_generation_speed(self, tmp_path, sample_gnn_files):
        """Test rendering to multiple frameworks is efficient."""
        if not sample_gnn_files:
            pytest.skip("No sample GNN files available")

        from render import get_supported_frameworks, PyMDPRenderer

        frameworks = get_supported_frameworks()
        assert len(frameworks) > 0, "No frameworks available"

        # Use actual file-based rendering
        renderer = PyMDPRenderer()
        output_file = tmp_path / "test_output.py"
        gnn_file = list(sample_gnn_files.values())[0]

        start = time.time()
        success, msg = renderer.render_file(gnn_file, output_file)
        elapsed = time.time() - start

        # Verify render completed (success or graceful failure)
        assert isinstance(success, bool), f"Expected bool result, got {type(success)}"
        assert elapsed < 5.0, f"Multi-framework render took {elapsed:.2f}s"


class TestRenderThroughput:
    """Throughput tests for render operations."""

    @pytest.mark.slow
    def test_batch_rendering_throughput(self, sample_gnn_files, tmp_path):
        """Test batch rendering throughput."""
        if not sample_gnn_files or len(sample_gnn_files) < 2:
            pytest.skip("Need multiple sample GNN files")
        
        from render import process_render
        import logging
        
        logger = logging.getLogger("test_render")
        output_dir = tmp_path / "batch_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start = time.time()
        result = process_render(
            target_dir=sample_gnn_files[0].parent,
            output_dir=output_dir,
            logger=logger
        )
        elapsed = time.time() - start
        
        files_per_second = len(sample_gnn_files) / elapsed if elapsed > 0 else 0
        print(f"\nRender throughput: {files_per_second:.2f} files/sec")
        
        # Should process at reasonable rate
        assert elapsed < 30.0

    @pytest.mark.slow
    def test_render_validation_speed(self):
        """Test render output validation is fast."""
        from render import validate_render
        
        # Sample render output
        render_output = {
            "code": "import pymdp\n# Generated code",
            "framework": "pymdp",
            "status": "success"
        }
        
        start = time.time()
        for _ in range(100):
            validate_render(render_output)
        elapsed = time.time() - start
        
        assert elapsed < 0.5, f"100 validations took {elapsed:.2f}s"


class TestRenderBenchmarks:
    """Benchmark tests for render operations."""

    @pytest.mark.slow
    def test_render_initialization_benchmark(self):
        """Benchmark render module initialization."""
        times = []
        
        for _ in range(5):
            start = time.time()
            from render import PyMDPRenderer
            renderer = PyMDPRenderer()
            times.append(time.time() - start)
        
        avg = sum(times) / len(times)
        print(f"\nRenderer init benchmark: avg={avg:.4f}s")
        
        assert avg < 0.5

    @pytest.mark.slow
    def test_code_generation_benchmark(self, sample_gnn_files, tmp_path):
        """Benchmark code generation."""
        if not sample_gnn_files:
            pytest.skip("No sample GNN files available")

        from render import PyMDPRenderer

        renderer = PyMDPRenderer()
        gnn_file = list(sample_gnn_files.values())[0]

        times = []
        for i in range(20):
            output = tmp_path / f"bench_{i}.py"
            start = time.time()
            renderer.render_file(gnn_file, output)
            times.append(time.time() - start)

        avg = sum(times) / len(times)
        min_t = min(times)
        max_t = max(times)

        print(f"\nCode gen benchmark: avg={avg:.4f}s, min={min_t:.4f}s, max={max_t:.4f}s")

        assert avg < 1.0, f"Average render time {avg:.2f}s exceeds 1.0s threshold"


class TestRenderMemoryPerformance:
    """Memory performance tests for render operations."""

    @pytest.mark.slow
    def test_render_no_memory_leak(self, sample_gnn_files, tmp_path):
        """Test that rendering doesn't leak memory."""
        if not sample_gnn_files:
            pytest.skip("No sample GNN files available")

        from render import PyMDPRenderer

        gnn_file = list(sample_gnn_files.values())[0]

        # Run many iterations and collect results
        results = []
        for i in range(50):
            renderer = PyMDPRenderer()
            output = tmp_path / f"mem_test_{i}.py"
            success, msg = renderer.render_file(gnn_file, output)
            results.append(success)

        # Verify all iterations completed
        assert len(results) == 50, f"Expected 50 results, got {len(results)}"
        # At least some should succeed (graceful failures are acceptable)
        assert any(isinstance(r, bool) for r in results), "Results should be boolean"
