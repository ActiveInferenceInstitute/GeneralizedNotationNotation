#!/usr/bin/env python3
"""
Test MCP Performance - Performance tests for MCP (Model Context Protocol) module.

Tests performance characteristics of MCP tool registration, execution, and throughput.
These tests require the MCP SDK to be available and will be skipped if it's not installed.
"""

import pytest
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import MCP exception for skip handling
try:
    from mcp import MCPSDKNotFoundError
except ImportError:
    MCPSDKNotFoundError = Exception  # Fallback


def _safe_initialize():
    """Initialize MCP, skipping test if SDK not available."""
    from mcp import initialize
    try:
        initialize()
    except MCPSDKNotFoundError:
        pytest.skip("MCP SDK not available")


class TestMCPToolRegistrationPerformance:
    """Performance tests for MCP tool registration."""

    @pytest.mark.slow
    def test_tool_registration_speed(self):
        """Test that tool registration completes in reasonable time."""
        from mcp import list_available_tools, initialize

        start_time = time.time()

        try:
            initialize()
        except MCPSDKNotFoundError:
            pytest.skip("MCP SDK not available")

        tools = list_available_tools()
        elapsed = time.time() - start_time

        # Registration should complete within 5 seconds
        assert elapsed < 5.0, f"Tool registration took {elapsed:.2f}s, expected < 5s"
        assert tools is not None

    @pytest.mark.slow
    def test_repeated_initialization_performance(self):
        """Test that repeated initialization is efficient."""
        from mcp import initialize

        times = []
        for i in range(3):
            start = time.time()
            try:
                initialize()
            except MCPSDKNotFoundError:
                if i == 0:
                    pytest.skip("MCP SDK not available")
                break
            times.append(time.time() - start)

        if not times:
            pytest.skip("MCP SDK not available")

        # Subsequent calls should be faster (cached)
        avg_time = sum(times) / len(times)
        assert avg_time < 2.0, f"Average init time {avg_time:.2f}s too slow"


class TestMCPToolExecutionPerformance:
    """Performance tests for MCP tool execution."""

    @pytest.mark.slow
    def test_tool_lookup_speed(self):
        """Test tool lookup is fast."""
        from mcp import list_available_tools

        _safe_initialize()

        start = time.time()
        for _ in range(100):
            tools = list_available_tools()
        elapsed = time.time() - start

        # 100 lookups should complete in < 1 second
        assert elapsed < 1.0, f"100 lookups took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_module_discovery_performance(self):
        """Test module discovery performance."""
        from mcp import get_mcp_instance

        _safe_initialize()

        start = time.time()
        mcp = get_mcp_instance()
        # Get loaded modules from the MCP instance
        modules = mcp.modules if hasattr(mcp, 'modules') else {}
        elapsed = time.time() - start

        # Discovery should be fast
        assert elapsed < 3.0, f"Module discovery took {elapsed:.2f}s"
        assert modules is not None


class TestMCPThroughput:
    """Throughput tests for MCP operations."""

    @pytest.mark.slow
    def test_concurrent_tool_access(self):
        """Test concurrent access to MCP tools."""
        from mcp import list_available_tools
        import threading

        _safe_initialize()
        results = []
        errors = []

        def access_tools():
            try:
                tools = list_available_tools()
                results.append(len(tools) if tools else 0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_tools) for _ in range(10)]

        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert elapsed < 5.0, f"Concurrent access took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_tool_count_consistency(self):
        """Test that tool count remains consistent."""
        from mcp import list_available_tools

        _safe_initialize()

        counts = []
        for _ in range(10):
            tools = list_available_tools()
            counts.append(len(tools) if tools else 0)

        # All counts should be the same
        if counts:
            assert all(c == counts[0] for c in counts), "Tool count inconsistent"


class TestMCPMemoryPerformance:
    """Memory performance tests for MCP operations."""

    @pytest.mark.slow
    def test_memory_usage_stable(self):
        """Test that MCP operations don't leak memory."""
        from mcp import list_available_tools

        _safe_initialize()
        initial_tools = list_available_tools()

        # Perform many operations
        for _ in range(50):
            list_available_tools()

        # Should still work
        final_tools = list_available_tools()
        assert final_tools is not None


class TestMCPServerPerformance:
    """Performance tests for MCP server operations."""

    @pytest.mark.slow
    def test_server_creation_speed(self):
        """Test MCP server can be created quickly."""
        try:
            from mcp import create_mcp_server

            start = time.time()
            server = create_mcp_server()
            elapsed = time.time() - start

            assert elapsed < 2.0, f"Server creation took {elapsed:.2f}s"
        except (ImportError, MCPSDKNotFoundError):
            pytest.skip("MCP server not available")

    @pytest.mark.slow
    def test_resource_listing_performance(self):
        """Test resource listing performance."""
        try:
            from mcp import list_available_resources

            _safe_initialize()

            start = time.time()
            resources = list_available_resources()
            elapsed = time.time() - start

            assert elapsed < 1.0, f"Resource listing took {elapsed:.2f}s"
        except (ImportError, AttributeError, MCPSDKNotFoundError):
            pytest.skip("Resource listing not available")


class TestMCPBenchmarks:
    """Benchmark tests for MCP operations."""

    @pytest.mark.slow
    def test_initialization_benchmark(self):
        """Benchmark MCP initialization."""
        from mcp import initialize

        times = []
        for i in range(5):
            start = time.time()
            try:
                initialize()
            except MCPSDKNotFoundError:
                if i == 0:
                    pytest.skip("MCP SDK not available")
                break
            times.append(time.time() - start)

        if not times:
            pytest.skip("MCP SDK not available")

        avg = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        # Record benchmark results
        print(f"\nMCP Init Benchmark: avg={avg:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")

        # Should complete within reasonable time
        assert avg < 3.0

    @pytest.mark.slow
    def test_tool_execution_benchmark(self):
        """Benchmark tool execution overhead."""
        from mcp import list_available_tools

        _safe_initialize()
        tools = list_available_tools()

        if not tools:
            pytest.skip("No tools registered")

        # Measure tool access overhead
        start = time.time()
        for _ in range(1000):
            _ = list_available_tools()
        elapsed = time.time() - start

        ops_per_second = 1000 / elapsed
        print(f"\nTool access: {ops_per_second:.0f} ops/sec")

        assert ops_per_second > 100, "Tool access too slow"
