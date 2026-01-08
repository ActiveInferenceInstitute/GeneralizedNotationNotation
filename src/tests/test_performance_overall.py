#!/usr/bin/env python3
"""
Test Performance Overall Tests

Real performance tests for the GNN pipeline covering timing,
memory usage, and resource monitoring.
"""

import pytest
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


class TestPerformanceBasics:
    """Basic performance characteristic tests."""

    def test_import_timing(self):
        """Test that core module imports complete in reasonable time."""
        start = time.perf_counter()
        
        # Import core modules
        import gnn
        import export
        import visualization
        
        elapsed = time.perf_counter() - start
        
        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Module imports took {elapsed:.2f}s, expected < 5s"

    def test_memory_tracking_available(self):
        """Test that memory tracking utilities are functional."""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            
            assert hasattr(mem_info, 'rss'), "Memory info should have RSS"
            assert mem_info.rss > 0, "RSS should be positive"
        except ImportError:
            pytest.skip("psutil not available for memory tracking")

    def test_gnn_parsing_performance(self, safe_filesystem):
        """Test GNN parsing completes in reasonable time."""
        from gnn import parse_gnn_file
        
        # Create a sample GNN file
        content = """
# Performance Test Model

## ModelName
perf_test

## StateSpaceBlock
s[10,1,type=float]
o[5,1,type=float]

## Connections
s -> o
"""
        gnn_file = safe_filesystem.create_file("perf_test.md", content)
        
        start = time.perf_counter()
        result = parse_gnn_file(gnn_file)
        elapsed = time.perf_counter() - start
        
        # Parsing should complete in under 1 second
        assert elapsed < 1.0, f"GNN parsing took {elapsed:.2f}s, expected < 1s"
        assert result is not None

    def test_export_performance(self, safe_filesystem):
        """Test export processing completes efficiently."""
        from export import get_supported_formats
        
        start = time.perf_counter()
        formats = get_supported_formats()
        elapsed = time.perf_counter() - start
        
        # Format retrieval should be fast
        assert elapsed < 0.5, f"Format lookup took {elapsed:.2f}s, expected < 0.5s"
        assert len(formats) > 0, "Should return supported formats"


class TestResourceMonitoring:
    """Tests for resource monitoring capabilities."""

    def test_cpu_count_detection(self):
        """Test CPU count detection works."""
        import os
        cpu_count = os.cpu_count()
        
        assert cpu_count is not None
        assert cpu_count > 0

    def test_disk_space_check(self, tmp_path):
        """Test disk space checking works."""
        import shutil
        
        total, used, free = shutil.disk_usage(tmp_path)
        
        assert total > 0
        assert free > 0
        assert used >= 0
