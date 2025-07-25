#!/usr/bin/env python3
"""
Pipeline Performance Tests

This module provides comprehensive performance testing for the GNN pipeline,
focusing on resource usage, timing, and scalability characteristics.

Key test areas:
1. GNN processing performance
2. Visualization generation timing
3. Memory usage patterns
4. Disk I/O performance
5. Network operation timing
6. Resource scaling characteristics
"""

import pytest
import time
import psutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, Mock

# Import test utilities
from . import (
    TEST_CONFIG,
    get_test_args,
    create_test_files,
    performance_tracker,
    TEST_DIR,
    SRC_DIR,
    PROJECT_ROOT
)

# Test markers
pytestmark = [pytest.mark.performance, pytest.mark.safe_to_fail]

# Performance thresholds
THRESHOLDS = {
    "small_model": {
        "processing_time": 1.0,  # seconds
        "memory_usage": 100,    # MB
        "disk_usage": 10        # MB
    },
    "medium_model": {
        "processing_time": 5.0,
        "memory_usage": 500,
        "disk_usage": 50
    },
    "large_model": {
        "processing_time": 20.0,
        "memory_usage": 2000,
        "disk_usage": 200
    }
}

@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Create necessary subdirectories
        (temp_path / "input").mkdir()
        (temp_path / "output").mkdir()
        yield temp_path

@pytest.fixture
def create_model_file(mock_environment):
    """Create a GNN model file of specified size."""
    def _create_file(size: str = "small") -> Path:
        sizes = {
            "small": 10,    # 10 states
            "medium": 100,  # 100 states
            "large": 1000   # 1000 states
        }
        num_states = sizes[size]
        
        content = [
            "# ModelName",
            f"Test{size.capitalize()}Model",
            "",
            "# StateSpaceBlock"
        ]
        
        # Add states
        for i in range(num_states):
            content.append(f"s{i}[3,1]")
            
        # Add connections
        content.append("")
        content.append("# Connections")
        for i in range(num_states - 1):
            content.append(f"s{i} -> s{i+1}")
            
        file_path = mock_environment / "input" / f"test_model_{size}.md"
        file_path.write_text("\n".join(content))
        return file_path
        
    return _create_file

class TestGNNProcessingPerformance:
    """Test suite for GNN processing performance."""
    
    @pytest.mark.parametrize("model_size", ["small", "medium", "large"])
    def test_processing_scaling(self, mock_environment, create_model_file, model_size):
        """Test GNN processing performance scaling."""
        from gnn import process_gnn_directory
        
        model_file = create_model_file(model_size)
        
        with performance_tracker() as tracker:
            result = process_gnn_directory(
                mock_environment / "input",
                mock_environment / "output"
            )
            
        assert result["status"] == "SUCCESS"
        assert tracker.duration < THRESHOLDS[f"{model_size}_model"]["processing_time"]
        assert tracker.max_memory_mb < THRESHOLDS[f"{model_size}_model"]["memory_usage"]
        
    def test_parallel_processing(self, mock_environment, create_model_file):
        """Test parallel GNN processing performance."""
        from gnn import process_gnn_directory
        
        # Create multiple models
        for size in ["small", "medium"]:
            create_model_file(size)
            
        with performance_tracker() as tracker:
            result = process_gnn_directory(
                mock_environment / "input",
                mock_environment / "output",
                parallel=True
            )
            
        assert result["status"] == "SUCCESS"
        assert len(result["processed_files"]) == 2
        # Should be faster than sequential processing
        assert tracker.duration < (
            THRESHOLDS["small_model"]["processing_time"] +
            THRESHOLDS["medium_model"]["processing_time"]
        )

class TestVisualizationPerformance:
    """Test suite for visualization performance."""
    
    @pytest.mark.parametrize("model_size", ["small", "medium", "large"])
    def test_visualization_scaling(self, mock_environment, create_model_file, model_size):
        """Test visualization generation performance scaling."""
        from visualization import generate_visualizations
        
        model_file = create_model_file(model_size)
        
        with performance_tracker() as tracker:
            result = generate_visualizations(
                model_file,
                mock_environment / "output"
            )
            
        assert result["status"] == "SUCCESS"
        assert tracker.duration < THRESHOLDS[f"{model_size}_model"]["processing_time"]
        assert tracker.max_memory_mb < THRESHOLDS[f"{model_size}_model"]["memory_usage"]
        
    def test_visualization_caching(self, mock_environment, create_model_file):
        """Test visualization caching performance."""
        from visualization import generate_visualizations
        
        model_file = create_model_file("medium")
        
        # First generation
        with performance_tracker() as tracker_1:
            result_1 = generate_visualizations(
                model_file,
                mock_environment / "output"
            )
            
        # Second generation (should use cache)
        with performance_tracker() as tracker_2:
            result_2 = generate_visualizations(
                model_file,
                mock_environment / "output"
            )
            
        assert result_1["status"] == result_2["status"] == "SUCCESS"
        assert tracker_2.duration < tracker_1.duration * 0.5  # Should be at least 50% faster

class TestMemoryUsagePatterns:
    """Test suite for memory usage patterns."""
    
    def test_memory_cleanup(self, mock_environment, create_model_file):
        """Test memory cleanup after processing."""
        from gnn import process_gnn_directory
        
        model_file = create_model_file("large")
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = process_gnn_directory(
            mock_environment / "input",
            mock_environment / "output"
        )
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_diff = abs(final_memory - initial_memory)
        
        assert result["status"] == "SUCCESS"
        assert memory_diff < 50  # Should not leak more than 50MB
        
    def test_peak_memory_tracking(self, mock_environment, create_model_file):
        """Test peak memory usage tracking."""
        from utils.resource_manager import track_peak_memory
        
        @track_peak_memory
        def memory_intensive_operation():
            # Simulate memory-intensive operation
            big_list = [0] * (1024 * 1024 * 10)  # 10MB
            time.sleep(0.1)  # Allow tracking to measure
            return "Success"
        
        result, peak_memory = memory_intensive_operation()
        
        assert result == "Success"
        assert isinstance(peak_memory, float)
        assert peak_memory > 0

class TestDiskIOPerformance:
    """Test suite for disk I/O performance."""
    
    def test_file_write_performance(self, mock_environment):
        """Test file write performance."""
        from utils.io_utils import batch_write_files
        
        test_data = {
            f"test_file_{i}.txt": "x" * 1024 * 1024  # 1MB each
            for i in range(10)
        }
        
        with performance_tracker() as tracker:
            result = batch_write_files(
                test_data,
                mock_environment / "output"
            )
            
        assert result["status"] == "SUCCESS"
        assert result["files_written"] == 10
        assert tracker.duration < 1.0  # Should write 10MB in under 1 second
        
    def test_export_performance(self, mock_environment, create_model_file):
        """Test export performance for different formats."""
        from export import export_model
        
        model_file = create_model_file("medium")
        formats = ["json", "xml", "yaml"]
        
        for fmt in formats:
            with performance_tracker() as tracker:
                result = export_model(
                    model_file,
                    mock_environment / "output",
                    format=fmt
                )
                
            assert result["status"] == "SUCCESS"
            assert tracker.duration < 1.0  # Each format should export quickly

class TestNetworkOperationTiming:
    """Test suite for network operation timing."""
    
    def test_api_request_timing(self, mock_environment):
        """Test API request timing."""
        from utils.network_utils import timed_request
        
        with performance_tracker() as tracker:
            result = timed_request("https://api.example.com/test")
            
        assert result["status"] in ["SUCCESS", "ERROR"]  # Allow for offline testing
        if result["status"] == "SUCCESS":
            assert result["response_time"] < 1.0  # Should respond within 1 second
            
    def test_batch_request_performance(self, mock_environment):
        """Test batch request performance."""
        from utils.network_utils import batch_request
        
        urls = [f"https://api.example.com/test/{i}" for i in range(5)]
        
        with performance_tracker() as tracker:
            results = batch_request(urls)
            
        assert isinstance(results, list)
        assert len(results) == 5
        # Batch should be faster than sequential
        assert tracker.duration < 5.0  # Allow 1 second per request

class TestResourceScaling:
    """Test suite for resource scaling characteristics."""
    
    @pytest.mark.parametrize("num_files", [1, 10, 100])
    def test_pipeline_scaling(self, mock_environment, create_model_file, num_files):
        """Test full pipeline scaling with number of files."""
        from main import run_pipeline
        
        # Create test files
        for i in range(num_files):
            create_model_file("small")
            
        with performance_tracker() as tracker:
            result = run_pipeline(
                target_dir=mock_environment / "input",
                output_dir=mock_environment / "output"
            )
            
        assert result["status"] == "SUCCESS"
        # Should scale sub-linearly due to parallelization
        assert tracker.duration < (num_files * 0.5)  # 0.5 seconds per file
        
    def test_resource_estimation(self, mock_environment, create_model_file):
        """Test resource estimation accuracy."""
        from utils.resource_manager import estimate_resources
        
        model_file = create_model_file("medium")
        
        estimate = estimate_resources(model_file)
        
        with performance_tracker() as tracker:
            result = run_pipeline(
                target_dir=mock_environment / "input",
                output_dir=mock_environment / "output"
            )
            
        # Estimate should be within 20% of actual
        assert 0.8 <= (tracker.duration / estimate["time"]) <= 1.2
        assert 0.8 <= (tracker.max_memory_mb / estimate["memory_mb"]) <= 1.2

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 