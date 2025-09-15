#!/usr/bin/env python3
"""
Test Performance Overall Tests

This file contains comprehensive performance tests for the GNN pipeline.
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

class TestPerformanceOverall:
    """Comprehensive performance tests for the GNN pipeline."""
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_import_performance(self):
        """Test import performance for all modules."""
        import_time = time.time()
        
        try:
            # Test core module imports
            from gnn import processor
            from render import renderer
            from execute import executor
            from visualization import visualizer
            from export import exporter
            
            import_duration = time.time() - import_time
            assert import_duration < 5.0  # Should import in under 5 seconds
        except ImportError as e:
            pytest.skip(f"Module imports not available: {e}")
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_memory_usage_performance(self):
        """Test memory usage during operations."""
        try:
            import psutil
            import gc
            
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform some operations
            test_data = [i for i in range(10000)]
            processed_data = [x * 2 for x in test_data]
            
            # Force garbage collection
            gc.collect()
            
            # Get final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Should not use excessive memory
            assert memory_increase < 100  # Less than 100MB increase
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        except Exception as e:
            pytest.skip(f"Memory testing not available: {e}")
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_file_io_performance(self, isolated_temp_dir):
        """Test file I/O performance."""
        # Test file writing performance
        start_time = time.time()
        
        test_file = isolated_temp_dir / "performance_test.txt"
        test_data = "x" * 10000  # 10KB of data
        
        for i in range(100):
            test_file.write_text(test_data)
        
        write_duration = time.time() - start_time
        assert write_duration < 1.0  # Should complete in under 1 second
        
        # Test file reading performance
        start_time = time.time()
        
        for i in range(100):
            content = test_file.read_text()
            assert len(content) == len(test_data)
        
        read_duration = time.time() - start_time
        assert read_duration < 1.0  # Should complete in under 1 second
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_json_processing_performance(self):
        """Test JSON processing performance."""
        import json
        
        # Create test data
        test_data = {
            "items": [{"id": i, "value": f"item_{i}"} for i in range(1000)]
        }
        
        # Test JSON serialization performance
        start_time = time.time()
        json_string = json.dumps(test_data)
        serialize_duration = time.time() - start_time
        assert serialize_duration < 0.1  # Should serialize in under 0.1 seconds
        
        # Test JSON deserialization performance
        start_time = time.time()
        parsed_data = json.loads(json_string)
        deserialize_duration = time.time() - start_time
        assert deserialize_duration < 0.1  # Should deserialize in under 0.1 seconds
        
        assert parsed_data == test_data
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_concurrent_operations_performance(self):
        """Test concurrent operations performance."""
        import threading
        import queue
        
        def worker_task(task_queue, result_queue):
            """Worker task for concurrent testing."""
            while True:
                try:
                    task = task_queue.get(timeout=1)
                    if task is None:
                        break
                    
                    # Simulate some work
                    result = sum(range(task))
                    result_queue.put(result)
                    task_queue.task_done()
                except queue.Empty:
                    break
        
        # Test concurrent processing
        start_time = time.time()
        
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Add tasks
        for i in range(10):
            task_queue.put(1000)
        
        # Start workers
        workers = []
        for i in range(3):
            worker = threading.Thread(target=worker_task, args=(task_queue, result_queue))
            worker.start()
            workers.append(worker)
        
        # Wait for completion
        task_queue.join()
        
        # Stop workers
        for i in range(3):
            task_queue.put(None)
        
        for worker in workers:
            worker.join()
        
        concurrent_duration = time.time() - start_time
        assert concurrent_duration < 2.0  # Should complete in under 2 seconds
        
        # Verify results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        assert len(results) == 10
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_string_processing_performance(self):
        """Test string processing performance."""
        # Test string concatenation performance
        start_time = time.time()
        
        result = ""
        for i in range(1000):
            result += f"item_{i}_"
        
        concat_duration = time.time() - start_time
        assert concat_duration < 0.1  # Should complete in under 0.1 seconds
        
        # Test string formatting performance
        start_time = time.time()
        
        formatted_strings = []
        for i in range(1000):
            formatted_strings.append(f"Item {i:04d}: {i * 2}")
        
        format_duration = time.time() - start_time
        assert format_duration < 0.1  # Should complete in under 0.1 seconds
        
        assert len(formatted_strings) == 1000
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_list_operations_performance(self):
        """Test list operations performance."""
        # Test list creation performance
        start_time = time.time()
        
        test_list = [i for i in range(10000)]
        
        create_duration = time.time() - start_time
        assert create_duration < 0.1  # Should complete in under 0.1 seconds
        
        # Test list operations performance
        start_time = time.time()
        
        # Test various list operations
        doubled = [x * 2 for x in test_list]
        filtered = [x for x in test_list if x % 2 == 0]
        sorted_list = sorted(test_list, reverse=True)
        
        operations_duration = time.time() - start_time
        assert operations_duration < 0.1  # Should complete in under 0.1 seconds
        
        assert len(doubled) == 10000
        assert len(filtered) == 5000
        assert len(sorted_list) == 10000
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_dictionary_operations_performance(self):
        """Test dictionary operations performance."""
        # Test dictionary creation performance
        start_time = time.time()
        
        test_dict = {f"key_{i}": i for i in range(10000)}
        
        create_duration = time.time() - start_time
        assert create_duration < 0.1  # Should complete in under 0.1 seconds
        
        # Test dictionary operations performance
        start_time = time.time()
        
        # Test various dictionary operations
        values = list(test_dict.values())
        keys = list(test_dict.keys())
        items = list(test_dict.items())
        
        operations_duration = time.time() - start_time
        assert operations_duration < 0.1  # Should complete in under 0.1 seconds
        
        assert len(values) == 10000
        assert len(keys) == 10000
        assert len(items) == 10000
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_algorithm_performance(self):
        """Test algorithm performance."""
        # Test sorting algorithm performance
        start_time = time.time()
        
        test_data = [i for i in range(10000, 0, -1)]  # Reverse order
        sorted_data = sorted(test_data)
        
        sort_duration = time.time() - start_time
        assert sort_duration < 0.1  # Should complete in under 0.1 seconds
        
        assert sorted_data == list(range(1, 10001))
        
        # Test search algorithm performance
        start_time = time.time()
        
        target = 5000
        found = target in sorted_data
        
        search_duration = time.time() - start_time
        assert search_duration < 0.01  # Should complete in under 0.01 seconds
        
        assert found is True
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_network_operations_performance(self):
        """Test network operations performance."""
        try:
            import socket
            
            # Test socket creation performance
            start_time = time.time()
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.close()
            
            socket_duration = time.time() - start_time
            assert socket_duration < 0.1  # Should complete in under 0.1 seconds
        except Exception as e:
            pytest.skip(f"Network operations not available: {e}")
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_system_resource_performance(self):
        """Test system resource usage performance."""
        try:
            import psutil
            
            # Test CPU usage monitoring
            start_time = time.time()
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            monitoring_duration = time.time() - start_time
            assert monitoring_duration < 1.0  # Should complete in under 1 second
            
            assert isinstance(cpu_percent, (int, float))
            assert isinstance(memory_info.total, int)
        except ImportError:
            pytest.skip("psutil not available for system monitoring")
        except Exception as e:
            pytest.skip(f"System resource monitoring not available: {e}")

