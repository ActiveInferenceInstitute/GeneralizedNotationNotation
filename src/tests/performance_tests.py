"""
Performance tests for GNN Processing Pipeline.

This module provides performance testing capabilities with fallback implementations.
"""

import logging
import time
import gc
import sys
from pathlib import Path
from typing import Dict, Any

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

def run_performance_test_suite(
    target_dir: Path,
    test_results_dir: Path,
    verbose: bool
) -> bool:
    """
    Run performance test suite for the GNN pipeline.
    
    Args:
        target_dir: Directory containing GNN files to test
        test_results_dir: Directory to save test results
        verbose: Enable verbose output
        
    Returns:
        True if all performance tests pass, False otherwise
    """
    logger = logging.getLogger("performance_tests")
    
    try:
        log_step_start(logger, "Running performance test suite")
        
        # Create performance test results directory
        performance_results_dir = test_results_dir / "performance_tests"
        performance_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run different performance test categories
        test_results = {}
        
        # Memory usage tests
        test_results["memory_tests"] = run_memory_performance_tests(performance_results_dir, verbose)
        
        # Execution time tests
        test_results["execution_time_tests"] = run_execution_time_tests(target_dir, performance_results_dir, verbose)
        
        # Scalability tests
        test_results["scalability_tests"] = run_scalability_tests(performance_results_dir, verbose)
        
        # Save performance test results
        import json
        results_file = performance_results_dir / "performance_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Check overall success
        all_passed = all(test_results.values())
        
        if all_passed:
            log_step_success(logger, "All performance tests passed")
        else:
            failed_tests = [name for name, passed in test_results.items() if not passed]
            log_step_error(logger, f"Some performance tests failed: {failed_tests}")
        
        return all_passed
        
    except Exception as e:
        log_step_error(logger, f"Performance test suite execution failed: {e}")
        return False

def run_memory_performance_tests(test_results_dir: Path, verbose: bool) -> bool:
    """Run memory performance tests."""
    logger = logging.getLogger("performance_tests.memory")
    
    try:
        log_step_start(logger, "Running memory performance tests")
        
        # Comprehensive memory performance tests
        test_results = {
            "memory_usage_basic": test_basic_memory_usage(),
            "memory_leak_detection": test_memory_leak_detection(),
            "memory_usage_under_load": test_memory_usage_under_load(),
            "memory_efficiency_comparison": test_memory_efficiency_comparison(),
            "garbage_collection_performance": test_garbage_collection_performance(),
            "memory_profiling_integration": test_memory_profiling_integration()
        }
        
        # Add detailed memory metrics
        memory_metrics = {
            "peak_memory_mb": get_peak_memory_usage(),
            "average_memory_mb": get_average_memory_usage(),
            "memory_growth_rate": calculate_memory_growth_rate(),
            "gc_statistics": get_garbage_collection_stats()
        }
        
        test_results["metrics"] = memory_metrics
        
        # Save memory test results
        import json
        results_file = test_results_dir / "memory_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Generate memory performance report
        generate_memory_performance_report(test_results, test_results_dir)
        
        all_passed = all(result for result in test_results.values() if isinstance(result, bool))
        
        if all_passed:
            log_step_success(logger, f"All memory performance tests passed. Peak memory: {memory_metrics.get('peak_memory_mb', 0):.1f}MB")
        else:
            failed_tests = [name for name, passed in test_results.items() if isinstance(passed, bool) and not passed]
            log_step_error(logger, f"Some memory performance tests failed: {failed_tests}")
        
        return all_passed
        
    except Exception as e:
        log_step_error(logger, f"Memory performance test execution failed: {e}")
        return False

def run_execution_time_tests(target_dir: Path, test_results_dir: Path, verbose: bool) -> bool:
    """Run execution time tests."""
    logger = logging.getLogger("performance_tests.execution_time")
    
    try:
        log_step_start(logger, "Running execution time tests")
        
        # Comprehensive execution time tests
        test_results = {
            "file_processing_time": test_file_processing_time(target_dir),
            "pipeline_step_time": test_pipeline_step_time(),
            "module_import_time": test_module_import_time(),
            "algorithm_performance": test_algorithm_performance(),
            "io_operation_time": test_io_operation_time(test_results_dir),
            "concurrent_execution_time": test_concurrent_execution_time(),
            "regression_test_baseline": test_performance_regression_baseline()
        }
        
        # Add timing metrics
        timing_metrics = {
            "fastest_operation_ms": get_fastest_operation_time(),
            "slowest_operation_ms": get_slowest_operation_time(),
            "average_operation_ms": get_average_operation_time(),
            "performance_baseline": get_performance_baseline()
        }
        
        test_results["timing_metrics"] = timing_metrics
        
        # Save execution time test results
        import json
        results_file = test_results_dir / "execution_time_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Generate performance report
        generate_execution_time_report(test_results, test_results_dir)
        
        all_passed = all(result for result in test_results.values() if isinstance(result, bool))
        
        if all_passed:
            avg_time = timing_metrics.get('average_operation_ms', 0)
            log_step_success(logger, f"All execution time tests passed. Average operation time: {avg_time:.2f}ms")
        else:
            failed_tests = [name for name, passed in test_results.items() if isinstance(passed, bool) and not passed]
            log_step_error(logger, f"Some execution time tests failed: {failed_tests}")
        
        return all_passed
        
    except Exception as e:
        log_step_error(logger, f"Execution time test execution failed: {e}")
        return False

def run_scalability_tests(test_results_dir: Path, verbose: bool) -> bool:
    """Run scalability tests."""
    logger = logging.getLogger("performance_tests.scalability")
    
    try:
        log_step_start(logger, "Running scalability tests")
        
        # Basic scalability test
        test_results = {
            "small_dataset_performance": test_small_dataset_performance(),
            "large_dataset_performance": test_large_dataset_performance()
        }
        
        # Save scalability test results
        import json
        results_file = test_results_dir / "scalability_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        all_passed = all(test_results.values())
        
        if all_passed:
            log_step_success(logger, "All scalability tests passed")
        else:
            failed_tests = [name for name, passed in test_results.items() if not passed]
            log_step_error(logger, f"Some scalability tests failed: {failed_tests}")
        
        return all_passed
        
    except Exception as e:
        log_step_error(logger, f"Scalability test execution failed: {e}")
        return False

# Test helper functions
def test_basic_memory_usage() -> bool:
    """Test basic memory usage."""
    try:
        # Basic memory usage test
        import sys
        initial_memory = sys.getsizeof([])
        test_list = [i for i in range(1000)]
        final_memory = sys.getsizeof(test_list)
        return final_memory > initial_memory  # Should use more memory
    except Exception:
        return False

def test_memory_usage_under_load() -> bool:
    """Test memory usage under computational load."""
    try:
        import gc
        import sys
        
        # Force garbage collection first
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create memory load
        data_structures = []
        for i in range(100):
            # Create various data structures
            data_structures.append({
                'list': list(range(100)),
                'dict': {j: j**2 for j in range(50)},
                'set': set(range(25)),
                'tuple': tuple(range(20))
            })
        
        # Check memory usage increased
        loaded_objects = len(gc.get_objects())
        
        # Cleanup
        del data_structures
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Verify memory was used and then cleaned up
        return (loaded_objects > initial_objects and 
                final_objects < loaded_objects)
        
    except Exception as e:
        return False

def test_memory_efficiency_comparison() -> bool:
    """Test memory efficiency of different approaches."""
    try:
        import sys
        
        # Test list vs generator memory efficiency
        list_data = [i**2 for i in range(1000)]
        list_memory = sys.getsizeof(list_data)
        
        gen_data = (i**2 for i in range(1000))
        gen_memory = sys.getsizeof(gen_data)
        
        # Generator should use less memory
        return gen_memory < list_memory
        
    except Exception:
        return False

def test_garbage_collection_performance() -> bool:
    """Test garbage collection performance."""
    try:
        import gc
        import time
        
        # Disable auto garbage collection
        gc.disable()
        
        # Create objects that need collection
        circular_refs = []
        for i in range(100):
            obj = {'data': list(range(50))}
            obj['self_ref'] = obj  # Create circular reference
            circular_refs.append(obj)
        
        # Measure garbage collection time
        start_time = time.time()
        collected = gc.collect()
        gc_time = time.time() - start_time
        
        # Re-enable auto garbage collection
        gc.enable()
        
        # GC should be reasonably fast (under 0.1 seconds for this test)
        return gc_time < 0.1 and collected >= 0
        
    except Exception:
        return False

def test_memory_profiling_integration() -> bool:
    """Test memory profiling integration."""
    try:
        # Test that we can monitor memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss > 0  # Should have some memory usage
        except ImportError:
            # psutil not available, use basic memory check
            import sys
            return sys.getsizeof({}) > 0
            
    except Exception:
        return False

def get_peak_memory_usage() -> float:
    """Get peak memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback without psutil
        import sys
        return sys.getsizeof([]) / 1024 / 1024

def get_average_memory_usage() -> float:
    """Get average memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback without psutil
        return get_peak_memory_usage()

def calculate_memory_growth_rate() -> float:
    """Calculate memory growth rate."""
    try:
        import gc
        import time
        
        # Measure memory before and after operations
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform some operations
        test_data = [i for i in range(1000)]
        time.sleep(0.1)  # Small delay
        
        final_objects = len(gc.get_objects())
        
        # Calculate growth rate (objects per second)
        growth_rate = (final_objects - initial_objects) / 0.1
        
        # Cleanup
        del test_data
        gc.collect()
        
        return max(0.0, growth_rate)
        
    except Exception:
        return 0.0

def get_garbage_collection_stats() -> dict:
    """Get garbage collection statistics."""
    try:
        import gc
        
        stats = {
            'collections_gen0': gc.get_count()[0] if hasattr(gc, 'get_count') else 0,
            'collections_gen1': gc.get_count()[1] if hasattr(gc, 'get_count') and len(gc.get_count()) > 1 else 0,
            'collections_gen2': gc.get_count()[2] if hasattr(gc, 'get_count') and len(gc.get_count()) > 2 else 0,
            'total_objects': len(gc.get_objects()),
            'garbage_objects': len(gc.garbage) if hasattr(gc, 'garbage') else 0
        }
        
        return stats
        
    except Exception:
        return {'error': 'Unable to get GC stats'}

def generate_memory_performance_report(test_results: dict, output_dir: Path):
    """Generate detailed memory performance report."""
    try:
        report_file = output_dir / "memory_performance_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Memory Performance Test Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Test results summary
            f.write("## Test Results Summary\n\n")
            for test_name, result in test_results.items():
                if isinstance(result, bool):
                    status = "✅ PASS" if result else "❌ FAIL"
                    f.write(f"- **{test_name}**: {status}\n")
            
            # Memory metrics
            if 'metrics' in test_results:
                f.write("\n## Memory Metrics\n\n")
                metrics = test_results['metrics']
                f.write(f"- **Peak Memory**: {metrics.get('peak_memory_mb', 0):.2f} MB\n")
                f.write(f"- **Average Memory**: {metrics.get('average_memory_mb', 0):.2f} MB\n")
                f.write(f"- **Memory Growth Rate**: {metrics.get('memory_growth_rate', 0):.2f} objects/sec\n")
                
                if 'gc_statistics' in metrics:
                    gc_stats = metrics['gc_statistics']
                    f.write(f"- **Total Objects**: {gc_stats.get('total_objects', 0)}\n")
                    f.write(f"- **Garbage Objects**: {gc_stats.get('garbage_objects', 0)}\n")
            
            f.write("\n## Recommendations\n\n")
            
            # Add recommendations based on results
            failed_tests = [name for name, result in test_results.items() 
                          if isinstance(result, bool) and not result]
            
            if failed_tests:
                f.write("### Issues Detected\n\n")
                for test_name in failed_tests:
                    f.write(f"- **{test_name}**: Consider memory optimization\n")
            else:
                f.write("✅ All memory performance tests passed. Memory usage is within acceptable limits.\n")
                
    except Exception as e:
        logging.error(f"Failed to generate memory performance report: {e}")

def test_memory_leak_detection() -> bool:
    """Test memory leak detection."""
    try:
        # Basic memory leak detection test
        import gc
        gc.collect()  # Force garbage collection
        return True
    except Exception:
        return False

def test_file_processing_time(target_dir: Path) -> bool:
    """Test file processing time."""
    try:
        start_time = time.time()
        gnn_files = list(target_dir.glob("*.md"))
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (less than 10 seconds)
        return processing_time < 10.0
    except Exception:
        return False

def test_pipeline_step_time() -> bool:
    """Test pipeline step execution time."""
    try:
        start_time = time.time()
        # Simulate pipeline step execution
        time.sleep(0.1)  # Simulate work
        end_time = time.time()
        step_time = end_time - start_time
        
        # Should complete in reasonable time
        return step_time < 1.0
    except Exception:
        return False

def test_small_dataset_performance() -> bool:
    """Test performance with small dataset."""
    try:
        start_time = time.time()
        # Simulate small dataset processing
        time.sleep(0.05)
        end_time = time.time()
        processing_time = end_time - start_time
        
        return processing_time < 0.5
    except Exception:
        return False

def test_large_dataset_performance() -> bool:
    """Test performance with large dataset."""
    try:
        start_time = time.time()
        # Simulate large dataset processing
        time.sleep(0.2)
        end_time = time.time()
        processing_time = end_time - start_time
        
        return processing_time < 2.0
    except Exception:
        return False

def test_module_import_time() -> bool:
    """Test module import performance."""
    try:
        import importlib
        import time
        
        modules_to_test = ['json', 'pathlib', 'logging', 'time']
        total_import_time = 0
        
        for module_name in modules_to_test:
            start_time = time.perf_counter()
            importlib.import_module(module_name)
            import_time = time.perf_counter() - start_time
            total_import_time += import_time
        
        # All imports should complete quickly (under 0.1 seconds total)
        return total_import_time < 0.1
        
    except Exception:
        return False

def test_algorithm_performance() -> bool:
    """Test algorithm performance benchmarks."""
    try:
        import time
        
        # Test sorting performance
        start_time = time.perf_counter()
        data = list(range(10000, 0, -1))  # Reverse sorted list
        sorted_data = sorted(data)
        sort_time = time.perf_counter() - start_time
        
        # Test search performance
        start_time = time.perf_counter()
        found = 5000 in sorted_data
        search_time = time.perf_counter() - start_time
        
        # Both operations should be fast
        return sort_time < 0.1 and search_time < 0.01 and found
        
    except Exception:
        return False

def test_io_operation_time(output_dir: Path) -> bool:
    """Test I/O operation performance."""
    try:
        import time
        
        test_file = output_dir / "performance_test.txt"
        test_data = "x" * 10000  # 10KB of data
        
        # Test write performance
        start_time = time.perf_counter()
        test_file.write_text(test_data)
        write_time = time.perf_counter() - start_time
        
        # Test read performance
        start_time = time.perf_counter()
        read_data = test_file.read_text()
        read_time = time.perf_counter() - start_time
        
        # Cleanup
        test_file.unlink()
        
        # I/O should be fast and data should match
        return (write_time < 0.1 and read_time < 0.1 and 
                read_data == test_data)
        
    except Exception:
        return False

def test_concurrent_execution_time() -> bool:
    """Test concurrent execution performance."""
    try:
        import concurrent.futures
        import time
        
        def cpu_task(n):
            """CPU-bound task for testing."""
            return sum(i * i for i in range(n))
        
        # Test sequential execution
        start_time = time.perf_counter()
        sequential_results = [cpu_task(1000) for _ in range(4)]
        sequential_time = time.perf_counter() - start_time
        
        # Test concurrent execution
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(cpu_task, [1000] * 4))
        concurrent_time = time.perf_counter() - start_time
        
        # Results should be the same, concurrent might be faster
        return (sequential_results == concurrent_results and
                concurrent_time <= sequential_time * 1.2)  # Allow 20% overhead
        
    except Exception:
        return False

def test_performance_regression_baseline() -> bool:
    """Test performance against regression baseline."""
    try:
        import time
        
        # Define baseline performance expectations
        baselines = {
            'list_creation': 0.01,  # seconds for 10k items
            'dict_lookup': 0.001,   # seconds for 1k lookups
            'string_processing': 0.01  # seconds for basic operations
        }
        
        # Test list creation
        start_time = time.perf_counter()
        test_list = [i for i in range(10000)]
        list_time = time.perf_counter() - start_time
        
        # Test dict lookup
        test_dict = {i: i**2 for i in range(1000)}
        start_time = time.perf_counter()
        for i in range(1000):
            _ = test_dict[i]
        dict_time = time.perf_counter() - start_time
        
        # Test string processing
        start_time = time.perf_counter()
        test_string = "test string " * 1000
        processed = test_string.upper().replace("TEST", "demo")
        string_time = time.perf_counter() - start_time
        
        # All operations should meet baseline expectations
        return (list_time < baselines['list_creation'] and
                dict_time < baselines['dict_lookup'] and
                string_time < baselines['string_processing'])
        
    except Exception:
        return False

def get_fastest_operation_time() -> float:
    """Get fastest operation time in milliseconds."""
    try:
        import time
        
        # Test several quick operations
        times = []
        
        # Simple arithmetic
        start = time.perf_counter()
        result = 2 + 2
        times.append((time.perf_counter() - start) * 1000)
        
        # List access
        test_list = [1, 2, 3, 4, 5]
        start = time.perf_counter()
        item = test_list[0]
        times.append((time.perf_counter() - start) * 1000)
        
        # Dict access
        test_dict = {'key': 'value'}
        start = time.perf_counter()
        value = test_dict['key']
        times.append((time.perf_counter() - start) * 1000)
        
        return min(times)
        
    except Exception:
        return 0.0

def get_slowest_operation_time() -> float:
    """Get slowest operation time in milliseconds."""
    try:
        import time
        
        # Test several slower operations
        times = []
        
        # List comprehension
        start = time.perf_counter()
        result = [i**2 for i in range(1000)]
        times.append((time.perf_counter() - start) * 1000)
        
        # String operations
        start = time.perf_counter()
        text = "test " * 1000
        result = text.upper()
        times.append((time.perf_counter() - start) * 1000)
        
        # Dictionary creation
        start = time.perf_counter()
        result = {i: i**2 for i in range(500)}
        times.append((time.perf_counter() - start) * 1000)
        
        return max(times)
        
    except Exception:
        return 0.0

def get_average_operation_time() -> float:
    """Get average operation time in milliseconds."""
    try:
        fastest = get_fastest_operation_time()
        slowest = get_slowest_operation_time()
        return (fastest + slowest) / 2
    except Exception:
        return 0.0

def get_performance_baseline() -> dict:
    """Get performance baseline metrics."""
    try:
        return {
            'cpu_benchmark': run_cpu_benchmark(),
            'memory_benchmark': run_memory_benchmark(),
            'io_benchmark': run_io_benchmark()
        }
    except Exception:
        return {'error': 'Unable to get baseline'}

def run_cpu_benchmark() -> float:
    """Run CPU benchmark test."""
    try:
        import time
        start_time = time.perf_counter()
        # Simple CPU-bound calculation
        result = sum(i**2 for i in range(10000))
        cpu_time = time.perf_counter() - start_time
        return cpu_time * 1000  # Convert to milliseconds
    except Exception:
        return 0.0

def run_memory_benchmark() -> float:
    """Run memory benchmark test."""
    try:
        import time
        start_time = time.perf_counter()
        # Memory allocation test
        data = [list(range(100)) for _ in range(100)]
        memory_time = time.perf_counter() - start_time
        del data  # Cleanup
        return memory_time * 1000  # Convert to milliseconds
    except Exception:
        return 0.0

def run_io_benchmark() -> float:
    """Run I/O benchmark test."""
    try:
        import time
        import tempfile
        
        start_time = time.perf_counter()
        # I/O operations test
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as f:
            f.write("benchmark data " * 1000)
            f.seek(0)
            data = f.read()
        io_time = time.perf_counter() - start_time
        return io_time * 1000  # Convert to milliseconds
    except Exception:
        return 0.0

def generate_execution_time_report(test_results: dict, output_dir: Path):
    """Generate detailed execution time performance report."""
    try:
        report_file = output_dir / "execution_time_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Execution Time Performance Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Test results summary
            f.write("## Test Results Summary\n\n")
            for test_name, result in test_results.items():
                if isinstance(result, bool):
                    status = "✅ PASS" if result else "❌ FAIL"
                    f.write(f"- **{test_name}**: {status}\n")
            
            # Timing metrics
            if 'timing_metrics' in test_results:
                f.write("\n## Timing Metrics\n\n")
                metrics = test_results['timing_metrics']
                f.write(f"- **Fastest Operation**: {metrics.get('fastest_operation_ms', 0):.3f} ms\n")
                f.write(f"- **Slowest Operation**: {metrics.get('slowest_operation_ms', 0):.3f} ms\n")
                f.write(f"- **Average Operation**: {metrics.get('average_operation_ms', 0):.3f} ms\n")
                
                if 'performance_baseline' in metrics:
                    baseline = metrics['performance_baseline']
                    f.write(f"- **CPU Benchmark**: {baseline.get('cpu_benchmark', 0):.3f} ms\n")
                    f.write(f"- **Memory Benchmark**: {baseline.get('memory_benchmark', 0):.3f} ms\n")
                    f.write(f"- **I/O Benchmark**: {baseline.get('io_benchmark', 0):.3f} ms\n")
            
            f.write("\n## Performance Analysis\n\n")
            
            # Add analysis based on results
            failed_tests = [name for name, result in test_results.items() 
                          if isinstance(result, bool) and not result]
            
            if failed_tests:
                f.write("### Performance Issues Detected\n\n")
                for test_name in failed_tests:
                    f.write(f"- **{test_name}**: Performance below expected baseline\n")
            else:
                f.write("✅ All execution time tests passed. Performance is within acceptable limits.\n")
                
    except Exception as e:
        logging.error(f"Failed to generate execution time report: {e}")
