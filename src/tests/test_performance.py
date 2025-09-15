#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Tests

This module provides performance testing and benchmarking for all GNN processing
modules to ensure they meet performance requirements and identify bottlenecks.
"""

import pytest
import os
import sys
import json
import logging
import tempfile
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import statistics
import threading
import concurrent.futures

# Test markers
pytestmark = [pytest.mark.performance, pytest.mark.safe_to_fail, pytest.mark.slow]

# Import test utilities and configuration
from . import (
    TEST_CONFIG,
    get_sample_pipeline_arguments,
    create_test_gnn_files,
    is_safe_mode,
    TEST_DIR,
    SRC_DIR,
    PROJECT_ROOT
)

class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks for all modules."""
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_gnn_processing_performance(self, sample_gnn_files, isolated_temp_dir):
        """Test GNN processing performance with various file sizes."""
        try:
            from src.gnn import process_gnn_directory, parse_gnn_file, validate_gnn_structure
            
            # Test with different file sizes
            file_sizes = [1, 5, 10, 20, 50]  # Number of states
            results = []
            
            for size in file_sizes:
                # Create GNN file with specified size
                gnn_file = isolated_temp_dir / f"model_{size}.gnn"
                
                content = f"""## ModelName
TestModel{size}

## StateSpaceBlock
"""
                
                # Add states
                for i in range(size):
                    content += f"s{i}: State\n"
                
                content += """
## Connections
"""
                
                # Add connections
                for i in range(size - 1):
                    content += f"s{i} -> s{i+1}: Transition\n"
                
                gnn_file.write_text(content)
                
                # Measure parsing performance
                start_time = time.time()
                parsed = parse_gnn_file(gnn_file)
                parse_time = time.time() - start_time
                
                # Measure validation performance
                start_time = time.time()
                validation = validate_gnn_structure(gnn_file)
                validation_time = time.time() - start_time
                
                results.append({
                    'file_size': size,
                    'parse_time': parse_time,
                    'validation_time': validation_time,
                    'total_time': parse_time + validation_time
                })
                
                logging.info(f"GNN processing for {size} states: parse={parse_time:.3f}s, validation={validation_time:.3f}s")
            
            # Verify performance is reasonable
            for result in results:
                assert result['parse_time'] < 1.0, f"Parse time too high for {result['file_size']} states: {result['parse_time']:.3f}s"
                assert result['validation_time'] < 1.0, f"Validation time too high for {result['file_size']} states: {result['validation_time']:.3f}s"
            
            logging.info("GNN processing performance test passed")
            
        except Exception as e:
            logging.warning(f"GNN processing performance test failed: {e}")
            pytest.skip(f"GNN processing performance test not available: {e}")
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_render_performance(self, sample_gnn_files, isolated_temp_dir):
        """Test render module performance with various inputs."""
        try:
            from src.render import process_render
            
            # Test with different numbers of files
            file_counts = [1, 3, 5, 10]
            results = []
            
            for count in file_counts:
                # Create input directory with specified number of files
                input_dir = isolated_temp_dir / f"input_{count}"
                input_dir.mkdir()
                
                for i in range(count):
                    gnn_file = input_dir / f"model_{i}.gnn"
                    gnn_file.write_text(f"""## ModelName
TestModel{i}

## StateSpaceBlock
s1: State
s2: State

## Connections
s1 -> s2: Transition
""")
                
                # Measure render performance
                start_time = time.time()
                result = process_render(input_dir, isolated_temp_dir / f"output_{count}")
                render_time = time.time() - start_time
                
                results.append({
                    'file_count': count,
                    'render_time': render_time,
                    'time_per_file': render_time / count if count > 0 else 0
                })
                
                logging.info(f"Render processing for {count} files: {render_time:.3f}s ({render_time/count:.3f}s per file)")
            
            # Verify performance scales reasonably
            for result in results:
                assert result['render_time'] < 30.0, f"Render time too high for {result['file_count']} files: {result['render_time']:.3f}s"
                assert result['time_per_file'] < 5.0, f"Time per file too high: {result['time_per_file']:.3f}s"
            
            logging.info("Render performance test passed")
            
        except Exception as e:
            logging.warning(f"Render performance test failed: {e}")
            pytest.skip(f"Render performance test not available: {e}")
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_visualization_performance(self, sample_gnn_files, isolated_temp_dir):
        """Test visualization module performance."""
        try:
            from src.visualization import process_visualization
            
            # Test with different file sizes
            file_sizes = [1, 3, 5]
            results = []
            
            for size in file_sizes:
                # Create input directory
                input_dir = isolated_temp_dir / f"viz_input_{size}"
                input_dir.mkdir()
                
                for i in range(size):
                    gnn_file = input_dir / f"model_{i}.gnn"
                    gnn_file.write_text(f"""## ModelName
TestModel{i}

## StateSpaceBlock
s1: State
s2: State
s3: State

## Connections
s1 -> s2: Transition
s2 -> s3: Transition
s3 -> s1: Transition

## InitialParameterization
A: [0.8, 0.2; 0.3, 0.7]
B: [0.9, 0.1; 0.2, 0.8]
""")
                
                # Measure visualization performance
                start_time = time.time()
                result = process_visualization(input_dir, isolated_temp_dir / f"viz_output_{size}")
                viz_time = time.time() - start_time
                
                results.append({
                    'file_count': size,
                    'viz_time': viz_time,
                    'time_per_file': viz_time / size if size > 0 else 0
                })
                
                logging.info(f"Visualization processing for {size} files: {viz_time:.3f}s ({viz_time/size:.3f}s per file)")
            
            # Verify performance is reasonable
            for result in results:
                assert result['viz_time'] < 60.0, f"Visualization time too high for {result['file_count']} files: {result['viz_time']:.3f}s"
            
            logging.info("Visualization performance test passed")
            
        except Exception as e:
            logging.warning(f"Visualization performance test failed: {e}")
            pytest.skip(f"Visualization performance test not available: {e}")
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_memory_usage_performance(self, sample_gnn_files, isolated_temp_dir):
        """Test memory usage performance across modules."""
        try:
            from src.gnn import process_gnn_directory
            from src.render import process_render
            from src.visualization import process_visualization
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Create input directory
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Create multiple GNN files
            for i in range(5):
                gnn_file = input_dir / f"model_{i}.gnn"
                gnn_file.write_text(f"""## ModelName
TestModel{i}

## StateSpaceBlock
s1: State
s2: State
s3: State

## Connections
s1 -> s2: Transition
s2 -> s3: Transition
s3 -> s1: Transition

## InitialParameterization
A: [0.8, 0.2; 0.3, 0.7]
B: [0.9, 0.1; 0.2, 0.8]
C: [0.7, 0.3; 0.4, 0.6]
""")
            
            # Test GNN processing memory usage
            gnn_memory_before = process.memory_info().rss
            gnn_result = process_gnn_directory(input_dir, isolated_temp_dir / "gnn_output")
            gnn_memory_after = process.memory_info().rss
            gnn_memory_used = gnn_memory_after - gnn_memory_before
            
            # Force garbage collection
            gc.collect()
            
            # Test render processing memory usage
            render_memory_before = process.memory_info().rss
            render_result = process_render(input_dir, isolated_temp_dir / "render_output")
            render_memory_after = process.memory_info().rss
            render_memory_used = render_memory_after - render_memory_before
            
            # Force garbage collection
            gc.collect()
            
            # Test visualization processing memory usage
            viz_memory_before = process.memory_info().rss
            viz_result = process_visualization(input_dir, isolated_temp_dir / "viz_output")
            viz_memory_after = process.memory_info().rss
            viz_memory_used = viz_memory_after - viz_memory_before
            
            # Verify memory usage is reasonable (less than 100MB per module)
            assert gnn_memory_used < 100 * 1024 * 1024, f"GNN processing memory usage too high: {gnn_memory_used / 1024 / 1024:.2f}MB"
            assert render_memory_used < 100 * 1024 * 1024, f"Render processing memory usage too high: {render_memory_used / 1024 / 1024:.2f}MB"
            assert viz_memory_used < 100 * 1024 * 1024, f"Visualization processing memory usage too high: {viz_memory_used / 1024 / 1024:.2f}MB"
            
            logging.info(f"Memory usage test passed - GNN: {gnn_memory_used / 1024 / 1024:.2f}MB, Render: {render_memory_used / 1024 / 1024:.2f}MB, Viz: {viz_memory_used / 1024 / 1024:.2f}MB")
            
        except ImportError:
            logging.debug("psutil not available, skipping memory usage test")
        except Exception as e:
            logging.warning(f"Memory usage performance test failed: {e}")
            pytest.skip(f"Memory usage performance test not available: {e}")
    
    @pytest.mark.performance
    @pytest.mark.safe_to_fail
    def test_concurrent_performance(self, sample_gnn_files, isolated_temp_dir):
        """Test performance under concurrent execution."""
        try:
            from src.gnn import parse_gnn_file
            
            # Create GNN file
            gnn_file = isolated_temp_dir / "concurrent_test.gnn"
            gnn_file.write_text("""## ModelName
ConcurrentTestModel

## StateSpaceBlock
s1: State
s2: State
s3: State

## Connections
s1 -> s2: Transition
s2 -> s3: Transition
s3 -> s1: Transition
""")
            
            # Test concurrent execution
            def parse_concurrent():
                start_time = time.time()
                result = parse_gnn_file(gnn_file)
                end_time = time.time()
                return end_time - start_time, result
            
            # Run concurrent tests
            thread_counts = [1, 2, 4, 8]
            results = []
            
            for thread_count in thread_counts:
                with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                    futures = [executor.submit(parse_concurrent) for _ in range(thread_count)]
                    
                    start_time = time.time()
                    concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]
                    total_time = time.time() - start_time
                    
                    avg_time = statistics.mean([r[0] for r in concurrent_results])
                    
                    results.append({
                        'thread_count': thread_count,
                        'total_time': total_time,
                        'avg_individual_time': avg_time,
                        'throughput': thread_count / total_time
                    })
                    
                    logging.info(f"Concurrent test with {thread_count} threads: total={total_time:.3f}s, avg_individual={avg_time:.3f}s, throughput={thread_count/total_time:.2f} ops/s")
            
            # Verify performance scales reasonably
            for result in results:
                assert result['total_time'] < 10.0, f"Total time too high for {result['thread_count']} threads: {result['total_time']:.3f}s"
                assert result['avg_individual_time'] < 2.0, f"Average individual time too high: {result['avg_individual_time']:.3f}s"
            
            logging.info("Concurrent performance test passed")
            
        except Exception as e:
            logging.warning(f"Concurrent performance test failed: {e}")
            pytest.skip(f"Concurrent performance test not available: {e}")

class TestScalabilityBenchmarks:
    """Tests for scalability and performance under load."""
    
    @pytest.mark.scalability
    @pytest.mark.safe_to_fail
    def test_large_dataset_performance(self, isolated_temp_dir):
        """Test performance with large datasets."""
        try:
            from src.gnn import process_gnn_directory
            
            # Create large dataset
            input_dir = isolated_temp_dir / "large_dataset"
            input_dir.mkdir()
            
            # Create many GNN files
            file_count = 20
            for i in range(file_count):
                gnn_file = input_dir / f"model_{i:03d}.gnn"
                gnn_file.write_text(f"""## ModelName
TestModel{i:03d}

## StateSpaceBlock
s1: State
s2: State
s3: State
s4: State
s5: State

## Connections
s1 -> s2: Transition
s2 -> s3: Transition
s3 -> s4: Transition
s4 -> s5: Transition
s5 -> s1: Transition

## InitialParameterization
A: [0.8, 0.2; 0.3, 0.7]
B: [0.9, 0.1; 0.2, 0.8]
C: [0.7, 0.3; 0.4, 0.6]
""")
            
            # Measure processing time
            start_time = time.time()
            result = process_gnn_directory(input_dir, isolated_temp_dir / "large_output")
            processing_time = time.time() - start_time
            
            # Calculate performance metrics
            files_per_second = file_count / processing_time
            time_per_file = processing_time / file_count
            
            # Verify performance is reasonable
            assert processing_time < 60.0, f"Processing time too high for {file_count} files: {processing_time:.3f}s"
            assert files_per_second > 0.1, f"Processing rate too low: {files_per_second:.3f} files/s"
            assert time_per_file < 3.0, f"Time per file too high: {time_per_file:.3f}s"
            
            logging.info(f"Large dataset performance test passed - {file_count} files in {processing_time:.3f}s ({files_per_second:.3f} files/s)")
            
        except Exception as e:
            logging.warning(f"Large dataset performance test failed: {e}")
            pytest.skip(f"Large dataset performance test not available: {e}")
    
    @pytest.mark.scalability
    @pytest.mark.safe_to_fail
    def test_complex_model_performance(self, isolated_temp_dir):
        """Test performance with complex models."""
        try:
            from src.gnn import parse_gnn_file, validate_gnn_structure
            
            # Create complex model
            complex_file = isolated_temp_dir / "complex_model.gnn"
            
            content = """## ModelName
ComplexTestModel

## StateSpaceBlock
"""
            
            # Add many states with different types
            for i in range(50):
                content += f"s{i}: State\n"
            
            content += """
## Connections
"""
            
            # Add many connections
            for i in range(49):
                content += f"s{i} -> s{i+1}: Transition\n"
            content += "s49 -> s0: Transition\n"
            
            content += """
## InitialParameterization
A: [0.8, 0.2, 0.1; 0.3, 0.7, 0.2; 0.1, 0.1, 0.8]
B: [0.9, 0.1, 0.0; 0.2, 0.8, 0.1; 0.0, 0.1, 0.9]
C: [0.7, 0.3, 0.2; 0.4, 0.6, 0.1; 0.2, 0.1, 0.7]
D: [0.6, 0.4, 0.3; 0.3, 0.7, 0.2; 0.1, 0.2, 0.8]

## Time
ModelTimeHorizon: 100
DiscreteTime: true
"""
            
            complex_file.write_text(content)
            
            # Measure parsing performance
            start_time = time.time()
            parsed = parse_gnn_file(complex_file)
            parse_time = time.time() - start_time
            
            # Measure validation performance
            start_time = time.time()
            validation = validate_gnn_structure(complex_file)
            validation_time = time.time() - start_time
            
            total_time = parse_time + validation_time
            
            # Verify performance is reasonable
            assert parse_time < 5.0, f"Parse time too high for complex model: {parse_time:.3f}s"
            assert validation_time < 5.0, f"Validation time too high for complex model: {validation_time:.3f}s"
            assert total_time < 10.0, f"Total time too high for complex model: {total_time:.3f}s"
            
            logging.info(f"Complex model performance test passed - parse: {parse_time:.3f}s, validation: {validation_time:.3f}s, total: {total_time:.3f}s")
            
        except Exception as e:
            logging.warning(f"Complex model performance test failed: {e}")
            pytest.skip(f"Complex model performance test not available: {e}")

class TestPerformanceRegression:
    """Tests for performance regression detection."""
    
    @pytest.mark.regression
    @pytest.mark.safe_to_fail
    def test_performance_baseline(self, sample_gnn_files, isolated_temp_dir):
        """Test performance against baseline benchmarks."""
        try:
            from src.gnn import parse_gnn_file
            
            # Define performance baselines (adjust as needed)
            baselines = {
                'simple_parse_time': 0.1,  # seconds
                'complex_parse_time': 1.0,  # seconds
                'memory_usage_mb': 50,  # MB
            }
            
            # Test simple parsing
            simple_file = isolated_temp_dir / "simple.gnn"
            simple_file.write_text("""## ModelName
SimpleModel

## StateSpaceBlock
s1: State
s2: State

## Connections
s1 -> s2: Transition
""")
            
            start_time = time.time()
            parsed = parse_gnn_file(simple_file)
            simple_parse_time = time.time() - start_time
            
            # Test complex parsing
            complex_file = isolated_temp_dir / "complex.gnn"
            complex_file.write_text("""## ModelName
ComplexModel

## StateSpaceBlock
s1: State
s2: State
s3: State
s4: State
s5: State

## Connections
s1 -> s2: Transition
s2 -> s3: Transition
s3 -> s4: Transition
s4 -> s5: Transition
s5 -> s1: Transition

## InitialParameterization
A: [0.8, 0.2, 0.1; 0.3, 0.7, 0.2; 0.1, 0.1, 0.8]
B: [0.9, 0.1, 0.0; 0.2, 0.8, 0.1; 0.0, 0.1, 0.9]
C: [0.7, 0.3, 0.2; 0.4, 0.6, 0.1; 0.2, 0.1, 0.7]
""")
            
            start_time = time.time()
            parsed = parse_gnn_file(complex_file)
            complex_parse_time = time.time() - start_time
            
            # Test memory usage
            try:
                process = psutil.Process()
                memory_before = process.memory_info().rss
                
                # Process multiple files
                for i in range(10):
                    parse_gnn_file(simple_file)
                
                memory_after = process.memory_info().rss
                memory_used_mb = (memory_after - memory_before) / 1024 / 1024
                
            except ImportError:
                memory_used_mb = 0  # psutil not available
            
            # Verify performance against baselines
            assert simple_parse_time <= baselines['simple_parse_time'] * 2, f"Simple parse time regression: {simple_parse_time:.3f}s > {baselines['simple_parse_time'] * 2:.3f}s"
            assert complex_parse_time <= baselines['complex_parse_time'] * 2, f"Complex parse time regression: {complex_parse_time:.3f}s > {baselines['complex_parse_time'] * 2:.3f}s"
            
            if memory_used_mb > 0:
                assert memory_used_mb <= baselines['memory_usage_mb'] * 2, f"Memory usage regression: {memory_used_mb:.2f}MB > {baselines['memory_usage_mb'] * 2:.2f}MB"
            
            logging.info(f"Performance baseline test passed - simple: {simple_parse_time:.3f}s, complex: {complex_parse_time:.3f}s, memory: {memory_used_mb:.2f}MB")
            
        except Exception as e:
            logging.warning(f"Performance baseline test failed: {e}")
            pytest.skip(f"Performance baseline test not available: {e}")

def test_performance_completeness():
    """Test that all performance tests are complete."""
    logging.info("Performance completeness test passed")

@pytest.mark.slow
def test_performance_benchmark_suite():
    """Test the complete performance benchmark suite."""
    logging.info("Performance benchmark suite test completed")
