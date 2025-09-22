"""
Performance Benchmarks for Generalized Notation Notation (GNN) Pipeline

This module contains comprehensive performance benchmarks to ensure the pipeline
maintains acceptable performance characteristics across different workloads.
"""

import pytest
import time
import psutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import gc

from src.gnn import GNNProcessor
from src.type_checker import GNNTypeChecker
from src.render import CodeRenderer
from src.utils.performance_tracker import PerformanceTracker


class TestGNNPerformanceBenchmarks:
    """Performance benchmarks for GNN processing."""

    @pytest.fixture
    def large_gnn_file(self):
        """Create a large GNN file for performance testing."""
        # Generate a large GNN model with many components
        components = []
        for i in range(50):  # 50 matrices
            components.append(f"A{i}[10,10],float  # Matrix A{i}")
        for i in range(30):  # 30 states
            components.append(f"s{i}[5,1],float  # State s{i}")
        for i in range(20):  # 20 connections
            components.append(f"s{i}-A{i}")

        gnn_content = f"""
        # Large GNN Performance Test Model
        ## GNNSection
        LargePerformanceTest

        ## ModelName
        Large Performance Test Model

        ## StateSpaceBlock
        {"\n".join(components)}

        ## Connections
        {"\n".join([f"A{i}>s{i}" for i in range(20)])}
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(gnn_content)
            f.flush()
            return Path(f.name)

    def test_gnn_parsing_performance(self, large_gnn_file):
        """Benchmark GNN parsing performance with large files."""
        from src.gnn import GNNProcessor

        processor = GNNProcessor()

        # Measure parsing time
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        with open(large_gnn_file, 'r') as f:
            content = f.read()

        parsed_model = processor.parse_gnn_content(content)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        parsing_time = end_time - start_time
        memory_increase = end_memory - start_memory

        # Performance assertions
        assert parsing_time < 5.0, f"Parsing took {parsing_time:.2f}s, should be < 5.0s"
        assert memory_increase < 100, f"Memory increase: {memory_increase:.1f}MB, should be < 100MB"
        assert parsed_model is not None

    def test_type_checking_performance(self, large_gnn_file):
        """Benchmark type checking performance with large models."""
        from src.type_checker import GNNTypeChecker

        checker = GNNTypeChecker()

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        result = checker.validate_single_gnn_file(large_gnn_file)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        type_check_time = end_time - start_time
        memory_increase = end_memory - start_memory

        # Performance assertions
        assert type_check_time < 3.0, f"Type checking took {type_check_time".2f"}s, should be < 3.0s"
        assert memory_increase < 50, f"Memory increase: {memory_increase".1f"}MB, should be < 50MB"
        assert result['valid'] is True

    def test_rendering_performance(self, large_gnn_file):
        """Benchmark rendering performance with large models."""
        from src.gnn import GNNProcessor
        from src.render.pymdp import PyMDPRenderer

        gnn_processor = GNNProcessor()
        renderer = PyMDPRenderer()

        with open(large_gnn_file, 'r') as f:
            content = f.read()

        parsed_model = gnn_processor.parse_gnn_content(content)

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        rendered_code = renderer.render_gnn_model(parsed_model)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        render_time = end_time - start_time
        memory_increase = end_memory - start_memory

        # Performance assertions
        assert render_time < 10.0, f"Rendering took {render_time".2f"}s, should be < 10.0s"
        assert memory_increase < 150, f"Memory increase: {memory_increase".1f"}MB, should be < 150MB"
        assert len(rendered_code) > 1000, "Rendered code should be substantial"


class TestMemoryEfficiencyBenchmarks:
    """Memory efficiency benchmarks."""

    def test_memory_efficiency_under_load(self):
        """Test memory efficiency when processing multiple files."""
        from src.gnn import GNNProcessor
        from src.type_checker import GNNTypeChecker

        processor = GNNProcessor()
        checker = GNNTypeChecker()

        # Create multiple test files
        test_files = []
        for i in range(10):
            content = f"""
            # Test Model {i}
            ## GNNSection
            TestModel{i}

            ## ModelName
            Test Model {i}

            ## StateSpaceBlock
            A{i}[3,3],float
            B{i}[3,3,3],float
            s{i}[3,1],float
            """
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(content)
                f.flush()
                test_files.append(Path(f.name))

        try:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Process all files
            for file_path in test_files:
                with open(file_path, 'r') as f:
                    content = f.read()

                parsed_model = processor.parse_gnn_content(content)
                type_result = checker.validate_single_gnn_file(file_path)

                # Clean up after each iteration
                del parsed_model
                del type_result
                gc.collect()

            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Performance assertions
            assert memory_increase < 200, f"Memory increase: {memory_increase".1f"}MB, should be < 200MB"

        finally:
            for file_path in test_files:
                file_path.unlink()

    def test_garbage_collection_efficiency(self):
        """Test that garbage collection works efficiently after processing."""
        import gc

        from src.gnn import GNNProcessor

        processor = GNNProcessor()

        # Create large content
        large_content = """
        # Large Test Model
        ## GNNSection
        LargeTest

        ## ModelName
        Large Test Model

        ## StateSpaceBlock
        """ + "\n".join([f"A{i}[5,5],float  # Matrix A{i}" for i in range(100)])

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Process and delete multiple times
        for _ in range(5):
            parsed_model = processor.parse_gnn_content(large_content)
            del parsed_model
            gc.collect()

        # Force garbage collection
        gc.collect()
        gc.collect()

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # Should not have significant memory leaks
        assert memory_increase < 50, f"Memory increase: {memory_increase".1f"}MB, should be < 50MB"


class TestScalabilityBenchmarks:
    """Scalability benchmarks for different model sizes."""

    @pytest.fixture
    def small_gnn_file(self):
        """Create a small GNN file for scalability testing."""
        content = """
        # Small GNN Model
        ## GNNSection
        SmallTest

        ## ModelName
        Small Test Model

        ## StateSpaceBlock
        A[2,2],float
        s[2,1],float
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            return Path(f.name)

    @pytest.fixture
    def medium_gnn_file(self):
        """Create a medium GNN file for scalability testing."""
        components = []
        for i in range(10):
            components.append(f"A{i}[5,5],float")
        for i in range(5):
            components.append(f"s{i}[3,1],float")

        content = f"""
        # Medium GNN Model
        ## GNNSection
        MediumTest

        ## ModelName
        Medium Test Model

        ## StateSpaceBlock
        {"\n".join(components)}
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            return Path(f.name)

    def test_linear_scaling_parsing(self, small_gnn_file, medium_gnn_file):
        """Test that parsing time scales linearly with model size."""
        from src.gnn import GNNProcessor

        processor = GNNProcessor()

        # Measure small model
        with open(small_gnn_file, 'r') as f:
            small_content = f.read()

        start_time = time.time()
        small_model = processor.parse_gnn_content(small_content)
        small_time = time.time() - start_time

        # Measure medium model
        with open(medium_gnn_file, 'r') as f:
            medium_content = f.read()

        start_time = time.time()
        medium_model = processor.parse_gnn_content(medium_content)
        medium_time = time.time() - start_time

        # Calculate scaling ratio
        size_ratio = len(medium_content) / len(small_content)  # ~5x larger
        time_ratio = medium_time / small_time

        # Should scale approximately linearly (within 2x of perfect scaling)
        assert time_ratio < size_ratio * 2, f"Time ratio {time_ratio".2f"} > 2x size ratio {size_ratio".2f"}"

    def test_constant_memory_scaling(self, small_gnn_file, medium_gnn_file):
        """Test that memory usage doesn't grow excessively with model size."""
        from src.gnn import GNNProcessor

        processor = GNNProcessor()

        # Measure small model memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        with open(small_gnn_file, 'r') as f:
            small_content = f.read()

        small_model = processor.parse_gnn_content(small_content)
        del small_model
        gc.collect()

        small_memory = psutil.Process().memory_info().rss / 1024 / 1024
        small_memory_increase = small_memory - initial_memory

        # Measure medium model memory
        with open(medium_gnn_file, 'r') as f:
            medium_content = f.read()

        medium_model = processor.parse_gnn_content(medium_content)
        del medium_model
        gc.collect()

        medium_memory = psutil.Process().memory_info().rss / 1024 / 1024
        medium_memory_increase = medium_memory - initial_memory

        # Memory should not increase more than 5x for 5x larger model
        size_ratio = len(medium_content) / len(small_content)
        memory_ratio = medium_memory_increase / small_memory_increase

        assert memory_ratio < size_ratio * 3, f"Memory ratio {memory_ratio".2f"} > 3x size ratio {size_ratio".2f"}"


class TestConcurrentProcessingBenchmarks:
    """Benchmarks for concurrent processing capabilities."""

    def test_concurrent_gnn_processing(self):
        """Test concurrent processing of multiple GNN files."""
        from src.gnn import GNNProcessor
        import threading
        from concurrent.futures import ThreadPoolExecutor

        processor = GNNProcessor()

        # Create multiple test files
        test_files = []
        for i in range(5):
            content = f"""
            # Concurrent Test Model {i}
            ## GNNSection
            ConcurrentTest{i}

            ## ModelName
            Concurrent Test Model {i}

            ## StateSpaceBlock
            A{i}[3,3],float
            s{i}[3,1],float
            """
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(content)
                f.flush()
                test_files.append(Path(f.name))

        try:
            start_time = time.time()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Process files concurrently
            results = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for file_path in test_files:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    future = executor.submit(processor.parse_gnn_content, content)
                    futures.append(future)

                for future in futures:
                    result = future.result()
                    results.append(result)

            end_time = time.time()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024

            concurrent_time = end_time - start_time
            memory_increase = final_memory - initial_memory

            # Performance assertions
            assert concurrent_time < 10.0, f"Concurrent processing took {concurrent_time".2f"}s, should be < 10.0s"
            assert memory_increase < 200, f"Memory increase: {memory_increase".1f"}MB, should be < 200MB"
            assert len(results) == 5, "All files should be processed"
            assert all(result is not None for result in results), "All results should be valid"

        finally:
            for file_path in test_files:
                file_path.unlink()

    def test_thread_safety(self):
        """Test thread safety of core components."""
        from src.gnn import GNNProcessor
        from src.type_checker import GNNTypeChecker
        import threading

        processor = GNNProcessor()
        checker = GNNTypeChecker()

        # Create test content
        test_content = """
        # Thread Safety Test
        ## GNNSection
        ThreadSafetyTest

        ## ModelName
        Thread Safety Test Model

        ## StateSpaceBlock
        A[2,2],float
        s[2,1],float
        """

        results = []
        errors = []

        def process_content():
            try:
                parsed = processor.parse_gnn_content(test_content)
                # Create a temporary file for type checking
                with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                    f.write(test_content)
                    f.flush()
                    temp_file = Path(f.name)

                try:
                    type_result = checker.validate_single_gnn_file(temp_file)
                    results.append((parsed, type_result))
                finally:
                    temp_file.unlink()
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=process_content)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Assertions
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert all(len(result) == 2 for result in results), "Each result should have parsed model and type result"
        assert all(result[0] is not None for result in results), "All parsed models should be valid"
        assert all(result[1]['valid'] for result in results), "All type checks should pass"


class TestPerformanceRegressionDetection:
    """Performance regression detection benchmarks."""

    def test_baseline_performance_establishment(self):
        """Establish baseline performance metrics."""
        from src.gnn import GNNProcessor
        from src.type_checker import GNNTypeChecker

        # Create a standard test model
        standard_content = """
        # Standard Performance Test Model
        ## GNNSection
        StandardTest

        ## ModelName
        Standard Performance Test Model

        ## StateSpaceBlock
        A[3,3],float
        B[3,3,3],float
        C[3],float
        D[3],float
        E[3],float
        s[3,1],float
        o[3,1],int
        π[3],float
        u[1],int
        F[π],float
        G[π],float
        t[1],int
        """

        processor = GNNProcessor()
        checker = GNNTypeChecker()

        # Measure multiple times for consistency
        times = []
        for _ in range(5):
            start_time = time.time()
            parsed = processor.parse_gnn_content(standard_content)
            type_result = checker.validate_single_gnn_file(
                Path(__file__).parent / "test_data" / "sample_gnn_model.md"
            )
            end_time = time.time()

            times.append(end_time - start_time)
            del parsed, type_result
            gc.collect()

        # Calculate baseline metrics
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)

        # Store baseline (in practice, this would be stored in a file/database)
        baseline = {
            'average_time': avg_time,
            'max_time': max_time,
            'min_time': min_time,
            'std_dev': (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        }

        # Assertions for reasonable baseline
        assert baseline['average_time'] < 2.0, f"Baseline too slow: {baseline['average_time']".2f"}s"
        assert baseline['max_time'] < 3.0, f"Max baseline too slow: {baseline['max_time']".2f"}s"
        assert baseline['std_dev'] < 0.5, f"Too much variation: {baseline['std_dev']".2f"}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
