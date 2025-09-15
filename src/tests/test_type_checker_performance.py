#!/usr/bin/env python3
"""
Performance and edge case tests for the Type Checker module.

This test suite focuses on performance characteristics, edge cases,
and stress testing for the type checker functionality.
"""

import pytest
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

# Test data generators for performance testing
def generate_large_variable_set(size: int) -> List[Dict[str, Any]]:
    """Generate a large set of variables for performance testing."""
    variables = []
    for i in range(size):
        variables.append({
            "name": f"var_{i}",
            "type": "belief" if i % 2 == 0 else "action",
            "data_type": "float" if i % 3 == 0 else "int",
            "dimensions": [10 + (i % 5), 1 + (i % 3)],
            "description": f"Variable {i} for performance testing"
        })
    return variables

def generate_large_connection_set(size: int) -> List[Dict[str, Any]]:
    """Generate a large set of connections for performance testing."""
    connections = []
    for i in range(size):
        connections.append({
            "type": "transition" if i % 2 == 0 else "observation",
            "source_variables": [f"var_{i}", f"var_{(i+1) % size}"],
            "target_variables": [f"var_{(i+2) % size}"],
            "description": f"Connection {i} for performance testing"
        })
    return connections

def generate_malformed_data() -> List[Dict[str, Any]]:
    """Generate malformed data for edge case testing."""
    return [
        None,  # None value
        "not_a_dict",  # String instead of dict
        {},  # Empty dict
        {"invalid": "structure"},  # Missing required fields
        {"name": None, "type": "belief"},  # None values
        {"name": "var", "type": 123},  # Wrong type
        {"name": "var", "dimensions": "not_a_list"},  # Wrong dimension type
        {"name": "var", "dimensions": [1, -1, 0]},  # Invalid dimensions
        {"name": "var", "dimensions": []},  # Empty dimensions
        {"name": "var", "dimensions": [1e10, 1e10, 1e10]},  # Very large dimensions
    ]

class TestTypeCheckerPerformance:
    """Test performance characteristics of the type checker."""
    
    def test_large_variable_analysis_performance(self):
        """Test performance with large variable sets."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        # Test with progressively larger datasets
        sizes = [100, 500, 1000, 2000]
        
        for size in sizes:
            variables = generate_large_variable_set(size)
            
            start_time = time.time()
            result = analyze_variable_types(variables)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Verify results
            assert isinstance(result, dict)
            assert result["total_variables"] == size
            assert "complexity_metrics" in result
            
            # Performance assertions
            assert processing_time < 5.0, f"Processing {size} variables took {processing_time:.2f}s (too slow)"
            
            # Memory efficiency check
            total_elements = result["complexity_metrics"]["total_elements"]
            assert total_elements > 0, "Should calculate total elements"
            
            print(f"Processed {size} variables in {processing_time:.3f}s")
    
    def test_large_connection_analysis_performance(self):
        """Test performance with large connection sets."""
        from src.type_checker.analysis_utils import analyze_connections
        
        sizes = [50, 200, 500, 1000]
        
        for size in sizes:
            connections = generate_large_connection_set(size)
            
            start_time = time.time()
            result = analyze_connections(connections)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Verify results
            assert isinstance(result, dict)
            assert result["total_connections"] == size
            assert "connectivity_metrics" in result
            
            # Performance assertions
            assert processing_time < 3.0, f"Processing {size} connections took {processing_time:.2f}s (too slow)"
            
            print(f"Processed {size} connections in {processing_time:.3f}s")
    
    def test_complexity_estimation_performance(self):
        """Test performance of complexity estimation with large models."""
        from src.type_checker.analysis_utils import (
            analyze_variable_types,
            analyze_connections,
            estimate_computational_complexity
        )
        
        # Large model
        variables = generate_large_variable_set(1000)
        connections = generate_large_connection_set(500)
        
        start_time = time.time()
        
        type_analysis = analyze_variable_types(variables)
        conn_analysis = analyze_connections(connections)
        complexity = estimate_computational_complexity(type_analysis, conn_analysis)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        assert isinstance(complexity, dict)
        assert "inference_complexity" in complexity
        assert "resource_requirements" in complexity
        
        # Performance assertion
        assert processing_time < 2.0, f"Complexity estimation took {processing_time:.2f}s (too slow)"
        
        print(f"Complexity estimation completed in {processing_time:.3f}s")
    
    def test_memory_efficiency_large_dimensions(self):
        """Test memory efficiency with very large dimensions."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        # Variables with very large dimensions
        large_variables = [
            {
                "name": "large_var_1",
                "type": "belief",
                "data_type": "float",
                "dimensions": [1000, 1000]  # 1 million elements
            },
            {
                "name": "large_var_2",
                "type": "action",
                "data_type": "int",
                "dimensions": [500, 500, 500]  # 125 million elements
            }
        ]
        
        start_time = time.time()
        result = analyze_variable_types(large_variables)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify results
        assert result["total_variables"] == 2
        assert result["complexity_metrics"]["total_elements"] == 1000000 + 125000000
        
        # Should handle large dimensions efficiently
        assert processing_time < 1.0, f"Large dimension processing took {processing_time:.2f}s (too slow)"
        
        # Check memory estimates
        memory_gb = result["complexity_metrics"]["estimated_memory_gb"]
        assert memory_gb > 0, "Should calculate memory usage"
        assert memory_gb < 1000, "Memory estimate should be reasonable"
        
        print(f"Processed large dimensions in {processing_time:.3f}s, estimated memory: {memory_gb:.2f} GB")
    
    def test_concurrent_processing_simulation(self):
        """Test handling of concurrent-like processing scenarios."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        # Simulate multiple analysis operations
        results = []
        
        start_time = time.time()
        
        for i in range(10):
            variables = generate_large_variable_set(100)
            result = analyze_variable_types(variables)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all results
        assert len(results) == 10
        for result in results:
            assert isinstance(result, dict)
            assert result["total_variables"] == 100
        
        # Should handle multiple operations efficiently
        assert total_time < 5.0, f"Concurrent simulation took {total_time:.2f}s (too slow)"
        
        print(f"Processed 10 concurrent operations in {total_time:.3f}s")


class TestTypeCheckerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_malformed_variable_data(self):
        """Test handling of malformed variable data."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        malformed_data = generate_malformed_data()
        
        # Should not crash with malformed data
        result = analyze_variable_types(malformed_data)
        
        assert isinstance(result, dict)
        assert result["total_variables"] == 0  # Should filter out invalid data
        assert "validation_issues" in result
        assert len(result["validation_issues"]) > 0  # Should report issues
    
    def test_malformed_connection_data(self):
        """Test handling of malformed connection data."""
        from src.type_checker.analysis_utils import analyze_connections
        
        malformed_data = generate_malformed_data()
        
        # Should not crash with malformed data
        result = analyze_connections(malformed_data)
        
        assert isinstance(result, dict)
        assert result["total_connections"] == 0  # Should filter out invalid data
        assert "validation_issues" in result
        assert len(result["validation_issues"]) > 0  # Should report issues
    
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        from src.type_checker.analysis_utils import (
            analyze_variable_types,
            analyze_connections,
            estimate_computational_complexity
        )
        
        # Empty variables
        var_result = analyze_variable_types([])
        assert var_result["total_variables"] == 0
        assert var_result["complexity_metrics"]["total_elements"] == 0
        
        # Empty connections
        conn_result = analyze_connections([])
        assert conn_result["total_connections"] == 0
        
        # Empty complexity estimation
        complexity = estimate_computational_complexity(var_result, conn_result)
        assert isinstance(complexity, dict)
        assert complexity["inference_complexity"]["operations_per_step"] == 0
    
    def test_none_inputs(self):
        """Test handling of None inputs."""
        from src.type_checker.analysis_utils import (
            analyze_variable_types,
            analyze_connections,
            estimate_computational_complexity
        )
        
        # None variables
        var_result = analyze_variable_types(None)
        assert isinstance(var_result, dict)
        assert var_result["total_variables"] == 0
        
        # None connections
        conn_result = analyze_connections(None)
        assert isinstance(conn_result, dict)
        assert conn_result["total_connections"] == 0
        
        # None complexity inputs
        complexity = estimate_computational_complexity(None, None)
        assert isinstance(complexity, dict)
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        # Variables with extreme values
        extreme_variables = [
            {
                "name": "extreme_var",
                "type": "belief",
                "data_type": "float",
                "dimensions": [1e6, 1e6, 1e6]  # Extremely large dimensions
            },
            {
                "name": "zero_var",
                "type": "action",
                "data_type": "int",
                "dimensions": [0, 0, 0]  # Zero dimensions
            },
            {
                "name": "negative_var",
                "type": "observation",
                "data_type": "float",
                "dimensions": [-1, -2, -3]  # Negative dimensions
            }
        ]
        
        result = analyze_variable_types(extreme_variables)
        
        assert isinstance(result, dict)
        assert "performance_warnings" in result
        assert len(result["performance_warnings"]) > 0  # Should warn about extreme values
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        unicode_variables = [
            {
                "name": "变量_1",  # Chinese characters
                "type": "belief",
                "data_type": "float",
                "dimensions": [10, 1]
            },
            {
                "name": "var_with_émojis_🚀",
                "type": "action",
                "data_type": "int",
                "dimensions": [5]
            },
            {
                "name": "var-with-special-chars!@#$%",
                "type": "observation",
                "data_type": "float",
                "dimensions": [3, 3]
            }
        ]
        
        result = analyze_variable_types(unicode_variables)
        
        assert isinstance(result, dict)
        assert result["total_variables"] == 3
        assert "validation_issues" in result  # May have validation issues with special chars


class TestTypeCheckerStressTests:
    """Stress tests for the type checker."""
    
    def test_memory_stress_test(self):
        """Test memory usage under stress conditions."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        # Create very large dataset
        large_variables = []
        for i in range(10000):  # 10k variables
            large_variables.append({
                "name": f"stress_var_{i}",
                "type": "belief",
                "data_type": "float",
                "dimensions": [100, 100]  # 10k elements each
            })
        
        start_time = time.time()
        result = analyze_variable_types(large_variables)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete without memory issues
        assert isinstance(result, dict)
        assert result["total_variables"] == 10000
        
        # Should complete in reasonable time
        assert processing_time < 30.0, f"Stress test took {processing_time:.2f}s (too slow)"
        
        print(f"Stress test completed: {result['total_variables']} variables in {processing_time:.2f}s")
    
    def test_rapid_successive_calls(self):
        """Test rapid successive calls to analysis functions."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        # Make many rapid calls
        start_time = time.time()
        
        for i in range(100):
            variables = generate_large_variable_set(50)
            result = analyze_variable_types(variables)
            assert isinstance(result, dict)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle rapid calls efficiently
        assert total_time < 10.0, f"Rapid calls took {total_time:.2f}s (too slow)"
        
        print(f"100 rapid calls completed in {total_time:.2f}s")
    
    def test_mixed_data_types_stress(self):
        """Test with mixed data types and structures."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        # Mix of valid and invalid data
        mixed_data = []
        
        # Add valid data
        for i in range(1000):
            mixed_data.append({
                "name": f"valid_var_{i}",
                "type": "belief",
                "data_type": "float",
                "dimensions": [10, 1]
            })
        
        # Add invalid data
        mixed_data.extend(generate_malformed_data())
        
        # Add more valid data
        for i in range(1000, 2000):
            mixed_data.append({
                "name": f"valid_var_{i}",
                "type": "action",
                "data_type": "int",
                "dimensions": [5]
            })
        
        start_time = time.time()
        result = analyze_variable_types(mixed_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should handle mixed data gracefully
        assert isinstance(result, dict)
        assert result["total_variables"] == 2000  # Only valid data
        assert len(result["validation_issues"]) > 0  # Should report issues
        
        # Should complete efficiently
        assert processing_time < 5.0, f"Mixed data processing took {processing_time:.2f}s (too slow)"
        
        print(f"Mixed data stress test: {result['total_variables']} valid variables in {processing_time:.2f}s")


class TestTypeCheckerProcessorPerformance:
    """Test performance of the GNNTypeChecker processor."""
    
    def test_large_file_processing(self):
        """Test processing of large GNN files."""
        from src.type_checker.processor import GNNTypeChecker
        
        # Create a large GNN file
        large_content = "# Large GNN Model\n\n## Variables\n"
        for i in range(1000):
            large_content += f"var_{i}: belief [10, 1]  # Variable {i}\n"
        
        large_content += "\n## Connections\n"
        for i in range(500):
            large_content += f"var_{i}, var_{i+1} -> var_{i+2}  # Connection {i}\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(large_content)
            f.flush()
            temp_file = Path(f.name)
        
        try:
            checker = GNNTypeChecker(verbose=False)
            
            start_time = time.time()
            result = checker.validate_single_gnn_file(temp_file, verbose=False)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Should complete successfully
            assert isinstance(result, dict)
            assert "valid" in result
            
            # Should complete in reasonable time
            assert processing_time < 10.0, f"Large file processing took {processing_time:.2f}s (too slow)"
            
            print(f"Large file processed in {processing_time:.2f}s")
            
        finally:
            temp_file.unlink(missing_ok=True)
    
    def test_directory_processing_performance(self):
        """Test performance of directory processing."""
        from src.type_checker.processor import GNNTypeChecker
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"
            
            # Create multiple GNN files
            for i in range(10):
                gnn_file = temp_path / f"model_{i}.md"
                content = f"# Model {i}\n\n## Variables\nvar_{i}: belief [10, 1]\n"
                gnn_file.write_text(content)
            
            checker = GNNTypeChecker(verbose=False)
            
            start_time = time.time()
            success = checker.validate_gnn_files(temp_path, output_dir, verbose=False)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Should complete successfully
            assert success
            assert output_dir.exists()
            assert (output_dir / "type_check_results").exists()
            
            # Should complete efficiently
            assert processing_time < 5.0, f"Directory processing took {processing_time:.2f}s (too slow)"
            
            print(f"Directory processing completed in {processing_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
