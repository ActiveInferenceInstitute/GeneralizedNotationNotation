#!/usr/bin/env python3
"""
Comprehensive test suite for the Type Checker module.

This test suite provides complete coverage for the type checker functionality,
including analysis utilities, processor methods, and integration testing.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import logging

# Test data and fixtures
@pytest.fixture
def sample_variables():
    """Sample variables for testing type analysis."""
    return [
        {
            "name": "state",
            "type": "belief",
            "data_type": "float",
            "dimensions": [10, 1],
            "description": "Agent belief state"
        },
        {
            "name": "action",
            "type": "action",
            "data_type": "int",
            "dimensions": [5],
            "description": "Available actions"
        },
        {
            "name": "observation",
            "type": "observation",
            "data_type": "float",
            "dimensions": [3, 3],
            "description": "Sensory observations"
        },
        {
            "name": "reward",
            "type": "reward",
            "data_type": "float",
            "dimensions": [1],
            "description": "Reward signal"
        }
    ]

@pytest.fixture
def sample_connections():
    """Sample connections for testing connection analysis."""
    return [
        {
            "type": "transition",
            "source_variables": ["state", "action"],
            "target_variables": ["state"],
            "description": "State transition function"
        },
        {
            "type": "observation",
            "source_variables": ["state"],
            "target_variables": ["observation"],
            "description": "Observation function"
        },
        {
            "type": "reward",
            "source_variables": ["state", "action"],
            "target_variables": ["reward"],
            "description": "Reward function"
        }
    ]

@pytest.fixture
def sample_gnn_content():
    """Sample GNN content for testing file processing."""
    return """
# Active Inference POMDP Agent

## Variables
state: belief [10, 1]  # Agent belief state
action: action [5]     # Available actions
observation: observation [3, 3]  # Sensory observations
reward: reward [1]     # Reward signal

## Connections
state, action -> state     # State transition
state -> observation      # Observation function
state, action -> reward   # Reward function

## Model Parameters
learning_rate: float = 0.01
discount_factor: float = 0.95
"""

@pytest.fixture
def temp_gnn_file(sample_gnn_content):
    """Create a temporary GNN file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(sample_gnn_content)
        f.flush()
        return Path(f.name)

class TestTypeCheckerAnalysisUtils:
    """Test the analysis utilities module."""
    
    def test_analyze_variable_types_basic(self, sample_variables):
        """Test basic variable type analysis."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        result = analyze_variable_types(sample_variables)
        
        assert isinstance(result, dict)
        assert result["total_variables"] == 4
        assert "belief" in result["type_distribution"]
        assert result["type_distribution"]["belief"] == 1
        assert result["type_distribution"]["action"] == 1
        assert result["type_distribution"]["observation"] == 1
        assert result["type_distribution"]["reward"] == 1
        
        assert result["dimension_analysis"]["max_dimensions"] == 2
        assert result["dimension_analysis"]["avg_dimensions"] == 1.5
        assert "2D" in result["dimension_analysis"]["dimension_distribution"]
        assert "1D" in result["dimension_analysis"]["dimension_distribution"]
        
        assert result["complexity_metrics"]["total_elements"] == 10 + 5 + 9 + 1  # 25
        assert result["complexity_metrics"]["estimated_memory_bytes"] == 25 * 8
        assert result["complexity_metrics"]["estimated_memory_mb"] == (25 * 8) / (1024 * 1024)
    
    def test_analyze_variable_types_empty(self):
        """Test variable type analysis with empty input."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        result = analyze_variable_types([])
        
        assert isinstance(result, dict)
        assert result["total_variables"] == 0
        assert result["type_distribution"] == {}
        assert result["dimension_analysis"]["max_dimensions"] == 0
        assert result["dimension_analysis"]["avg_dimensions"] == 0
        assert result["complexity_metrics"]["total_elements"] == 0
    
    def test_analyze_variable_types_missing_fields(self):
        """Test variable type analysis with missing fields."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        variables = [
            {"name": "var1"},  # Missing type, data_type, dimensions
            {"name": "var2", "type": "belief", "dimensions": [5, 3]},  # Missing data_type
        ]
        
        result = analyze_variable_types(variables)
        
        assert result["total_variables"] == 2
        assert "unknown" in result["type_distribution"]
        assert result["type_distribution"]["unknown"] == 1
        assert result["type_distribution"]["belief"] == 1
        assert result["complexity_metrics"]["total_elements"] == 16  # 1 + 5*3 + 1 (default for invalid)
    
    def test_analyze_connections_basic(self, sample_connections):
        """Test basic connection analysis."""
        from src.type_checker.analysis_utils import analyze_connections
        
        result = analyze_connections(sample_connections)
        
        assert isinstance(result, dict)
        assert result["total_connections"] == 3
        assert "transition" in result["connection_type_distribution"]
        assert "observation" in result["connection_type_distribution"]
        assert "reward" in result["connection_type_distribution"]
        
        assert result["connectivity_metrics"]["avg_connections_per_variable"] > 0
        assert result["connectivity_metrics"]["max_connections_per_variable"] > 0
        assert result["connectivity_metrics"]["isolated_variables"] == 0
    
    def test_analyze_connections_empty(self):
        """Test connection analysis with empty input."""
        from src.type_checker.analysis_utils import analyze_connections
        
        result = analyze_connections([])
        
        assert isinstance(result, dict)
        assert result["total_connections"] == 0
        assert result["connection_type_distribution"] == {}
        assert result["connectivity_metrics"]["avg_connections_per_variable"] == 0
        assert result["connectivity_metrics"]["max_connections_per_variable"] == 0
        assert result["connectivity_metrics"]["isolated_variables"] == 0
    
    def test_analyze_connections_missing_fields(self):
        """Test connection analysis with missing fields."""
        from src.type_checker.analysis_utils import analyze_connections
        
        connections = [
            {"type": "transition"},  # Missing source_variables, target_variables
            {"source_variables": ["state"], "target_variables": ["action"]},  # Missing type
        ]
        
        result = analyze_connections(connections)
        
        assert result["total_connections"] == 2
        assert "unknown" in result["connection_type_distribution"]
        assert result["connection_type_distribution"]["unknown"] == 1
        assert result["connection_type_distribution"]["transition"] == 1
    
    def test_estimate_computational_complexity(self, sample_variables, sample_connections):
        """Test computational complexity estimation."""
        from src.type_checker.analysis_utils import (
            analyze_variable_types,
            analyze_connections,
            estimate_computational_complexity
        )
        
        type_analysis = analyze_variable_types(sample_variables)
        connection_analysis = analyze_connections(sample_connections)
        complexity = estimate_computational_complexity(type_analysis, connection_analysis)
        
        assert isinstance(complexity, dict)
        assert "inference_complexity" in complexity
        assert "learning_complexity" in complexity
        assert "resource_requirements" in complexity
        
        assert complexity["inference_complexity"]["operations_per_step"] > 0
        assert complexity["inference_complexity"]["memory_bandwidth_gb_s"] >= 0
        assert complexity["inference_complexity"]["parallelization_potential"] in ["very_low", "low", "medium", "high", "very_high"]
        
        assert complexity["resource_requirements"]["cpu_cores_recommended"] >= 1
        assert complexity["resource_requirements"]["ram_gb_recommended"] >= 1
        assert complexity["resource_requirements"]["gpu_memory_gb_recommended"] >= 0
    
    def test_estimate_computational_complexity_high_complexity(self):
        """Test complexity estimation for high-complexity models."""
        from src.type_checker.analysis_utils import (
            analyze_variable_types,
            analyze_connections,
            estimate_computational_complexity
        )
        
        # High complexity variables
        variables = [
            {"name": f"var_{i}", "type": "belief", "data_type": "float", "dimensions": [100, 100]}
            for i in range(10)
        ]
        
        # High complexity connections
        connections = [
            {"type": "transition", "source_variables": [f"var_{i}" for i in range(5)], 
             "target_variables": [f"var_{i}" for i in range(5, 10)]}
            for _ in range(5)
        ]
        
        type_analysis = analyze_variable_types(variables)
        connection_analysis = analyze_connections(connections)
        complexity = estimate_computational_complexity(type_analysis, connection_analysis)
        
        assert complexity["inference_complexity"]["parallelization_potential"] in ["medium", "high", "very_high"]
        assert complexity["resource_requirements"]["cpu_cores_recommended"] >= 2
        assert complexity["resource_requirements"]["ram_gb_recommended"] >= 1
        assert complexity["resource_requirements"]["gpu_memory_gb_recommended"] >= 0


class TestTypeCheckerProcessor:
    """Test the type checker processor module."""
    
    def test_gnn_type_checker_initialization(self):
        """Test GNNTypeChecker initialization."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker()
        assert checker is not None
        assert hasattr(checker, 'validation_rules')
        assert isinstance(checker.validation_rules, dict)
        assert "valid_types" in checker.validation_rules
        assert "type_patterns" in checker.validation_rules
    
    def test_validate_single_gnn_file(self, temp_gnn_file):
        """Test single GNN file validation."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker()
        result = checker.validate_single_gnn_file(temp_gnn_file, verbose=True)
        
        assert isinstance(result, dict)
        assert "file_path" in result
        assert "file_name" in result
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "type_issues" in result
        assert "validation_timestamp" in result
        
        assert result["file_path"] == str(temp_gnn_file)
        assert result["file_name"] == temp_gnn_file.name
        assert isinstance(result["valid"], bool)
        assert isinstance(result["errors"], list)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["type_issues"], list)
    
    def test_validate_single_gnn_file_nonexistent(self):
        """Test validation of non-existent file."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker()
        nonexistent_file = Path("/nonexistent/file.md")
        result = checker.validate_single_gnn_file(nonexistent_file)
        
        assert isinstance(result, dict)
        assert result["valid"] == False
        assert len(result["errors"]) > 0
        assert "No such file or directory" in str(result["errors"][0])
    
    def test_validate_gnn_files_basic(self, temp_gnn_file):
        """Test basic GNN files validation."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"
            
            # Copy test file to temp directory
            test_file = temp_path / "test_gnn.md"
            test_file.write_text(temp_gnn_file.read_text())
            
            result = checker.validate_gnn_files(temp_path, output_dir, verbose=True)
            
            assert isinstance(result, bool)
            assert output_dir.exists()
            assert (output_dir / "type_check_results").exists()
            assert (output_dir / "type_check_results" / "type_check_results.json").exists()
            assert (output_dir / "type_check_results" / "type_check_summary.md").exists()
    
    def test_validate_gnn_files_no_files(self):
        """Test validation with no GNN files."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"
            
            result = checker.validate_gnn_files(temp_path, output_dir)
            
            assert result == False  # Should fail when no files found
            assert output_dir.exists()
            assert (output_dir / "type_check_results").exists()
    
    def test_validation_rules(self):
        """Test validation rules configuration."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker()
        rules = checker._get_validation_rules()
        
        assert isinstance(rules, dict)
        assert "valid_types" in rules
        assert "type_patterns" in rules
        
        assert isinstance(rules["valid_types"], list)
        assert len(rules["valid_types"]) > 0
        assert "int" in rules["valid_types"]
        assert "float" in rules["valid_types"]
        assert "string" in rules["valid_types"]
        
        assert isinstance(rules["type_patterns"], dict)
        assert "numeric" in rules["type_patterns"]
        assert "identifier" in rules["type_patterns"]
        assert "array" in rules["type_patterns"]
    
    def test_validate_type(self):
        """Test individual type validation."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker()
        
        # Valid type
        valid_type = {"name": "state", "type": "belief"}
        result = checker._validate_type(valid_type)
        assert result["valid"] == True
        assert result["message"] == ""
        
        # Invalid type
        invalid_type = {"name": "state", "type": "invalid_type"}
        result = checker._validate_type(invalid_type)
        assert result["valid"] == False
        assert "Unknown type" in result["message"]
        
        # Invalid variable name
        invalid_name = {"name": "123invalid", "type": "belief"}
        result = checker._validate_type(invalid_name)
        assert result["valid"] == False
        assert "Invalid variable name" in result["message"]
    
    def test_check_type_consistency(self):
        """Test type consistency checking."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker()
        
        # Consistent types
        consistent_types = [
            {"name": "state", "type": "belief"},
            {"name": "action", "type": "action"}
        ]
        result = checker._check_type_consistency(consistent_types)
        assert result["consistent"] == True
        assert result["message"] == ""
        
        # Duplicate variable names
        duplicate_types = [
            {"name": "state", "type": "belief"},
            {"name": "state", "type": "action"}
        ]
        result = checker._check_type_consistency(duplicate_types)
        assert result["consistent"] == False
        assert "Duplicate variable names" in result["message"]
        assert "state" in result["message"]
    
    def test_analyze_types(self, temp_gnn_file):
        """Test type analysis functionality."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker()
        result = checker._analyze_types(temp_gnn_file, verbose=True)
        
        assert isinstance(result, dict)
        assert "file_path" in result
        assert "file_name" in result
        assert "types_found" in result
        assert "type_distribution" in result
        assert "total_variables" in result
        assert "analysis_timestamp" in result
        
        assert result["file_path"] == str(temp_gnn_file)
        assert result["file_name"] == temp_gnn_file.name
        assert isinstance(result["types_found"], list)
        assert isinstance(result["type_distribution"], dict)
        assert isinstance(result["total_variables"], int)
    
    def test_generate_type_check_summary(self):
        """Test type check summary generation."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker()
        
        # Test with sample results
        results = {
            "processed_files": 2,
            "success": True,
            "errors": [],
            "validation_results": [
                {"valid": True},
                {"valid": False}
            ],
            "type_analysis": [
                {"total_variables": 5},
                {"total_variables": 3}
            ]
        }
        
        summary = checker._generate_type_check_summary(results)
        
        assert isinstance(summary, str)
        assert "Type Check Summary" in summary
        assert "Files Processed" in summary
        assert "Success" in summary
        assert "Files Validated" in summary
        assert "Valid Files" in summary
        assert "Invalid Files" in summary
        assert "Total Variables" in summary


class TestTypeCheckerIntegration:
    """Test integration between type checker components."""
    
    def test_end_to_end_processing(self, temp_gnn_file):
        """Test end-to-end type checking process."""
        from src.type_checker.processor import GNNTypeChecker
        from src.type_checker.analysis_utils import (
            analyze_variable_types,
            analyze_connections,
            estimate_computational_complexity
        )
        
        checker = GNNTypeChecker()
        
        # Validate file
        validation_result = checker.validate_single_gnn_file(temp_gnn_file)
        assert isinstance(validation_result, dict)
        
        # Analyze types
        type_analysis = checker._analyze_types(temp_gnn_file)
        assert isinstance(type_analysis, dict)
        
        # Test with sample data
        sample_variables = [
            {"name": "state", "type": "belief", "data_type": "float", "dimensions": [10, 1]},
            {"name": "action", "type": "action", "data_type": "int", "dimensions": [5]}
        ]
        
        sample_connections = [
            {"type": "transition", "source_variables": ["state", "action"], "target_variables": ["state"]}
        ]
        
        # Test analysis utilities
        var_analysis = analyze_variable_types(sample_variables)
        conn_analysis = analyze_connections(sample_connections)
        complexity = estimate_computational_complexity(var_analysis, conn_analysis)
        
        assert isinstance(var_analysis, dict)
        assert isinstance(conn_analysis, dict)
        assert isinstance(complexity, dict)
    
    def test_error_handling_robustness(self):
        """Test error handling robustness."""
        from src.type_checker.processor import GNNTypeChecker
        from src.type_checker.analysis_utils import analyze_variable_types
        
        checker = GNNTypeChecker()
        
        # Test with malformed data
        malformed_variables = [
            None,  # None value
            {"name": "var1"},  # Missing required fields
            {"invalid": "data"},  # Wrong structure
            [],  # Empty list
        ]
        
        # Should not crash
        result = analyze_variable_types(malformed_variables)
        assert isinstance(result, dict)
        assert result["total_variables"] == 2  # Should handle None and invalid data gracefully, keeping valid ones
        
        # Test with non-existent file
        nonexistent_file = Path("/nonexistent/file.md")
        result = checker._analyze_types(nonexistent_file)
        assert isinstance(result, dict)
        assert "error" in result


class TestTypeCheckerPerformance:
    """Test performance characteristics of type checker."""
    
    def test_large_file_processing(self):
        """Test processing of large GNN files."""
        from src.type_checker.processor import GNNTypeChecker
        from src.type_checker.analysis_utils import analyze_variable_types
        
        # Create large variable set
        large_variables = [
            {
                "name": f"var_{i}",
                "type": "belief",
                "data_type": "float",
                "dimensions": [10, 10]
            }
            for i in range(1000)
        ]
        
        # Should complete without timeout
        result = analyze_variable_types(large_variables)
        assert isinstance(result, dict)
        assert result["total_variables"] == 1000
        assert result["complexity_metrics"]["total_elements"] == 1000 * 100  # 100,000 elements
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        from src.type_checker.analysis_utils import analyze_variable_types
        
        # Test with very large dimensions
        large_variables = [
            {
                "name": "large_var",
                "type": "belief",
                "data_type": "float",
                "dimensions": [1000, 1000, 1000]  # 1 billion elements
            }
        ]
        
        result = analyze_variable_types(large_variables)
        assert isinstance(result, dict)
        assert result["total_variables"] == 1
        assert result["complexity_metrics"]["total_elements"] == 1000**3
        assert result["complexity_metrics"]["estimated_memory_mb"] > 0


@pytest.mark.integration
class TestTypeCheckerPipelineIntegration:
    """Test integration with pipeline step 5."""
    
    def test_step5_integration(self, temp_gnn_file):
        """Test integration with step 5 orchestrator."""
        # This would test the actual step 5 script integration
        # For now, we test the key functions it uses
        from src.type_checker.analysis_utils import (
            analyze_variable_types,
            analyze_connections,
            estimate_computational_complexity
        )
        
        # Test with realistic data that step 5 would process
        variables = [
            {"name": "state", "type": "belief", "data_type": "float", "dimensions": [10, 1]},
            {"name": "action", "type": "action", "data_type": "int", "dimensions": [5]},
        ]
        
        connections = [
            {"type": "transition", "source_variables": ["state", "action"], "target_variables": ["state"]}
        ]
        
        var_analysis = analyze_variable_types(variables)
        conn_analysis = analyze_connections(connections)
        complexity = estimate_computational_complexity(var_analysis, conn_analysis)
        
        assert isinstance(var_analysis, dict)
        assert isinstance(conn_analysis, dict)
        assert isinstance(complexity, dict)
        
        # Verify expected structure for pipeline integration
        assert "total_variables" in var_analysis
        assert "total_connections" in conn_analysis
        assert "inference_complexity" in complexity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
