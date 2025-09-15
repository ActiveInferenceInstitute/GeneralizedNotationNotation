#!/usr/bin/env python3
"""
POMDP Integration Tests

This module tests POMDP-specific functionality across multiple pipeline steps,
ensuring proper integration and data flow between different components.

Test Categories:
- POMDP file discovery and parsing
- POMDP model registry integration
- POMDP validation across steps
- POMDP export and visualization
- POMDP ontology processing
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from type_checker.pomdp_analyzer import POMDPAnalyzer, load_ontology_terms
from type_checker.processor import GNNTypeChecker


class TestPOMDPIntegration:
    """Test POMDP integration across pipeline steps."""
    
    @pytest.fixture
    def sample_pomdp_content(self):
        """Sample POMDP content for testing."""
        return """
# Active Inference POMDP Agent

## Variables
- A[3,3],float: Likelihood matrix
- B[3,3,3],float: Transition matrix  
- C[3],float: Preference vector
- D[3],float: Prior vector
- E[3],float: Habit vector
- s[3,3],float: Hidden state
- o[3,3],int: Observation
- π[3],float: Policy
- u[3],int: Action
- F[π],float: Free energy
- G[π],float: Expected free energy
- t[3],int: Time

## Connections
- A -> o: Likelihood mapping
- B -> s: State transitions
- C -> π: Preference influence
- D -> s: Prior state
- E -> π: Habit influence
- s -> o: State-observation mapping
- π -> u: Policy to action
- F -> π: Free energy minimization
- G -> π: Expected free energy minimization
"""

    @pytest.fixture
    def sample_ontology_terms(self):
        """Sample ontology terms for testing."""
        return {
            "pomdp_components": {
                "likelihood_matrix": {
                    "description": "Maps hidden states to observations",
                    "required": True,
                    "dimensions": [2, 2]
                },
                "transition_matrix": {
                    "description": "State transition probabilities",
                    "required": True,
                    "dimensions": [3, 3, 3]
                },
                "preference_vector": {
                    "description": "Agent preferences over outcomes",
                    "required": True,
                    "dimensions": [1]
                }
            },
            "active_inference_terms": {
                "free_energy": {
                    "description": "Variational free energy",
                    "required": True
                },
                "expected_free_energy": {
                    "description": "Expected variational free energy",
                    "required": True
                }
            }
        }

    def test_pomdp_analyzer_initialization(self, sample_ontology_terms):
        """Test POMDP analyzer initialization with ontology terms."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_ontology_terms, f)
            ontology_file = f.name
        
        try:
            analyzer = POMDPAnalyzer(ontology_file=ontology_file)
            assert analyzer.ontology_terms is not None
            assert "pomdp_components" in analyzer.ontology_terms
            assert "active_inference_terms" in analyzer.ontology_terms
        finally:
            Path(ontology_file).unlink()

    def test_pomdp_structure_analysis(self, sample_pomdp_content, sample_ontology_terms):
        """Test POMDP structure analysis."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_ontology_terms, f)
            ontology_file = f.name
        
        try:
            analyzer = POMDPAnalyzer(ontology_file=ontology_file)
            result = analyzer.analyze_pomdp_structure(sample_pomdp_content)
            
            assert result["valid"] is True
            assert "components_found" in result
            assert "dimensions" in result
            assert "pomdp_specific" in result
            
            # Check that required components are found
            components = result["components_found"]
            assert "likelihood_matrix" in components
            assert "transition_matrix" in components
            assert "preference_vector" in components
            assert "free_energy" in components
            
        finally:
            Path(ontology_file).unlink()

    def test_pomdp_model_validation(self, sample_pomdp_content, sample_ontology_terms):
        """Test POMDP model validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_ontology_terms, f)
            ontology_file = f.name
        
        try:
            analyzer = POMDPAnalyzer(ontology_file=ontology_file)
            result = analyzer.validate_pomdp_model(sample_pomdp_content)
            
            assert result.get("overall_valid", False) is True
            assert "validation_results" in result
            assert "pomdp_specific" in result
            
            # Check POMDP-specific validation
            pomdp_specific = result["pomdp_specific"]
            assert "state_space_size" in pomdp_specific
            assert "observation_space_size" in pomdp_specific
            assert "action_space_size" in pomdp_specific
            
        finally:
            Path(ontology_file).unlink()

    def test_pomdp_complexity_estimation(self, sample_pomdp_content, sample_ontology_terms):
        """Test POMDP complexity estimation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_ontology_terms, f)
            ontology_file = f.name
        
        try:
            analyzer = POMDPAnalyzer(ontology_file=ontology_file)
            result = analyzer.estimate_pomdp_complexity(sample_pomdp_content)
            
            assert "complexity_score" in result
            assert "resource_estimates" in result
            assert "recommendations" in result
            
            # Check that complexity score is reasonable
            assert 0 <= result["complexity_score"] <= 100
            
        finally:
            Path(ontology_file).unlink()

    def test_pomdp_type_checker_integration(self, sample_pomdp_content, sample_ontology_terms):
        """Test POMDP integration with type checker."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_ontology_terms, f)
            ontology_file = f.name
        
        try:
            # Test type checker with POMDP mode
            type_checker = GNNTypeChecker(pomdp_mode=True, ontology_file=ontology_file)
            assert type_checker.pomdp_mode is True
            assert type_checker.pomdp_analyzer is not None
            
            # Test POMDP file validation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(sample_pomdp_content)
                pomdp_file = f.name
            
            try:
                result = type_checker.validate_pomdp_file(Path(pomdp_file))
                assert "pomdp_analysis" in result
                assert result["pomdp_analysis"]["valid"] is True
            finally:
                Path(pomdp_file).unlink()
                
        finally:
            Path(ontology_file).unlink()

    def test_pomdp_ontology_loading(self, sample_ontology_terms):
        """Test POMDP ontology loading functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_ontology_terms, f)
            ontology_file = f.name
        
        try:
            # Test direct ontology loading
            loaded_terms = load_ontology_terms(ontology_file)
            assert loaded_terms == sample_ontology_terms
            
            # Test POMDP analyzer ontology loading
            analyzer = POMDPAnalyzer(ontology_file=ontology_file)
            assert analyzer.ontology_terms == sample_ontology_terms
            
        finally:
            Path(ontology_file).unlink()

    def test_pomdp_error_handling(self):
        """Test POMDP error handling for invalid inputs."""
        analyzer = POMDPAnalyzer()
        
        # Test with invalid content
        invalid_content = "This is not a POMDP model"
        result = analyzer.analyze_pomdp_structure(invalid_content)
        assert result["valid"] is False
        assert "errors" in result
        
        # Test with missing ontology file
        analyzer_no_ontology = POMDPAnalyzer(ontology_file="nonexistent.json")
        result = analyzer_no_ontology.analyze_pomdp_structure("A[1,1],float")
        # Should still work but without ontology validation
        assert "components_found" in result

    def test_pomdp_mcp_integration(self):
        """Test POMDP MCP tool integration."""
        from type_checker.mcp import MCP_TOOLS
        
        # Check that POMDP tools are registered
        pomdp_tools = [tool for tool in MCP_TOOLS if "pomdp" in tool["name"].lower()]
        assert len(pomdp_tools) >= 3  # Should have at least 3 POMDP tools
        
        # Check tool schemas
        for tool in pomdp_tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert "properties" in tool["parameters"]

    def test_pomdp_performance_metrics(self, sample_pomdp_content, sample_ontology_terms):
        """Test POMDP performance metrics collection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_ontology_terms, f)
            ontology_file = f.name
        
        try:
            analyzer = POMDPAnalyzer(ontology_file=ontology_file)
            
            # Test structure analysis performance
            import time
            start_time = time.time()
            result = analyzer.analyze_pomdp_structure(sample_pomdp_content)
            analysis_time = time.time() - start_time
            
            assert analysis_time < 1.0  # Should complete quickly
            assert result["valid"] is True
            
            # Test complexity estimation performance
            start_time = time.time()
            complexity_result = analyzer.estimate_pomdp_complexity(sample_pomdp_content)
            complexity_time = time.time() - start_time
            
            assert complexity_time < 1.0  # Should complete quickly
            assert "complexity_score" in complexity_result
            
        finally:
            Path(ontology_file).unlink()

    def test_pomdp_data_consistency(self, sample_pomdp_content, sample_ontology_terms):
        """Test POMDP data consistency across different analysis methods."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_ontology_terms, f)
            ontology_file = f.name
        
        try:
            analyzer = POMDPAnalyzer(ontology_file=ontology_file)
            
            # Get structure analysis
            structure_result = analyzer.analyze_pomdp_structure(sample_pomdp_content)
            
            # Get model validation
            validation_result = analyzer.validate_pomdp_model(sample_pomdp_content)
            
            # Get complexity estimation
            complexity_result = analyzer.estimate_pomdp_complexity(sample_pomdp_content)
            
            # Check consistency between results
            assert structure_result["valid"] == validation_result.get("overall_valid", False)
            
            # Check that dimensions are consistent
            if "dimensions" in structure_result and "pomdp_specific" in validation_result:
                structure_dims = structure_result["dimensions"]
                validation_dims = validation_result["pomdp_specific"]
                
                # State space size should be consistent
                if "state_space_size" in validation_dims:
                    assert validation_dims["state_space_size"] > 0
                
                # Observation space size should be consistent
                if "observation_space_size" in validation_dims:
                    assert validation_dims["observation_space_size"] > 0
                    
        finally:
            Path(ontology_file).unlink()


class TestPOMDPPipelineIntegration:
    """Test POMDP integration with pipeline steps."""
    
    def test_pomdp_step_2_tests(self):
        """Test POMDP integration with step 2 (tests)."""
        # This test verifies that POMDP tests are discoverable
        from tests.runner import MODULAR_TEST_CATEGORIES
        
        assert "pomdp" in MODULAR_TEST_CATEGORIES
        pomdp_config = MODULAR_TEST_CATEGORIES["pomdp"]
        
        assert pomdp_config["name"] == "POMDP Module Tests"
        assert "pomdp" in pomdp_config["markers"]
        assert "test_type_checker_pomdp.py" in pomdp_config["files"]

    def test_pomdp_step_5_type_checker(self):
        """Test POMDP integration with step 5 (type checker)."""
        # Test that type checker supports POMDP mode
        type_checker = GNNTypeChecker(pomdp_mode=True)
        assert type_checker.pomdp_mode is True
        assert type_checker.pomdp_analyzer is not None

    def test_pomdp_configuration_integration(self):
        """Test POMDP configuration integration."""
        # Test that POMDP configuration is properly loaded
        from pipeline.config import get_pipeline_config
        
        config = get_pipeline_config()
        
        # Check that POMDP settings are available
        if hasattr(config, 'get_step_config'):
            step_config = config.get_step_config("5_type_checker")
            if step_config and "pomdp" in step_config:
                pomdp_config = step_config["pomdp"]
                assert "enabled" in pomdp_config
                assert "ontology_file" in pomdp_config

    def test_pomdp_argument_parsing(self):
        """Test POMDP argument parsing integration."""
        from utils.argument_utils import ArgumentParser
        
        # Check that POMDP arguments are defined for type checker
        if "5_type_checker" in ArgumentParser.STEP_ARGUMENTS:
            args = ArgumentParser.STEP_ARGUMENTS["5_type_checker"]
            assert "pomdp_mode" in args
            assert "ontology_file" in args

    def test_pomdp_mcp_tool_registration(self):
        """Test POMDP MCP tool registration."""
        from type_checker.mcp import MCP_TOOL_REGISTRY
        
        # Check that POMDP tools are registered
        pomdp_tools = [name for name in MCP_TOOL_REGISTRY.keys() if "pomdp" in name.lower()]
        assert len(pomdp_tools) >= 3  # Should have at least 3 POMDP tools
        
        # Check specific tool names
        expected_tools = [
            "validate_pomdp_file",
            "analyze_pomdp_structure", 
            "estimate_pomdp_complexity"
        ]
        
        for tool_name in expected_tools:
            assert tool_name in MCP_TOOL_REGISTRY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
