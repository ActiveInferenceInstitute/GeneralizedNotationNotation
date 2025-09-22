#!/usr/bin/env python3
"""
POMDP-specific tests for the Type Checker module.

This test suite focuses on POMDP-specific functionality, including
Active Inference model validation, ontology compliance checking,
and POMDP-specific analysis capabilities.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List

# Sample POMDP GNN content for testing
SAMPLE_POMDP_CONTENT = """
# GNN Example: Active Inference POMDP Agent
# GNN Version: 1.0

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Classic Active Inference POMDP Agent v1

## StateSpaceBlock
# Likelihood matrix: A[observation_outcomes, hidden_states]
A[3,3],float   # Likelihood mapping hidden states to observations

# Transition matrix: B[states_next, states_previous, actions]
B[3,3,3],float   # State transitions given previous state and action

# Preference vector: C[observation_outcomes]
C[3],float       # Log-preferences over observations

# Prior vector: D[states]
D[3],float       # Prior over initial hidden states

# Habit vector: E[actions]
E[3],float       # Initial policy prior (habit) over actions

# Hidden State
s[3,1],float     # Current hidden state distribution
s_prime[3,1],float # Next hidden state distribution
F[π],float       # Variational Free Energy for belief updating from observations

# Observation
o[3,1],int     # Current observation (integer index)

# Policy and Control
π[3],float       # Policy (distribution over actions), no planning
u[1],int         # Action taken
G[π],float       # Expected Free Energy (per policy)

# Time
t[1],int         # Discrete time step

## Connections
D>s
s-A
s>s_prime
A-o
s-B
C>G
E>π
G>π
π>u
B>u
u>s_prime

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
E=Habit
F=VariationalFreeEnergy
G=ExpectedFreeEnergy
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector
u=Action
t=Time

## ModelParameters
num_hidden_states: 3
num_obs: 3
num_actions: 3
"""

INVALID_POMDP_CONTENT = """
# Invalid POMDP - missing required components
A[3,3],float
# Missing B, C, D, E matrices
"""

@pytest.fixture
def sample_pomdp_file():
    """Create a temporary POMDP GNN file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(SAMPLE_POMDP_CONTENT)
        f.flush()
        return Path(f.name)

@pytest.fixture
def invalid_pomdp_file():
    """Create a temporary invalid POMDP GNN file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(INVALID_POMDP_CONTENT)
        f.flush()
        return Path(f.name)

@pytest.fixture
def sample_ontology_file():
    """Create a temporary ontology file for testing."""
    ontology_data = {
        "state_space": "The set of all possible states of a system",
        "observation_space": "The set of all possible observations",
        "action_space": "The set of all possible actions",
        "likelihood_matrix": "Matrix A mapping hidden states to observations",
        "transition_matrix": "Matrix B describing state transitions given actions"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ontology_data, f)
        f.flush()
        return Path(f.name)

class TestPOMDPAnalyzer:
    """Test the POMDP analyzer functionality."""
    
    def test_pomdp_analyzer_initialization(self):
        """Test POMDP analyzer initialization."""
        from src.type_checker.pomdp_analyzer import POMDPAnalyzer
        
        analyzer = POMDPAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'ontology_terms')
        assert hasattr(analyzer, 'logger')
    
    def test_pomdp_analyzer_with_ontology(self, sample_ontology_file):
        """Test POMDP analyzer with custom ontology."""
        from src.type_checker.pomdp_analyzer import POMDPAnalyzer

        analyzer = POMDPAnalyzer(ontology_file=sample_ontology_file)

        assert "state_space" in analyzer.ontology_terms
    
    def test_analyze_pomdp_structure(self):
        """Test POMDP structure analysis."""
        from src.type_checker.pomdp_analyzer import POMDPAnalyzer
        
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(SAMPLE_POMDP_CONTENT)
        
        assert isinstance(result, dict)
        assert result["model_type"] == "POMDP"
        assert "components_found" in result
        assert "dimensions" in result
        assert "validation_results" in result
        assert "pomdp_specific" in result
        
        # Check that required components are found
        components = result["components_found"]
        assert "likelihood_matrix" in components
        assert "transition_matrix" in components
        assert "preference_vector" in components
        assert "prior_vector" in components
        assert "habit_vector" in components
        assert "hidden_state" in components
        assert "observation" in components
        assert "policy" in components
        assert "action" in components
        assert "free_energy" in components
        assert "expected_free_energy" in components
        assert "time" in components
    
    def test_analyze_pomdp_structure_invalid(self):
        """Test POMDP structure analysis with invalid content."""
        from src.type_checker.pomdp_analyzer import POMDPAnalyzer
        
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(INVALID_POMDP_CONTENT)
        
        assert isinstance(result, dict)
        assert result["model_type"] == "POMDP"
        assert not result["validation_results"]["structure_valid"]
        assert len(result["validation_results"]["missing_components"]) > 0
    
    def test_validate_pomdp_model(self, sample_pomdp_file):
        """Test complete POMDP model validation."""
        from src.type_checker.pomdp_analyzer import POMDPAnalyzer
        
        analyzer = POMDPAnalyzer()
        result = analyzer.validate_pomdp_model(sample_pomdp_file)
        
        assert isinstance(result, dict)
        assert "file_path" in result
        assert "file_name" in result
        assert "validation_results" in result
        assert "pomdp_specific" in result
        
        # Check validation results
        assert result["validation_results"]["structure_valid"]
        assert result["validation_results"]["dimension_consistency"]
        assert result["validation_results"]["ontology_compliance"]
        assert result["overall_valid"]
    
    def test_validate_pomdp_model_invalid(self, invalid_pomdp_file):
        """Test POMDP model validation with invalid file."""
        from src.type_checker.pomdp_analyzer import POMDPAnalyzer
        
        analyzer = POMDPAnalyzer()
        result = analyzer.validate_pomdp_model(invalid_pomdp_file)
        
        assert isinstance(result, dict)
        assert not result["overall_valid"]
        assert not result["validation_results"]["structure_valid"]
    
    def test_estimate_pomdp_complexity(self):
        """Test POMDP complexity estimation."""
        from src.type_checker.pomdp_analyzer import POMDPAnalyzer
        
        analyzer = POMDPAnalyzer()
        analysis = analyzer.analyze_pomdp_structure(SAMPLE_POMDP_CONTENT)
        complexity = analyzer.estimate_pomdp_complexity(analysis)
        
        assert isinstance(complexity, dict)
        assert "inference_complexity" in complexity
        assert "policy_complexity" in complexity
        assert "memory_requirements" in complexity
        assert "scalability" in complexity
        
        # Check that complexity values are calculated
        assert complexity["inference_complexity"]["total_inference_ops"] > 0
        assert complexity["policy_complexity"]["total_policy_ops"] > 0
        assert complexity["memory_requirements"]["total_memory_bytes"] > 0


class TestPOMDPTypeChecker:
    """Test POMDP-specific type checker functionality."""
    
    def test_gnn_type_checker_pomdp_mode(self, sample_ontology_file):
        """Test GNNTypeChecker in POMDP mode."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker(pomdp_mode=True, ontology_file=sample_ontology_file)
        assert checker.pomdp_mode
        assert checker.pomdp_analyzer is not None
        assert checker.ontology_file == sample_ontology_file
    
    def test_gnn_type_checker_pomdp_mode_no_ontology(self):
        """Test GNNTypeChecker in POMDP mode without ontology file."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker(pomdp_mode=True, ontology_file=None)
        assert checker.pomdp_mode
        assert checker.pomdp_analyzer is not None
        assert checker.ontology_file is None
    
    def test_validate_pomdp_file(self, sample_pomdp_file, sample_ontology_file):
        """Test POMDP file validation."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker(pomdp_mode=True, ontology_file=sample_ontology_file)
        result = checker.validate_pomdp_file(sample_pomdp_file)
        
        assert isinstance(result, dict)
        assert "pomdp_analysis" in result
        assert "validation_results" in result["pomdp_analysis"]
        assert "pomdp_specific" in result["pomdp_analysis"]

        # Check that complexity estimation is included for valid models
        if result["pomdp_analysis"].get("validation_results", {}).get("overall_valid", False):
            assert "complexity_estimation" in result["pomdp_analysis"]
    
    def test_validate_pomdp_file_not_pomdp_mode(self, sample_pomdp_file):
        """Test POMDP file validation when not in POMDP mode."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker(pomdp_mode=False)
        result = checker.validate_pomdp_file(sample_pomdp_file)
        
        assert isinstance(result, dict)
        assert "error" in result
        assert "POMDP mode not enabled" in result["error"]
        assert not result["valid"]
    
    def test_get_pomdp_analysis_summary(self, sample_ontology_file):
        """Test POMDP analysis summary generation."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker(pomdp_mode=True, ontology_file=sample_ontology_file)
        
        # Create mock results
        results = {
            "pomdp_analysis": [
                {
                    "file_name": "test_model.md",
                    "pomdp_specific": {
                        "state_space_size": 3,
                        "observation_space_size": 3,
                        "action_space_size": 3,
                        "model_complexity": "low"
                    },
                    "validation_results": {
                        "overall_valid": True,
                        "structure_valid": True,
                        "dimension_consistency": True,
                        "ontology_compliance": True
                    },
                    "errors": [],
                    "warnings": []
                }
            ],
            "summary_statistics": {
                "pomdp_metrics": {
                    "total_state_space_size": 3,
                    "total_observation_space_size": 3,
                    "total_action_space_size": 3,
                    "pomdp_models_valid": 1
                }
            }
        }
        
        summary = checker.get_pomdp_analysis_summary(results)
        
        assert isinstance(summary, str)
        assert "POMDP Analysis Summary" in summary
        assert "test_model.md" in summary
        assert "**State Space**: 3" in summary
        assert "**Observation Space**: 3" in summary
        assert "**Action Space**: 3" in summary
    
    def test_get_pomdp_analysis_summary_no_pomdp_mode(self):
        """Test POMDP analysis summary when not in POMDP mode."""
        from src.type_checker.processor import GNNTypeChecker
        
        checker = GNNTypeChecker(pomdp_mode=False)
        results = {"some": "data"}
        
        summary = checker.get_pomdp_analysis_summary(results)
        
        assert summary == "POMDP analysis not available"


class TestPOMDPIntegration:
    """Test POMDP integration with the main type checker."""
    
    def test_pomdp_analysis_in_main_workflow(self, sample_pomdp_file, sample_ontology_file):
        """Test POMDP analysis integrated into main type checker workflow."""
        from src.type_checker.processor import GNNTypeChecker
        
        # Test with POMDP mode enabled
        checker = GNNTypeChecker(pomdp_mode=True, ontology_file=sample_ontology_file, verbose=True)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"
            
            # Copy test file to temp directory
            test_file = temp_path / "test_pomdp.md"
            test_file.write_text(sample_pomdp_file.read_text())
            
            # Validate files
            success = checker.validate_gnn_files(temp_path, output_dir, verbose=True)
            
            assert isinstance(success, bool)
            assert output_dir.exists()
            assert (output_dir / "type_check_results").exists()
            
            # Check that POMDP analysis was performed
            results_file = output_dir / "type_check_results" / "type_check_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                assert "pomdp_analysis" in results
                assert len(results["pomdp_analysis"]) > 0
    
    def test_pomdp_ontology_loading(self, sample_ontology_file):
        """Test ontology loading functionality."""
        from src.type_checker.pomdp_analyzer import load_ontology_terms
        
        ontology_terms = load_ontology_terms(sample_ontology_file)
        
        assert isinstance(ontology_terms, dict)
        assert "state_space" in ontology_terms
        assert "observation_space" in ontology_terms
        assert "action_space" in ontology_terms
        assert "likelihood_matrix" in ontology_terms
        assert "transition_matrix" in ontology_terms
    
    def test_pomdp_ontology_loading_nonexistent(self):
        """Test ontology loading with nonexistent file."""
        from src.type_checker.pomdp_analyzer import load_ontology_terms
        
        nonexistent_file = Path("/nonexistent/ontology.json")
        ontology_terms = load_ontology_terms(nonexistent_file)

        # Should return empty dict when file doesn't exist
        assert isinstance(ontology_terms, dict)
        assert len(ontology_terms) == 0


class TestPOMDPValidationRules:
    """Test POMDP-specific validation rules."""
    
    def test_required_components_validation(self):
        """Test validation of required POMDP components."""
        from src.type_checker.pomdp_analyzer import POMDPAnalyzer
        
        analyzer = POMDPAnalyzer()
        
        # Test with complete POMDP
        complete_result = analyzer.analyze_pomdp_structure(SAMPLE_POMDP_CONTENT)
        assert complete_result["validation_results"]["structure_valid"]
        assert len(complete_result["validation_results"]["missing_components"]) == 0
        
        # Test with incomplete POMDP
        incomplete_result = analyzer.analyze_pomdp_structure(INVALID_POMDP_CONTENT)
        assert not incomplete_result["validation_results"]["structure_valid"]
        assert len(incomplete_result["validation_results"]["missing_components"]) > 0
    
    def test_dimension_consistency_validation(self):
        """Test dimension consistency validation."""
        from src.type_checker.pomdp_analyzer import POMDPAnalyzer
        
        analyzer = POMDPAnalyzer()
        
        # Test with consistent dimensions
        result = analyzer.analyze_pomdp_structure(SAMPLE_POMDP_CONTENT)
        assert result["validation_results"]["dimension_consistency"]
        
        # Test with inconsistent dimensions
        inconsistent_content = """
        A[3,3],float   # 3x3 likelihood matrix
        B[2,2,2],float # 2x2x2 transition matrix (inconsistent with A)
        C[3],float     # 3-element preference vector
        """
        
        inconsistent_result = analyzer.analyze_pomdp_structure(inconsistent_content)
        # Should detect inconsistency
        assert not inconsistent_result["validation_results"]["dimension_consistency"]
    
    def test_ontology_compliance_validation(self):
        """Test ontology compliance validation."""
        from src.type_checker.pomdp_analyzer import POMDPAnalyzer
        
        analyzer = POMDPAnalyzer()
        
        # Test with compliant ontology
        result = analyzer.analyze_pomdp_structure(SAMPLE_POMDP_CONTENT)
        assert result["validation_results"]["ontology_compliance"]
        
        # Test without ontology section
        no_ontology_content = """
        A[3,3,type=float]
        B[3,3,3,type=float]
        C[3,type=float]
        D[3,type=float]
        E[3,type=float]
        s[3,1,type=float]
        o[3,1,type=int]
        π[3,type=float]
        u[1,type=int]
        F[π,type=float]
        G[π,type=float]
        t[1,type=int]
        """
        
        no_ontology_result = analyzer.analyze_pomdp_structure(no_ontology_content)
        # Content without ontology annotation should still be valid for basic POMDP structure
        assert no_ontology_result["validation_results"]["ontology_compliance"]


class TestPOMDPPerformance:
    """Test POMDP-specific performance characteristics."""
    
    def test_large_pomdp_analysis(self):
        """Test analysis of large POMDP models."""
        from src.type_checker.pomdp_analyzer import POMDPAnalyzer
        
        # Create a larger POMDP model
        large_pomdp_content = """
        A[10,10],float   # 10x10 likelihood matrix
        B[10,10,10],float # 10x10x10 transition matrix
        C[10],float       # 10-element preference vector
        D[10],float       # 10-element prior vector
        E[10],float       # 10-element habit vector
        s[10,1],float     # 10-element hidden state
        o[10,1],int       # 10-element observation
        π[10],float       # 10-element policy
        u[1],int          # Action
        F[π],float        # Free energy
        G[π],float        # Expected free energy
        t[1],int          # Time
        """
        
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(large_pomdp_content)
        
        assert isinstance(result, dict)
        assert result["validation_results"]["structure_valid"]
        
        # Check that complexity is calculated
        pomdp_specific = result["pomdp_specific"]
        assert pomdp_specific["state_space_size"] == 10
        assert pomdp_specific["observation_space_size"] == 10
        assert pomdp_specific["action_space_size"] == 10
        assert pomdp_specific["model_complexity"] in ["low", "medium", "high"]
    
    def test_pomdp_complexity_estimation(self):
        """Test POMDP complexity estimation."""
        from src.type_checker.pomdp_analyzer import POMDPAnalyzer
        
        analyzer = POMDPAnalyzer()
        analysis = analyzer.analyze_pomdp_structure(SAMPLE_POMDP_CONTENT)
        complexity = analyzer.estimate_pomdp_complexity(analysis)
        
        # Verify complexity structure
        assert "inference_complexity" in complexity
        assert "policy_complexity" in complexity
        assert "memory_requirements" in complexity
        assert "scalability" in complexity
        
        # Verify that operations are calculated
        assert complexity["inference_complexity"]["total_inference_ops"] > 0
        assert complexity["policy_complexity"]["total_policy_ops"] > 0
        assert complexity["memory_requirements"]["total_memory_bytes"] > 0
        
        # Verify scaling characteristics
        assert complexity["scalability"]["overall_scaling"] in ["linear", "polynomial", "exponential"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
