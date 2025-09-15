#!/usr/bin/env python3
"""
POMDP Validation Tests

This module tests POMDP-specific validation rules and consistency checking
across different pipeline steps.

Test Categories:
- POMDP structure validation
- POMDP dimension consistency
- POMDP ontology compliance
- POMDP semantic validation
- POMDP error handling and recovery
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


class TestPOMDPStructureValidation:
    """Test POMDP structure validation rules."""
    
    @pytest.fixture
    def valid_pomdp_content(self):
        """Valid POMDP content for testing."""
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
    def invalid_pomdp_content(self):
        """Invalid POMDP content for testing."""
        return """
# Invalid POMDP Model

## Variables
- A[3,3],float: Some matrix
- B[2,2],float: Another matrix
- C[1],float: Vector

## Connections
- A -> B: Some connection
"""

    def test_valid_pomdp_structure(self, valid_pomdp_content):
        """Test validation of valid POMDP structure."""
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(valid_pomdp_content)
        
        assert result["valid"] is True
        assert "components_found" in result
        assert "dimensions" in result
        assert "pomdp_specific" in result
        
        # Check required components
        components = result["components_found"]
        required_components = [
            "likelihood_matrix", "transition_matrix", "preference_vector",
            "prior_vector", "habit_vector", "hidden_state", "observation",
            "policy", "action", "free_energy", "expected_free_energy"
        ]
        
        for component in required_components:
            assert component in components, f"Missing required component: {component}"

    def test_invalid_pomdp_structure(self, invalid_pomdp_content):
        """Test validation of invalid POMDP structure."""
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(invalid_pomdp_content)
        
        assert result["valid"] is False
        assert "errors" in result or "validation_results" in result

    def test_missing_required_components(self):
        """Test validation when required components are missing."""
        incomplete_content = """
# Incomplete POMDP Model

## Variables
- A[3,3],float: Likelihood matrix
- B[3,3,3],float: Transition matrix

## Connections
- A -> B: Some connection
"""
        
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(incomplete_content)
        
        assert result["valid"] is False
        # Should identify missing components
        if "missing_components" in result:
            missing = result["missing_components"]
            assert len(missing) > 0

    def test_invalid_dimension_syntax(self):
        """Test validation with invalid dimension syntax."""
        invalid_dim_content = """
# POMDP with Invalid Dimensions

## Variables
- A[3,3,3,3],float: Too many dimensions
- B[],float: No dimensions
- C[abc],float: Non-numeric dimensions
- D[3,3],string: Wrong type

## Connections
- A -> B: Some connection
"""
        
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(invalid_dim_content)
        
        # Should handle invalid syntax gracefully
        assert "components_found" in result
        # May or may not be valid depending on error handling

    def test_pomdp_component_detection(self, valid_pomdp_content):
        """Test detection of POMDP-specific components."""
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(valid_pomdp_content)
        
        components = result["components_found"]
        
        # Test specific POMDP components
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

    def test_pomdp_dimension_extraction(self, valid_pomdp_content):
        """Test extraction of POMDP dimensions."""
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(valid_pomdp_content)
        
        dimensions = result["dimensions"]
        
        # Check that dimensions are properly extracted
        assert "likelihood_matrix" in dimensions
        assert dimensions["likelihood_matrix"] == [3, 3]
        
        assert "transition_matrix" in dimensions
        assert dimensions["transition_matrix"] == [3, 3, 3]
        
        assert "preference_vector" in dimensions
        assert dimensions["preference_vector"] == [3]

    def test_pomdp_specific_metrics(self, valid_pomdp_content):
        """Test POMDP-specific metrics calculation."""
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(valid_pomdp_content)
        
        pomdp_specific = result["pomdp_specific"]
        
        # Check that POMDP-specific metrics are calculated
        assert "state_space_size" in pomdp_specific
        assert "observation_space_size" in pomdp_specific
        assert "action_space_size" in pomdp_specific
        
        # Check that sizes are reasonable
        assert pomdp_specific["state_space_size"] > 0
        assert pomdp_specific["observation_space_size"] > 0
        assert pomdp_specific["action_space_size"] > 0


class TestPOMDPDimensionConsistency:
    """Test POMDP dimension consistency validation."""
    
    def test_consistent_dimensions(self):
        """Test validation with consistent dimensions."""
        consistent_content = """
# POMDP with Consistent Dimensions

## Variables
- A[3,3],float: Likelihood matrix (3 states, 3 observations)
- B[3,3,3],float: Transition matrix (3 states, 3 states, 3 actions)
- C[3],float: Preference vector (3 outcomes)
- D[3],float: Prior vector (3 states)
- E[3],float: Habit vector (3 actions)
- s[3,3],float: Hidden state (3 states, 3 time steps)
- o[3,3],int: Observation (3 observations, 3 time steps)
- π[3],float: Policy (3 actions)
- u[3],int: Action (3 actions)
- F[π],float: Free energy
- G[π],float: Expected free energy
- t[3],int: Time (3 time steps)

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
        
        analyzer = POMDPAnalyzer()
        result = analyzer.validate_pomdp_model(consistent_content)
        
        assert result.get("overall_valid", False) is True
        assert "validation_results" in result
        
        validation_results = result["validation_results"]
        assert "dimension_consistency" in validation_results
        assert validation_results["dimension_consistency"] is True

    def test_inconsistent_dimensions(self):
        """Test validation with inconsistent dimensions."""
        inconsistent_content = """
# POMDP with Inconsistent Dimensions

## Variables
- A[3,3],float: Likelihood matrix (3 states, 3 observations)
- B[2,2,2],float: Transition matrix (2 states, 2 states, 2 actions) - INCONSISTENT
- C[3],float: Preference vector (3 outcomes)
- D[2],float: Prior vector (2 states) - INCONSISTENT
- E[3],float: Habit vector (3 actions)
- s[3,3],float: Hidden state (3 states, 3 time steps)
- o[2,2],int: Observation (2 observations, 2 time steps) - INCONSISTENT
- π[3],float: Policy (3 actions)
- u[2],int: Action (2 actions) - INCONSISTENT
- F[π],float: Free energy
- G[π],float: Expected free energy
- t[3],int: Time (3 time steps)

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
        
        analyzer = POMDPAnalyzer()
        result = analyzer.validate_pomdp_model(inconsistent_content)
        
        # Should detect dimension inconsistencies
        assert "validation_results" in result
        validation_results = result["validation_results"]
        
        if "dimension_consistency" in validation_results:
            assert validation_results["dimension_consistency"] is False

    def test_missing_dimension_validation(self):
        """Test validation when dimension information is missing."""
        missing_dim_content = """
# POMDP with Missing Dimensions

## Variables
- A,float: Likelihood matrix (no dimensions)
- B,float: Transition matrix (no dimensions)
- C,float: Preference vector (no dimensions)

## Connections
- A -> B: Some connection
"""
        
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(missing_dim_content)
        
        # Should handle missing dimensions gracefully
        assert "components_found" in result
        # May or may not be valid depending on error handling


class TestPOMDPOntologyCompliance:
    """Test POMDP ontology compliance validation."""
    
    @pytest.fixture
    def sample_ontology_terms(self):
        """Sample ontology terms for testing."""
        return {
            "pomdp_components": {
                "likelihood_matrix": {
                    "description": "Maps hidden states to observations",
                    "required": True,
                    "dimensions": [2, 2],
                    "type": "float"
                },
                "transition_matrix": {
                    "description": "State transition probabilities",
                    "required": True,
                    "dimensions": [3, 3, 3],
                    "type": "float"
                },
                "preference_vector": {
                    "description": "Agent preferences over outcomes",
                    "required": True,
                    "dimensions": [1],
                    "type": "float"
                }
            },
            "active_inference_terms": {
                "free_energy": {
                    "description": "Variational free energy",
                    "required": True,
                    "type": "float"
                },
                "expected_free_energy": {
                    "description": "Expected variational free energy",
                    "required": True,
                    "type": "float"
                }
            }
        }

    def test_ontology_compliant_pomdp(self, sample_ontology_terms):
        """Test validation of ontology-compliant POMDP."""
        compliant_content = """
# Ontology-Compliant POMDP

## Variables
- A[2,2],float: Likelihood matrix
- B[2,2,2],float: Transition matrix
- C[2],float: Preference vector
- D[2],float: Prior vector
- E[2],float: Habit vector
- s[2,2],float: Hidden state
- o[2,2],int: Observation
- π[2],float: Policy
- u[2],int: Action
- F[π],float: Free energy
- G[π],float: Expected free energy
- t[2],int: Time

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
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_ontology_terms, f)
            ontology_file = f.name
        
        try:
            analyzer = POMDPAnalyzer(ontology_file=ontology_file)
            result = analyzer.validate_pomdp_model(compliant_content)
            
            assert result.get("overall_valid", False) is True
            assert "validation_results" in result
            
            validation_results = result["validation_results"]
            if "ontology_compliance" in validation_results:
                assert validation_results["ontology_compliance"] is True
                
        finally:
            Path(ontology_file).unlink()

    def test_ontology_non_compliant_pomdp(self, sample_ontology_terms):
        """Test validation of non-ontology-compliant POMDP."""
        non_compliant_content = """
# Non-Ontology-Compliant POMDP

## Variables
- A[3,3],float: Likelihood matrix (wrong dimensions)
- B[2,2,2],float: Transition matrix (wrong dimensions)
- C[2],float: Preference vector (wrong dimensions)
- D[2],float: Prior vector
- E[2],float: Habit vector
- s[2,2],float: Hidden state
- o[2,2],int: Observation
- π[2],float: Policy
- u[2],int: Action
- F[π],float: Free energy
- G[π],float: Expected free energy
- t[2],int: Time
- X[1],float: Unknown component

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
- A -> B: Some invalid connection

## ActInfOntologyAnnotation
- likelihood_matrix: A[3,3],float (should be A[2,2],float)
- transition_matrix: B[2,2,2],float (should be B[3,3,3],float)
- preference_vector: C[2],float (should be C[3],float)
- unknown_component: X[1],float (not in ontology)
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_ontology_terms, f)
            ontology_file = f.name
        
        try:
            analyzer = POMDPAnalyzer(ontology_file=ontology_file)
            result = analyzer.validate_pomdp_model(non_compliant_content)
            
            # Should detect ontology violations
            assert "validation_results" in result
            validation_results = result["validation_results"]
            
            if "ontology_compliance" in validation_results:
                assert validation_results["ontology_compliance"] is False
                
        finally:
            Path(ontology_file).unlink()

    def test_missing_ontology_file(self):
        """Test validation when ontology file is missing."""
        content = """
# POMDP without Ontology

## Variables
- A[3,3],float: Likelihood matrix
- B[3,3,3],float: Transition matrix

## Connections
- A -> B: Some connection
"""
        
        # Test with missing ontology file
        analyzer = POMDPAnalyzer(ontology_file="nonexistent.json")
        result = analyzer.validate_pomdp_model(content)
        
        # Should still work but without ontology validation
        assert "validation_results" in result

    def test_ontology_loading_error_handling(self):
        """Test error handling for invalid ontology files."""
        # Test with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            invalid_ontology_file = f.name
        
        try:
            analyzer = POMDPAnalyzer(ontology_file=invalid_ontology_file)
            # Should handle invalid JSON gracefully
            assert analyzer.ontology_terms is None or analyzer.ontology_terms == {}
        finally:
            Path(invalid_ontology_file).unlink()


class TestPOMDPSemanticValidation:
    """Test POMDP semantic validation rules."""
    
    def test_semantic_consistency(self):
        """Test semantic consistency validation."""
        consistent_content = """
# Semantically Consistent POMDP

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
        
        analyzer = POMDPAnalyzer()
        result = analyzer.validate_pomdp_model(consistent_content)
        
        assert result.get("overall_valid", False) is True
        assert "validation_results" in result
        
        validation_results = result["validation_results"]
        if "semantic_consistency" in validation_results:
            assert validation_results["semantic_consistency"] is True

    def test_semantic_inconsistency(self):
        """Test detection of semantic inconsistencies."""
        inconsistent_content = """
# Semantically Inconsistent POMDP

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
- A -> s: WRONG - Likelihood should map to observations
- B -> o: WRONG - Transitions should map to states
- C -> u: WRONG - Preferences should influence policy
- D -> o: WRONG - Prior should be for states
- E -> u: WRONG - Habits should influence policy
- s -> π: WRONG - States should map to observations
- π -> s: WRONG - Policy should map to actions
- F -> u: WRONG - Free energy should influence policy
- G -> u: WRONG - Expected free energy should influence policy
"""
        
        analyzer = POMDPAnalyzer()
        result = analyzer.validate_pomdp_model(inconsistent_content)
        
        # Should detect semantic inconsistencies
        assert "validation_results" in result
        validation_results = result["validation_results"]
        
        if "semantic_consistency" in validation_results:
            assert validation_results["semantic_consistency"] is False

    def test_connection_validation(self):
        """Test validation of POMDP connections."""
        content = """
# POMDP with Connections

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
        
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(content)
        
        assert result["valid"] is True
        assert "connections" in result
        
        connections = result["connections"]
        assert len(connections) > 0
        
        # Check that connections are properly parsed
        for connection in connections:
            assert "source" in connection
            assert "target" in connection
            assert "line" in connection
            assert "pattern" in connection


class TestPOMDPErrorHandling:
    """Test POMDP error handling and recovery."""
    
    def test_invalid_input_handling(self):
        """Test handling of invalid input."""
        analyzer = POMDPAnalyzer()
        
        # Test with None input
        result = analyzer.analyze_pomdp_structure(None)
        assert result["valid"] is False
        assert "errors" in result
        
        # Test with empty input
        result = analyzer.analyze_pomdp_structure("")
        assert result["valid"] is False
        
        # Test with non-string input
        result = analyzer.analyze_pomdp_structure(123)
        assert result["valid"] is False

    def test_malformed_content_handling(self):
        """Test handling of malformed content."""
        malformed_content = """
# Malformed POMDP

## Variables
- A[3,3],float: Likelihood matrix
- B[3,3,3],float: Transition matrix
- C[3],float: Preference vector

## Connections
- A -> B: Some connection
- Invalid connection syntax
- Another -> invalid -> connection
"""
        
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(malformed_content)
        
        # Should handle malformed content gracefully
        assert "components_found" in result
        # May or may not be valid depending on error handling

    def test_large_content_handling(self):
        """Test handling of large content."""
        # Generate large content
        large_content = "# Large POMDP\n\n## Variables\n"
        for i in range(1000):
            large_content += f"- A{i}[3,3],float: Matrix {i}\n"
        
        large_content += "\n## Connections\n"
        for i in range(100):
            large_content += f"- A{i} -> A{i+1}: Connection {i}\n"
        
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(large_content)
        
        # Should handle large content without crashing
        assert "components_found" in result

    def test_unicode_content_handling(self):
        """Test handling of unicode content."""
        unicode_content = """
# POMDP with Unicode

## Variables
- A[3,3],float: Matrice de vraisemblance
- B[3,3,3],float: Matrice de transition
- C[3],float: Vecteur de préférence
- D[3],float: Vecteur prior
- E[3],float: Vecteur d'habitude
- s[3,3],float: État caché
- o[3,3],int: Observation
- π[3],float: Politique
- u[3],int: Action
- F[π],float: Énergie libre
- G[π],float: Énergie libre attendue
- t[3],int: Temps

## Connections
- A -> o: Mapping de vraisemblance
- B -> s: Transitions d'état
- C -> π: Influence des préférences
- D -> s: État prior
- E -> π: Influence d'habitude
- s -> o: Mapping état-observation
- π -> u: Politique vers action
- F -> π: Minimisation de l'énergie libre
- G -> π: Minimisation de l'énergie libre attendue
"""
        
        analyzer = POMDPAnalyzer()
        result = analyzer.analyze_pomdp_structure(unicode_content)
        
        # Should handle unicode content
        assert "components_found" in result
        assert result["valid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
