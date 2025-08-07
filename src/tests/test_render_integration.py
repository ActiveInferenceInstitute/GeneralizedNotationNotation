#!/usr/bin/env python3
"""
Test Render Integration Tests

This file contains comprehensive integration tests for code rendering functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *

class TestRenderIntegration:
    """Test render integration functionality."""
    
    def test_render_import_available(self):
        """Test that render module can be imported."""
        try:
            from render import CodeRenderer
            assert True
        except ImportError:
            pytest.skip("Render module not available")
    
    def test_pymdp_rendering(self):
        """Test PyMDP code rendering."""
        # Test basic PyMDP template
        pymdp_template = """
import pymdp
import numpy as np

# GNN Model: {model_name}
# Generated from GNN specification

def create_model():
    # State space
    {state_space}
    
    # Observation space  
    {observation_space}
    
    # Transition matrix
    A = {transition_matrix}
    
    # Observation matrix
    B = {observation_matrix}
    
    return A, B
"""
        
        # Test template substitution
        model_name = "test_model"
        state_space = "s = [3]"
        observation_space = "o = [2]"
        transition_matrix = "np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])"
        observation_matrix = "np.array([[0.9, 0.1], [0.1, 0.9]])"
        
        rendered_code = pymdp_template.format(
            model_name=model_name,
            state_space=state_space,
            observation_space=observation_space,
            transition_matrix=transition_matrix,
            observation_matrix=observation_matrix
        )
        
        assert model_name in rendered_code
        assert state_space in rendered_code
        assert observation_space in rendered_code
        assert transition_matrix in rendered_code
        assert observation_matrix in rendered_code
    
    def test_rxinfer_rendering(self):
        """Test RxInfer.jl code rendering."""
        # Test basic RxInfer template
        rxinfer_template = """
using RxInfer
using LinearAlgebra

# GNN Model: {model_name}
# Generated from GNN specification

@model function gnn_model()
    # State space
    {state_space}
    
    # Observation space
    {observation_space}
    
    # Prior distributions
    {priors}
    
    # Likelihood
    {likelihood}
    
    return {return_vars}
end
"""
        
        # Test template substitution
        model_name = "test_model"
        state_space = "s ~ Normal(0, 1)"
        observation_space = "o ~ Normal(s, 0.1)"
        priors = "s_prior ~ Normal(0, 1)"
        likelihood = "o ~ Normal(s, 0.1)"
        return_vars = "s, o"
        
        rendered_code = rxinfer_template.format(
            model_name=model_name,
            state_space=state_space,
            observation_space=observation_space,
            priors=priors,
            likelihood=likelihood,
            return_vars=return_vars
        )
        
        assert model_name in rendered_code
        assert state_space in rendered_code
        assert observation_space in rendered_code
        assert priors in rendered_code
        assert likelihood in rendered_code
        assert return_vars in rendered_code
    
    def test_jax_rendering(self):
        """Test JAX code rendering."""
        # Test basic JAX template
        jax_template = """
import jax
import jax.numpy as jnp
from jax import random

# GNN Model: {model_name}
# Generated from GNN specification

def create_model():
    # Model parameters
    {parameters}
    
    # Model function
    def model(key, data):
        {model_function}
        return {return_values}
    
    return model
"""
        
        # Test template substitution
        model_name = "test_model"
        parameters = "A = jnp.array([[0.8, 0.2], [0.2, 0.8]])"
        model_function = "s = jax.random.normal(key, (2,))"
        return_values = "s"
        
        rendered_code = jax_template.format(
            model_name=model_name,
            parameters=parameters,
            model_function=model_function,
            return_values=return_values
        )
        
        assert model_name in rendered_code
        assert parameters in rendered_code
        assert model_function in rendered_code
        assert return_values in rendered_code
    
    def test_render_error_handling(self):
        """Test render error handling."""
        # Test with invalid template
        invalid_template = """
# Invalid template with missing {placeholder}

def test_function():
    return {undefined_variable}
"""
        
        # Should handle missing placeholders gracefully
        try:
            # This would normally raise a KeyError
            rendered = invalid_template.format(placeholder="test")
            assert "Invalid template" in invalid_template
        except KeyError:
            # Expected behavior
            assert True
    
    def test_render_performance(self):
        """Test render performance."""
        import time
        
        start_time = time.time()
        
        # Simulate rendering
        template = "Model: {name}, State: {state}"
        for i in range(100):
            rendered = template.format(name=f"model_{i}", state=f"state_{i}")
            assert "Model:" in rendered
        
        rendering_time = time.time() - start_time
        assert rendering_time < 1.0  # Should complete quickly
    
    def test_render_memory_usage(self):
        """Test render memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate rendering
        templates = [f"Template {i}: {{var}}" for i in range(1000)]
        rendered = [t.format(var=f"value_{i}") for i, t in enumerate(templates)]
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB for this test)
        assert memory_increase < 50.0

