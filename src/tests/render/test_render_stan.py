"""
Tests for the Stan renderer component of GNN.
"""
import pytest

from render.stan.stan_renderer import _stan_type, render_stan


def test_stan_type_mapping():
    """Test mapping of GNN types and dimensions to Stan types."""
    assert _stan_type("real", []) == "real"
    assert _stan_type("int", []) == "int"
    assert _stan_type("float", [5]) == "vector[5]"
    assert _stan_type("double", [3, 4]) == "matrix[3, 4]"
    assert _stan_type("real", [2, 3, 4]) == "array[2] matrix[3, 4]"

def test_render_stan_basic():
    """Test generation of basic Stan code."""
    variables = [
        {"name": "obs1", "dimensions": [5], "dtype": "real"},
        {"name": "theta", "dimensions": [3], "dtype": "real"}
    ]
    connections = [
        {"source": "theta", "target": "obs1", "directed": True}
    ]
    
    code = render_stan(variables, connections, model_name="test_stan_model")
    
    assert "data {" in code
    assert "vector[5] obs1;" in code
    assert "parameters {" in code
    assert "vector[3] theta;" in code
    assert "model {" in code
    assert "obs1 ~ normal(theta, 1.0);" in code
    assert "test_stan_model" in code

def test_render_stan_empty():
    """Test generating Stan code with empty data."""
    code = render_stan([], [])
    
    assert "data {" in code
    assert "No observed variables declared" in code
    assert "parameters {" in code
    assert "No parameters declared" in code
    assert "model {" in code
    assert "No connections to model" in code
