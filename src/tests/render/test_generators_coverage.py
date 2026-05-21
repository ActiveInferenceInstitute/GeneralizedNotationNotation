from typing import Any

import pytest

from render.generators import (
    _matrix_to_julia,
    _sanitize_identifier,
    _to_pascal_case,
    generate_activeinference_jl_code,
    generate_bnlearn_code,
    generate_discopy_code,
    generate_pymdp_code,
    generate_rxinfer_code,
)


def _explicit_pomdp_spec() -> dict[str, Any]:
    return {
        "model_name": "TestModel",
        "initialparameterization": {
            "A": [[0.9, 0.1], [0.1, 0.9]],
            "B": [[[0.8, 0.2], [0.2, 0.8]]],
            "C": [0.0, 1.0],
            "D": [0.5, 0.5],
            "E": [1.0],
        },
        "model_parameters": {"num_hidden_states": 2, "num_obs": 2, "num_actions": 1},
    }


def test_bnlearn_generator() -> Any:
    res = generate_bnlearn_code({"model_name": "TestModel"})
    assert isinstance(res, str)
    assert "TestModel" in res


def test_pymdp_generator() -> Any:
    res = generate_pymdp_code({"model_name": "TestModel"})
    assert isinstance(res, str)
    # The template might enforce certain things


def test_activeinference_jl_generator() -> Any:
    res = generate_activeinference_jl_code(_explicit_pomdp_spec())
    assert isinstance(res, str)
    assert "TestModel" in res


def test_rxinfer_generator() -> Any:
    res = generate_rxinfer_code(_explicit_pomdp_spec())
    assert isinstance(res, str)
    assert "TestModel" in res


def test_discopy_generator() -> Any:
    res = generate_discopy_code({"model_name": "TestModel"})
    assert isinstance(res, str)
    assert "TestModel" in res


def test_matrix_to_julia() -> Any:
    assert _matrix_to_julia([1, 2, 3]) == "[1, 2, 3]"
    assert _matrix_to_julia([[1, 2], [3, 4]]) == "[1 2; 3 4]"
    assert "cat(" in _matrix_to_julia([[[1, 2]], [[3, 4]]])


def test_sanitizers() -> Any:
    assert _sanitize_identifier("Test Model 1") == "test_model_1"
    assert _to_pascal_case("test model") == "TestModel"
