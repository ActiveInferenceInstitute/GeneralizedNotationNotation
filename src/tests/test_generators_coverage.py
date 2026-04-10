import pytest

from render.generators import (
    _matrix_to_julia,
    _sanitize_identifier,
    _to_pascal_case,
    generate_activeinference_jl_code,
    generate_bnlearn_code,
    generate_discopy_code,
    generate_pymdp_code,
)


def test_bnlearn_generator():
    res = generate_bnlearn_code({"model_name": "TestModel"})
    assert isinstance(res, str)
    assert "TestModel" in res

def test_pymdp_generator():
    res = generate_pymdp_code({"model_name": "TestModel"})
    assert isinstance(res, str)
    # The template might enforce certain things

def test_activeinference_jl_generator():
    res = generate_activeinference_jl_code({"model_name": "TestModel"})
    assert isinstance(res, str)
    assert "TestModel" in res

def test_discopy_generator():
    res = generate_discopy_code({"model_name": "TestModel"})
    assert isinstance(res, str)
    assert "TestModel" in res

def test_matrix_to_julia():
    assert _matrix_to_julia([1, 2, 3]) == "[1, 2, 3]"
    assert _matrix_to_julia([[1, 2], [3, 4]]) == "[1 2; 3 4]"
    assert "cat(" in _matrix_to_julia([[[1, 2]], [[3, 4]]])

def test_sanitizers():
    assert _sanitize_identifier("Test Model 1") == "test_model_1"
    assert _to_pascal_case("test model") == "TestModel"
