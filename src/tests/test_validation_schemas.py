#!/usr/bin/env python3
"""Tests for src/utils/validation_schemas.py. Zero-mock, uses real tempdirs."""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.validation_schemas import (  # noqa: E402
    FRAMEWORK_PRESETS,
    KNOWN_FRAMEWORKS,
    normalize_pomdp_columns,
    validate_frameworks_arg,
    validate_model_data,
    validate_target_dir,
)


# --- validate_model_data -------------------------------------------------

def test_validate_model_data_passes_through_valid_dict():
    data = {"model_name": "m", "other": 1}
    assert validate_model_data(data) is data


def test_validate_model_data_rejects_none():
    with pytest.raises(ValueError, match="is None"):
        validate_model_data(None)


def test_validate_model_data_rejects_non_dict():
    with pytest.raises(ValueError, match="must be a dict"):
        validate_model_data("not a dict")  # type: ignore[arg-type]


def test_validate_model_data_rejects_missing_required_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        validate_model_data({"other": 1}, required_keys=("model_name",))


def test_validate_model_data_accepts_extra_required_keys():
    data = {"model_name": "x", "version": "1.0"}
    assert validate_model_data(data, required_keys=("model_name", "version")) is data


def test_validate_model_data_context_included_in_error():
    with pytest.raises(ValueError, match=r"\[my_generator\]"):
        validate_model_data(None, context="my_generator")


# --- validate_target_dir -------------------------------------------------

def test_validate_target_dir_returns_path_for_existing_dir(tmp_path):
    result = validate_target_dir(tmp_path)
    assert result == tmp_path
    assert isinstance(result, Path)


def test_validate_target_dir_accepts_str_input(tmp_path):
    result = validate_target_dir(str(tmp_path))
    assert result == tmp_path


def test_validate_target_dir_rejects_none():
    with pytest.raises(ValueError, match="is None"):
        validate_target_dir(None)


def test_validate_target_dir_raises_when_missing(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        validate_target_dir(missing, must_exist=True)


def test_validate_target_dir_allows_missing_when_must_exist_false(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert validate_target_dir(missing, must_exist=False) == missing


def test_validate_target_dir_rejects_file_path(tmp_path):
    f = tmp_path / "a_file.txt"
    f.write_text("hi")
    with pytest.raises(NotADirectoryError):
        validate_target_dir(f)


# --- validate_frameworks_arg ---------------------------------------------

def test_validate_frameworks_arg_defaults_to_all_for_empty():
    assert validate_frameworks_arg("") == "all"
    assert validate_frameworks_arg(None) == "all"


@pytest.mark.parametrize("preset", list(FRAMEWORK_PRESETS.keys()))
def test_validate_frameworks_arg_accepts_presets(preset):
    assert validate_frameworks_arg(preset) == preset


def test_validate_frameworks_arg_accepts_comma_list():
    result = validate_frameworks_arg("pymdp,jax")
    assert result == "pymdp,jax"


def test_validate_frameworks_arg_rejects_non_string():
    with pytest.raises(ValueError, match="must be a string"):
        validate_frameworks_arg(123)  # type: ignore[arg-type]


def test_validate_frameworks_arg_rejects_all_unknown_frameworks():
    with pytest.raises(ValueError, match="no known frameworks"):
        validate_frameworks_arg("nope,nada,zilch")


def test_validate_frameworks_arg_accepts_mixed_known_and_unknown():
    # As long as at least one is known, the full list passes to parse_frameworks_parameter
    # which filters invalid entries with a warning.
    result = validate_frameworks_arg("pymdp,bogus")
    assert "pymdp" in result


def test_known_frameworks_contains_all_runners():
    required = {"jax", "numpyro", "pytorch", "discopy", "bnlearn", "pymdp"}
    assert required.issubset(set(KNOWN_FRAMEWORKS))


# --- normalize_pomdp_columns ---------------------------------------------

def test_normalize_pomdp_columns_2d_produces_stochastic_columns():
    M = np.array([[1.0, 3.0], [1.0, 1.0]])
    out = normalize_pomdp_columns(M)
    assert np.allclose(out.sum(axis=0), [1.0, 1.0])


def test_normalize_pomdp_columns_zero_sum_becomes_uniform():
    M = np.array([[0.0, 1.0], [0.0, 0.0]])
    out = normalize_pomdp_columns(M)
    # Column 0 was all zeros -> becomes uniform (0.5, 0.5)
    assert np.allclose(out[:, 0], [0.5, 0.5])
    # Column 1 had one nonzero entry -> becomes (1.0, 0.0)
    assert np.allclose(out[:, 1], [1.0, 0.0])


def test_normalize_pomdp_columns_passes_through_non_2d():
    M = np.zeros((2, 2, 2))
    out = normalize_pomdp_columns(M)
    assert out.shape == (2, 2, 2)


def test_normalize_pomdp_columns_accepts_list_input():
    out = normalize_pomdp_columns([[1.0, 1.0], [1.0, 1.0]])
    assert np.allclose(out, [[0.5, 0.5], [0.5, 0.5]])
