#!/usr/bin/env python3
"""Regression tests for total/degenerate-input handling in analysis/math_utils.py.

Prior to these fixes, several Active Inference metric helpers were not total
on edge inputs:

- ``compute_expected_free_energy`` raised a ``ValueError`` (numpy matmul
  dimension mismatch) when given empty beliefs.
- ``analyze_active_inference_metrics`` raised ``IndexError`` on a flat
  (single-vector) belief list because it indexed ``beliefs_array.shape[1]``.
- ``compute_shannon_entropy(NaN)`` and ``compute_kl_divergence`` propagated
  NaN into downstream metrics instead of returning a guarded value.

These tests pin the total behaviour so the metric helpers never crash and
never leak NaN.
"""

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis import math_utils as mu


def test_shannon_entropy_empty_is_zero() -> None:
    assert mu.compute_shannon_entropy(np.array([])) == 0.0


def test_shannon_entropy_zero_distribution_is_finite() -> None:
    value = mu.compute_shannon_entropy(np.zeros(4))
    assert np.isfinite(value)
    assert value >= 0.0


def test_shannon_entropy_nan_input_does_not_propagate() -> None:
    value = mu.compute_shannon_entropy(np.array([np.nan, np.nan]))
    assert np.isfinite(value)


def test_shannon_entropy_uniform_is_log_of_dimension() -> None:
    value = mu.compute_shannon_entropy(np.full(5, 0.2))
    assert value == pytest.approx(np.log(5), abs=1e-4)


def test_shannon_entropy_certain_distribution_is_near_zero() -> None:
    one_hot = np.array([0.0, 1.0, 0.0, 0.0])
    assert mu.compute_shannon_entropy(one_hot) == pytest.approx(0.0, abs=1e-7)


def test_kl_divergence_empty_returns_zero() -> None:
    assert mu.compute_kl_divergence(np.array([]), np.array([])) == 0.0


def test_kl_divergence_zero_sum_p_is_finite() -> None:
    value = mu.compute_kl_divergence(np.zeros(3), np.full(3, 1 / 3))
    assert np.isfinite(value)


def test_kl_divergence_identical_is_zero() -> None:
    p = np.array([0.2, 0.3, 0.5])
    assert mu.compute_kl_divergence(p, p) == pytest.approx(0.0, abs=1e-4)


def test_kl_divergence_nan_input_finite_result() -> None:
    value = mu.compute_kl_divergence(
        np.array([np.nan, 0.5]), np.array([0.5, 0.5])
    )
    assert np.isfinite(value)


def test_variational_free_energy_empty_beliefs_is_zero() -> None:
    assert mu.compute_variational_free_energy(
        np.array([]), np.array([]), np.ones((2, 2))
    ) == 0.0


def test_expected_free_energy_empty_beliefs_does_not_crash() -> None:
    value = mu.compute_expected_free_energy(
        np.array([]), np.ones((2, 2)), np.ones((2, 2, 2)), np.ones(2), 0
    )
    assert value == 0.0


def test_expected_free_energy_empty_matrices_is_zero() -> None:
    value = mu.compute_expected_free_energy(
        np.array([0.5, 0.5]), np.array([]), np.array([]), np.array([]), 0
    )
    assert value == 0.0
    assert np.isfinite(value)


def test_expected_free_energy_valid_inputs_is_finite() -> None:
    A = np.array([[0.9, 0.2], [0.1, 0.8]])
    B = np.array([[[0.8, 0.3], [0.2, 0.7]], [[0.2, 0.3], [0.8, 0.7]]])
    value = mu.compute_expected_free_energy(
        np.array([0.5, 0.5]), A, B, np.array([0.0, 0.0]), 0
    )
    assert np.isfinite(value)


def test_flat_belief_list_does_not_crash_certainty() -> Any:
    result = mu.analyze_active_inference_metrics(
        cast(Any, [0.5, 0.5, 0.5, 0.5]), [1.0, 2.0], [0, 1], "flat-model"
    )
    certainty = result["metrics"]["certainty"]
    assert np.isfinite(certainty["mean"])
    assert len(certainty["trajectory"]) == 4


def test_empty_trajectory_returns_early() -> Any:
    result = mu.analyze_active_inference_metrics([], [], [], "empty-model")
    assert result["num_timesteps"] == 0
    assert result["metrics"] == {}