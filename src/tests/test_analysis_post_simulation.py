#!/usr/bin/env python3
"""
Functional tests for the Analysis post_simulation module.

Tests mathematical functions (Shannon entropy, KL divergence, variational free energy,
expected free energy), framework data extractors, simulation trace analysis,
free energy analysis, policy convergence, state distributions, and cross-framework
comparison.
"""

import pytest
import sys
import json
import math
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

pytestmark = pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")

from analysis.post_simulation import (
    compute_shannon_entropy,
    compute_kl_divergence,
    compute_variational_free_energy,
    compute_expected_free_energy,
    compute_information_gain,
    analyze_simulation_traces,
    analyze_free_energy,
    analyze_policy_convergence,
    analyze_state_distributions,
    compare_framework_results,
    extract_pymdp_data,
    extract_rxinfer_data,
    extract_jax_data,
    extract_discopy_data,
    extract_activeinference_jl_data,
)


# ---------------------------------------------------------------------------
# Mathematical functions
# ---------------------------------------------------------------------------
class TestShannonEntropy:
    """Test compute_shannon_entropy."""

    @pytest.mark.unit
    def test_uniform_distribution(self):
        """Uniform distribution should have maximum entropy for its size."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = compute_shannon_entropy(p)
        expected = -np.sum(p * np.log(p))
        assert abs(entropy - expected) < 1e-6

    @pytest.mark.unit
    def test_deterministic_distribution(self):
        """A near-deterministic distribution should have near-zero entropy."""
        p = np.array([1.0, 0.0, 0.0])
        entropy = compute_shannon_entropy(p)
        assert entropy < 0.01  # close to 0

    @pytest.mark.unit
    def test_binary_half(self):
        """Binary fair coin should have entropy ln(2)."""
        p = np.array([0.5, 0.5])
        entropy = compute_shannon_entropy(p)
        assert abs(entropy - np.log(2)) < 1e-4

    @pytest.mark.unit
    def test_non_normalized_input(self):
        """Function should normalize input before computing."""
        p = np.array([2.0, 2.0, 2.0])
        entropy = compute_shannon_entropy(p)
        expected = np.log(3)  # uniform over 3
        assert abs(entropy - expected) < 1e-4

    @pytest.mark.unit
    def test_single_element(self):
        """Single-element distribution should have zero entropy."""
        p = np.array([1.0])
        entropy = compute_shannon_entropy(p)
        assert entropy < 1e-6


class TestKLDivergence:
    """Test compute_kl_divergence."""

    @pytest.mark.unit
    def test_identical_distributions(self):
        """KL divergence of identical distributions should be near zero."""
        p = np.array([0.3, 0.3, 0.4])
        kl = compute_kl_divergence(p, p)
        assert abs(kl) < 1e-6

    @pytest.mark.unit
    def test_asymmetry(self):
        """KL(P||Q) should differ from KL(Q||P)."""
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        kl_pq = compute_kl_divergence(p, q)
        kl_qp = compute_kl_divergence(q, p)
        assert kl_pq != kl_qp

    @pytest.mark.unit
    def test_kl_non_negative(self):
        """KL divergence should always be non-negative."""
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.3, 0.6])
        kl = compute_kl_divergence(p, q)
        assert kl >= -1e-10  # allow small numerical error

    @pytest.mark.unit
    def test_kl_with_zero_elements(self):
        """Should handle distributions with zero elements gracefully."""
        p = np.array([1.0, 0.0])
        q = np.array([0.5, 0.5])
        kl = compute_kl_divergence(p, q)
        assert np.isfinite(kl)


class TestVariationalFreeEnergy:
    """Test compute_variational_free_energy."""

    @pytest.mark.unit
    def test_basic_computation(self):
        """Should return a finite float for valid inputs."""
        obs = np.array([1.0, 0.0, 0.0])
        beliefs = np.array([0.5, 0.3, 0.2])
        A = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])
        fe = compute_variational_free_energy(obs, beliefs, A)
        assert isinstance(fe, float)
        assert np.isfinite(fe)

    @pytest.mark.unit
    def test_with_explicit_prior(self):
        """Should accept an explicit prior distribution."""
        obs = np.array([1.0, 0.0, 0.0])
        beliefs = np.array([0.8, 0.1, 0.1])
        A = np.eye(3)
        prior = np.array([0.5, 0.25, 0.25])
        fe = compute_variational_free_energy(obs, beliefs, A, prior=prior)
        assert isinstance(fe, float)
        assert np.isfinite(fe)

    @pytest.mark.unit
    def test_uniform_prior_default(self):
        """With no prior supplied, a uniform prior should be used."""
        obs = np.array([1.0, 0.0])
        beliefs = np.array([0.6, 0.4])
        A = np.eye(2)
        fe = compute_variational_free_energy(obs, beliefs, A)
        assert isinstance(fe, float)
        assert np.isfinite(fe)

    @pytest.mark.unit
    def test_free_energy_changes_with_beliefs(self):
        """Different beliefs should yield different free energy values."""
        obs = np.array([1.0, 0.0, 0.0])
        A = np.eye(3)
        fe1 = compute_variational_free_energy(obs, np.array([0.9, 0.05, 0.05]), A)
        fe2 = compute_variational_free_energy(obs, np.array([0.33, 0.33, 0.34]), A)
        assert fe1 != fe2


class TestExpectedFreeEnergy:
    """Test compute_expected_free_energy."""

    @pytest.mark.unit
    def test_basic_efe(self):
        """Should return a finite float for valid inputs."""
        beliefs = np.array([0.5, 0.3, 0.2])
        A = np.eye(3)
        B = np.stack([np.eye(3)] * 2, axis=-1)  # 3x3x2
        C = np.array([1.0, 0.0, 0.0])  # prefer first observation
        efe = compute_expected_free_energy(beliefs, A, B, C, policy=0)
        assert isinstance(efe, float)
        assert np.isfinite(efe)

    @pytest.mark.unit
    def test_different_policies_different_efe(self):
        """Different policy indices should generally give different EFE."""
        beliefs = np.array([0.5, 0.3, 0.2])
        A = np.eye(3)
        # Make B different for each action to ensure different outcomes
        B0 = np.array([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        B1 = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
        B = np.stack([B0, B1], axis=-1)
        C = np.array([2.0, 0.0, 0.0])
        efe0 = compute_expected_free_energy(beliefs, A, B, C, policy=0)
        efe1 = compute_expected_free_energy(beliefs, A, B, C, policy=1)
        # With different transition matrices, EFE should differ
        assert efe0 != efe1

    @pytest.mark.unit
    def test_efe_with_2d_B(self):
        """Should handle a 2D B matrix (single action)."""
        beliefs = np.array([0.5, 0.5])
        A = np.eye(2)
        B = np.eye(2)
        C = np.array([1.0, 0.0])
        efe = compute_expected_free_energy(beliefs, A, B, C, policy=0)
        assert isinstance(efe, float)
        assert np.isfinite(efe)


class TestInformationGain:
    """Test compute_information_gain."""

    @pytest.mark.unit
    def test_no_update(self):
        """Same prior and posterior should give near-zero information gain."""
        p = np.array([0.5, 0.3, 0.2])
        ig = compute_information_gain(p, p)
        assert abs(ig) < 1e-6

    @pytest.mark.unit
    def test_positive_gain(self):
        """Information gain should be non-negative."""
        prior = np.array([0.33, 0.33, 0.34])
        posterior = np.array([0.9, 0.05, 0.05])
        ig = compute_information_gain(prior, posterior)
        assert ig >= -1e-10


# ---------------------------------------------------------------------------
# Framework data extractors
# ---------------------------------------------------------------------------
class TestExtractPymdpData:
    """Test extract_pymdp_data."""

    @pytest.mark.unit
    def test_empty_result(self):
        """Should return empty lists for an empty execution result."""
        data = extract_pymdp_data({})
        assert data["traces"] == []
        assert data["free_energy"] == []
        assert data["beliefs"] == []

    @pytest.mark.unit
    def test_with_simulation_data(self):
        """Should extract simulation data when present."""
        result = {
            "simulation_data": {
                "traces": [1, 2, 3],
                "beliefs": [[0.5, 0.5], [0.6, 0.4]],
                "actions": [0, 1],
                "observations": [0, 1],
                "free_energy": [1.2, 0.8],
            }
        }
        data = extract_pymdp_data(result)
        assert data["traces"] == [1, 2, 3]
        assert data["beliefs"] == [[0.5, 0.5], [0.6, 0.4]]
        assert data["actions"] == [0, 1]
        assert data["free_energy"] == [1.2, 0.8]


class TestExtractRxinferData:
    """Test extract_rxinfer_data."""

    @pytest.mark.unit
    def test_empty_result(self):
        """Should return empty lists for an empty execution result."""
        data = extract_rxinfer_data({})
        assert data["beliefs"] == []
        assert data["observations"] == []

    @pytest.mark.unit
    def test_with_simulation_data(self):
        """Should extract RxInfer-specific fields."""
        result = {
            "simulation_data": {
                "beliefs": [[0.3, 0.7]],
                "true_states": [1],
                "observations": [0],
                "posterior": [0.6, 0.4],
            }
        }
        data = extract_rxinfer_data(result)
        assert data["beliefs"] == [[0.3, 0.7]]
        assert data["true_states"] == [1]
        assert data["posterior"] == [0.6, 0.4]


class TestExtractJaxData:
    """Test extract_jax_data."""

    @pytest.mark.unit
    def test_empty_result(self):
        """Should handle empty execution result without error."""
        data = extract_jax_data({})
        assert isinstance(data, dict)


class TestExtractDiscopyData:
    """Test extract_discopy_data."""

    @pytest.mark.unit
    def test_empty_result(self):
        """Should handle empty execution result without error."""
        data = extract_discopy_data({})
        assert isinstance(data, dict)


class TestExtractActiveInferenceJlData:
    """Test extract_activeinference_jl_data."""

    @pytest.mark.unit
    def test_empty_result(self):
        """Should handle empty execution result without error."""
        data = extract_activeinference_jl_data({})
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Simulation analysis functions
# ---------------------------------------------------------------------------
class TestAnalyzeSimulationTraces:
    """Test analyze_simulation_traces."""

    @pytest.mark.unit
    def test_empty_traces(self):
        """Should handle empty trace list."""
        result = analyze_simulation_traces([], "pymdp", "test_model")
        assert result["trace_count"] == 0
        assert result["framework"] == "pymdp"

    @pytest.mark.unit
    def test_list_traces(self):
        """Should calculate trace lengths for list-based traces."""
        traces = [[1, 2, 3], [4, 5]]
        result = analyze_simulation_traces(traces, "pymdp", "test_model")
        assert result["trace_count"] == 2
        assert result["trace_lengths"] == [3, 2]
        assert result["avg_trace_length"] == 2.5

    @pytest.mark.unit
    def test_dict_traces(self):
        """Should handle dict-based traces with states key."""
        traces = [
            {"states": [0, 1, 2]},
            {"states": [0, 1]},
        ]
        result = analyze_simulation_traces(traces, "rxinfer", "model_a")
        assert result["trace_count"] == 2
        assert result["trace_lengths"] == [3, 2]

    @pytest.mark.unit
    def test_model_name_preserved(self):
        """Should preserve model name and framework in result."""
        result = analyze_simulation_traces([], "jax", "my_model")
        assert result["model_name"] == "my_model"
        assert result["framework"] == "jax"


class TestAnalyzeFreeEnergy:
    """Test analyze_free_energy."""

    @pytest.mark.unit
    def test_empty_values(self):
        """Should handle empty free energy list."""
        result = analyze_free_energy([], "pymdp", "test")
        assert result["free_energy_count"] == 0

    @pytest.mark.unit
    def test_basic_statistics(self):
        """Should compute mean, std, min, max of free energy."""
        values = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = analyze_free_energy(values, "pymdp", "test")
        assert abs(result["mean_free_energy"] - 3.0) < 1e-6
        assert result["min_free_energy"] == 1.0
        assert result["max_free_energy"] == 5.0

    @pytest.mark.unit
    def test_decreasing_trend(self):
        """Should detect a decreasing free energy trend."""
        values = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0]
        result = analyze_free_energy(values, "pymdp", "test")
        assert result["free_energy_decreasing"] == True
        assert result["free_energy_trend"] < 0

    @pytest.mark.unit
    def test_convergence_detection(self):
        """Should detect convergence when late values have low variance."""
        values = [10.0, 5.0, 2.5, 1.2, 1.1, 1.05, 1.02, 1.01, 1.005, 1.001]
        result = analyze_free_energy(values, "pymdp", "test")
        assert "converged" in result


class TestAnalyzePolicyConvergence:
    """Test analyze_policy_convergence."""

    @pytest.mark.unit
    def test_empty_policies(self):
        """Should handle empty policy list."""
        result = analyze_policy_convergence([], "pymdp", "test")
        assert result["policy_count"] == 0
        assert result["policy_entropy"] == []

    @pytest.mark.unit
    def test_deterministic_policies(self):
        """Near-deterministic policies should have low entropy."""
        policies = [
            [0.99, 0.005, 0.005],
            [0.99, 0.005, 0.005],
        ]
        result = analyze_policy_convergence(policies, "pymdp", "test")
        assert len(result["policy_entropy"]) == 2
        for e in result["policy_entropy"]:
            assert e < 0.5  # low entropy

    @pytest.mark.unit
    def test_uniform_policies(self):
        """Uniform policies should have high entropy."""
        policies = [
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
        ]
        result = analyze_policy_convergence(policies, "pymdp", "test")
        for e in result["policy_entropy"]:
            assert e > 0.9  # high entropy


class TestAnalyzeStateDistributions:
    """Test analyze_state_distributions."""

    @pytest.mark.unit
    def test_empty_states(self):
        """Should handle empty state list."""
        result = analyze_state_distributions([], "pymdp", "test")
        assert result["state_count"] == 0
        assert result["state_entropy"] == []

    @pytest.mark.unit
    def test_state_entropy_calculation(self):
        """Should compute entropy for each state distribution."""
        states = [
            [0.5, 0.5],
            [0.9, 0.1],
        ]
        result = analyze_state_distributions(states, "pymdp", "test")
        assert len(result["state_entropy"]) == 2
        # Uniform should have higher entropy than peaked
        assert result["state_entropy"][0] > result["state_entropy"][1]

    @pytest.mark.unit
    def test_diversity_metrics(self):
        """Should compute mean and std of state entropy."""
        states = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
        result = analyze_state_distributions(states, "pymdp", "test")
        assert "mean_entropy" in result["state_diversity"]
        assert "std_entropy" in result["state_diversity"]
        assert result["state_diversity"]["std_entropy"] < 1e-6  # all same


class TestCompareFrameworkResults:
    """Test compare_framework_results."""

    @pytest.mark.unit
    def test_single_framework(self):
        """Should note that comparison needs at least 2 frameworks."""
        results = {"pymdp": {"success": True}}
        comparison = compare_framework_results(results, "test_model")
        assert "Need at least 2 frameworks" in comparison.get("message", "")

    @pytest.mark.unit
    def test_two_frameworks(self):
        """Should compare two frameworks and identify fastest."""
        results = {
            "pymdp": {"success": True, "execution_time": 1.5},
            "jax": {"success": True, "execution_time": 0.8},
        }
        comparison = compare_framework_results(results, "test_model")
        assert comparison["framework_count"] == 2
        assert "pymdp" in comparison["frameworks_compared"]
        assert "jax" in comparison["frameworks_compared"]
        assert comparison["comparisons"]["fastest_execution"]["framework"] == "jax"

    @pytest.mark.unit
    def test_success_rates_comparison(self):
        """Should compare success rates across frameworks."""
        results = {
            "pymdp": {"success": True, "execution_time": 1.0},
            "rxinfer": {"success": False, "execution_time": 2.0},
        }
        comparison = compare_framework_results(results, "model")
        assert comparison["comparisons"]["success_rates"]["pymdp"] is True
        assert comparison["comparisons"]["success_rates"]["rxinfer"] is False

    @pytest.mark.unit
    def test_empty_results(self):
        """Should handle empty results dict."""
        comparison = compare_framework_results({}, "test_model")
        assert comparison["framework_count"] == 0
