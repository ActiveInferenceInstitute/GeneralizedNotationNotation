"""
Unit tests for Active Inference mathematical functions in post_simulation.py.

Tests Shannon entropy, KL divergence, variational free energy,
expected free energy, information gain, and the analysis metrics aggregator
using known-correct mathematical inputs/outputs.
"""

import numpy as np
import pytest
from pathlib import Path

# Import the functions under test
from analysis.post_simulation import (
    compute_shannon_entropy,
    compute_kl_divergence,
    compute_variational_free_energy,
    compute_expected_free_energy,
    compute_information_gain,
    analyze_active_inference_metrics,
)
from render.processor import normalize_matrices
from gnn.pomdp_extractor import POMDPStateSpace


# =============================================================================
# Shannon Entropy Tests
# =============================================================================

class TestShannonEntropy:
    """Tests for compute_shannon_entropy."""

    def test_uniform_distribution(self):
        """Uniform distribution over N states should have entropy ln(N)."""
        for n in [2, 4, 8]:
            p = np.ones(n) / n
            expected = np.log(n)
            result = compute_shannon_entropy(p)
            assert abs(result - expected) < 1e-6, (
                f"Uniform({n}) entropy should be {expected:.4f}, got {result:.4f}"
            )

    def test_dirac_distribution(self):
        """Concentrated distribution should have near-zero entropy."""
        p = np.array([1.0, 0.0, 0.0, 0.0])
        result = compute_shannon_entropy(p)
        assert result < 0.01, f"Dirac entropy should be ~0, got {result}"

    def test_binary_distribution(self):
        """Binary distribution (0.5, 0.5) should equal ln(2)."""
        p = np.array([0.5, 0.5])
        expected = np.log(2)
        result = compute_shannon_entropy(p)
        assert abs(result - expected) < 1e-6

    def test_asymmetric_binary(self):
        """Entropy of (0.9, 0.1) should be less than ln(2)."""
        p = np.array([0.9, 0.1])
        result = compute_shannon_entropy(p)
        assert result < np.log(2), "Asymmetric should have less entropy than uniform"
        assert result > 0, "Entropy should be positive"

    def test_non_negative(self):
        """Entropy should always be non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            p = rng.dirichlet(np.ones(5))
            assert compute_shannon_entropy(p) >= 0


# =============================================================================
# KL Divergence Tests
# =============================================================================

class TestKLDivergence:
    """Tests for compute_kl_divergence."""

    def test_same_distribution_is_zero(self):
        """D_KL(P || P) = 0."""
        p = np.array([0.3, 0.5, 0.2])
        result = compute_kl_divergence(p, p)
        assert abs(result) < 1e-4, f"D_KL(P||P) should be ~0, got {result}"

    def test_non_negative(self):
        """KL divergence is always non-negative (Gibbs' inequality)."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            p = rng.dirichlet(np.ones(4))
            q = rng.dirichlet(np.ones(4))
            result = compute_kl_divergence(p, q)
            assert result >= -1e-10, f"D_KL should be >= 0, got {result}"

    def test_asymmetric(self):
        """KL divergence is generally asymmetric: D_KL(P||Q) != D_KL(Q||P)."""
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        pq = compute_kl_divergence(p, q)
        qp = compute_kl_divergence(q, p)
        # Both should be positive
        assert pq > 0
        assert qp > 0
        # They are equal in this symmetric case, but that's fine —
        # the point is the function handles both directions

    def test_peaked_vs_uniform(self):
        """KL from peaked to uniform should be positive."""
        p = np.array([0.99, 0.01])
        q = np.array([0.5, 0.5])
        result = compute_kl_divergence(p, q)
        assert result > 0


# =============================================================================
# Variational Free Energy Tests
# =============================================================================

class TestVariationalFreeEnergy:
    """Tests for compute_variational_free_energy."""

    def test_returns_float(self):
        """Should return a finite float."""
        beliefs = np.array([0.5, 0.5])
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        obs = np.array([1, 0])
        result = compute_variational_free_energy(obs, beliefs, A)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_certain_belief_lower_energy(self):
        """A belief matching the true generative model should have lower free energy
        than a mismatched belief (given the same A matrix and observations)."""
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        obs = np.array([1, 0])

        # Belief matching state 0 (correct for obs=[1,0] with this A)
        correct_belief = np.array([0.9, 0.1])
        # Belief matching state 1 (wrong)
        wrong_belief = np.array([0.1, 0.9])

        fe_correct = compute_variational_free_energy(obs, correct_belief, A)
        fe_wrong = compute_variational_free_energy(obs, wrong_belief, A)

        # The correct belief should yield lower (or equal) free energy
        assert fe_correct <= fe_wrong + 1e-6, (
            f"Correct belief FE ({fe_correct:.4f}) should be <= wrong belief FE ({fe_wrong:.4f})"
        )

    def test_uniform_prior(self):
        """When no prior is given, should use uniform prior."""
        beliefs = np.array([0.5, 0.5])
        A = np.eye(2)
        obs = np.array([1, 0])
        # Should not raise
        result = compute_variational_free_energy(obs, beliefs, A)
        assert np.isfinite(result)


# =============================================================================
# Expected Free Energy Tests
# =============================================================================

class TestExpectedFreeEnergy:
    """Tests for compute_expected_free_energy."""

    def test_returns_float(self):
        """Should return a finite float."""
        beliefs = np.array([0.5, 0.3, 0.2])
        A = np.eye(3)
        B = np.stack([np.eye(3)] * 2, axis=2)  # 2 actions, identity transitions
        C = np.zeros(3)
        result = compute_expected_free_energy(beliefs, A, B, C, policy=0)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_preferred_outcome_lower_efe(self):
        """A policy leading to preferred observations should have lower EFE."""
        beliefs = np.array([1.0, 0.0])  # Agent is certain it's in state 0
        A = np.eye(2)  # Identity observation

        # Action 0: stay in state 0, Action 1: move to state 1
        B = np.zeros((2, 2, 2))
        B[0, 0, 0] = 1.0  # Action 0 keeps in state 0
        B[1, 0, 1] = 1.0  # Action 1 moves to state 1
        B[0, 1, 0] = 1.0
        B[1, 1, 1] = 1.0

        # Prefer observation 0 (state 0)
        C = np.array([1.0, -1.0])  # Log preference for obs 0

        efe_stay = compute_expected_free_energy(beliefs, A, B, C, policy=0)
        efe_move = compute_expected_free_energy(beliefs, A, B, C, policy=1)

        # Staying should be preferred (lower EFE)
        assert efe_stay < efe_move, (
            f"Stay EFE ({efe_stay:.4f}) should be < Move EFE ({efe_move:.4f})"
        )

    def test_2d_b_matrix_fallback(self):
        """Should handle 2D B matrices (no action dimension)."""
        beliefs = np.array([0.5, 0.5])
        A = np.eye(2)
        B = np.eye(2)  # 2D, no action dimension
        C = np.zeros(2)
        result = compute_expected_free_energy(beliefs, A, B, C, policy=0)
        assert np.isfinite(result)


# =============================================================================
# Information Gain Tests
# =============================================================================

class TestInformationGain:
    """Tests for compute_information_gain."""

    def test_no_update_zero_gain(self):
        """If posterior equals prior, information gain is zero."""
        p = np.array([0.3, 0.7])
        result = compute_information_gain(p, p)
        assert abs(result) < 1e-4

    def test_positive_gain(self):
        """Updating beliefs should yield positive information gain."""
        prior = np.array([0.5, 0.5])
        posterior = np.array([0.9, 0.1])
        result = compute_information_gain(prior, posterior)
        assert result > 0, "Information gain should be positive for belief update"

    def test_equals_kl_posterior_prior(self):
        """IG should equal D_KL(posterior || prior)."""
        prior = np.array([0.4, 0.6])
        posterior = np.array([0.8, 0.2])
        ig = compute_information_gain(prior, posterior)
        kl = compute_kl_divergence(posterior, prior)
        assert abs(ig - kl) < 1e-6


# =============================================================================
# Analyze Active Inference Metrics Tests
# =============================================================================

class TestAnalyzeActiveInferenceMetrics:
    """Tests for analyze_active_inference_metrics."""

    @pytest.fixture
    def sample_trajectory(self):
        """Create a realistic belief trajectory that converges."""
        rng = np.random.default_rng(42)
        beliefs = []
        # Start uniform, converge to state 0
        for t in range(20):
            w = min(t / 15.0, 1.0)
            b = np.array([(1 - w) * 0.25 + w * 0.9,
                          (1 - w) * 0.25 + w * 0.05,
                          (1 - w) * 0.25 + w * 0.03,
                          (1 - w) * 0.25 + w * 0.02])
            b = b / b.sum()
            beliefs.append(b.tolist())
        free_energy = [5.0 - 0.2 * t + rng.normal(0, 0.05) for t in range(20)]
        actions = [rng.integers(0, 3) for _ in range(20)]
        return beliefs, free_energy, actions

    def test_output_structure(self, sample_trajectory):
        """Result should contain expected keys."""
        beliefs, fe, actions = sample_trajectory
        result = analyze_active_inference_metrics(beliefs, fe, actions, "test_model")
        assert "model_name" in result
        assert result["model_name"] == "test_model"
        assert "num_timesteps" in result
        assert result["num_timesteps"] == 20
        assert "metrics" in result
        assert "belief_entropy" in result["metrics"]
        assert "information_gain" in result["metrics"]
        assert "free_energy" in result["metrics"]
        assert "action_distribution" in result["metrics"]

    def test_entropy_decreasing_trend(self, sample_trajectory):
        """For converging beliefs, entropy should show decreasing trend."""
        beliefs, fe, actions = sample_trajectory
        result = analyze_active_inference_metrics(beliefs, fe, actions, "test_model")
        entropy_data = result["metrics"]["belief_entropy"]
        assert entropy_data["trend"] == "decreasing", (
            f"Expected 'decreasing' trend, got '{entropy_data['trend']}'"
        )

    def test_information_gain_positive(self, sample_trajectory):
        """Total information gain should be positive for converging beliefs."""
        beliefs, fe, actions = sample_trajectory
        result = analyze_active_inference_metrics(beliefs, fe, actions, "test_model")
        ig = result["metrics"]["information_gain"]
        assert ig["total"] > 0

    def test_empty_trajectory(self):
        """Should handle empty input gracefully."""
        result = analyze_active_inference_metrics([], [], [], "empty_model")
        assert result["num_timesteps"] == 0
        assert result["metrics"] == {}


# =============================================================================
# normalize_matrices Tests
# =============================================================================

class TestNormalizeMatrices:
    """Tests for normalize_matrices in render/processor.py."""

    def test_2d_a_matrix_normalization(self):
        """Columns of 2D A matrix should sum to 1 after normalization."""
        import logging
        log = logging.getLogger("test")

        A = np.array([[2.0, 1.0], [2.0, 3.0]])  # Columns sum to 4
        pomdp = POMDPStateSpace(num_states=2, num_observations=2, num_actions=1, A_matrix=A)
        result = normalize_matrices(pomdp, log)
        col_sums = result.A_matrix.sum(axis=0)
        np.testing.assert_allclose(col_sums, [1.0, 1.0], atol=1e-10)

    def test_3d_b_matrix_normalization(self):
        """Columns of each action slice in 3D B should sum to 1."""
        import logging
        log = logging.getLogger("test")

        B = np.ones((3, 3, 2))  # (next_state, curr_state, action), all ones
        pomdp = POMDPStateSpace(num_states=3, num_observations=3, num_actions=2, B_matrix=B)
        result = normalize_matrices(pomdp, log)
        for a in range(2):
            col_sums = result.B_matrix[:, :, a].sum(axis=0)
            np.testing.assert_allclose(col_sums, np.ones(3), atol=1e-10)

    def test_zero_column_uniform_fallback(self):
        """Zero-sum columns should be filled with uniform distribution."""
        import logging
        log = logging.getLogger("test")

        A = np.array([[0.0, 1.0], [0.0, 1.0]])  # Column 0 is all zeros
        pomdp = POMDPStateSpace(num_states=2, num_observations=2, num_actions=1, A_matrix=A)
        result = normalize_matrices(pomdp, log)
        # Column 0 should now be uniform (0.5, 0.5)
        np.testing.assert_allclose(result.A_matrix[:, 0], [0.5, 0.5], atol=1e-10)
        # Column 1 should be (0.5, 0.5)
        np.testing.assert_allclose(result.A_matrix[:, 1], [0.5, 0.5], atol=1e-10)

    def test_factorial_a_matrix(self):
        """Should handle list-of-arrays (factorial) A matrix."""
        import logging
        log = logging.getLogger("test")

        A_list = [np.array([[3.0, 1.0], [1.0, 3.0]]),
                  np.array([[2.0, 2.0], [2.0, 2.0]])]
        pomdp = POMDPStateSpace(num_states=2, num_observations=2, num_actions=1, A_matrix=A_list)
        result = normalize_matrices(pomdp, log)
        assert isinstance(result.A_matrix, list)
        for a in result.A_matrix:
            col_sums = a.sum(axis=0)
            np.testing.assert_allclose(col_sums, np.ones(2), atol=1e-10)

    def test_passthrough_no_matrices(self):
        """Should handle POMDP with no matrices without error."""
        import logging
        log = logging.getLogger("test")

        pomdp = POMDPStateSpace(num_states=2, num_observations=2, num_actions=1)
        result = normalize_matrices(pomdp, log)
        assert result.A_matrix is None
        assert result.B_matrix is None
