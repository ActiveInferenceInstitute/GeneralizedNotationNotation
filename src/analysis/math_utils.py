"""
Active Inference mathematical utility functions.

Provides Shannon entropy, KL divergence, variational free energy,
expected free energy, information gain, and comprehensive Active Inference
metrics analysis.

Extracted from post_simulation.py for maintainability.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def compute_shannon_entropy(distribution: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.

    Args:
        distribution: Probability distribution (must sum to 1)

    Returns:
        Shannon entropy in nats
    """
    # Ensure valid probability distribution
    p = np.asarray(distribution, dtype=np.float64)
    p = np.clip(p, 1e-10, 1.0)
    p = p / np.sum(p)  # Normalize
    return float(-np.sum(p * np.log(p + 1e-10)))


def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL divergence D_KL(P || Q).

    Args:
        p: First probability distribution (P)
        q: Second probability distribution (Q)

    Returns:
        KL divergence in nats
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Ensure valid probability distributions
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)

    return float(np.sum(p * np.log((p + 1e-10) / (q + 1e-10))))


def compute_variational_free_energy(
    observations: np.ndarray,
    beliefs: np.ndarray,
    A_matrix: np.ndarray,
    prior: Optional[np.ndarray] = None
) -> float:
    """
    Compute variational free energy: F = E_q[ln q(s)] - E_q[ln p(o,s)]

    The variational free energy consists of:
    - Energy: -E_q[ln p(o|s)] - E_q[ln p(s)]
    - Entropy: -E_q[ln q(s)]

    Args:
        observations: Current observations
        beliefs: Current belief distribution over states q(s)
        A_matrix: Observation likelihood matrix p(o|s)
        prior: Prior distribution p(s), defaults to uniform

    Returns:
        Variational free energy value
    """
    q_s = np.asarray(beliefs, dtype=np.float64)
    q_s = np.clip(q_s, 1e-10, 1.0)
    q_s = q_s / np.sum(q_s)

    if prior is None:
        prior = np.ones_like(q_s) / len(q_s)
    prior = np.clip(prior, 1e-10, 1.0)
    prior = prior / np.sum(prior)

    # Entropy term: -E_q[ln q(s)]
    entropy = -np.sum(q_s * np.log(q_s + 1e-10))

    # Prior term: E_q[ln p(s)]
    prior_term = np.sum(q_s * np.log(prior + 1e-10))

    # Likelihood term: E_q[ln p(o|s)]
    A = np.asarray(A_matrix, dtype=np.float64)
    if A.ndim == 2 and len(observations) <= A.shape[0]:
        # Compute expected log likelihood
        log_likelihood = 0.0
        for s_idx in range(len(q_s)):
            if s_idx < A.shape[1]:
                obs_prob = A[:, s_idx]
                obs_prob = np.clip(obs_prob, 1e-10, 1.0)
                obs_prob = obs_prob / np.sum(obs_prob)
                log_likelihood += q_s[s_idx] * np.sum(np.log(obs_prob + 1e-10))
    else:
        log_likelihood = 0.0

    # F = -Entropy - Prior_term - Likelihood_term
    # F = E_q[ln q(s)] - E_q[ln p(o,s)]
    free_energy = -entropy - prior_term - log_likelihood

    return float(free_energy)


def compute_expected_free_energy(
    beliefs: np.ndarray,
    A_matrix: np.ndarray,
    B_matrix: np.ndarray,
    C_vector: np.ndarray,
    policy: int,
    horizon: int = 1
) -> float:
    """
    Compute expected free energy G for a given policy.

    G = E_pi[D_KL(q(o|pi) || p(o))] + E_pi[H[p(o|s)]]

    This combines:
    - Epistemic value: Information gain about hidden states
    - Pragmatic value: Expected utility/preference satisfaction

    Args:
        beliefs: Current belief distribution q(s)
        A_matrix: Observation likelihood matrix p(o|s)
        B_matrix: Transition matrix p(s'|s,a) - 3D array [s', s, a]
        C_vector: Preference distribution (log preferences)
        policy: Action index
        horizon: Planning horizon (default 1)

    Returns:
        Expected free energy value
    """
    q_s = np.asarray(beliefs, dtype=np.float64)
    q_s = np.clip(q_s, 1e-10, 1.0)
    q_s = q_s / np.sum(q_s)

    A = np.asarray(A_matrix, dtype=np.float64)
    B = np.asarray(B_matrix, dtype=np.float64)
    C = np.asarray(C_vector, dtype=np.float64)

    # Predict next state distribution under policy
    if B.ndim == 3 and policy < B.shape[2]:
        B_policy = B[:, :, policy]
    elif B.ndim == 2:
        B_policy = B
    else:
        B_policy = np.eye(len(q_s))

    q_s_next = B_policy @ q_s
    q_s_next = np.clip(q_s_next, 1e-10, 1.0)
    q_s_next = q_s_next / np.sum(q_s_next)

    # Predicted observation distribution
    if A.ndim == 2:
        q_o = A @ q_s_next
    else:
        q_o = np.ones(A.shape[0]) / A.shape[0]
    q_o = np.clip(q_o, 1e-10, 1.0)
    q_o = q_o / np.sum(q_o)

    # Pragmatic value: negative cross-entropy with preferences
    # C is log preferences, so p(o) proportional to exp(C)
    p_o_preferred = np.exp(C - np.max(C))
    p_o_preferred = p_o_preferred / np.sum(p_o_preferred)
    pragmatic_value = -np.sum(q_o * np.log(p_o_preferred + 1e-10))

    # Epistemic value: expected information gain
    # Approximate as entropy of predicted states
    epistemic_value = compute_shannon_entropy(q_s_next)

    # G = pragmatic + epistemic (both are "costs" to minimize)
    G = pragmatic_value + epistemic_value

    return float(G)


def compute_information_gain(
    prior_beliefs: np.ndarray,
    posterior_beliefs: np.ndarray
) -> float:
    """
    Compute information gain from prior to posterior beliefs.

    IG = D_KL(posterior || prior)

    Args:
        prior_beliefs: Prior belief distribution
        posterior_beliefs: Posterior belief distribution

    Returns:
        Information gain in nats
    """
    return compute_kl_divergence(posterior_beliefs, prior_beliefs)


def analyze_active_inference_metrics(
    beliefs_trajectory: List[List[float]],
    free_energy_trajectory: List[float],
    actions: List[int],
    model_name: str
) -> Dict[str, Any]:
    """
    Compute comprehensive Active Inference metrics from simulation data.

    Args:
        beliefs_trajectory: Belief distributions over time
        free_energy_trajectory: Free energy values over time
        actions: Actions taken over time
        model_name: Name of the model

    Returns:
        Dictionary with Active Inference analysis metrics
    """
    analysis = {
        "model_name": model_name,
        "num_timesteps": len(beliefs_trajectory),
        "metrics": {}
    }

    if not beliefs_trajectory:
        return analysis

    # Convert to numpy arrays
    beliefs_array = np.array(beliefs_trajectory)

    # Belief entropy over time
    entropy_trajectory = [compute_shannon_entropy(b) for b in beliefs_trajectory]
    analysis["metrics"]["belief_entropy"] = {
        "trajectory": entropy_trajectory,
        "mean": float(np.mean(entropy_trajectory)),
        "std": float(np.std(entropy_trajectory)),
        "final": entropy_trajectory[-1] if entropy_trajectory else 0.0,
        "trend": "decreasing" if len(entropy_trajectory) > 1 and entropy_trajectory[-1] < entropy_trajectory[0] else "stable"
    }

    # Information gain between consecutive timesteps
    if len(beliefs_trajectory) > 1:
        info_gain = []
        for t in range(1, len(beliefs_trajectory)):
            ig = compute_information_gain(
                np.array(beliefs_trajectory[t - 1]),
                np.array(beliefs_trajectory[t])
            )
            info_gain.append(ig)

        analysis["metrics"]["information_gain"] = {
            "trajectory": info_gain,
            "total": float(np.sum(info_gain)),
            "mean": float(np.mean(info_gain)),
            "peak_timestep": int(np.argmax(info_gain)) + 1
        }

    # Free energy analysis
    if free_energy_trajectory:
        fe_array = np.array(free_energy_trajectory)
        analysis["metrics"]["free_energy"] = {
            "trajectory": list(fe_array),
            "initial": float(fe_array[0]),
            "final": float(fe_array[-1]),
            "min": float(np.min(fe_array)),
            "reduction": float(fe_array[0] - fe_array[-1]) if len(fe_array) > 1 else 0.0,
            "converged": bool(np.std(fe_array[-5:]) < 0.01) if len(fe_array) >= 5 else False
        }

    # Action analysis
    if actions:
        action_counts = {}
        for a in actions:
            action_counts[str(a)] = action_counts.get(str(a), 0) + 1
        analysis["metrics"]["action_distribution"] = action_counts
        analysis["metrics"]["action_entropy"] = compute_shannon_entropy(
            np.array(list(action_counts.values()))
        )

    # Belief certainty (1 - entropy normalized)
    max_entropy = np.log(beliefs_array.shape[1]) if beliefs_array.shape[1] > 1 else 1.0
    certainty_trajectory = [1.0 - (e / max_entropy) for e in entropy_trajectory]
    analysis["metrics"]["certainty"] = {
        "trajectory": certainty_trajectory,
        "mean": float(np.mean(certainty_trajectory)),
        "final": certainty_trajectory[-1] if certainty_trajectory else 0.0
    }

    return analysis
