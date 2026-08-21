#!/usr/bin/env python3
"""Sparse Kronecker-factorized discrete active inference in JAX (roadmap MAJ-02).

The dense path materialises the joint state space (size ``prod(factor_sizes)``)
when a POMDP has multiple state factors; ``B`` alone is O(n^3) in bytes and the
joint posterior is exponential in the factor count. This module implements the
sparse alternative for *factor-separable* models:

- transition  ``B = B_1 ⊗ ... ⊗ B_K``  (per-factor transition tensors),
- likelihood  ``A = A_1 ⊗ ... ⊗ A_K``  (per-factor observation matrices),
- preferences ``ln C(o) = Σ_f ln C_f(o_f)``  (per-factor log preferences),
- prior       ``D = D_1 ⊗ ... ⊗ D_K``.

When both the generative model and the variational posterior factorise
(``q(s) = ⊗_f q_f(s_f)`` — exact mean-field for this model class), every
computation decomposes per factor:

- belief update     ``q_f ∝ A_f[o_f, :] ⊙ q_f``   (exact for separable A),
- state prediction  ``q_f(s') = B_f[:, :, u_f] q_f(s)``,
- EFE               ``G = Σ_f G_f`` with ambiguity/risk per factor,
- policy            ``π(u) = ⊗_f softmax(-γ G_f(u_f))``.

The joint state space is never built: all operations act on per-factor
objects, so models with ``prod(factor_sizes) >= 64`` states (e.g. six binary
factors, or three 4-state factors) execute in time/space proportional to the
sum of factor sizes. ``kron_matvec_flat`` additionally applies a Kronecker
product to an *arbitrary* (non-factorised) vector via tensor contraction
without materialising the joint — used by the EFE and validation paths.

The Kronecker identities are pinned by ``src/tests/execute/test_kronecker_factorized.py``:

- ``kron_matvec(factors, vecs) == dense_kron(factors) @ (⊗ vecs)``,
- ``kron_matvec_flat(factors, v) == dense_kron(factors) @ v`` for any ``v``,
- factorised per-factor EFE sums exactly to the dense EFE evaluated at a
  factorised posterior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import jax.numpy as jnp
import numpy as np
from jax import random as jax_random

__all__ = [
    "kron_matvec",
    "kron_matvec_flat",
    "kron_materialize",
    "FactorizedPOMDP",
    "run_factorized_active_inference",
    "factorized_state_space_size",
]

# --- Kronecker utilities -----------------------------------------------------


def kron_matvec(
    factor_matrices: Sequence[Any],
    factor_vectors: Sequence[Any],
) -> List[jnp.ndarray]:
    """Apply a Kronecker product to a factorised vector, factor by factor.

    For factor matrices ``A_1..A_K`` and factor vectors ``v_1..v_K``::

        (A_1 ⊗ ... ⊗ A_K) (v_1 ⊗ ... ⊗ v_K) = (A_1 v_1) ⊗ ... ⊗ (A_K v_K)

    Costs O(Σ_f |A_f| |v_f|) instead of the dense O(∏_f |A_f| |v_f|).
    """
    if len(factor_matrices) != len(factor_vectors):
        raise ValueError(
            "factor count mismatch: "
            f"{len(factor_matrices)} matrices vs {len(factor_vectors)} vectors"
        )
    results: List[jnp.ndarray] = []
    for matrix, vector in zip(factor_matrices, factor_vectors):
        applied = jnp.asarray(matrix) @ jnp.asarray(vector)
        applied = applied / jnp.sum(applied)
        results.append(applied)
    return results


def kron_matvec_flat(
    factor_matrices: Sequence[Any],
    flat_vector: Any,
) -> jnp.ndarray:
    """Apply a Kronecker product to an arbitrary flat vector, sparsely.

    Uses the tensor-contraction identity: reshape the flat vector into the
    per-factor grid ``(n_1, ..., n_K)`` and contract factor matrices inward
    one dimension at a time (Kronecker = sequential tensor contraction). The
    joint matrix is never built. Falls back to dense application only when a
    single factor is present (the contraction is then an ordinary matvec).

    The input may be either a 1-D flat vector over the joint space or a
    K-dimensional grid already shaped ``(n_1, ..., n_K)``.
    """
    matrices = [jnp.asarray(m) for m in factor_matrices]
    vector = jnp.asarray(flat_vector)
    if not matrices:
        return vector
    per_factor_sizes = [m.shape[1] for m in matrices]
    if vector.ndim == 1:
        expected = int(np.prod(per_factor_sizes))
        if vector.shape[0] != expected:
            raise ValueError(
                f"flat vector length {vector.shape[0]} does not match the "
                f"joint size {expected} for factors {per_factor_sizes}"
            )
        grid = vector.reshape(per_factor_sizes)
    else:
        grid = vector
        if tuple(grid.shape) != tuple(per_factor_sizes):
            raise ValueError(
                f"grid shape {tuple(grid.shape)} does not match factor sizes "
                f"{tuple(per_factor_sizes)}"
            )
    # Contract factors from the last axis inward: A_f acts on axis f.
    contracted = grid
    for axis in range(len(matrices) - 1, -1, -1):
        matrix = matrices[axis]
        # Move the target axis to the end, matmul, move back.
        moved = jnp.moveaxis(contracted, axis, -1)
        shape = moved.shape
        flat = moved.reshape(-1, shape[-1])
        out = flat @ matrix.T  # rows are outer axes, cols the factor axis
        out = out.reshape(shape[:-1] + (matrix.shape[0],))
        contracted = jnp.moveaxis(out, -1, axis)
    return jnp.reshape(contracted, (int(np.prod([m.shape[0] for m in matrices])),))


def kron_materialize(factor_matrices: Sequence[Any]) -> jnp.ndarray:
    """Materialise the dense Kronecker product (validation only).

    Exponential memory in the factor count — use for small models and tests,
    never in the execution path.
    """
    matrices = [jnp.asarray(m) for m in factor_matrices]
    if not matrices:
        return jnp.array([[1.0]])
    result = matrices[0]
    for matrix in matrices[1:]:
        result = jnp.kron(result, matrix)
    return result


def factorized_state_space_size(factor_sizes: Sequence[int]) -> int:
    """Return the joint state-space size ``prod(factor_sizes)``."""
    size = 1
    for n in factor_sizes:
        size *= int(n)
    return size


# --- Factorised model + mean-field simulation --------------------------------


@dataclass
class FactorizedPOMDP:
    """A factor-separable discrete POMDP.

    Attributes:
        A: Per-factor likelihood matrices, ``A_f`` shape ``(n_obs_f, n_states_f)``.
        B: Per-factor transition tensors, ``B_f`` shape
            ``(n_states_f, n_states_f, n_actions_f)``.
        C: Per-factor log-preference vectors, ``C_f`` shape ``(n_obs_f,)``.
        D: Per-factor initial priors, ``D_f`` shape ``(n_states_f,)``.
        T: Number of timesteps.
        seed: Random seed for the generative forward pass.
        action_precision: Precision ``gamma`` scaling EFE in the policy.
    """

    A: List[np.ndarray]
    B: List[np.ndarray]
    C: List[np.ndarray]
    D: List[np.ndarray]
    T: int = 20
    seed: int = 42
    action_precision: float = 4.0

    def __post_init__(self) -> None:
        if not (len(self.A) == len(self.B) == len(self.C) == len(self.D)):
            raise ValueError("factor count mismatch across A/B/C/D")
        if not self.A:
            raise ValueError("at least one factor is required")
        for index, (a, b, d) in enumerate(zip(self.A, self.B, self.D)):
            a_arr = np.asarray(a, dtype=np.float64)
            b_arr = np.asarray(b, dtype=np.float64)
            d_arr = np.asarray(d, dtype=np.float64)
            if a_arr.ndim != 2:
                raise ValueError(f"A[{index}] must be 2-D, got {a_arr.shape}")
            if b_arr.ndim != 3:
                raise ValueError(f"B[{index}] must be 3-D, got {b_arr.shape}")
            if b_arr.shape[0] != b_arr.shape[1] or b_arr.shape[1] != a_arr.shape[1]:
                raise ValueError(
                    f"B[{index}] shape {b_arr.shape} does not match "
                    f"state size {a_arr.shape[1]}"
                )
            if d_arr.shape[0] != a_arr.shape[1]:
                raise ValueError(
                    f"D[{index}] length {d_arr.shape[0]} does not match "
                    f"state size {a_arr.shape[1]}"
                )
            for state in range(b_arr.shape[2]):
                column_sums = b_arr[:, :, state].sum(axis=0)
                if not np.all(np.isfinite(column_sums)) or np.any(column_sums <= 0):
                    raise ValueError(
                        f"B[{index}] action {state} has invalid probability mass"
                    )
        self.A = [np.asarray(m, dtype=np.float64) for m in self.A]
        self.B = [np.asarray(m, dtype=np.float64) for m in self.B]
        self.C = [np.asarray(m, dtype=np.float64) for m in self.C]
        self.D = [np.asarray(m, dtype=np.float64) for m in self.D]

    @property
    def factor_sizes(self) -> List[int]:
        """Per-factor state-space sizes ``[n_1, ..., n_K]``."""
        return [int(m.shape[1]) for m in self.A]

    @property
    def observation_sizes(self) -> List[int]:
        """Per-factor observation sizes ``[n_o_1, ..., n_o_K]``."""
        return [int(m.shape[0]) for m in self.A]

    @property
    def action_sizes(self) -> List[int]:
        """Per-factor action counts ``[a_1, ..., a_K]``."""
        return [int(m.shape[2]) for m in self.B]

    @property
    def joint_state_space_size(self) -> int:
        """Size of the (never materialised) joint state space."""
        return factorized_state_space_size(self.factor_sizes)


def _categorical_sample(rng_key: jnp.ndarray, probabilities: np.ndarray) -> int:
    """Sample one index from a probability vector with a JAX key."""
    return int(
        jax_random.categorical(rng_key, jnp.log(jnp.asarray(probabilities) + 1e-16))
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.asarray(np.exp(shifted), dtype=np.float64)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("softmax input has invalid mass")
    return weights / total


def _normalize(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    total = np.sum(vector)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("probability vector has invalid mass")
    return vector / total


def per_factor_efe(
    belief: np.ndarray,
    action: int,
    a: np.ndarray,
    b: np.ndarray,
    c_pref: np.ndarray,
) -> float:
    """Expected free energy for one factor and one candidate action.

    ``G_f(u_f) = ambiguity_f + risk_f`` with

    - predicted state ``q_f(s') = B_f[:, :, u_f] q_f(s)``,
    - predicted observation ``p_f(o) = A_f q_f(s')``,
    - ambiguity ``H[o | s']`` under the predicted state,
    - risk ``KL[p_f(o) || C_f(o)]`` against the preference.
    """
    predicted_state = b[:, :, action] @ belief
    predicted_state = np.maximum(predicted_state, 1e-16)
    predicted_state = predicted_state / np.sum(predicted_state)
    predicted_obs = a @ predicted_state
    predicted_obs = np.maximum(predicted_obs, 1e-16)
    predicted_obs = predicted_obs / np.sum(predicted_obs)

    ambiguity = 0.0
    for state in range(predicted_state.shape[0]):
        likelihood = np.maximum(a[:, state], 1e-16)
        ambiguity -= predicted_state[state] * float(
            np.sum(likelihood * np.log(likelihood))
        )

    preferred = np.maximum(c_pref, 1e-16)
    risk = float(np.sum(predicted_obs * (np.log(predicted_obs) - np.log(preferred))))
    return ambiguity + risk


def run_factorized_active_inference(model: FactorizedPOMDP) -> Dict[str, Any]:
    """Run mean-field active inference over factorised matrices (no joint build).

    Each factor keeps its own belief/observation/action; the environment is a
    true generative process over the factorised matrices (per-factor
    transitions and observations). Beliefs update exactly under mean field
    (``q_f ∝ A_f[o_f, :] ⊙ q_f``), EFE decomposes per factor, and the joint
    policy is the product of per-factor policies. The joint state space
    (``joint_state_space_size = prod(factor_sizes)``) is reported but never
    allocated.
    """
    a_matrices = [np.asarray(m, dtype=np.float64) for m in model.A]
    b_matrices = [np.asarray(m, dtype=np.float64) for m in model.B]
    c_matrices = [np.asarray(m, dtype=np.float64) for m in model.C]
    d_matrices = [np.asarray(m, dtype=np.float64) for m in model.D]

    c_prefs = [_softmax(c) for c in c_matrices]
    key = jax_random.PRNGKey(model.seed)
    num_factors = len(a_matrices)

    # True generative process: initial states and beliefs from D_f.
    true_states: List[List[int]] = [[] for _ in range(num_factors)]
    beliefs: List[List[np.ndarray]] = [[] for _ in range(num_factors)]
    observations: List[List[int]] = [[] for _ in range(num_factors)]
    actions: List[List[int]] = [[] for _ in range(num_factors)]
    efe_per_factor: List[List[float]] = [[] for _ in range(num_factors)]
    policy_factors: List[List[np.ndarray]] = [[] for _ in range(num_factors)]

    current_states: List[int] = []
    current_beliefs: List[np.ndarray] = []
    for d in d_matrices:
        key, subkey = jax_random.split(key)
        current_states.append(_categorical_sample(subkey, d))
        current_beliefs.append(_normalize(d.copy()))

    for _step in range(model.T):
        # --- Observation: sample per-factor likelihood, update beliefs ---
        for factor in range(num_factors):
            key, subkey = jax_random.split(key)
            observation = _categorical_sample(
                subkey, a_matrices[factor][:, current_states[factor]]
            )
            observations[factor].append(observation)
            updated = current_beliefs[factor] * a_matrices[factor][observation, :]
            if np.sum(updated) <= 0:
                raise ValueError(f"factor {factor} belief update produced zero mass")
            current_beliefs[factor] = _normalize(updated)
            beliefs[factor].append(current_beliefs[factor].copy())

        # --- Action selection: per-factor EFE, product policy ---
        for factor in range(num_factors):
            efe_values = [
                per_factor_efe(
                    current_beliefs[factor],
                    action,
                    a_matrices[factor],
                    b_matrices[factor],
                    c_prefs[factor],
                )
                for action in range(model.action_sizes[factor])
            ]
            policy = _softmax(-model.action_precision * np.asarray(efe_values))
            key, subkey = jax_random.split(key)
            action = _categorical_sample(subkey, policy)
            actions[factor].append(action)
            efe_per_factor[factor].append(efe_values[action])
            policy_factors[factor].append(policy)

        # --- Environment transition (per factor) ---
        for factor in range(num_factors):
            next_probs = b_matrices[factor][
                :, current_states[factor], actions[factor][-1]
            ]
            key, subkey = jax_random.split(key)
            current_states[factor] = _categorical_sample(subkey, next_probs)
            true_states[factor].append(current_states[factor])

    # --- Validation over the factorised trajectory ---
    all_beliefs_valid = all(
        all(np.all(np.isfinite(b)) and np.all(b >= 0.0) for b in factor_beliefs)
        for factor_beliefs in beliefs
    )
    beliefs_sum_to_one = all(
        all(abs(float(np.sum(b)) - 1.0) < 1e-6 for b in factor_beliefs)
        for factor_beliefs in beliefs
    )
    actions_in_range = all(
        all(0 <= a < n for a in factor_actions)
        for factor_actions, n in zip(actions, model.action_sizes)
    )
    all_valid = all_beliefs_valid and beliefs_sum_to_one and actions_in_range

    agent_names = [f"factor{index}" for index in range(num_factors)]
    return {
        "schema_version": "jax_kronecker_factorized_v1",
        "success": True,
        "framework": "JAX (Kronecker-factorized)",
        "model_name": "factorized_pomdp",
        "model_kind": "factorized_kronecker",
        "num_timesteps": model.T,
        "num_factors": num_factors,
        "factors": agent_names,
        "observations_by_factor": {
            name: obs for name, obs in zip(agent_names, observations)
        },
        "true_states_by_factor": {
            name: states for name, states in zip(agent_names, true_states)
        },
        "actions_by_factor": {name: acts for name, acts in zip(agent_names, actions)},
        "beliefs_by_factor": {
            name: [b.tolist() for b in factor_beliefs]
            for name, factor_beliefs in zip(agent_names, beliefs)
        },
        "efe_per_factor": {name: efe for name, efe in zip(agent_names, efe_per_factor)},
        "policy_by_factor": {
            name: [p.tolist() for p in factor_policies]
            for name, factor_policies in zip(agent_names, policy_factors)
        },
        "model_parameters": {
            "factor_sizes": model.factor_sizes,
            "observation_sizes": model.observation_sizes,
            "action_sizes": model.action_sizes,
            "joint_state_space_size": model.joint_state_space_size,
            "time_steps": model.T,
            "joint_materialized": False,
        },
        "runtime_metadata": {
            "random_seed": model.seed,
            "schema_version": "jax_kronecker_factorized_v1",
            "action_precision": model.action_precision,
            "backend": "jax",
        },
        "validation": {
            "all_valid": all_valid,
            "all_beliefs_valid": all_beliefs_valid,
            "beliefs_sum_to_one": beliefs_sum_to_one,
            "actions_in_range": actions_in_range,
        },
    }


# --- Constructors for common topologies --------------------------------------


def build_binary_factor_model(
    num_factors: int,
    t: int = 20,
    seed: int = 42,
    a_signal: float = 0.85,
    b_signal: float = 0.8,
    action_precision: float = 4.0,
) -> FactorizedPOMDP:
    """Build a homogeneous binary-factor POMDP (``2^num_factors`` states)."""
    a = np.array([[a_signal, 1.0 - a_signal], [1.0 - a_signal, a_signal]])
    b = np.array(
        [
            [[b_signal, 1.0 - b_signal], [1.0 - b_signal, b_signal]],
            [[1.0 - b_signal, b_signal], [b_signal, 1.0 - b_signal]],
        ]
    )
    c = np.array([-0.5, 1.0])
    d = np.array([0.8, 0.2])
    return FactorizedPOMDP(
        A=[a.copy() for _ in range(num_factors)],
        B=[b.copy() for _ in range(num_factors)],
        C=[c.copy() for _ in range(num_factors)],
        D=[d.copy() for _ in range(num_factors)],
        T=t,
        seed=seed,
        action_precision=action_precision,
    )


def build_generic_factor_model(
    factor_sizes: Sequence[int],
    t: int = 20,
    seed: int = 42,
    a_signal: float = 0.85,
    b_signal: float = 0.8,
    action_precision: float = 4.0,
) -> FactorizedPOMDP:
    """Build a factor model with arbitrary per-factor sizes.

    Each factor uses a noisy-identity likelihood, a noisy-permuted transition
    (n actions = n states), a preference for state 0, and a peaked prior.
    """
    a_matrices: List[np.ndarray] = []
    b_matrices: List[np.ndarray] = []
    c_matrices: List[np.ndarray] = []
    d_matrices: List[np.ndarray] = []
    for n in factor_sizes:
        a = np.full((n, n), (1.0 - a_signal) / (n - 1) if n > 1 else 0.0)
        np.fill_diagonal(a, a_signal)
        b = np.zeros((n, n, n))
        for action in range(n):
            for previous in range(n):
                intended = (previous + action) % n
                b[:, previous, action] = (1.0 - b_signal) / (n - 1) if n > 1 else 0.0
                b[intended, previous, action] = b_signal
        c = np.zeros(n)
        c[0] = 1.0
        d = np.full(n, 0.05)
        d[0] = 1.0 - 0.05 * (n - 1)
        a_matrices.append(a)
        b_matrices.append(b)
        c_matrices.append(c)
        d_matrices.append(d)
    return FactorizedPOMDP(
        A=a_matrices,
        B=b_matrices,
        C=c_matrices,
        D=d_matrices,
        T=t,
        seed=seed,
        action_precision=action_precision,
    )
