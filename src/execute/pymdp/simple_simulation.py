#!/usr/bin/env python3
"""
Simple PyMDP 1.0.0 simulation for GNN POMDP specifications.

pymdp 1.0.0 (https://github.com/infer-actively/pymdp) is a JAX-first rewrite.
The Agent accepts batched ``list[jax.Array]`` models and the public loop is:

    qs, info       = agent.infer_states(obs, empirical_prior=prior, return_info=True)
    q_pi, neg_efe  = agent.infer_policies(qs)
    action         = agent.sample_action(q_pi, rng_key=keys)
    prior          = agent.update_empirical_prior(action, qs)

This module converts a GNN specification (whose matrices are plain nested
lists / numpy arrays) into the pymdp 1.0.0 batched list-of-arrays format,
runs a rollout, and writes a ``simulation_results.json`` file that the
analysis step (``src/analysis/pymdp/``) consumes.

Architecture note:
    This module is part of EXECUTE (step 12). It ONLY runs simulations and
    logs raw data. ALL visualisations belong to ANALYSIS (step 16).
    Flow: Render (prepare scripts) → Execute (run + log raw) → Analysis (plot).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GNN matrix normalisation helpers (pure numpy; framework-agnostic)
# ---------------------------------------------------------------------------

def _normalise_prob_vector(v: np.ndarray) -> np.ndarray:
    """Return a 1-D probability vector summing to 1 (robust to GNN rounding)."""
    arr = np.asarray(v, dtype=np.float64).flatten()
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0:
        n = max(len(arr), 1)
        return np.ones(n, dtype=np.float64) / n
    return arr / total


def _normalise_columns(mat: np.ndarray) -> np.ndarray:
    """Normalise each column so it sums to 1 (fallback to uniform)."""
    out = np.asarray(mat, dtype=np.float64).copy()
    if out.ndim != 2:
        raise ValueError(f"_normalise_columns expected 2D, got {out.ndim}D")
    col_sums = out.sum(axis=0, keepdims=True)
    zero_cols = col_sums <= 0
    col_sums = np.where(zero_cols, 1.0, col_sums)
    out = out / col_sums
    if zero_cols.any():
        rows = out.shape[0]
        for j in np.where(zero_cols.flatten())[0]:
            out[:, j] = 1.0 / rows
    return out


def _canonicalise_A(A_data: Any, fallback_shape: Tuple[int, int]) -> np.ndarray:
    """Return an A matrix of shape ``(num_obs, num_states)`` with columns summing to 1."""
    if A_data is None:
        num_obs, num_states = fallback_shape
        return _normalise_columns(np.eye(num_obs, num_states) * 0.9 + 0.1 / max(num_obs, 1))
    mat = np.asarray(A_data, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"A matrix must be 2D (num_obs, num_states); got shape {mat.shape}")
    return _normalise_columns(mat)


def _canonicalise_B(B_data: Any, num_states: int, num_actions: int) -> np.ndarray:
    """
    Return a B tensor with PyMDP shape ``(next_state, prev_state, action)``.

    Accepts GNN raw formats:
      * 3-D ``(action, prev, next)`` — transposed to ``(next, prev, action)``
      * 3-D ``(next, prev, action)`` — kept as-is if shape matches
      * 2-D ``(next, prev)`` — promoted to single-action ``(next, prev, 1)``
      * ``None`` — identity per action
    Each slice ``B[:, :, a]`` is column-normalised.
    """
    if B_data is None:
        tensor = np.stack([np.eye(num_states) for _ in range(max(num_actions, 1))], axis=-1)
        return tensor

    raw = np.asarray(B_data, dtype=np.float64)

    if raw.ndim == 2:
        tensor = raw[:, :, np.newaxis]
    elif raw.ndim == 3:
        # If leading dim matches num_actions, assume GNN (action, prev, next)
        if raw.shape[0] == num_actions and raw.shape[1] == raw.shape[2]:
            tensor = raw.transpose(2, 1, 0)
        elif raw.shape[-1] == num_actions and raw.shape[0] == raw.shape[1]:
            tensor = raw
        else:
            # Best-effort: treat as (next, prev, action)
            tensor = raw
    else:
        raise ValueError(f"B matrix must be 2D or 3D; got shape {raw.shape}")

    # Normalise each action slice by column (over next_state).
    tensor = tensor.copy()
    for a in range(tensor.shape[2]):
        tensor[:, :, a] = _normalise_columns(tensor[:, :, a])
    return tensor


def _canonicalise_C(C_data: Any, num_obs: int) -> np.ndarray:
    if C_data is None:
        return np.zeros(num_obs, dtype=np.float64)
    vec = np.asarray(C_data, dtype=np.float64).flatten()
    if vec.shape[0] != num_obs:
        padded = np.zeros(num_obs, dtype=np.float64)
        k = min(num_obs, vec.shape[0])
        padded[:k] = vec[:k]
        vec = padded
    return vec


def _canonicalise_D(D_data: Any, num_states: int) -> np.ndarray:
    if D_data is None:
        return np.ones(num_states, dtype=np.float64) / max(num_states, 1)
    return _normalise_probability_vector_safe(D_data, num_states)


def _normalise_probability_vector_safe(v: Any, expected_len: int) -> np.ndarray:
    vec = np.asarray(v, dtype=np.float64).flatten()
    if vec.shape[0] != expected_len:
        padded = np.ones(expected_len, dtype=np.float64) / max(expected_len, 1)
        k = min(expected_len, vec.shape[0])
        padded[:k] = vec[:k]
        vec = padded
    return _normalise_prob_vector(vec)


def _canonicalise_E(E_data: Any, expected_policies: Optional[int]) -> Optional[np.ndarray]:
    if E_data is None:
        return None
    vec = np.asarray(E_data, dtype=np.float64).flatten()
    if expected_policies is not None and vec.shape[0] != expected_policies:
        # Re-scale / truncate / pad to match policy count (pymdp asserts on this).
        padded = np.ones(expected_policies, dtype=np.float64) / max(expected_policies, 1)
        k = min(expected_policies, vec.shape[0])
        padded[:k] = vec[:k]
        vec = padded
    return _normalise_prob_vector(vec)


# ---------------------------------------------------------------------------
# pymdp 1.0.0 (JAX-first) import + Agent construction
# ---------------------------------------------------------------------------

def _require_pymdp_1():
    """
    Import pymdp 1.0.0 (JAX-first). We probe for the new surface explicitly so
    we fail fast with an actionable error if an old 0.x wheel is installed.
    """
    try:
        import jax.numpy as jnp  # noqa: F401
        import jax.random as jr  # noqa: F401
        from pymdp.agent import Agent  # noqa: F401
        from pymdp import utils as pymdp_utils  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pymdp 1.0.0 (JAX-first) is required. Install with:\n"
            "    uv pip install 'inferactively-pymdp>=1.0.0'\n"
            f"(original error: {e})"
        )

    # pymdp 1.0.0 has Agent.update_empirical_prior; 0.x does not.
    if not hasattr(Agent, "update_empirical_prior"):
        raise ImportError(
            "Detected legacy pymdp (<1.0.0). This module requires pymdp 1.0.0. "
            "Upgrade with: uv pip install --upgrade 'inferactively-pymdp>=1.0.0'"
        )
    return Agent, pymdp_utils, jnp, jr


def _to_jax_batched(mat_np: np.ndarray, batch_size: int):
    """Add a leading batch dim and convert to jnp float32, pymdp 1.0.0 convention."""
    import jax.numpy as jnp

    arr = jnp.asarray(mat_np, dtype=jnp.float32)
    if batch_size == 1:
        return arr[None, ...]
    return jnp.broadcast_to(arr[None, ...], (batch_size, *arr.shape))


def _build_pymdp_agent(
    *,
    A_np: np.ndarray,
    B_np: np.ndarray,
    C_np: np.ndarray,
    D_np: np.ndarray,
    E_np: Optional[np.ndarray],
    batch_size: int = 1,
    policy_len: int = 1,
    gamma: float = 16.0,
    alpha: float = 16.0,
):
    """
    Build a pymdp 1.0.0 ``Agent`` from canonical GNN numpy matrices.

    Parameters
    ----------
    A_np : (num_obs, num_states) column-normalised
    B_np : (num_states, num_states, num_actions) column-normalised per action
    C_np : (num_obs,)
    D_np : (num_states,)
    E_np : (num_policies,) or None
    """
    Agent, _, _, _ = _require_pymdp_1()

    num_states = int(A_np.shape[1])
    num_actions = int(B_np.shape[-1])

    A_list = [_to_jax_batched(A_np, batch_size)]
    B_list = [_to_jax_batched(B_np, batch_size)]
    C_list = [_to_jax_batched(C_np, batch_size)]
    D_list = [_to_jax_batched(D_np, batch_size)]

    agent_kwargs: Dict[str, Any] = dict(
        A=A_list,
        B=B_list,
        C=C_list,
        D=D_list,
        num_controls=[num_actions],
        policy_len=policy_len,
        gamma=gamma,
        alpha=alpha,
        batch_size=batch_size,
    )

    # pymdp 1.0.0 asserts ``num_controls[fi] > 1`` for every factor listed in
    # ``control_fac_idx``. A pure HMM (num_actions == 1) must therefore omit
    # control_fac_idx entirely; pymdp defaults to an empty / passive control set.
    if num_actions > 1:
        agent_kwargs["control_fac_idx"] = [0]

    if E_np is not None:
        # E is a plain Array of shape (batch, num_policies) — not a list.
        agent_kwargs["E"] = _to_jax_batched(E_np, batch_size)

    agent = Agent(**agent_kwargs)
    return agent


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_simple_pymdp_simulation(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run a discrete POMDP active inference rollout with real pymdp 1.0.0.

    Parameters
    ----------
    gnn_spec : dict
        Parsed GNN specification. Matrices live under
        ``gnn_spec["initialparameterization"]`` with keys A, B, C, D, E.
        Scalar runtime knobs live under ``gnn_spec["model_parameters"]``
        (``num_timesteps``, ``random_seed``, ``batch_size``, ``policy_len``,
        ``gamma``, ``alpha``).
    output_dir : Path
        Directory for ``simulation_results.json``. Created if missing.

    Returns
    -------
    (success, results) : (bool, dict)
        On success, ``results`` matches the legacy schema consumed by
        ``src/analysis/pymdp/framework_extractors``:
        keys include ``observations``, ``actions``, ``beliefs``,
        ``true_states``, ``simulation_trace``, ``validation``, ``metrics``,
        ``model_parameters``, ``framework == "PyMDP"``.
    """
    try:
        Agent, pymdp_utils, jnp, jr = _require_pymdp_1()
    except ImportError as e:
        logger.error(str(e))
        return False, {
            "success": False,
            "error": str(e),
            "suggestion": "Install with: uv pip install 'inferactively-pymdp>=1.0.0'",
        }

    try:
        import importlib.metadata as _ilm

        pymdp_version = _ilm.version("inferactively-pymdp")
    except Exception:  # pragma: no cover - metadata missing is non-fatal
        pymdp_version = "unknown"

    logger.info("Starting pymdp %s rollout (JAX backend)", pymdp_version)

    init_params = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization", {}
    )
    model_params = gnn_spec.get("model_parameters", {}) or {}

    # Determine fallback dims from A if present
    a_raw = init_params.get("A")
    if a_raw is not None:
        a_np_tmp = np.asarray(a_raw, dtype=np.float64)
        if a_np_tmp.ndim != 2:
            return False, {
                "success": False,
                "error": f"A matrix must be 2D; got shape {a_np_tmp.shape}",
            }
        fallback_shape = (int(a_np_tmp.shape[0]), int(a_np_tmp.shape[1]))
    else:
        fallback_shape = (
            int(model_params.get("num_obs", 3)),
            int(model_params.get("num_hidden_states", 3)),
        )

    A_np = _canonicalise_A(a_raw, fallback_shape)
    num_obs, num_states = A_np.shape

    # Derive num_actions from B or model params
    b_raw = init_params.get("B")
    if b_raw is not None:
        b_np_tmp = np.asarray(b_raw, dtype=np.float64)
        if b_np_tmp.ndim == 3:
            # Heuristic: if leading dim equals any plausible action count, prefer it
            if b_np_tmp.shape[0] == b_np_tmp.shape[1] and b_np_tmp.shape[0] == num_states:
                num_actions_guess = int(b_np_tmp.shape[-1])
            else:
                num_actions_guess = int(b_np_tmp.shape[0])
        elif b_np_tmp.ndim == 2:
            num_actions_guess = 1
        else:
            num_actions_guess = int(model_params.get("num_actions", 1))
    else:
        num_actions_guess = int(model_params.get("num_actions", 1))

    num_actions = max(1, int(model_params.get("num_actions", num_actions_guess)))

    B_np = _canonicalise_B(b_raw, num_states, num_actions)
    # Ensure num_actions reflects the canonicalised tensor.
    num_actions = int(B_np.shape[2])
    C_np = _canonicalise_C(init_params.get("C"), num_obs)
    D_np = _canonicalise_D(init_params.get("D"), num_states)
    E_np = _canonicalise_E(init_params.get("E"), expected_policies=num_actions)

    num_timesteps = int(model_params.get("num_timesteps", 20))
    batch_size = int(model_params.get("batch_size", 1))
    policy_len = int(model_params.get("policy_len", 1))
    gamma = float(model_params.get("gamma", 16.0))
    alpha = float(model_params.get("alpha", 16.0))
    seed = int(model_params.get("random_seed", 0))

    logger.info(
        "Dimensions: No=%d, Ns=%d, Nu=%d | T=%d, batch=%d, policy_len=%d",
        num_obs,
        num_states,
        num_actions,
        num_timesteps,
        batch_size,
        policy_len,
    )

    agent = _build_pymdp_agent(
        A_np=A_np,
        B_np=B_np,
        C_np=C_np,
        D_np=D_np,
        E_np=E_np,
        batch_size=batch_size,
        policy_len=policy_len,
        gamma=gamma,
        alpha=alpha,
    )
    logger.info("pymdp 1.0.0 Agent built (batch_size=%d)", batch_size)

    # RNG keys: numpy for the "environment" draws, jax for pymdp sampling.
    np_rng = np.random.default_rng(seed)
    jax_key = jr.PRNGKey(seed)

    # Initial true state
    true_state = int(np_rng.choice(num_states, p=D_np))
    true_states: List[int] = [true_state]
    observations: List[int] = []
    actions: List[int] = []
    beliefs: List[List[float]] = []
    efe_history: List[List[float]] = []
    vfe_history: List[float] = []

    empirical_prior = agent.D

    for t in range(num_timesteps):
        # Environment: sample observation from A given true state
        obs_probs = A_np[:, true_state]
        obs_idx = int(np_rng.choice(num_obs, p=_normalise_prob_vector(obs_probs)))
        observations.append(obs_idx)

        obs_jax = [jnp.array([obs_idx], dtype=jnp.int32)]

        # Agent inference
        qs, info = agent.infer_states(
            obs_jax,
            empirical_prior=empirical_prior,
            return_info=True,
        )

        # qs[f] shape: (batch, time, num_states[f]) — take most recent
        belief_vec = np.asarray(qs[0][0, -1], dtype=np.float64).flatten()
        beliefs.append(belief_vec.tolist())

        try:
            vfe_history.append(float(np.asarray(info["vfe"]).mean()))
        except Exception:  # noqa: BLE001 - informational only
            vfe_history.append(0.0)

        q_pi, neg_efe = agent.infer_policies(qs)
        # q_pi / neg_efe shape: (batch, num_policies)
        efe_history.append(np.asarray(neg_efe[0], dtype=np.float64).flatten().tolist())

        jax_key, subkey = jr.split(jax_key)
        action_keys = jr.split(subkey, batch_size + 1)
        action = agent.sample_action(q_pi, rng_key=action_keys[1:])
        # action shape: (batch, num_factors) — single control factor
        action_idx = int(np.asarray(action)[0, 0])
        actions.append(action_idx)

        # Environment: sample next true state from B
        next_probs = _normalise_prob_vector(B_np[:, true_state, action_idx])
        true_state = int(np_rng.choice(num_states, p=next_probs))
        true_states.append(true_state)

        # Update empirical prior for next step (pymdp 1.0.0 canonical helper)
        empirical_prior = agent.update_empirical_prior(action, qs)

        logger.info(
            "t=%02d obs=%d belief=%s action=%d next_state=%d",
            t,
            obs_idx,
            np.round(belief_vec, 3).tolist(),
            action_idx,
            true_state,
        )

    # ---------------------------------------------------------------------
    # Assemble results (schema-compatible with analysis step)
    # ---------------------------------------------------------------------
    model_name = gnn_spec.get("model_name") or gnn_spec.get("name") or "pymdp_model"

    results: Dict[str, Any] = {
        "success": True,
        "framework": "PyMDP",
        "pymdp_version": pymdp_version,
        "backend": "jax",
        "model_name": model_name,
        "num_timesteps": num_timesteps,
        "simulation_trace": {
            "observations": observations,
            "true_states": true_states,
            "beliefs": beliefs,
            "actions": actions,
            "efe_history": efe_history,
            "vfe_history": vfe_history,
            "belief_confidence": [float(max(b)) if b else 0.0 for b in beliefs],
        },
        "observations": observations,
        "true_states": true_states,
        "beliefs": beliefs,
        "actions": actions,
        "model_parameters": {
            "A_shape": list(A_np.shape),
            "B_shape": list(B_np.shape),
            "C_shape": list(C_np.shape),
            "D_shape": list(D_np.shape),
            "num_states": int(num_states),
            "num_observations": int(num_obs),
            "num_actions": int(num_actions),
            "batch_size": batch_size,
            "policy_len": policy_len,
            "gamma": gamma,
            "alpha": alpha,
        },
        "metrics": {
            "expected_free_energy": efe_history,
            "variational_free_energy": vfe_history,
            "belief_confidence": [float(max(b)) if b else 0.0 for b in beliefs],
            "cumulative_preference": [float(C_np[obs]) for obs in observations],
        },
        "validation": {
            "all_beliefs_valid": all(0.0 <= v <= 1.0 for b in beliefs for v in b),
            "beliefs_sum_to_one": all(
                abs(sum(b) - 1.0) < 1e-2 for b in beliefs if b
            ),
            "actions_in_range": all(0 <= a < num_actions for a in actions),
            "pymdp_version_ge_1_0_0": _is_version_ge(pymdp_version, (1, 0, 0)),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "simulation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Saved simulation results to %s", results_file)
    return True, results


def _is_version_ge(ver: str, target: Sequence[int]) -> bool:
    try:
        parts = tuple(int(p) for p in ver.split(".")[: len(target)] if p.isdigit())
    except Exception:  # noqa: BLE001
        return False
    return parts >= tuple(target)
