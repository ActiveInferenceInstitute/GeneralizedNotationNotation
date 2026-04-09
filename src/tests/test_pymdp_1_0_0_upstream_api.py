#!/usr/bin/env python3
"""
Regression tests for pymdp 1.0.0 (JAX-first) as used by this repository.

These tests call the **installed** ``inferactively-pymdp`` wheel, not mocks.
They lock the exact API surface that ``execute/pymdp/simple_simulation`` and
``execute/pymdp/pymdp_simulation`` depend on:

* List-of-``jax.Array`` models with leading batch dim
* ``Agent.infer_states(obs, empirical_prior=prior, return_info=True)``
  returns ``(qs, info)`` where ``info`` exposes ``vfe``
* ``Agent.infer_policies(qs)`` returns ``(q_pi, neg_efe)`` per batch
* ``Agent.sample_action(q_pi, rng_key=keys)`` expects a JAX PRNG slice
* ``Agent.update_empirical_prior(action, qs)`` closes the rollout loop

See also:
  * doc/pymdp/pymdp_1_0_0_alignment_matrix.md — upstream mapping
  * src/execute/pymdp/simple_simulation.py — the code under test
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, Tuple

import numpy as np
import pytest

pytestmark = [pytest.mark.integration]


def _inferactively_pymdp_version() -> str:
    try:
        return importlib.metadata.version("inferactively-pymdp")
    except importlib.metadata.PackageNotFoundError:
        return ""


def _require_pymdp_1_0_0() -> Tuple[Any, Any, Any, Any, str]:
    """
    Import pymdp 1.0.0 (JAX-first) or skip the test. We probe the presence of
    ``Agent.update_empirical_prior``, which only exists on the new Agent, so a
    stale 0.x wheel gets flagged explicitly.
    """
    try:
        import jax.numpy as jnp
        import jax.random as jr
        from pymdp import utils
        from pymdp.agent import Agent
    except ImportError:
        pytest.skip("inferactively-pymdp (>=1.0.0) not installed")

    if not hasattr(Agent, "update_empirical_prior"):
        pytest.skip(
            "installed inferactively-pymdp is <1.0.0 (legacy NumPy API) — "
            "upgrade with `uv pip install --upgrade 'inferactively-pymdp>=1.0.0'`"
        )

    ver = _inferactively_pymdp_version()
    if not ver:
        pytest.skip("inferactively-pymdp metadata missing")
    return Agent, utils, jnp, jr, ver


def _normalise_columns(a: np.ndarray) -> np.ndarray:
    m = np.asarray(a, dtype=np.float64)
    norm = m.sum(axis=0, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return m / norm


def _canonicalise_b_3d(b_raw: np.ndarray, num_states: int) -> np.ndarray:
    """Convert a GNN (action, prev, next) tensor to pymdp (next, prev, action)."""
    if b_raw.ndim != 3:
        raise ValueError("expected 3D B_raw")
    if b_raw.shape[0] == num_states and b_raw.shape[1] == num_states:
        b = b_raw.transpose(2, 1, 0)
    else:
        b = b_raw
    out = b.copy()
    for a in range(out.shape[2]):
        col_sums = out[:, :, a].sum(axis=0, keepdims=True)
        zero = col_sums <= 0
        col_sums = np.where(zero, 1.0, col_sums)
        out[:, :, a] = out[:, :, a] / col_sums
        if zero.any():
            rows = out.shape[0]
            for j in np.where(zero.flatten())[0]:
                out[:, j, a] = 1.0 / rows
    return out


def _to_batched_jax(mat: np.ndarray, jnp_mod: Any):
    return jnp_mod.asarray(mat, dtype=jnp_mod.float32)[None, ...]


def _minimal_agent(Agent: Any, jnp_mod: Any, *, with_e: bool = False) -> Any:
    """2 states, 2 observations, 2 actions — small but multi-policy."""
    a = _normalise_columns(np.array([[0.85, 0.15], [0.15, 0.85]], dtype=np.float64))
    b_raw = np.array(
        [
            [[0.88, 0.12], [0.12, 0.88]],
            [[0.12, 0.88], [0.88, 0.12]],
        ],
        dtype=np.float64,
    )
    b = _canonicalise_b_3d(b_raw, num_states=2)
    c = np.array([0.0, 1.0], dtype=np.float64)
    d = np.array([0.5, 0.5], dtype=np.float64)

    kwargs: dict = dict(
        A=[_to_batched_jax(a, jnp_mod)],
        B=[_to_batched_jax(b, jnp_mod)],
        C=[_to_batched_jax(c, jnp_mod)],
        D=[_to_batched_jax(d, jnp_mod)],
        num_controls=[2],
        control_fac_idx=[0],
        policy_len=1,
        batch_size=1,
    )
    if with_e:
        e = np.array([0.5, 0.5], dtype=np.float64)
        kwargs["E"] = _to_batched_jax(e, jnp_mod)
    return Agent(**kwargs)


def test_inferactively_pymdp_version_is_1_0_0_plus() -> None:
    """Sanity: the installed wheel exposes the 1.0.0+ JAX-first API."""
    Agent, utils, _, _, ver = _require_pymdp_1_0_0()
    assert Agent is not None and utils is not None
    major = int(ver.split(".")[0]) if ver and ver.split(".")[0].isdigit() else 0
    assert major >= 1, f"expected pymdp >=1.0.0, got {ver!r}"


def test_utils_public_surface_exists() -> None:
    """Key ``pymdp.utils`` functions that the pipeline uses must be present."""
    _, utils, _, _, _ = _require_pymdp_1_0_0()
    for name in (
        "random_A_array",
        "random_B_array",
        "list_array_uniform",
        "norm_dist",
        "list_array_norm_dist",
    ):
        assert hasattr(utils, name), f"pymdp.utils missing: {name}"


def test_single_step_infer_states_infer_policies_sample_action() -> None:
    """One timestep end-to-end via the public pymdp 1.0.0 methods."""
    Agent, _, jnp, jr, _ = _require_pymdp_1_0_0()
    agent = _minimal_agent(Agent, jnp, with_e=False)

    # Initial empirical prior is agent.D (list per factor)
    prior = agent.D
    assert isinstance(prior, (list, tuple))
    assert prior[0].shape == (1, 2)  # (batch, num_states[0])

    obs = [jnp.array([0], dtype=jnp.int32)]
    qs, info = agent.infer_states(obs, empirical_prior=prior, return_info=True)

    assert isinstance(qs, (list, tuple))
    assert qs[0].shape == (1, 1, 2), f"qs shape was {qs[0].shape}"
    belief = np.asarray(qs[0][0, -1]).flatten()
    assert belief.shape == (2,)
    assert np.isclose(belief.sum(), 1.0, atol=1e-4)
    assert "vfe" in info

    q_pi, neg_efe = agent.infer_policies(qs)
    assert q_pi.shape == (1, 2)
    assert neg_efe.shape == (1, 2)
    assert np.isclose(float(q_pi[0].sum()), 1.0, atol=1e-4)

    key = jr.PRNGKey(0)
    action_keys = jr.split(key, agent.batch_size + 1)
    action = agent.sample_action(q_pi, rng_key=action_keys[1:])
    assert action.shape == (1, 1)  # (batch, num_factors)
    assert 0 <= int(np.asarray(action)[0, 0]) < 2


def test_agent_with_e_matches_policy_count() -> None:
    """Optional E (habit) length must match the number of policies."""
    Agent, _, jnp, _, _ = _require_pymdp_1_0_0()
    agent = _minimal_agent(Agent, jnp, with_e=True)
    obs = [jnp.array([1], dtype=jnp.int32)]
    qs, _info = agent.infer_states(obs, empirical_prior=agent.D, return_info=True)
    q_pi, _ = agent.infer_policies(qs)
    assert q_pi.shape[-1] == int(agent.E.shape[-1])


def test_multi_step_rollout_closes_via_update_empirical_prior() -> None:
    """Several steps: infer_states → infer_policies → sample_action → update_empirical_prior."""
    Agent, _, jnp, jr, _ = _require_pymdp_1_0_0()
    agent = _minimal_agent(Agent, jnp, with_e=False)

    rng_np = np.random.default_rng(7)
    jax_key = jr.PRNGKey(7)
    prior = agent.D
    history: list = []

    for _ in range(5):
        obs_idx = int(rng_np.integers(0, 2))
        obs = [jnp.array([obs_idx], dtype=jnp.int32)]
        qs, info = agent.infer_states(obs, empirical_prior=prior, return_info=True)
        q_pi, neg_efe = agent.infer_policies(qs)
        jax_key, sub = jr.split(jax_key)
        ak = jr.split(sub, agent.batch_size + 1)
        action = agent.sample_action(q_pi, rng_key=ak[1:])
        history.append(int(np.asarray(action)[0, 0]))
        prior = agent.update_empirical_prior(action, qs)

        # Sanity: prior must be list-of-(batch, num_states) arrays matching agent.D
        assert isinstance(prior, (list, tuple))
        assert prior[0].shape == agent.D[0].shape

    assert len(history) == 5
    assert all(0 <= a < 2 for a in history)


def test_control_and_inference_submodules_importable() -> None:
    """``pymdp.control`` / ``pymdp.inference`` are used internally by the Agent."""
    _require_pymdp_1_0_0()
    import pymdp.control  # noqa: F401
    import pymdp.inference  # noqa: F401


def test_pymdp_legacy_namespace_still_shipped() -> None:
    """
    pymdp 1.0.0 ships a ``pymdp.legacy`` namespace holding the 0.x NumPy API
    for migration. This repository does not consume it, but we assert it is
    importable so the migration doc stays accurate.
    """
    _require_pymdp_1_0_0()
    try:
        import pymdp.legacy  # noqa: F401
    except ImportError:
        pytest.skip("pymdp.legacy not shipped in this build")
