#!/usr/bin/env python3
"""
Regression tests for the PyMDP ``Agent`` API used by this repository.

These tests call the **installed** ``inferactively-pymdp`` wheel (version from
importlib.metadata), not mocks. They lock the contract assumed by
``execute.pymdp.simple_simulation`` and the upstream ``Agent`` docstrings
(e.g. ``infer_policies`` returns policy posterior and per-policy negative
expected free energy).

Upstream 1.0.0 release notes are tracked in ``doc/pymdp/pymdp_1_0_0_alignment_matrix.md``.
The PyPI wheel may report a pre-1.0 version while exposing the same ``Agent``
surface; tests do not require ``1.0.0`` specifically.
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


def _require_agent_and_utils() -> Tuple[Any, Any, str]:
    try:
        from pymdp.agent import Agent
        from pymdp import utils
    except ImportError:
        pytest.skip("inferactively-pymdp not installed (pip: inferactively-pymdp)")

    ver = _inferactively_pymdp_version()
    if not ver:
        pytest.skip("inferactively-pymdp metadata missing")
    return Agent, utils, ver


def _transpose_and_normalize_b(b_raw: np.ndarray) -> np.ndarray:
    """Match ``run_simple_pymdp_simulation`` GNN (action,prev,next) → PyMDP layout."""
    if b_raw.ndim != 3:
        raise ValueError("expected 3D B_raw")
    b = b_raw.transpose(2, 1, 0)
    for action_idx in range(b.shape[2]):
        col_sums = b[:, :, action_idx].sum(axis=0)
        for col in range(len(col_sums)):
            if col_sums[col] > 0 and abs(col_sums[col] - 1.0) > 1e-6:
                b[:, col, action_idx] /= col_sums[col]
            elif col_sums[col] == 0:
                n = b.shape[0]
                b[:, col, action_idx] = 1.0 / n
    return b


def _column_normalize_a(a: np.ndarray) -> np.ndarray:
    m = np.asarray(a, dtype=np.float64)
    norm = m.sum(axis=0, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return m / norm


def _minimal_agent(
    Agent: Any,
    utils: Any,
    *,
    with_e: bool = False,
) -> Any:
    """2 states, 2 observations, 2 actions — small but multi-policy."""
    a = _column_normalize_a(np.array([[0.85, 0.15], [0.15, 0.85]], dtype=np.float64))
    b_raw = np.array(
        [
            [[0.88, 0.12], [0.12, 0.88]],
            [[0.12, 0.88], [0.88, 0.12]],
        ],
        dtype=np.float64,
    )
    b = _transpose_and_normalize_b(b_raw)
    c = np.array([0.0, 1.0], dtype=np.float64)
    d = np.array([0.5, 0.5], dtype=np.float64)

    a_obj = utils.obj_array(1)
    a_obj[0] = a
    b_obj = utils.obj_array(1)
    b_obj[0] = b
    c_obj = utils.obj_array(1)
    c_obj[0] = c
    d_obj = utils.obj_array(1)
    d_obj[0] = d

    if with_e:
        # Policy count for 2 controls with 2 actions each, policy_len=1 → 2 policies
        e = np.array([0.5, 0.5], dtype=np.float64)
        return Agent(A=a_obj, B=b_obj, C=c_obj, D=d_obj, E=e)
    return Agent(A=a_obj, B=b_obj, C=c_obj, D=d_obj)


def test_inferactively_pymdp_version_reported() -> None:
    """Sanity: distribution is installed and version string is readable."""
    Agent, utils, ver = _require_agent_and_utils()
    assert Agent is not None and utils is not None
    parts = [int(p) for p in ver.split(".") if p.isdigit()]
    assert parts, f"unexpected version string: {ver!r}"


def test_utils_obj_array_and_is_normalized() -> None:
    """``utils.obj_array`` / ``utils.is_normalized`` — required for Agent construction."""
    _, utils, _ = _require_agent_and_utils()
    a = _column_normalize_a(np.eye(3))
    b = _transpose_and_normalize_b(np.stack([np.eye(3), np.eye(3)], axis=0))
    ao = utils.obj_array(1)
    ao[0] = a
    bo = utils.obj_array(1)
    bo[0] = b
    assert utils.is_normalized(ao)
    assert utils.is_normalized(bo)


def test_agent_reset_infer_states_infer_policies_sample_action() -> None:
    """One timestep: public methods used by ``simple_simulation``."""
    Agent, utils, _ = _require_agent_and_utils()
    agent = _minimal_agent(Agent, utils, with_e=False)
    qs0 = agent.reset()
    assert qs0 is not None

    # Same observation container style as ``run_simple_pymdp_simulation`` (one modality).
    obs = np.array([0])
    qs = agent.infer_states(obs)
    assert qs is not None
    belief = np.asarray(qs[0]).flatten()
    assert belief.shape == (2,)
    assert np.isclose(belief.sum(), 1.0, atol=1e-4)

    q_pi, neg_efe = agent.infer_policies()
    assert q_pi.ndim == 1
    assert neg_efe.ndim == 1
    assert q_pi.shape == neg_efe.shape
    assert np.isclose(float(np.sum(q_pi)), 1.0, atol=1e-4)

    action = agent.sample_action()
    assert isinstance(action, np.ndarray)
    assert action.size >= 1
    assert 0 <= int(action.flat[0]) < 2


def test_agent_with_e_vector_matches_policy_count() -> None:
    """Optional E (habit) length must match number of policies (PyMDP assert)."""
    Agent, utils, _ = _require_agent_and_utils()
    agent = _minimal_agent(Agent, utils, with_e=True)
    agent.reset()
    agent.infer_states(np.array([1]))
    q_pi, _g = agent.infer_policies()
    assert q_pi.shape[0] == len(agent.E)


def test_multi_step_rollout_matches_simple_simulation_pattern() -> None:
    """Several steps: infer_states → infer_policies → sample_action loop."""
    np.random.seed(7)
    Agent, utils, _ = _require_agent_and_utils()
    agent = _minimal_agent(Agent, utils, with_e=False)
    agent.reset()

    for _ in range(5):
        obs_idx = int(np.random.randint(0, 2))
        agent.infer_states(np.array([obs_idx]))
        agent.infer_policies()
        act = agent.sample_action()
        assert 0 <= int(act.flat[0]) < 2


def test_pymdp_legacy_module_present_or_skip() -> None:
    """``pymdp.legacy`` is upstream migration surface; optional in some wheels."""
    _require_agent_and_utils()
    try:
        import pymdp.legacy  # noqa: F401
    except ImportError:
        pytest.skip("pymdp.legacy not shipped in this inferactively-pymdp build")


def test_control_and_inference_submodules_importable() -> None:
    """Agent delegates to ``pymdp.inference`` / ``pymdp.control`` — ensure imports work."""
    _require_agent_and_utils()
    import pymdp.control  # noqa: F401
    import pymdp.inference  # noqa: F401
