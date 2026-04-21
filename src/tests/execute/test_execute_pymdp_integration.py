#!/usr/bin/env python3
"""
Real pymdp 1.0.0 (JAX-first) execution tests for the GNN pipeline.

This module exercises the actual installed ``inferactively-pymdp`` wheel with
no mocks. Tests skip cleanly if pymdp is not installed or if the installed
wheel predates 1.0.0.

Upstream: https://github.com/infer-actively/pymdp
Local contract: ``src/execute/pymdp/simple_simulation.py``
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Check if pymdp 1.0.0+ is available
try:
    import jax.numpy as jnp
    import jax.random as jr
    from pymdp import utils as pymdp_utils
    from pymdp.agent import Agent

    PYMDP_AVAILABLE = True
    PYMDP_IS_1_0_0_PLUS = hasattr(Agent, "update_empirical_prior")
except ImportError:
    PYMDP_AVAILABLE = False
    PYMDP_IS_1_0_0_PLUS = False

try:
    from execute.pymdp.package_detector import (
        detect_pymdp_installation,
        is_correct_pymdp_package,
        validate_pymdp_for_execution,
    )
    from execute.pymdp.simple_simulation import run_simple_pymdp_simulation

    EXECUTE_MODULE_AVAILABLE = True
except ImportError as e:
    EXECUTE_MODULE_AVAILABLE = False
    IMPORT_ERROR = str(e)


# ---------------------------------------------------------------------------
# Helpers — build a pymdp 1.0.0 Agent using the batched JAX convention
# ---------------------------------------------------------------------------
def _to_batched(mat: np.ndarray):
    return jnp.asarray(mat, dtype=jnp.float32)[None, ...]


def _build_minimal_agent(num_actions: int = 2) -> Any:
    """Return a 2-state / 2-obs Agent suitable for smoke tests."""
    A = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    if num_actions == 1:
        B = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)[:, :, None]
    else:
        B = np.stack(
            [
                np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64),
                np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float64),
            ],
            axis=-1,
        )
    C = np.array([0.0, 1.0], dtype=np.float64)
    D = np.array([0.5, 0.5], dtype=np.float64)

    kwargs: dict = dict(
        A=[_to_batched(A)],
        B=[_to_batched(B)],
        C=[_to_batched(C)],
        D=[_to_batched(D)],
        num_controls=[num_actions],
        policy_len=1,
        batch_size=1,
    )
    if num_actions > 1:
        kwargs["control_fac_idx"] = [0]
    return Agent(**kwargs)


# ---------------------------------------------------------------------------
@pytest.mark.skipif(not PYMDP_AVAILABLE, reason="inferactively-pymdp not installed")
@pytest.mark.skipif(not PYMDP_IS_1_0_0_PLUS, reason="pymdp < 1.0.0 (JAX Agent missing)")
@pytest.mark.skipif(not EXECUTE_MODULE_AVAILABLE, reason="Execute module not available")
class TestPyMDPRealExecution:
    """Exercise real pymdp 1.0.0 via the JAX-first Agent."""

    def test_pymdp_agent_import(self) -> None:
        """The 1.0.0 JAX-first Agent is importable and callable."""
        assert Agent is not None and pymdp_utils is not None
        assert callable(Agent)
        assert hasattr(Agent, "update_empirical_prior")

    def test_pymdp_agent_creation(self) -> None:
        """Build a minimal 2×2 Agent (1 action = pure HMM)."""
        agent = _build_minimal_agent(num_actions=1)
        assert agent is not None
        assert hasattr(agent, "infer_states")
        assert hasattr(agent, "infer_policies")
        assert hasattr(agent, "sample_action")
        assert hasattr(agent, "update_empirical_prior")
        # D is a list of (batch, num_states[f]) arrays
        assert isinstance(agent.D, (list, tuple))
        assert agent.D[0].shape == (1, 2)

    def test_pymdp_simple_simulation_execution(self, tmp_path: Any) -> None:
        """End-to-end rollout via the pipeline's real pymdp 1.0.0 runner."""
        gnn_spec = {
            "model_name": "test_pymdp_model",
            "initialparameterization": {
                "A": [[0.9, 0.1], [0.1, 0.9]],
                # GNN B in (action, prev, next) form; simple_simulation will
                # transpose to the pymdp (next, prev, action) convention.
                "B": [[[0.8, 0.2], [0.2, 0.8]]],
                "C": [0.0, 0.0],
                "D": [0.5, 0.5],
            },
            "model_parameters": {"num_timesteps": 4},
        }
        output_dir = tmp_path / "pymdp_output"
        output_dir.mkdir()

        success, results = run_simple_pymdp_simulation(gnn_spec, output_dir)
        assert success is True
        assert results["success"] is True
        assert results["framework"] == "PyMDP"
        assert results["backend"] == "jax"
        assert results["pymdp_version"].startswith("1.")
        assert len(results["observations"]) == 4
        assert len(results["actions"]) == 4
        assert len(results["beliefs"]) == 4

    def test_pymdp_package_detection_with_real_installation(self) -> None:
        """The package detector should correctly identify the 1.0.0 install."""
        detection = detect_pymdp_installation()
        assert detection["installed"] is True
        assert detection["correct_package"] is True
        assert detection["has_agent"] is True
        assert detection["wrong_package"] is False

    def test_is_correct_pymdp_package_with_real_installation(self) -> None:
        assert is_correct_pymdp_package() is True

    def test_validate_pymdp_for_execution_with_real_installation(self) -> None:
        validation = validate_pymdp_for_execution()
        assert validation["ready"] is True
        assert validation["detection"]["correct_package"] is True


# ---------------------------------------------------------------------------
@pytest.mark.skipif(not EXECUTE_MODULE_AVAILABLE, reason="Execute module not available")
class TestPyMDPErrorHandling:
    """Package detection returns sensible structure even in odd environments."""

    def test_package_detection_structure(self) -> None:
        detection = detect_pymdp_installation()
        assert isinstance(detection, dict)
        assert "installed" in detection
        assert "correct_package" in detection
        assert "wrong_package" in detection
        assert isinstance(detection["installed"], bool)

    def test_validation_structure(self) -> None:
        validation = validate_pymdp_for_execution()
        assert isinstance(validation, dict)
        assert "ready" in validation
        assert "detection" in validation
        assert "instructions" in validation
        assert isinstance(validation["ready"], bool)


# ---------------------------------------------------------------------------
@pytest.mark.skipif(not PYMDP_AVAILABLE, reason="inferactively-pymdp not installed")
@pytest.mark.skipif(not PYMDP_IS_1_0_0_PLUS, reason="pymdp < 1.0.0 (JAX Agent missing)")
class TestPyMDPJAXFirstAPI:
    """Regression tests for the 1.0.0 JAX-first API used by the pipeline."""

    def test_modern_import_works(self) -> None:
        assert Agent is not None
        assert pymdp_utils is not None

    def test_agent_has_required_methods(self) -> None:
        agent = _build_minimal_agent(num_actions=2)
        for name in (
            "infer_states",
            "infer_policies",
            "sample_action",
            "update_empirical_prior",
        ):
            assert hasattr(agent, name), f"Agent missing {name}"
            assert callable(getattr(agent, name))

    def test_simulation_step_execution(self) -> None:
        """A single rollout step via the JAX-first Agent."""
        agent = _build_minimal_agent(num_actions=2)

        obs = [jnp.array([0], dtype=jnp.int32)]
        qs, info = agent.infer_states(obs, empirical_prior=agent.D, return_info=True)
        assert isinstance(qs, (list, tuple))
        assert qs[0].shape == (1, 1, 2)
        assert "vfe" in info

        q_pi, neg_efe = agent.infer_policies(qs)
        assert q_pi.shape == (1, 2)
        assert neg_efe.shape == (1, 2)

        rng = jr.PRNGKey(0)
        action_keys = jr.split(rng, agent.batch_size + 1)
        action = agent.sample_action(q_pi, rng_key=action_keys[1:])
        assert action.shape == (1, 1)

        new_prior = agent.update_empirical_prior(action, qs)
        assert isinstance(new_prior, (list, tuple))
        assert new_prior[0].shape == agent.D[0].shape
