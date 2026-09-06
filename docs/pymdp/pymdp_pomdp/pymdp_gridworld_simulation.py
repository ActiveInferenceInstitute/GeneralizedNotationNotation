#!/usr/bin/env python3
"""
Reference PyMDP gridworld-style example.

This is a documentation example, not the pipeline execution entrypoint.
Authoritative runtime modules are in src/execute/pymdp.
"""

from __future__ import annotations

import numpy as np

try:
    from pymdp import utils
    from pymdp.agent import Agent
except ImportError as exc:  # pragma: no cover - docs reference script
    raise SystemExit(
        "PyMDP is required for this example. Install with: uv pip install inferactively-pymdp"
    ) from exc


def run_reference_example(steps: int = 10, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)

    # Minimal 2-state/2-observation/1-action model.
    A = utils.obj_array(1)
    B = utils.obj_array(1)
    C = utils.obj_array(1)
    D = utils.obj_array(1)

    A[0] = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    B[0] = np.zeros((2, 2, 1), dtype=float)
    B[0][:, :, 0] = np.array([[0.85, 0.15], [0.15, 0.85]], dtype=float)
    C[0] = np.array([0.0, 1.0], dtype=float)
    D[0] = np.array([0.5, 0.5], dtype=float)

    agent = Agent(A=A, B=B, C=C, D=D)

    observations = []
    actions = []
    neg_efe_trace = []

    for _ in range(steps):
        obs = [np.array([int(rng.integers(0, 2))])]
        _qs = agent.infer_states(obs)
        _q_pi, neg_efe = agent.infer_policies()
        action = agent.sample_action()

        observations.append(int(obs[0][0]))
        actions.append(int(action[0]) if hasattr(action, "__len__") else int(action))
        neg_efe_trace.append(np.asarray(neg_efe).tolist())

    return {
        "steps": steps,
        "observations": observations,
        "actions": actions,
        "neg_efe": neg_efe_trace,
    }


if __name__ == "__main__":
    result = run_reference_example()
    print(f"Ran {result['steps']} steps. First action: {result['actions'][0]}")
