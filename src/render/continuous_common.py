"""Shared helpers for rendering continuous-state (linear-Gaussian) GNN models.

Every framework renderer consumes the same ``gnn_spec`` produced by
``render.pomdp_processor`` for ``model_kind == "continuous"``:

``initialparameterization`` holds ``F`` (n×n), ``H`` (m×n), ``Q`` (n×n),
``R`` (m×m), ``prior_mean`` (n), ``prior_cov`` (n×n) and optionally
``goal_mean`` (n) + ``control_gain`` (scalar). ``model_parameters`` holds
``num_timesteps``, ``dt`` and ``random_seed``.

The generative model each generated script simulates and filters:

    x_1 ~ N(prior_mean, prior_cov)
    x_t = F x_{t-1} + u_{t-1} + N(0, Q)
    y_t = H x_t + N(0, R)
    u_t = control_gain * (goal_mean - mu_t)   (mu_t = filtered mean; else 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

REQUIRED_KEYS = ("F", "H", "Q", "R", "prior_mean", "prior_cov")


def is_continuous_spec(gnn_spec: Dict[str, Any]) -> bool:
    """True when the spec is a continuous linear-Gaussian model."""
    if gnn_spec.get("model_kind") == "continuous":
        return True
    try:
        from render.pomdp_contract import ModelKind, detect_model_kind

        return detect_model_kind(gnn_spec) == ModelKind.CONTINUOUS
    except Exception:
        return False


@dataclass
class ContinuousSpec:
    """Validated numeric view of a continuous GNN spec."""

    model_name: str
    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    prior_mean: np.ndarray
    prior_cov: np.ndarray
    goal_mean: Optional[np.ndarray]
    control_gain: Optional[float]
    num_timesteps: int
    dt: float
    random_seed: int

    @property
    def n(self) -> int:
        return int(self.F.shape[0])

    @property
    def m(self) -> int:
        return int(self.H.shape[0])

    @property
    def has_control(self) -> bool:
        return self.goal_mean is not None and self.control_gain is not None


def _scalar(value: Any) -> float:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return float(value)


def extract_continuous_spec(gnn_spec: Dict[str, Any]) -> ContinuousSpec:
    """Parse and shape-check the continuous parameter block."""
    initial = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization"
    )
    if not isinstance(initial, dict):
        raise ValueError("continuous spec requires an initialparameterization mapping")
    missing = [key for key in REQUIRED_KEYS if key not in initial]
    if missing:
        raise ValueError(f"continuous spec is missing {missing}")

    F = np.asarray(initial["F"], dtype=float)
    H = np.asarray(initial["H"], dtype=float)
    Q = np.asarray(initial["Q"], dtype=float)
    R = np.asarray(initial["R"], dtype=float)
    prior_mean = np.asarray(initial["prior_mean"], dtype=float).reshape(-1)
    prior_cov = np.asarray(initial["prior_cov"], dtype=float)
    n = F.shape[0]
    if F.shape != (n, n):
        raise ValueError(f"F must be square, got {F.shape}")
    m = H.shape[0]
    if H.shape != (m, n):
        raise ValueError(f"H must be [m, n]={m, n}, got {H.shape}")
    if Q.shape != (n, n) or R.shape != (m, m) or prior_cov.shape != (n, n):
        raise ValueError(
            f"covariance shapes mismatch: Q{Q.shape} R{R.shape} prior_cov{prior_cov.shape}"
        )
    if prior_mean.shape != (n,):
        raise ValueError(f"prior_mean must have {n} entries, got {prior_mean.shape}")

    goal_mean: Optional[np.ndarray] = None
    control_gain: Optional[float] = None
    if "goal_mean" in initial and "control_gain" in initial:
        goal_mean = np.asarray(initial["goal_mean"], dtype=float).reshape(-1)
        if goal_mean.shape != (n,):
            raise ValueError(f"goal_mean must have {n} entries, got {goal_mean.shape}")
        control_gain = _scalar(initial["control_gain"])

    params = gnn_spec.get("model_parameters") or {}
    num_timesteps = int(params.get("num_timesteps", 20))
    dt = float(params.get("dt", 1.0))
    seed = int(params.get("random_seed", params.get("seed", 42)))
    name = str(gnn_spec.get("model_name") or gnn_spec.get("name") or "continuous_model")
    return ContinuousSpec(
        model_name=name,
        F=F,
        H=H,
        Q=Q,
        R=R,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        goal_mean=goal_mean,
        control_gain=control_gain,
        num_timesteps=num_timesteps,
        dt=dt,
        random_seed=seed,
    )


def py_literal(arr: np.ndarray) -> str:
    """Render an array as a nested Python list literal with full precision."""
    return repr(np.asarray(arr, dtype=float).tolist())


def literal_block(spec: ContinuousSpec) -> Dict[str, str]:
    """Literals for every parameter, ready to splice into a template."""
    goal = py_literal(spec.goal_mean) if spec.goal_mean is not None else "None"
    gain = repr(float(spec.control_gain)) if spec.control_gain is not None else "None"
    return {
        "F": py_literal(spec.F),
        "H": py_literal(spec.H),
        "Q": py_literal(spec.Q),
        "R": py_literal(spec.R),
        "prior_mean": py_literal(spec.prior_mean),
        "prior_cov": py_literal(spec.prior_cov),
        "goal_mean": goal,
        "control_gain": gain,
    }


RESULT_KEYS: List[str] = [
    "model_name",
    "framework",
    "model_kind",
    "num_timesteps",
    "num_states",
    "num_observations",
    "beliefs",
    "posterior_cov",
    "true_states_continuous",
    "observations_continuous",
    "controls",
    "rmse_vs_true",
    "validation",
]
