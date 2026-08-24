# Copyright (c) 2026 GNN Pipeline contributors.
# SPDX-License-Identifier: MIT
"""First-class Step 12 executor entry for sparse Kronecker-factorised inference.

Roadmap MAJ-02 residual: route the factorised execution (which the scaling
script previously ran directly via ``run_factorized_active_inference``) through
the numbered pipeline so a constructed factorised model produces a
``simulation_results.json`` carrying the ``jax_kronecker_factorized_v1`` schema
in the standard ``simulation_data/`` location that Step 16 analysis consumes.

This is a *thin* executor: it validates its inputs, runs the factor-separable
mean-field path from ``execute.jax.kronecker_factorized``, and writes the
schema artifact (plus a slim execution summary) — never a re-implementation of
the inference itself.

.. note::
   The joint state space is never materialised; ``joint_materialized`` is
   reported ``False`` by the underlying path.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "execute_kronecker_factorized",
    "run_kronecker_factorized_execution",
]

_KRONECKER_SCHEMA_VERSION = "jax_kronecker_factorized_v1"


def _build_factor_model(
    factor_sizes: List[int],
    t: int,
    seed: int,
    a_signal: float = 0.85,
    b_signal: float = 0.8,
    action_precision: float = 4.0,
) -> Any:
    """Build the factorised model, deferring the concrete builder to the kernel.

    Eight-or-fewer homogeneous binary factors use the homogeneous binary
    builder; arbitrary per-factor sizes route through the generic
    noisy-identity/permuted builder.
    """
    from execute.jax.kronecker_factorized import (
        build_binary_factor_model,
        build_generic_factor_model,
    )

    if all(size == 2 for size in factor_sizes):
        return build_binary_factor_model(
            len(factor_sizes),
            t=t,
            seed=seed,
            a_signal=a_signal,
            b_signal=b_signal,
            action_precision=action_precision,
        )
    return build_generic_factor_model(
        list(factor_sizes),
        t=t,
        seed=seed,
        a_signal=a_signal,
        b_signal=b_signal,
        action_precision=action_precision,
    )


def run_kronecker_factorized_execution(
    model: Any,
    output_dir: Union[str, Path],
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run factorised Kronecker inference and write the schema artifact.

    Args:
        model: A ``FactorizedPOMDP`` instance from
            ``execute.jax.kronecker_factorized``.
        output_dir: Destination directory. ``simulation_data/simulation_results.json``
            carries the ``jax_kronecker_factorized_v1`` schema; a compact
            ``kronecker_execution_summary.json`` records runtime metadata.
        model_name: Optional display name stamped into the summary (defaults to
            the model's ``model_name`` or ``model_kind``).

    Returns:
        The execution envelope dict: ``success``, ``schema_version``,
        ``execution_time``, ``output_files``, ``simulation`` (the full schema
        dict written to disk), and ``summary``.
    """
    from execute.jax.kronecker_factorized import run_factorized_active_inference

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    simulation_dir = out / "simulation_data"
    simulation_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    simulation: Dict[str, Any] = run_factorized_active_inference(model)
    elapsed = time.time() - started

    simulation.setdefault("schema_version", _KRONECKER_SCHEMA_VERSION)

    results_file = simulation_dir / "simulation_results.json"
    results_file.write_text(
        json.dumps(simulation, indent=2, default=str), encoding="utf-8"
    )

    summary: Dict[str, Any] = {
        "schema_version": _KRONECKER_SCHEMA_VERSION,
        "model_name": model_name
        or str(
            simulation.get("model_name")
            or simulation.get("model_kind", "factorized_pomdp")
        ),
        "success": bool(simulation.get("success", False)),
        "joint_state_space_size": (
            simulation.get("model_parameters", {}).get("joint_state_space_size")
        ),
        "joint_materialized": (
            simulation.get("model_parameters", {}).get("joint_materialized")
        ),
        "num_factors": simulation.get("num_factors"),
        "num_timesteps": simulation.get("num_timesteps"),
        "all_valid": bool(simulation.get("validation", {}).get("all_valid", False)),
        "execution_time_seconds": round(elapsed, 4),
        "written_at": datetime.now().isoformat(),
        "simulation_results_relative": str(results_file.relative_to(out)),
    }
    summary_file = out / "kronecker_execution_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "success": bool(simulation.get("success", False)),
        "schema_version": _KRONECKER_SCHEMA_VERSION,
        "execution_time": elapsed,
        "output_files": [str(results_file), str(summary_file)],
        "simulation": simulation,
        "summary": summary,
    }


def execute_kronecker_factorized(
    config: Dict[str, Any],
    output_dir: Union[str, Path],
    *,
    factor_sizes: Optional[List[int]] = None,
    t: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute a factorised Kronecker run from a configuration dict.

    This is the Step-12-style entry point: it turns an execution ``config``
    (or explicit per-factor sizes / timesteps / seed) into a runnable
    ``FactorizedPOMDP``, executes it, and writes the analysis-consumable schema
    artifact. It returns the same envelope as
    :func:`run_kronecker_factorized_execution`.

    Args:
        config: Execution config. Recognised keys ``factor_sizes`` (list of
            per-factor state sizes), ``t``/``timesteps``, ``seed``,
            ``a_signal``, ``b_signal``, ``action_precision``, ``model_name``.
        output_dir: Result directory.
        factor_sizes: Explicit override for ``config['factor_sizes']``.
        t: Explicit timestep override.
        seed: Explicit seed override.
        **kwargs: Additional overrides passed to the model builder.

    Returns:
        Envelope dict from :func:`run_kronecker_factorized_execution`.
    """
    if not isinstance(config, dict):
        raise ValueError("execute_kronecker_factorized expects a config dict")

    sizes = (
        list(factor_sizes) if factor_sizes is not None else config.get("factor_sizes")
    )
    if not sizes:
        raise ValueError("factor_sizes is required (config or argument)")
    sizes = [int(n) for n in sizes]
    if any(n <= 0 for n in sizes):
        raise ValueError("factor sizes must be positive integers")

    t_val: Any = t if t is not None else config.get("t", config.get("timesteps", 20))
    seed_val: Any = seed if seed is not None else config.get("seed", 42)
    resolution_t: int = int(t_val if t_val is not None else 20)
    resolution_seed: int = int(seed_val if seed_val is not None else 42)
    a_signal_val: Any = kwargs.get("a_signal", config.get("a_signal", 0.85))
    b_signal_val: Any = kwargs.get("b_signal", config.get("b_signal", 0.8))
    precision_val: Any = kwargs.get(
        "action_precision", config.get("action_precision", 4.0)
    )
    a_signal = float(a_signal_val if a_signal_val is not None else 0.85)
    b_signal = float(b_signal_val if b_signal_val is not None else 0.8)
    action_precision = float(precision_val if precision_val is not None else 4.0)
    model = _build_factor_model(
        sizes,
        t=resolution_t,
        seed=resolution_seed,
        a_signal=a_signal,
        b_signal=b_signal,
        action_precision=action_precision,
    )
    model_name = str(config.get("model_name") or "factorized_kronecker")
    return run_kronecker_factorized_execution(model, output_dir, model_name=model_name)
