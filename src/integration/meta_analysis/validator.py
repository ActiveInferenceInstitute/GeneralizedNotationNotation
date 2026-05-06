"""Sweep grid and record consistency checks for meta-analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .collector import SweepRecord

logger = logging.getLogger(__name__)


def validate_sweep_records(records: List[SweepRecord]) -> Dict[str, Any]:
    """Return structured validation issues (non-fatal; for reporting only).

    Checks:
    - Full Cartesian grid coverage over observed N × T values.
    - Simulation JSON timestep length vs sweep label when files exist.
    - Benchmark repeat coherence when repeats > 1.
    """
    issues: List[Dict[str, Any]] = []
    schema_version = "1.0"

    if not records:
        return {
            "schema_version": schema_version,
            "issues": [],
            "summary": {"info": 0, "warning": 0, "error": 0},
        }

    n_vals = sorted({r.num_states for r in records if r.num_states is not None})
    t_vals = sorted({r.num_timesteps for r in records if r.num_timesteps is not None})

    if n_vals and t_vals:
        expected: Set[Tuple[int, int]] = {(n, t) for n in n_vals for t in t_vals}
        actual: Set[Tuple[int, int]] = {
            (r.num_states, r.num_timesteps)
            for r in records
            if r.num_states is not None and r.num_timesteps is not None
        }
        missing = sorted(expected - actual)
        for n, t in missing:
            issues.append(
                {
                    "severity": "warning",
                    "code": "GRID_MISSING_CELL",
                    "message": f"No sweep record for expected grid cell N={n}, T={t}",
                    "num_states": n,
                    "num_timesteps": t,
                }
            )

    for r in records:
        if not r.success and r.execution_time and r.execution_time > 0:
            issues.append(
                {
                    "severity": "info",
                    "code": "TIMING_WITHOUT_SUCCESS",
                    "message": f"Non-success record reports execution_time={r.execution_time}",
                    "model_name": r.model_name,
                    "framework": r.framework,
                }
            )

        if r.execution_benchmark_repeats > 1:
            samples = r.execution_time_samples or []
            if len(samples) < 2:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "BENCHMARK_SAMPLES_SHORT",
                        "message": (
                            f"execution_benchmark_repeats={r.execution_benchmark_repeats} "
                            f"but samples length is {len(samples)}"
                        ),
                        "model_name": r.model_name,
                        "framework": r.framework,
                    }
                )
            if not r.execution_time_std or r.execution_time_std <= 0:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "BENCHMARK_STD_MISSING",
                        "message": "Repeats > 1 but execution_time_std missing or zero",
                        "model_name": r.model_name,
                        "framework": r.framework,
                    }
                )

        path_str = r.simulation_results_path
        if not path_str or r.num_timesteps is None:
            continue
        p = Path(path_str)
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(
                {
                    "severity": "warning",
                    "code": "SIM_JSON_READ",
                    "message": f"Could not read simulation_results: {exc}",
                    "model_name": r.model_name,
                    "framework": r.framework,
                    "path": path_str,
                }
            )
            continue

        obs = data.get("observations") or []
        obs_len = len(obs) if isinstance(obs, list) else 0
        declared = data.get("num_timesteps")
        try:
            effective = int(declared) if declared is not None else obs_len
        except (TypeError, ValueError):
            effective = obs_len
        if effective and r.num_timesteps != effective:
            issues.append(
                {
                    "severity": "warning",
                    "code": "TIMESTEP_MISMATCH",
                    "message": (
                        f"Sweep label T={r.num_timesteps} vs simulation "
                        f"num_timesteps/observations length={effective}"
                    ),
                    "model_name": r.model_name,
                    "framework": r.framework,
                    "path": path_str,
                }
            )

    summary = {"info": 0, "warning": 0, "error": 0}
    for item in issues:
        sev = item.get("severity", "warning")
        if sev in summary:
            summary[sev] += 1

    return {
        "schema_version": schema_version,
        "issues": issues,
        "summary": summary,
        "grid": {
            "n_values": n_vals,
            "t_values": t_vals,
            "expected_cells": len(n_vals) * len(t_vals) if n_vals and t_vals else 0,
            "record_cells": len(
                {
                    (r.num_states, r.num_timesteps)
                    for r in records
                    if r.num_states is not None and r.num_timesteps is not None
                }
            ),
        },
    }
