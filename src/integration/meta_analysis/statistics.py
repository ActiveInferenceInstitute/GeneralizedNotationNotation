"""Aggregate numeric summaries for sweep meta-analysis."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from .collector import SweepRecord

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


def compute_meta_statistics(records: List[SweepRecord]) -> Dict[str, Any]:
    """Compute per-framework stats, best-framework per (N,T), and log-log slopes."""
    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy not available — returning minimal meta_statistics")
        return {"schema_version": SCHEMA_VERSION, "error": "numpy_missing"}

    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "per_framework": {},
        "per_cell_best_framework": [],
        "loglog_runtime_vs_n_by_T": {},
    }

    fw_groups: Dict[str, List[SweepRecord]] = {}
    for r in records:
        fw_groups.setdefault(r.framework, []).append(r)

    for fw, group in sorted(fw_groups.items()):
        ok = [x for x in group if x.success and x.execution_time and x.execution_time > 0]
        times = [x.execution_time for x in ok]
        total = len(group)
        result["per_framework"][fw] = {
            "records_total": total,
            "successful_runs": len(ok),
            "success_rate": round(len(ok) / total, 4) if total else 0.0,
            "runtime_median_s": float(np.median(times)) if times else None,
            "runtime_mean_s": float(np.mean(times)) if times else None,
            "runtime_min_s": float(min(times)) if times else None,
            "runtime_max_s": float(max(times)) if times else None,
            "sum_median_proxy_s": float(sum(times)) if times else 0.0,
        }

    # Best framework per (N, T) by median runtime (single record each typical)
    cells: Dict[Tuple[int, int], List[SweepRecord]] = {}
    for r in records:
        if r.num_states is None or r.num_timesteps is None:
            continue
        if not r.success or not r.execution_time or r.execution_time <= 0:
            continue
        cells.setdefault((r.num_states, r.num_timesteps), []).append(r)

    for (n, t), group in sorted(cells.items()):
        best = min(group, key=lambda x: x.execution_time)
        result["per_cell_best_framework"].append(
            {
                "num_states": n,
                "num_timesteps": t,
                "framework": best.framework,
                "execution_time_s": best.execution_time,
            }
        )

    pymdp = [
        r
        for r in records
        if r.framework == "pymdp"
        and r.success
        and r.execution_time
        and r.execution_time > 0
        and r.num_states is not None
        and r.num_timesteps is not None
    ]
    t_values = sorted({r.num_timesteps for r in pymdp if r.num_timesteps is not None})
    for t in t_values:
        subset = [r for r in pymdp if r.num_timesteps == t]
        if len(subset) < 2:
            continue
        subset.sort(key=lambda r: r.num_states)
        n_arr = np.array([float(r.num_states) for r in subset], dtype=float)
        rt_arr = np.array([r.execution_time for r in subset], dtype=float)
        valid = (n_arr > 0) & (rt_arr > 0)
        if np.sum(valid) < 2:
            continue
        log_x = np.log(n_arr[valid])
        log_y = np.log(rt_arr[valid])
        coeffs = np.polyfit(log_x, log_y, 1)
        y_hat = coeffs[0] * log_x + coeffs[1]
        ss_res = float(np.sum((log_y - y_hat) ** 2))
        y_mean = float(np.mean(log_y))
        ss_tot = float(np.sum((log_y - y_mean) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        result["loglog_runtime_vs_n_by_T"][str(int(t))] = {
            "slope": float(coeffs[0]),
            "intercept": float(coeffs[1]),
            "r_squared": float(r2),
            "n_points": int(np.sum(valid)),
        }

    return result
