#!/usr/bin/env python3
"""
Kronecker-factorized (MAJ-02) JAX script generation for GNN Step 11.

Extracted from ``render.jax.jax_renderer``.
"""

import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import numpy as np

from .jax_spec_extract import _jax_model_name

# --- Kronecker-factorized (MAJ-02) helpers ---

_FACTOR_MATRIX_RE = re.compile(r"^([ABCD])_f(\d+)$")


def _is_factorized_spec(gnn_spec: Dict[str, Any]) -> bool:
    """True when the spec declares per-factor ``A_fN``/``B_fN`` matrices."""
    matrices = (gnn_spec.get("structured_pomdp") or {}).get("matrices") or {}
    a_factor_keys = [
        key
        for key in matrices
        if _FACTOR_MATRIX_RE.match(str(key)) and str(key).startswith("A_f")
    ]
    b_factor_keys = [
        key
        for key in matrices
        if _FACTOR_MATRIX_RE.match(str(key)) and str(key).startswith("B_f")
    ]
    return len(a_factor_keys) >= 2 and len(b_factor_keys) >= 2


def _factor_matrix_groups(
    gnn_spec: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """Group per-factor A/B/C/D matrices keyed by factor index."""
    matrices = (gnn_spec.get("structured_pomdp") or {}).get("matrices") or {}
    groups: Dict[int, Dict[str, Any]] = {}
    for key, value in matrices.items():
        match = _FACTOR_MATRIX_RE.match(str(key))
        if not match:
            continue
        letter, index = match.group(1), int(match.group(2))
        groups.setdefault(index, {})[letter] = value
    return {
        index: group
        for index, group in sorted(groups.items())
        if set(group) >= {"A", "B", "C", "D"}
    }


def _canonicalise_factor_b(value: Any, num_actions: int) -> np.ndarray:
    """Canonicalise a per-factor B tensor to ``(next, previous, action)``.

    Mirrors ``POMDPRenderProcessor._canonicalise_factored_B``: action-major
    raw tensors (the generator layout) are transposed and each action slice
    is column-normalised.
    """
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 2:
        tensor = raw[:, :, np.newaxis]
    elif raw.ndim == 3 and raw.shape[0] == num_actions and raw.shape[1] == raw.shape[2]:
        tensor = raw.transpose(2, 1, 0)
    elif raw.ndim == 3 and raw.shape[0] == 1 and raw.shape[1] == raw.shape[2]:
        tensor = raw.transpose(2, 1, 0)
    elif raw.ndim == 3 and raw.shape[0] == raw.shape[1]:
        tensor = raw
    elif raw.ndim == 3:
        tensor = raw
    else:
        raise ValueError(f"B factor must be 2D or 3D, got shape {raw.shape}")
    tensor = tensor.astype(np.float64, copy=True)
    for action in range(tensor.shape[2]):
        column_sums = tensor[:, :, action].sum(axis=0)
        if np.any(~np.isfinite(column_sums)) or np.any(column_sums <= 0):
            raise ValueError(
                f"B factor column has invalid probability mass for action {action}"
            )
        tensor[:, :, action] /= column_sums[np.newaxis, :]
    return tensor


def _generate_jax_factorized_code(
    gnn_spec: Dict[str, Any], options: Optional[Dict[str, Any]]
) -> str:
    """Generate a standalone sparse Kronecker-factorized JAX script."""
    groups = _factor_matrix_groups(gnn_spec)
    if not groups:
        raise ValueError("No complete per-factor A/B/C/D groups found")
    model_name = _jax_model_name(gnn_spec, "GNNModel")
    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("._")

    a_lists: List[Any] = []
    b_lists: List[Any] = []
    c_lists: List[Any] = []
    d_lists: List[Any] = []
    for index in sorted(groups):
        group = groups[index]
        b_canonical = _canonicalise_factor_b(
            group["B"], _factor_action_count(group["B"])
        )
        a_lists.append(np.asarray(group["A"], dtype=np.float64).tolist())
        b_lists.append(b_canonical.tolist())
        c_lists.append(np.asarray(group["C"], dtype=np.float64).flatten().tolist())
        d_lists.append(np.asarray(group["D"], dtype=np.float64).flatten().tolist())

    model_params = gnn_spec.get("model_parameters", {}) or {}
    num_timesteps = int(
        model_params.get("num_timesteps")
        or gnn_spec.get("initialparameterization", {}).get("num_timesteps")
        or 20
    )
    seed = int((options or {}).get("seed", 42))
    action_precision = float((options or {}).get("action_precision", 4.0))

    joint_size = 1
    for factor_list in a_lists:
        joint_size *= len(factor_list[0])

    # nosec
    code = f'''#!/usr/bin/env python3
"""
Kronecker-factorized JAX active inference for: {model_name}

Generated from a GNN specification (roadmap MAJ-02). This script runs sparse
mean-field active inference over per-factor matrices — the joint state space
({joint_size} states) is reported in the results but never allocated. It
requires ``jax`` and ``numpy`` and the GNN repository on the path.

Results are written as ``simulation_results.json`` (schema
``jax_kronecker_factorized_v1``) under ``GNN_OUTPUT_DIR`` (Step 12 sets it
for the ``jax`` framework; defaults to a local output path otherwise).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root resolution (prefer GNN_PROJECT_ROOT; else walk upwards)
# ---------------------------------------------------------------------------
_gnn_root = os.environ.get("GNN_PROJECT_ROOT")
if _gnn_root:
    _repo = Path(_gnn_root).resolve()
    sys.path.insert(0, str(_repo / "src"))
else:
    _cur = Path(__file__).resolve().parent
    _found = None
    for _ in range(24):
        if (_cur / "pyproject.toml").is_file() and (_cur / "src").is_dir():
            _found = _cur
            break
        if _cur.parent == _cur:
            break
        _cur = _cur.parent
    if _found is None:
        print(
            "ERROR: Cannot locate GNN repository root. Run via the pipeline "
            "execute step, or set GNN_PROJECT_ROOT to the checkout root.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.path.insert(0, str(_found / "src"))

from gnn.execute.jax.kronecker_factorized import (  # noqa: E402
    FactorizedPOMDP,
    run_factorized_active_inference,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    """Run sparse mean-field active inference for the embedded factor model."""
    model = FactorizedPOMDP(
        A={a_lists!r},
        B={b_lists!r},
        C={c_lists!r},
        D={d_lists!r},
        T={num_timesteps},
        seed={seed},
        action_precision={action_precision},
    )
    results = run_factorized_active_inference(model)

    out_dir = Path(
        os.environ.get(
            "GNN_OUTPUT_DIR",
            "output/jax_simulations/{safe_model_name}/simulation_data",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "simulation_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))

    logger.info("Simulation completed successfully")
    logger.info("  framework:    %s", results.get("framework"))
    logger.info("  model_kind:   %s", results.get("model_kind"))
    logger.info("  num_factors:  %s", results.get("num_factors"))
    logger.info("  joint_states: %s", results.get("model_parameters", {{}}).get("joint_state_space_size"))
    logger.info("  timesteps:    %s", results.get("num_timesteps"))
    logger.info("  results:      %s", results_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
    return code


def _factor_action_count(value: Any) -> int:
    """Per-factor action count from an action-major raw B tensor."""
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 2:
        return 1
    if raw.ndim == 3 and raw.shape[1] == raw.shape[2]:
        return max(1, int(raw.shape[0]))
    if raw.ndim == 3:
        return max(1, int(raw.shape[-1]))
    return 1
