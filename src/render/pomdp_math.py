#!/usr/bin/env python3
"""
POMDP math helpers for the render module.

Numeric normalization helpers (probability vectors, column-stochastic
matrices) and Kronecker factored-spec detection / joint-action decoding used
by ``pomdp_processor`` when composing factored POMDPs into joint PyMDP-style
matrices.
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    from gnn.pomdp_extractor import POMDPStateSpace


def _is_kronecker_factorized_spec(pomdp_space: "POMDPStateSpace") -> bool:
    """Detect specs with independent per-factor action spaces (MAJ-02).

    The Kronecker-factorized generator (``scripts/pymdp_spec_generator.py::
    generate_factorized_gnn_file``) declares one action variable ``u_fN`` per
    factor and does **not** surface shared control factors, whereas
    shared-control factored specs (gridworld) and multi-agent specs populate
    ``control_factors``. For the former, the joint action space is the
    *product* of the per-factor action counts and the joint model is the
    Kronecker composition; for the latter, one joint action index is shared
    across all factors.
    """
    matrices = getattr(pomdp_space, "matrices", None) or {}
    b_factor_keys = [key for key in matrices if re.match(r"^B_f\d+$", str(key))]
    if len(b_factor_keys) < 2:
        return False
    if getattr(pomdp_space, "control_factors", None):
        return False
    return True


def _factor_action_counts(matrices: Dict[str, Any], b_keys: List[str]) -> List[int]:
    """Per-factor action counts from action-major ``B_fN`` tensors."""
    counts: List[int] = []
    for key in b_keys:
        raw = np.asarray(matrices[key], dtype=np.float64)
        if raw.ndim == 2:
            counts.append(1)
        elif raw.ndim == 3 and raw.shape[1] == raw.shape[2]:
            counts.append(max(1, int(raw.shape[0])))
        elif raw.ndim == 3 and raw.shape[0] == raw.shape[1]:
            counts.append(max(1, int(raw.shape[2])))
        else:
            counts.append(max(1, int(raw.shape[-1])))
    return counts


def _mixed_radix_digit(action: int, radices: List[int], index: int) -> int:
    """Decode a flat joint action into one factor's action index (LSB-first)."""
    for factor_index, radix in enumerate(radices):
        digit = action % radix
        if factor_index == index:
            return digit
        action //= radix
    return 0


def _normalise_prob_vector(values: np.ndarray) -> np.ndarray:
    """Normalize prob vector."""
    vector = np.asarray(values, dtype=np.float64).flatten()
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0:
        return np.ones(max(vector.shape[0], 1), dtype=np.float64) / max(
            vector.shape[0], 1
        )
    return vector / total


def _normalise_columns(matrix: np.ndarray) -> np.ndarray:
    """Normalize columns."""
    out = np.asarray(matrix, dtype=np.float64).copy()
    if out.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {out.shape}")
    column_sums = out.sum(axis=0, keepdims=True)
    zero_columns = column_sums <= 0
    column_sums = np.where(zero_columns, 1.0, column_sums)
    out = out / column_sums
    if zero_columns.any():
        rows = out.shape[0]
        for column in np.where(zero_columns.flatten())[0]:
            out[:, column] = 1.0 / rows
    return out
