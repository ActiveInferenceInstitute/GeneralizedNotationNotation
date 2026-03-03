#!/usr/bin/env python3
"""
Shared matrix utilities for GNN renderers.

Provides common matrix operations used by all framework renderers:
- Column normalization (stochastic matrix enforcement)
- ABCD shape validation for POMDP models

Addresses improvement items:
  P-1: Matrix normalization extracted from PyMDP renderer
  P-4: ABCD shape validation before simulation
"""
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Normalize matrix columns to sum to 1 (stochastic matrix).

    Each column is divided by its sum. Zero-sum columns are left unchanged.

    Args:
        matrix: 2D numpy array to normalize.

    Returns:
        Column-normalized copy of the matrix.
    """
    if matrix.ndim != 2:
        return matrix
    result = matrix.copy().astype(float)
    col_sums = result.sum(axis=0)
    nonzero = col_sums > 0
    result[:, nonzero] = result[:, nonzero] / col_sums[nonzero]
    return result


def validate_abcd_shapes(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
) -> Tuple[bool, str]:
    """Validate POMDP matrix shapes for consistency.

    Checks:
      - A is 2D with shape (num_obs, num_states)
      - B is 2D (num_states, num_states) or 3D (num_states, num_states, num_actions)
      - C is 1D with length num_obs
      - D is 1D with length num_states

    Args:
        A: Observation model matrix.
        B: Transition model matrix.
        C: Preference vector (log-preferences over observations).
        D: Prior belief vector over states.

    Returns:
        Tuple of (is_valid, message). message is "ok" if valid.
    """
    issues = []

    if A.ndim != 2:
        issues.append(f"A should be 2D, got {A.ndim}D shape {A.shape}")
    if B.ndim not in (2, 3):
        issues.append(f"B should be 2D or 3D, got {B.ndim}D shape {B.shape}")

    if C.ndim != 1:
        issues.append(f"C should be 1D, got {C.ndim}D shape {C.shape}")
    if D.ndim != 1:
        issues.append(f"D should be 1D, got {D.ndim}D shape {D.shape}")

    if A.ndim == 2 and D.ndim == 1:
        num_states = A.shape[1]
        if D.shape[0] != num_states:
            issues.append(
                f"D length ({D.shape[0]}) != A columns ({num_states})"
            )

    if A.ndim == 2 and C.ndim == 1:
        num_obs = A.shape[0]
        if C.shape[0] != num_obs:
            issues.append(
                f"C length ({C.shape[0]}) != A rows ({num_obs})"
            )

    if A.ndim == 2 and B.ndim >= 2:
        if B.shape[0] != A.shape[1]:
            issues.append(
                f"B rows ({B.shape[0]}) != A columns ({A.shape[1]})"
            )
        if B.shape[1] != A.shape[1]:
            issues.append(
                f"B cols ({B.shape[1]}) != num_states ({A.shape[1]})"
            )

    if issues:
        msg = "; ".join(issues)
        logger.warning(f"ABCD shape validation failed: {msg}")
        return False, msg

    return True, "ok"
