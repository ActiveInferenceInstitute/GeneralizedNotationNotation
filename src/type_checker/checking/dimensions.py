"""
Dimension extraction and compatibility validation for GNN models.

Provides functions to parse the ``StateSpaceBlock`` from GNN file content
and validate Active Inference POMDP dimensional constraints (A, B, C, D).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

_logger = logging.getLogger(__name__)


def extract_gnn_dimensions(content: str) -> Dict[str, Any]:
    """Extract variable dimensions from GNN StateSpaceBlock content.

    Parses patterns like: ``A[3,3,type=float]``, ``s[3,1,type=float]``.

    Args:
        content: Full GNN file content as string.

    Returns:
        Dict mapping variable names to their dimension lists.
    """
    variables: Dict[str, Any] = {}

    # Match: varname[dim1,dim2,...,type=xxx] or varname[dim1,dim2,...]
    pattern = r'^([A-Za-z_][A-Za-z0-9_\']*)\s*\[([^\]]+)\]'

    in_state_space = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## StateSpaceBlock"):
            in_state_space = True
            continue
        elif stripped.startswith("##") and in_state_space:
            in_state_space = False
            continue

        if in_state_space:
            match = re.match(pattern, stripped)
            if match:
                var_name = match.group(1)
                dim_str = match.group(2)
                # Parse dimensions (skip type=xxx entries)
                dims: list[int] = []
                for part in dim_str.split(","):
                    part = part.strip()
                    if part.startswith("type=") or part.startswith("π") or not part:
                        continue
                    try:
                        dims.append(int(part))
                    except ValueError:
                        _logger.log(
                            5,
                            "Skipping non-integer dimension token: %s",
                            part,
                        )
                if dims:
                    variables[var_name] = dims

    return variables


def validate_dimension_compatibility(variables: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that matrix/tensor dimensions are compatible in a GNN model.

    Checks Active Inference POMDP constraints:

    - Likelihood matrix ``A[obs, states]``: columns must match hidden state count
    - Transition tensor ``B[states, states, actions]``: first two dims must match
    - Preference ``C[obs]``: length must match A's first dimension
    - Prior ``D[states]``: length must match A's second dimension

    Args:
        variables: Dict mapping variable names to their dimension specs,
                   e.g. ``{"A": [3,3], "B": [3,3,3], "s": [3,1]}``.

    Returns:
        Dict with keys: ``compatible`` (bool), ``issues`` (list), ``warnings`` (list),
        ``variables_checked`` (list), ``dimension_map`` (dict).
    """
    issues: list[str] = []
    warnings: list[str] = []

    # Parse dimension specs: extract variables with numeric dimensions
    dims: Dict[str, list[int]] = {}
    for name, spec in variables.items():
        if isinstance(spec, (list, tuple)) and all(isinstance(d, int) for d in spec):
            dims[name] = list(spec)

    # Check A-s compatibility: A[obs, states], s[states, 1]
    if "A" in dims and "s" in dims:
        a_dims = dims["A"]
        s_dims = dims["s"]
        if len(a_dims) >= 2 and len(s_dims) >= 1:
            if a_dims[1] != s_dims[0]:
                issues.append(
                    f"Dimension mismatch: A[{a_dims[0]},{a_dims[1]}] column count ({a_dims[1]}) "
                    f"!= s[{s_dims[0]},...] row count ({s_dims[0]}). "
                    f"A's columns must equal the number of hidden states."
                )

    # Check B symmetry: B[states, states, actions] -- first two dims must match
    if "B" in dims:
        b_dims = dims["B"]
        if len(b_dims) >= 2 and b_dims[0] != b_dims[1]:
            issues.append(
                f"Transition matrix B[{','.join(str(d) for d in b_dims)}]: "
                f"first two dimensions must match (got {b_dims[0]} != {b_dims[1]}). "
                f"B[next_states, prev_states, actions] requires next_states == prev_states."
            )

    # Check A-B state dimension consistency
    if "A" in dims and "B" in dims:
        a_dims = dims["A"]
        b_dims = dims["B"]
        if len(a_dims) >= 2 and len(b_dims) >= 1:
            if a_dims[1] != b_dims[0]:
                issues.append(
                    f"State dimension mismatch between A and B: "
                    f"A has {a_dims[1]} hidden states, B has {b_dims[0]} states. "
                    f"Must be equal."
                )

    # Check C-A observation compatibility
    if "C" in dims and "A" in dims:
        c_dims = dims["C"]
        a_dims = dims["A"]
        if len(c_dims) >= 1 and len(a_dims) >= 1:
            c_obs = c_dims[0]
            a_obs = a_dims[0]
            if c_obs != a_obs:
                issues.append(
                    f"Preference vector C[{c_obs}] length != A observation dimension A[{a_obs},...]. "
                    f"C must have one entry per observation outcome."
                )

    # Check D-s prior compatibility
    if "D" in dims and "s" in dims:
        d_dims = dims["D"]
        s_dims = dims["s"]
        if len(d_dims) >= 1 and len(s_dims) >= 1:
            if d_dims[0] != s_dims[0]:
                issues.append(
                    f"Prior D[{d_dims[0]}] length != hidden state s[{s_dims[0]},...] count. "
                    f"D must have one entry per hidden state."
                )

    # Warn about very large dimensions (tractability)
    for name, d in dims.items():
        total_elements = 1
        for dim in d:
            total_elements *= dim
        if total_elements > 10000:
            warnings.append(
                f"Variable {name} with dimensions {d} has {total_elements} total elements. "
                f"Consider dimensionality reduction for tractable inference."
            )

    return {
        "compatible": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "variables_checked": list(dims.keys()),
        "dimension_map": dims,
    }
