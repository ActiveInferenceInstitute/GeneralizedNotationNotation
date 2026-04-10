#!/usr/bin/env python3
"""
Stan Renderer — Generates Stan probabilistic programming models from GNN specs.

Produces Stan model code with data{}, parameters{}, and model{} blocks
from parsed GNN state-space variables and connections.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def render_stan(
    variables: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    model_name: str = "gnn_model",
) -> str:
    """
    Generate Stan model code from GNN variables and connections.

    Args:
        variables: Parsed variable dicts with name, dimensions, dtype.
        connections: Parsed connection dicts with source, target, directed.
        model_name: Model identifier for comments.

    Returns:
        Stan model code string.
    """
    data_vars = []
    param_vars = []
    model_lines = []

    # Classify variables
    for v in variables:
        name = v.get("name", "x")
        dims = v.get("dimensions", [])
        dtype = v.get("dtype", "real")

        stan_type = _stan_type(dtype, dims)

        # Heuristic: observed variables → data, latent → parameters
        if name.lower() in ("o", "obs", "y", "data", "u", "t"):
            data_vars.append(f"  {stan_type} {name};")
        else:
            param_vars.append(f"  {stan_type} {name};")

    # Generate model block from connections
    for conn in connections:
        source = conn.get("source", "")
        target = conn.get("target", "")
        directed = conn.get("directed", True)

        if directed:
            model_lines.append(f"  // {source} → {target}")
            model_lines.append(f"  {target} ~ normal({source}, 1.0);")

    # Assemble
    lines = [
        f"// Stan model generated from GNN: {model_name}",
        f"// Variables: {len(variables)}, Connections: {len(connections)}",
        "",
        "data {",
    ]
    lines.extend(data_vars if data_vars else ["  // No observed variables declared"])
    lines.append("}")
    lines.append("")
    lines.append("parameters {")
    lines.extend(param_vars if param_vars else ["  // No parameters declared"])
    lines.append("}")
    lines.append("")
    lines.append("model {")
    lines.extend(model_lines if model_lines else ["  // No connections to model"])
    lines.append("}")

    code = "\n".join(lines)
    logger.info(f"🔧 Stan model generated: {len(data_vars)} data, {len(param_vars)} params")
    return code


def _stan_type(dtype: str, dims: list) -> str:
    """Map GNN type+dims to Stan type declaration."""
    base = "real" if dtype in ("float", "double", "real") else "int"

    # Defensively parsing dims to ignore malformed dimensions
    valid_dims = [d for d in dims if isinstance(d, int) and d > 0]

    if not valid_dims:
        return base
    elif len(valid_dims) == 1:
        return f"vector[{valid_dims[0]}]"
    elif len(valid_dims) == 2:
        return f"matrix[{valid_dims[0]}, {valid_dims[1]}]"
    else:
        # Stan doesn't have native 3D arrays, use array syntax
        inner = f"matrix[{valid_dims[-2]}, {valid_dims[-1]}]"
        for d in reversed(valid_dims[:-2]):
            inner = f"array[{d}] {inner}"
        return inner
