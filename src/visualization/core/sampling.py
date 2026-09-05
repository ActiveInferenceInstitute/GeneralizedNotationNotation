"""Downsampling for very large GNN models before visualization.

Pure helpers — no matplotlib/networkx imports — so they can be unit-tested in
isolation and reused by any caller that wants to cap variable/matrix counts
before rendering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, TypedDict, cast

VARIABLE_SAMPLE_LIMIT = 100
MATRIX_SAMPLE_LIMIT = 5


class SamplingSummary(TypedDict):
    """Before/after counts recorded on ``parsed_data["_sampling_applied"]``."""

    original_variables: int
    sampled_variables: int
    original_connections: int
    sampled_connections: int


def _filter_connections(
    connections: List[Dict[str, Any]], var_names: Set[str]
) -> List[Dict[str, Any]]:
    """Keep connections whose source and target variables survive sampling."""
    out: List[Dict[str, Any]] = []
    # Imported lazily: advanced_visualization._shared imports back into this
    # package (visualization.matrix_visualizer), and sampling is loaded from
    # visualization.core.process during package init — a top-level import
    # here would be circular.
    from advanced_visualization._shared import normalize_connection_format

    for conn in connections:
        if not isinstance(conn, dict):
            continue
        normalized = normalize_connection_format(conn)
        sources = normalized.get("source_variables") or []
        targets = normalized.get("target_variables") or []
        if any(s in var_names for s in sources) and any(
            t in var_names for t in targets
        ):
            out.append(conn)
    return out


def sample_parsed_data(
    parsed_data: Dict[str, Any],
    variable_limit: int = VARIABLE_SAMPLE_LIMIT,
    matrix_limit: int = MATRIX_SAMPLE_LIMIT,
) -> bool:
    """Downsample ``parsed_data`` in place when it exceeds display limits.

    Returns ``True`` when sampling was applied. When applied, sets
    ``parsed_data["_sampling_applied"]`` to a :class:`SamplingSummary` with
    the before/after counts so callers can surface a note to the user.
    """
    variables = parsed_data.get("variables")
    if not isinstance(variables, list) or len(variables) <= variable_limit:
        return False

    original_variables = len(variables)
    original_connections = len(parsed_data.get("connections") or [])

    truncated = variables[:variable_limit]
    var_names: Set[str] = {
        cast(str, var["name"])
        for var in truncated
        if isinstance(var, dict) and var.get("name")
    }
    parsed_data["variables"] = truncated
    parsed_data["connections"] = _filter_connections(
        parsed_data.get("connections") or [], var_names
    )

    matrices = parsed_data.get("matrices")
    if isinstance(matrices, list) and len(matrices) > matrix_limit:
        parsed_data["matrices"] = matrices[:matrix_limit]

    summary: SamplingSummary = {
        "original_variables": original_variables,
        "sampled_variables": len(parsed_data["variables"]),
        "original_connections": original_connections,
        "sampled_connections": len(parsed_data.get("connections") or []),
    }
    parsed_data["_sampling_applied"] = summary
    return True
