"""Numeric extraction from GNN parameter payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from visualization.compat.viz_compat import np

NUMPY_AVAILABLE = np is not None


def convert_to_matrix(value: Any, name: str = "") -> Optional[Any]:
    """Convert nested lists / tuples to a numpy array, or None."""
    if not NUMPY_AVAILABLE or np is None:
        return None
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)):
            if len(value) > 0 and isinstance(value[0], (list, tuple)):
                matrix = np.array(value, dtype=float)
                if matrix.size > 0:
                    return matrix
            matrix = np.array(value, dtype=float)
            if matrix.size > 0:
                return matrix
        matrix = np.array(value, dtype=float)
        if matrix.size > 0:
            return matrix
    except (ValueError, TypeError):
        return None
    return None


def extract_matrix_data_from_parameters(
    parameters: Union[List[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Map parameter names to numpy arrays (same rules as MatrixVisualizer)."""
    matrices: Dict[str, Any] = {}
    if not parameters:
        return matrices
    if isinstance(parameters, dict):
        for param_name, param_value in parameters.items():
            matrix = convert_to_matrix(param_value, param_name)
            if matrix is not None:
                matrices[param_name] = matrix
        return matrices
    for param in parameters:
        if not isinstance(param, dict):
            continue
        param_name = param.get("name", "")
        if not param_name:
            continue
        param_value = param.get("value")
        if param_value is None:
            continue
        matrix = convert_to_matrix(param_value, param_name)
        if matrix is not None:
            matrices[param_name] = matrix
    return matrices
