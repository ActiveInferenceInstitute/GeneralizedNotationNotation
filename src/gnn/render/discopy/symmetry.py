"""Matrix permutation metadata for DisCoPy symmetry rendering."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def nested_shape(value: Any) -> List[int]:
    """Return the rectangular nested-list shape for a matrix-like value."""
    shape: List[int] = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else []
    return shape


def validate_permutation(shape: Sequence[int], permutation: Sequence[int]) -> None:
    """Validate a row-axis permutation against a matrix/tensor shape."""
    if not shape:
        raise ValueError("Cannot permute scalar matrix data")
    expected = int(shape[0])
    if len(permutation) != expected:
        raise ValueError(
            f"Permutation length {len(permutation)} does not match first matrix dimension {expected}"
        )
    if sorted(int(item) for item in permutation) != list(range(expected)):
        raise ValueError(
            f"Permutation must contain each row index exactly once: {permutation}"
        )


def validate_matrix_permutation_metadata(metadata: Dict[str, Dict[str, Any]]) -> None:
    """Validate generated matrix permutation metadata records."""
    if not isinstance(metadata, dict):
        raise ValueError("Matrix permutation metadata must be a dictionary")
    for matrix_name, record in metadata.items():
        if not isinstance(record, dict):
            raise ValueError(f"Permutation metadata for {matrix_name} must be a dict")
        if record.get("axis") != "rows":
            raise ValueError(f"Unsupported permutation axis for {matrix_name}")
        shape = record.get("shape")
        permutation = record.get("permutation")
        if not isinstance(shape, list) or not isinstance(permutation, list):
            raise ValueError(
                f"Permutation metadata for {matrix_name} requires shape and permutation lists"
            )
        validate_permutation(shape, permutation)


def build_matrix_permutation_metadata(
    gnn_spec: Dict[str, Any],
    matrix_permutations: Dict[str, Sequence[int]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Build permutation metadata from canonical parsed GNN parameter matrices."""
    params = (
        gnn_spec.get("initial_parameterization")
        or gnn_spec.get("initialparameterization")
        or {}
    )
    permutations = matrix_permutations or params.get("matrix_permutations") or {}
    metadata: Dict[str, Dict[str, Any]] = {}
    for matrix_name, permutation in permutations.items():
        if matrix_name not in params:
            raise ValueError(f"Permutation references missing matrix '{matrix_name}'")
        shape = nested_shape(params[matrix_name])
        validate_permutation(shape, permutation)
        metadata[matrix_name] = {
            "axis": "rows",
            "shape": shape,
            "permutation": [int(item) for item in permutation],
        }
    return metadata
