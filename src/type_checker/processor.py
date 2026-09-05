"""Type checker public facade for the GNN pipeline."""

# Re-export from the checking subpackage.
from typing import Any

from .checking import (
    GNNTypeChecker,
    ResourceEstimate,
    ValidationSummary,
    estimate_file_resources,
    extract_b_matrix_evidence,
    extract_gnn_dimensions,
    extract_gnn_dimensions_with_diagnostics,
    parse_state_variables,
    summarize_type_check_results,
    validate_dimension_compatibility,
)

__all__: list[Any] = [
    "GNNTypeChecker",
    "ResourceEstimate",
    "ValidationSummary",
    "estimate_file_resources",
    "extract_b_matrix_evidence",
    "extract_gnn_dimensions",
    "extract_gnn_dimensions_with_diagnostics",
    "parse_state_variables",
    "summarize_type_check_results",
    "validate_dimension_compatibility",
]
