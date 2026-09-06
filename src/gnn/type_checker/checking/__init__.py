"""
Type checking subsystem.

Contains the core logic for GNN type validation, dimension checking,
and structural consistency analysis.
"""

from typing import Any

from .core import GNNTypeChecker, ResourceEstimate, estimate_file_resources
from .dimensions import (
    extract_b_matrix_evidence,
    extract_gnn_dimensions,
    extract_gnn_dimensions_with_diagnostics,
    parse_state_variables,
    validate_dimension_compatibility,
)
from .rules import (
    VALID_TYPES,
    check_type_consistency,
    extract_types_from_content,
    get_validation_rules,
    validate_type,
)
from .sections import (
    CANONICAL_GNN_SECTIONS,
    connection_group,
    detect_time_dynamics,
    extract_markdown_section,
    parse_resource_connections,
    section_presence,
)
from .summary import ValidationSummary, summarize_type_check_results

__all__: list[Any] = [
    "CANONICAL_GNN_SECTIONS",
    "GNNTypeChecker",
    "ResourceEstimate",
    "VALID_TYPES",
    "ValidationSummary",
    "check_type_consistency",
    "connection_group",
    "detect_time_dynamics",
    "estimate_file_resources",
    "extract_b_matrix_evidence",
    "extract_gnn_dimensions",
    "extract_gnn_dimensions_with_diagnostics",
    "extract_markdown_section",
    "extract_types_from_content",
    "get_validation_rules",
    "parse_resource_connections",
    "parse_state_variables",
    "section_presence",
    "summarize_type_check_results",
    "validate_dimension_compatibility",
    "validate_type",
]
