"""
Type checking subsystem.

Contains the core logic for GNN type validation, dimension checking,
and structural consistency analysis.
"""

from .core import GNNTypeChecker, estimate_file_resources
from .dimensions import extract_gnn_dimensions, validate_dimension_compatibility
from .rules import (
    check_type_consistency,
    extract_types_from_content,
    get_validation_rules,
    validate_type,
)

__all__ = [
    "GNNTypeChecker",
    "check_type_consistency",
    "estimate_file_resources",
    "extract_gnn_dimensions",
    "extract_types_from_content",
    "get_validation_rules",
    "validate_dimension_compatibility",
    "validate_type",
]
