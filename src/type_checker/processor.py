"""Type checker public facade for the GNN pipeline."""

# Re-export from the checking subpackage.
from .checking.core import GNNTypeChecker, estimate_file_resources
from .checking.dimensions import (
    extract_gnn_dimensions,
    validate_dimension_compatibility,
)

__all__ = [
    "GNNTypeChecker",
    "estimate_file_resources",
    "extract_gnn_dimensions",
    "validate_dimension_compatibility",
]
