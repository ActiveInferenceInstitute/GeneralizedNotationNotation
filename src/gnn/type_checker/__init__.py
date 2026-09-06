"""
Type checker module for GNN Processing Pipeline.

This module provides GNN syntax validation and resource estimation.
"""

from typing import Any

__version__ = "1.7.0"
FEATURES: dict[str, Any] = {
    "syntax_validation": True,
    "resource_estimation": True,
    "type_checking": True,
    "mcp_integration": True,
    "content_validation": True,
    "strict_mode": True,
    "validation_summary": True,
}

from .checking import (
    GNNTypeChecker,
    ResourceEstimate,
    ValidationSummary,
    estimate_file_resources,
    extract_gnn_dimensions,
    summarize_type_check_results,
    validate_dimension_compatibility,
)

__all__: list[Any] = [
    "FEATURES",
    "GNNTypeChecker",
    "ResourceEstimate",
    "ValidationSummary",
    "__version__",
    "estimate_file_resources",
    "extract_gnn_dimensions",
    "summarize_type_check_results",
    "validate_dimension_compatibility",
]


def get_module_info() -> dict[str, Any]:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "type_checker",
        "version": __version__,
        "description": "Static type analysis and resource estimation for GNN models",
        "features": FEATURES,
    }
