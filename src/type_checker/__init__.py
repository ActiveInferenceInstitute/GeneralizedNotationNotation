"""
Type checker module for GNN Processing Pipeline.

This module provides GNN syntax validation and resource estimation.
"""

__version__ = "1.6.0"
FEATURES = {
    "syntax_validation": True,
    "resource_estimation": True,
    "type_checking": True,
    "mcp_integration": True
}

from .checking import GNNTypeChecker, estimate_file_resources

__all__ = [
    '__version__',
    'FEATURES',
    'GNNTypeChecker',
    'estimate_file_resources'
]


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "type_checker",
        "version": __version__,
        "description": "Static type analysis and resource estimation for GNN models",
        "features": FEATURES,
    }
