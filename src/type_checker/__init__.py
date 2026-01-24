"""
Type checker module for GNN Processing Pipeline.

This module provides GNN syntax validation and resource estimation.
"""

__version__ = "1.1.3"
FEATURES = {
    "syntax_validation": True,
    "resource_estimation": True,
    "type_checking": True,
    "mcp_integration": True
}

from .processor import GNNTypeChecker, estimate_file_resources

__all__ = [
    '__version__',
    'FEATURES',
    'GNNTypeChecker',
    'estimate_file_resources'
] 