"""
Model Registry Module

This module provides model versioning, registry management, and metadata handling
for GNN model specifications.
"""

__version__ = "1.1.3"
FEATURES = {
    "model_versioning": True,
    "registry_management": True,
    "metadata_handling": True,
    "mcp_integration": True
}

from pathlib import Path
from typing import Dict, Any, Optional, List

# Import core registry functionality
from .registry import ModelRegistry, process_model_registry

# Re-export main classes and functions
__all__ = [
    '__version__',
    'FEATURES',
    'ModelRegistry',
    'process_model_registry'
]
