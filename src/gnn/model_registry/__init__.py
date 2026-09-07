"""
Model Registry Module

This module provides model versioning, registry management, and metadata handling
for GNN model specifications.
"""

from typing import Any

__version__ = "3.3.0"
FEATURES: dict[str, Any] = {
    "model_versioning": True,
    "registry_management": True,
    "metadata_handling": True,
    "mcp_integration": True,
}

from pathlib import Path
from typing import Any, Dict, List, Optional

# Import core registry functionality
from .registry import (
    ModelEntry,
    ModelRegistry,
    ModelVersion,
    process_model_registry,
)

# Re-export main classes and functions
__all__: list[Any] = [
    "__version__",
    "FEATURES",
    "ModelEntry",
    "ModelRegistry",
    "ModelVersion",
    "process_model_registry",
]


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "model_registry",
        "version": __version__,
        "description": "Model versioning, metadata management, and lifecycle tracking",
        "features": FEATURES,
    }
