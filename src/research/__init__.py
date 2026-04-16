"""
research module for GNN Processing Pipeline.

This module provides research capabilities with recovery implementations.
"""

# Import processor functions - single source of truth
from .processor import process_research

# Module metadata
__version__ = "1.6.0"
__author__ = "Active Inference Institute"
__description__ = "research processing for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'basic_processing': True,
    'fallback_mode': True
}


__all__ = [
    'process_research',
    'FEATURES',
    '__version__'
]


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "research",
        "version": __version__,
        "description": "Research workflow management and experimental tools",
        "features": FEATURES,
    }
