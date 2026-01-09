"""
GNN Pipeline Core Module

This module provides the core functionality for the GNN processing pipeline.
"""

from pathlib import Path
from typing import List

# Package-level metadata expected by tests
__version__ = "1.1.1"
FEATURES = {
    "pipeline_orchestration": True,
    "mcp_integration": True,
}


def _discover_top_level_modules() -> List[str]:
    """Discover all top-level subpackages under the src package.

    Returns a sorted list of directory names that contain an __init__.py file
    and do not start with an underscore.
    """
    base_dir = Path(__file__).parent
    module_names: List[str] = []
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith('_'):
            continue
        if (entry / '__init__.py').exists():
            module_names.append(name)
    return sorted(module_names)


def get_module_info() -> dict[str, object]:
    """Get information about the core GNN package.

    Returns a dictionary with high-level metadata about the overall package.
    """
    return {
        "name": "GNN Pipeline Core",
        "version": __version__,
        "description": "Core functionality for GNN processing pipeline",
        "modules": _discover_top_level_modules(),
        "features": [
            "Pipeline orchestration",
            "GNN processing",
            "Analysis and statistics",
            "LLM integration",
            "Code generation",
            "Website generation",
            "Security validation",
            "Advanced visualization",
        ],
    }

# Expose submodules expected by tests and users via `import src`
# Use lazy/guarded import to avoid import-time failures when optional deps are missing in test isolation
import importlib

# Import SAPF module directly now that it is fully implemented and robust
try:
    sapf = importlib.import_module('src.sapf')
except ImportError:
    # Fallback to relative import if package context differs
    from . import sapf

__all__ = [
    'get_module_info',
    'sapf',
    '__version__',
    'FEATURES',
]