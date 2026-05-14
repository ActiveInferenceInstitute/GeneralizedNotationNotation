"""
integration module for GNN Processing Pipeline.

This module provides integration capabilities with recovery implementations,
including:
- System-level consistency checks and dependency graph analysis
- Meta-analysis of parameter sweep runtime and simulation outputs
"""

# Import processor functions - single source of truth
from .processor import process_integration

# Meta-analysis submodule.
try:
    from .meta_analysis import SweepDataCollector, SweepRecord, run_meta_analysis

    _META_ANALYSIS_AVAILABLE = True
except ImportError as exc:
    _META_ANALYSIS_AVAILABLE = False
    _META_ANALYSIS_IMPORT_ERROR = exc

    def run_meta_analysis(*args, **kwargs):
        """Raise a clear dependency error when meta-analysis cannot load."""
        raise RuntimeError(
            "integration.meta_analysis is unavailable"
        ) from _META_ANALYSIS_IMPORT_ERROR


# Module metadata
__version__ = "1.6.0"
__author__ = "Active Inference Institute"
__description__ = "integration processing for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    "basic_processing": True,
    "meta_analysis": _META_ANALYSIS_AVAILABLE,
}


__all__ = [
    "process_integration",
    "run_meta_analysis",
    "FEATURES",
    "__version__",
]


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "integration",
        "version": __version__,
        "description": "Cross-module system integration and coordination",
        "features": FEATURES,
    }
