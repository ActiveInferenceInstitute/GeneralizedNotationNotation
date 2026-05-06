"""
integration module for GNN Processing Pipeline.

This module provides integration capabilities with recovery implementations,
including:
- System-level consistency checks and dependency graph analysis
- Meta-analysis of parameter sweep runtime and simulation outputs
"""

# Import processor functions - single source of truth
from .processor import process_integration

# Meta-analysis submodule (lazy import to avoid hard dependency on matplotlib)
try:
    from .meta_analysis import run_meta_analysis, SweepDataCollector, SweepRecord
    _META_ANALYSIS_AVAILABLE = True
except ImportError:
    _META_ANALYSIS_AVAILABLE = False

    def run_meta_analysis(*args, **kwargs):
        """Stub when meta_analysis dependencies are unavailable."""
        import logging
        logging.getLogger(__name__).warning("meta_analysis submodule unavailable")
        return None

# Module metadata
__version__ = "1.6.0"
__author__ = "Active Inference Institute"
__description__ = "integration processing for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'basic_processing': True,
    'fallback_mode': True,
    'meta_analysis': _META_ANALYSIS_AVAILABLE,
}


__all__ = [
    'process_integration',
    'run_meta_analysis',
    'FEATURES',
    '__version__',
]


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "integration",
        "version": __version__,
        "description": "Cross-module system integration and coordination",
        "features": FEATURES,
    }
