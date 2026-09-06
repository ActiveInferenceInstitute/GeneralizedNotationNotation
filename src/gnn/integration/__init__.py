"""
integration module for GNN Processing Pipeline.

This module provides integration capabilities including:
- System-level consistency checks and dependency graph analysis
- Meta-analysis of parameter sweep runtime and simulation outputs
"""

from typing import Any

from .graph import (
    SystemAnalysis,
    SystemGraphStats,
    analyze_system,
    build_system_graph,
    export_dependency_graph,
    verify_references,
)
from .meta_analysis import SweepDataCollector, SweepRecord, run_meta_analysis
from .processor import process_integration

_META_ANALYSIS_AVAILABLE = True


# Module metadata
__version__ = "1.7.0"
__author__ = "Active Inference Institute"
__description__ = "integration processing for GNN Processing Pipeline"

# Feature availability flags
FEATURES: dict[str, Any] = {
    "basic_processing": True,
    "meta_analysis": _META_ANALYSIS_AVAILABLE,
    "system_graph_export": True,
}


__all__: list[Any] = [
    "process_integration",
    "analyze_system",
    "build_system_graph",
    "verify_references",
    "export_dependency_graph",
    "SystemAnalysis",
    "SystemGraphStats",
    "run_meta_analysis",
    "SweepDataCollector",
    "SweepRecord",
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
