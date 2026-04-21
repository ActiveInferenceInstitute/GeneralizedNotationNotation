"""
research module for GNN Processing Pipeline.

Rule-based static analysis (no external LLM required). All dependencies
are core Python / stdlib — imports are unconditional per Phase 6.
"""

from .processor import process_research

__version__ = "1.6.0"
__author__ = "Active Inference Institute"
__description__ = "research processing for GNN Processing Pipeline"

FEATURES = {
    'basic_processing': True,
    'fallback_mode': True,  # Documented in CLAUDE.md: rule-based, no LLM required
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
