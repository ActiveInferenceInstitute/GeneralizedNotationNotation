"""
research module for GNN Processing Pipeline.

This module provides research capabilities with recovery implementations.
"""

import logging as _logging
from pathlib import Path as _Path
from typing import Any as _Any

# Phase 1.2: soft-import the processor. Step 19 is NOT a hard-import step,
# so missing optional deps must degrade to a warning (exit-code 2) rather
# than crash at module load. Mirrors the pattern used in src/execute/__init__.py.
try:
    from .processor import process_research
    _PROCESSOR_AVAILABLE = True
except ImportError as _err:  # pragma: no cover - exercised via test_research_soft_import
    _PROCESSOR_AVAILABLE = False
    _IMPORT_ERROR = str(_err)

    def process_research(target_dir: _Path, output_dir: _Path, **_kwargs: _Any) -> int:  # type: ignore[misc]
        """Recovery stub when the real processor can't be imported.

        Returns exit-code 2 (warnings/skipped) so the pipeline template surfaces
        a warning log line without marking the step as a hard failure.
        """
        _logging.getLogger(__name__).warning(
            f"Research processor unavailable; skipping step 19. Cause: {_IMPORT_ERROR}"
        )
        return 2

# Module metadata
__version__ = "1.6.0"
__author__ = "Active Inference Institute"
__description__ = "research processing for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'basic_processing': _PROCESSOR_AVAILABLE,
    'fallback_mode': True,
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
