#!/usr/bin/env python3
"""
Shared visualization utilities for the analysis module.

Centralizes matplotlib backend setup, availability checking, and common
plotting helpers used across all framework-specific analyzers.

Usage:
    from analysis.viz_base import plt, np, MATPLOTLIB_AVAILABLE, safe_savefig
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# --- Centralized matplotlib setup ---
MATPLOTLIB_AVAILABLE = False
plt = None
np = None
patches = None
sns = None

try:
    import numpy as _np
    np = _np
except ImportError:
    logger.debug("numpy not available")

try:
    import matplotlib as _mpl
    _mpl.use('Agg')
    import matplotlib.pyplot as _plt
    plt = _plt
    MATPLOTLIB_AVAILABLE = True
except (ImportError, RecursionError):
    logger.debug("matplotlib not available")

try:
    import matplotlib.patches as _patches
    patches = _patches
except (ImportError, AttributeError):
    pass

try:
    import seaborn as _sns
    sns = _sns
except ImportError:
    pass


def safe_savefig(
    output_path: Path,
    dpi: int = 300,
    bbox_inches: str = 'tight',
    log: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    Save the current matplotlib figure and close it.

    Wraps plt.savefig + plt.close with error handling and logging.

    Args:
        output_path: Path to save the figure
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box specification
        log: Optional logger instance

    Returns:
        String path to saved file, or None on failure
    """
    _log = log or logger
    if not MATPLOTLIB_AVAILABLE or plt is None:
        _log.warning("matplotlib not available, cannot save figure")
        return None

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=dpi, bbox_inches=bbox_inches)
        plt.close()
        _log.info(f"Saved visualization: {output_path.name}")
        return str(output_path)
    except Exception as e:
        _log.warning(f"Failed to save figure to {output_path}: {e}")
        try:
            plt.close()
        except Exception:
            pass
        return None


def check_matplotlib() -> bool:
    """Check if matplotlib is available for visualization."""
    return MATPLOTLIB_AVAILABLE


__all__ = [
    'plt',
    'np',
    'patches',
    'sns',
    'MATPLOTLIB_AVAILABLE',
    'safe_savefig',
    'check_matplotlib',
]
