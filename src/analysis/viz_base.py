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

# --- Centralized matplotlib setup (shared via visualization._viz_compat) ---
from visualization._viz_compat import MATPLOTLIB_AVAILABLE, np, plt, sns

patches = None
try:
    import matplotlib.patches as _patches
    patches = _patches
except (ImportError, AttributeError) as e:
    logger.debug("matplotlib.patches not available: %s", e)


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
        except (OSError, ValueError) as e:
            logger.debug(f"plt.close() failed (non-fatal): {e}")
            pass  # nosec B110 -- intentional: plt.close() failure is non-fatal
        return None


__all__ = [
    'plt',
    'np',
    'patches',
    'sns',
    'MATPLOTLIB_AVAILABLE',
    'safe_savefig',
]
