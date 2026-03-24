"""Shared matplotlib save helpers for visualization (no imports from core.process)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except (ImportError, RecursionError):
    plt = None
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


def save_plot_safely(plot_path: Path, dpi: int = 300, **savefig_kwargs: Any) -> bool:
    """Save current figure with DPI fallbacks."""
    if plt is None:
        return False

    def _safe_dpi_value(dpi_input: Any) -> int:
        try:
            dpi_val = int(dpi_input) if isinstance(dpi_input, (int, float)) else 150
            return max(50, min(dpi_val, 600))
        except (ValueError, TypeError, OverflowError):
            return 150

    safe_dpi = _safe_dpi_value(dpi)
    try:
        plt.savefig(plot_path, dpi=safe_dpi, **savefig_kwargs)
        logger.debug("Saved plot with DPI %s", safe_dpi)
        return True
    except Exception as e:
        logger.debug("Error saving with DPI %s: %s", safe_dpi, e)
        try:
            fallback_dpi = _safe_dpi_value(matplotlib.rcParams.get("savefig.dpi", 100))
            plt.savefig(plot_path, dpi=fallback_dpi, **savefig_kwargs)
            return True
        except Exception as e2:
            logger.debug("Recovery DPI failed: %s", e2)
            try:
                plt.savefig(plot_path, **savefig_kwargs)
                return True
            except Exception as e3:
                logger.error("Failed to save plot %s: %s", plot_path, e3)
                return False


def safe_tight_layout() -> None:
    """Apply tight_layout with warning suppression."""
    if plt is None:
        return
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=".*[Tt]ight.?layout.*"
            )
            plt.tight_layout()
    except (ValueError, RuntimeError):
        logger.debug("tight_layout skipped (non-critical)")


# Backward-compatible names for tests / processor shims
_save_plot_safely = save_plot_safely
_safe_tight_layout = safe_tight_layout
