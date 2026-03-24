"""Matplotlib helpers shared across visualization subpackages."""

from .utils import (
    MATPLOTLIB_AVAILABLE,
    _safe_tight_layout,
    _save_plot_safely,
    plt,
    safe_tight_layout,
    save_plot_safely,
)

__all__ = [
    "MATPLOTLIB_AVAILABLE",
    "plt",
    "save_plot_safely",
    "safe_tight_layout",
    "_save_plot_safely",
    "_safe_tight_layout",
]
