"""Matplotlib helpers shared across visualization subpackages."""

from .utils import (
    MATPLOTLIB_AVAILABLE,
    plt,
    save_plot_safely,
    safe_tight_layout,
    _save_plot_safely,
    _safe_tight_layout,
)

__all__ = [
    "MATPLOTLIB_AVAILABLE",
    "plt",
    "save_plot_safely",
    "safe_tight_layout",
    "_save_plot_safely",
    "_safe_tight_layout",
]
