"""Matplotlib helpers shared across visualization subpackages."""

from typing import Any

from .utils import (
    MATPLOTLIB_AVAILABLE,
    plt,
    safe_tight_layout,
    save_plot_safely,
)

__all__: list[Any] = [
    "MATPLOTLIB_AVAILABLE",
    "plt",
    "save_plot_safely",
    "safe_tight_layout",
]
