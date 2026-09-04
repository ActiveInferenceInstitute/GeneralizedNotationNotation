"""Backend availability reporting for the visualization module.

Centralises the matplotlib / numpy / seaborn / networkx / plotly availability
probes so a single call answers "which visualization backends are
installed?" — the diagnostic the module AGENTS troubleshooting guide
prescribes for the "no visualizations generated" case. Matplotlib availability
is read from :mod:`visualization.compat.viz_compat`, which performs the Agg
backend setup and reports ``MATPLOTLIB_AVAILABLE``; the remaining backends are
probed with :func:`importlib.util.find_spec` (no import side effects).
"""

from __future__ import annotations

import importlib.util
from typing import Dict

from .compat.viz_compat import MATPLOTLIB_AVAILABLE


def backend_status() -> Dict[str, bool]:
    """Return the availability of every visualization backend.

    Keys: ``matplotlib``, ``numpy``, ``seaborn``, ``networkx``, ``plotly``.
    A backend is ``True`` only when it is importable in the current
    environment (and, for matplotlib, when the Agg backend initialised
    successfully via :mod:`visualization.compat.viz_compat`).
    """
    return {
        "matplotlib": MATPLOTLIB_AVAILABLE,
        "numpy": _has_module("numpy"),
        "seaborn": _has_module("seaborn"),
        "networkx": _has_module("networkx"),
        "plotly": _has_module("plotly"),
    }


def _has_module(name: str) -> bool:
    """Return ``True`` when ``name`` is importable without importing it."""
    return importlib.util.find_spec(name) is not None
