"""Matplotlib backend setup for subprocess and headless pipelines.

Call ``apply_env_backend_if_set()`` immediately before ``import matplotlib.pyplot``
when optional plotting modules load. If ``MPLBACKEND`` is set in the environment,
matplotlib switches to that backend before pyplot initializes GUI state.
"""

from __future__ import annotations

import os


def apply_env_backend_if_set() -> None:
    """Invoke ``matplotlib.use`` only when ``MPLBACKEND`` is present."""
    backend = os.environ.get("MPLBACKEND")
    if not backend:
        return
    import matplotlib

    matplotlib.use(backend)
