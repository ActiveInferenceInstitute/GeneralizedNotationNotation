"""Public API for the compat package.

Re-exports Any, MATPLOTLIB_AVAILABLE, np, plt, sns from submodules.
"""

from typing import Any

from .viz_compat import MATPLOTLIB_AVAILABLE, np, plt

__all__: list[Any] = ["MATPLOTLIB_AVAILABLE", "np", "plt", "sns"]


def __getattr__(name: str) -> Any:
    if name == "sns":
        from .viz_compat import sns

        return sns
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
