"""Public API for the compat package.

Re-exports Any, MATPLOTLIB_AVAILABLE, np, plt, sns from submodules.
"""

from typing import Any

from .viz_compat import MATPLOTLIB_AVAILABLE, np, plt, sns

__all__: list[Any] = ["MATPLOTLIB_AVAILABLE", "np", "plt", "sns"]
