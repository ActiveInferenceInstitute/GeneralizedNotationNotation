"""Re-export shared plotting imports (implementation in compat.viz_compat)."""

from typing import Any

from .compat.viz_compat import MATPLOTLIB_AVAILABLE, np, plt, sns

__all__: list[Any] = ["MATPLOTLIB_AVAILABLE", "np", "plt", "sns"]
