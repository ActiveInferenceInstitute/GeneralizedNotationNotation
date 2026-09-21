"""Re-export shared plotting imports (implementation in compat.viz_compat).

The seaborn module resolves lazily via ``sns``/``get_sns()`` on first access.
"""

import logging
from typing import Any

from .compat.viz_compat import MATPLOTLIB_AVAILABLE, get_sns, np, plt

logger = logging.getLogger(__name__)


sns: Any
__all__: list[Any] = ["MATPLOTLIB_AVAILABLE", "np", "plt", "sns"]


def __getattr__(name: str) -> Any:
    if name == "sns":
        return get_sns()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
