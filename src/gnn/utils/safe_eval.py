"""Earlier name; implementation moved to
``gnn/utils/runtime_safety/safe_eval.py`` (S2-33 Step 4)."""

import warnings

warnings.warn(
    "gnn.utils.safe_eval is the earlier name; import gnn.utils.runtime_safety.safe_eval instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.runtime_safety.safe_eval import (  # noqa: E402,F401
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_LEN,
    MATRIX_MAX_LEN,
    safe_literal_eval,
)

__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_LEN",
    "MATRIX_MAX_LEN",
    "safe_literal_eval",
]
