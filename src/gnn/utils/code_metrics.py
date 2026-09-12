"""Earlier name; implementation moved to
``gnn/utils/config_io/code_metrics.py`` (S2-33 Step 7, family 2/3)."""

import warnings

warnings.warn(
    "gnn.utils.code_metrics is the earlier name; import gnn.utils.config_io.code_metrics instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.config_io.code_metrics import count_code_metrics  # noqa: E402,F401

__all__ = [
    "count_code_metrics",
]
