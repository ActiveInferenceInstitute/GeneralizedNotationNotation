"""Earlier name; implementation moved to
``gnn/utils/config_io/path_utils.py`` (S2-33 Step 7, family 2/3)."""

import warnings

warnings.warn(
    "gnn.utils.path_utils is the earlier name; import gnn.utils.config_io.path_utils instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.config_io.path_utils import (  # noqa: E402,F401
    get_relative_path_if_possible,
)

__all__ = [
    "get_relative_path_if_possible",
]
