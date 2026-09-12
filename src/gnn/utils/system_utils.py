"""Earlier name; implementation moved to
``gnn/utils/system_env/system_utils.py`` (S2-33 Step 7, family 1/3)."""

import warnings

warnings.warn(
    "gnn.utils.system_utils is the earlier name; import gnn.utils.system_env.system_utils instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.system_env.system_utils import (  # noqa: E402,F401
    PSUTIL_AVAILABLE,
    get_system_info,
)

__all__ = [
    "PSUTIL_AVAILABLE",
    "get_system_info",
]