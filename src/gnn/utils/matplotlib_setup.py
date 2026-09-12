"""Earlier name; implementation moved to
``gnn/utils/system_env/matplotlib_setup.py`` (S2-33 Step 7, family 1/3)."""

import warnings

warnings.warn(
    "gnn.utils.matplotlib_setup is the earlier name; import gnn.utils.system_env.matplotlib_setup instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.system_env.matplotlib_setup import (  # noqa: E402,F401
    apply_env_backend_if_set,
)

__all__ = [
    "apply_env_backend_if_set",
]
