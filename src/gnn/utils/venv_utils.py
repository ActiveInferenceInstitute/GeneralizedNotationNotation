"""Earlier name; implementation moved to
``gnn/utils/system_env/venv_utils.py`` (S2-33 Step 7, family 1/3)."""

import warnings

warnings.warn(
    "gnn.utils.venv_utils is the earlier name; import gnn.utils.system_env.venv_utils instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.system_env.venv_utils import (  # noqa: E402,F401
    get_venv_python,
)

__all__ = [
    "get_venv_python",
]