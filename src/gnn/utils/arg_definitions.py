"""Earlier name; implementation moved to ``gnn/utils/arguments/arg_definitions.py``
(S2-33 Step 2)."""

import warnings

warnings.warn(
    "gnn.utils.arg_definitions is the earlier name; import gnn.utils.arguments.arg_definitions instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.arguments.arg_definitions import (  # noqa: E402,F401
    ArgumentDefinition,
)

__all__ = ["ArgumentDefinition"]
