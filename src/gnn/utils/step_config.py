"""Earlier name; implementation moved to ``gnn/utils/arguments/step_config.py``
(S2-33 Step 2)."""

import warnings

warnings.warn(
    "gnn.utils.step_config is the earlier name; import gnn.utils.arguments.step_config instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.arguments.step_config import (  # noqa: E402,F401
    StepConfiguration,
)

__all__ = ["StepConfiguration"]
