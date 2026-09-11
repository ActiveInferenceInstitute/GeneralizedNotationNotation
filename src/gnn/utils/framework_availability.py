"""Earlier name; implementation moved to
``gnn/utils/runtime_safety/framework_availability.py`` (S2-33 Step 4)."""

import warnings

warnings.warn(
    "gnn.utils.framework_availability is the earlier name; import gnn.utils.runtime_safety.framework_availability instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.runtime_safety.framework_availability import (  # noqa: E402,F401
    FRAMEWORK_IMPORT_CHECK,
    FRAMEWORK_PROBE_STATEMENT,
    FrameworkStatus,
    check_framework,
    is_framework_available,
)

__all__ = [
    "FRAMEWORK_IMPORT_CHECK",
    "FRAMEWORK_PROBE_STATEMENT",
    "FrameworkStatus",
    "check_framework",
    "is_framework_available",
]
