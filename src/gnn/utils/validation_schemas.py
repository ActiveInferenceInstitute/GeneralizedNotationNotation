"""Earlier name; implementation moved to
``gnn/utils/runtime_safety/validation_schemas.py`` (S2-33 Step 4)."""

import warnings

warnings.warn(
    "gnn.utils.validation_schemas is the earlier name; import gnn.utils.runtime_safety.validation_schemas instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.runtime_safety.validation_schemas import (  # noqa: E402,F401
    FRAMEWORK_PRESETS,
    KNOWN_FRAMEWORKS,
    normalize_pomdp_columns,
    validate_frameworks_arg,
    validate_model_data,
    validate_target_dir,
)

__all__ = [
    "FRAMEWORK_PRESETS",
    "KNOWN_FRAMEWORKS",
    "normalize_pomdp_columns",
    "validate_frameworks_arg",
    "validate_model_data",
    "validate_target_dir",
]
