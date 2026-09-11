"""Earlier name; implementation moved to
``gnn/utils/arguments/path_conversion.py`` (S2-33 Step 2)."""

import warnings

warnings.warn(
    "gnn.utils.path_conversion is the earlier name; import gnn.utils.arguments.path_conversion instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.arguments.path_conversion import (  # noqa: E402,F401
    convert_path_arguments,
    logger,
    validate_and_convert_paths,
    validate_pipeline_configuration,
)

__all__ = [
    "convert_path_arguments",
    "logger",
    "validate_and_convert_paths",
    "validate_pipeline_configuration",
]
