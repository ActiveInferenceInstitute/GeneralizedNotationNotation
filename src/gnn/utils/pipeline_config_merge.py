"""Earlier name; implementation moved to
``gnn/utils/arguments/pipeline_config_merge.py`` (S2-33 Step 2)."""

import warnings

warnings.warn(
    "gnn.utils.pipeline_config_merge is the earlier name; import gnn.utils.arguments.pipeline_config_merge instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.arguments.pipeline_config_merge import (  # noqa: E402,F401
    apply_input_config_defaults,
)

__all__ = ["apply_input_config_defaults"]
