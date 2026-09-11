"""Earlier name; implementation moved to
``gnn/utils/pipeline_orchestration/execution_utils.py`` (S2-33 Step 3)."""

import warnings

warnings.warn(
    "gnn.utils.execution_utils is the earlier name; import gnn.utils.pipeline_orchestration.execution_utils instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.pipeline_orchestration.execution_utils import (  # noqa: E402,F401
    execute_command_streaming,
)

__all__ = ["execute_command_streaming"]
