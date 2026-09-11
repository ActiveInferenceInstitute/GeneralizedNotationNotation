"""Earlier name; implementation moved to ``gnn/utils/mcp/dispatch.py``
(S2-33 Step 6)."""

import warnings

warnings.warn(
    "gnn.utils.mcp_dispatch is the earlier name; import gnn.utils.mcp.dispatch instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.mcp.dispatch import (  # noqa: E402,F401
    run_pipeline_step_mcp,
    run_tool_envelope,
)

__all__ = [
    "run_pipeline_step_mcp",
    "run_tool_envelope",
]
