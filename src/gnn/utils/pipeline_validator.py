"""Earlier name; implementation moved to
``gnn/utils/pipeline_orchestration/pipeline_validator.py`` (S2-33 Step 3)."""

import warnings

warnings.warn(
    "gnn.utils.pipeline_validator is the earlier name; import gnn.utils.pipeline_orchestration.pipeline_validator instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.pipeline_orchestration.pipeline_validator import (  # noqa: E402,F401
    check_pipeline_readiness,
    validate_pipeline_step_sequence,
    validate_step_outputs,
    validate_step_prerequisites,
)

__all__ = [
    "check_pipeline_readiness",
    "validate_pipeline_step_sequence",
    "validate_step_outputs",
    "validate_step_prerequisites",
]
