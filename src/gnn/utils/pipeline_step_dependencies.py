"""Earlier name; implementation moved to
``gnn/utils/pipeline_orchestration/pipeline_step_dependencies.py``
(S2-33 Step 3)."""

import warnings

warnings.warn(
    "gnn.utils.pipeline_step_dependencies is the earlier name; import gnn.utils.pipeline_orchestration.pipeline_step_dependencies instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.pipeline_orchestration.pipeline_step_dependencies import (  # noqa: E402,F401
    PIPELINE_SCRIPT_STEPS,
    PIPELINE_STEP_DEPENDENCIES,
    PIPELINE_STEP_SCRIPTS,
    dependency_scripts_for_script,
    dependency_steps_for_step,
    normalize_script_name,
    resolve_step_dependencies,
    step_number_for_script,
)

__all__ = [
    "PIPELINE_SCRIPT_STEPS",
    "PIPELINE_STEP_DEPENDENCIES",
    "PIPELINE_STEP_SCRIPTS",
    "dependency_scripts_for_script",
    "dependency_steps_for_step",
    "normalize_script_name",
    "resolve_step_dependencies",
    "step_number_for_script",
]
