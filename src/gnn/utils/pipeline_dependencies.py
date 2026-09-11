"""Earlier name; implementation moved to
``gnn/utils/pipeline_orchestration/pipeline_dependencies.py`` (S2-33 Step 3)."""

import warnings

warnings.warn(
    "gnn.utils.pipeline_dependencies is the earlier name; import gnn.utils.pipeline_orchestration.pipeline_dependencies instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.pipeline_orchestration.pipeline_dependencies import (  # noqa: E402,F401
    DependencyResult,
    PipelineDependencyManager,
    StepDependencyInfo,
    get_pipeline_dependency_manager,
)

__all__ = [
    "DependencyResult",
    "PipelineDependencyManager",
    "StepDependencyInfo",
    "get_pipeline_dependency_manager",
]
