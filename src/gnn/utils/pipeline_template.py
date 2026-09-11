"""Earlier name; implementation moved to
``gnn/utils/pipeline_orchestration/pipeline_template.py`` (S2-33 Step 3).

The historical ``get_output_dir_for_script`` re-export keeps its MAJ-05
deprecation behavior (``gnn.pipeline.config`` is the canonical home)."""

import warnings
from typing import Any

warnings.warn(
    "gnn.utils.pipeline_template is the earlier name; import gnn.utils.pipeline_orchestration.pipeline_template instead",
    DeprecationWarning,
    stacklevel=2,
)

from gnn.utils.pipeline_orchestration.pipeline_template import (  # noqa: E402,F401
    UTILS_AVAILABLE,
    create_standardized_pipeline_script,
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)


def __getattr__(name: str) -> Any:
    """Warn on the incidental ``get_output_dir_for_script`` re-export.

    ``gnn.pipeline.config`` is the canonical home; the historical
    ``gnn.utils.pipeline_template`` re-export now warns so internal callers
    migrate (MAJ-05 migration pattern).
    """
    if name == "get_output_dir_for_script":
        warnings.warn(
            "gnn.utils.pipeline_template.get_output_dir_for_script is "
            "superseded; import it from gnn.pipeline.config instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gnn.pipeline.config import get_output_dir_for_script as _f

        return _f
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UTILS_AVAILABLE",
    "create_standardized_pipeline_script",
    "log_step_error",
    "log_step_start",
    "log_step_success",
    "log_step_warning",
]
