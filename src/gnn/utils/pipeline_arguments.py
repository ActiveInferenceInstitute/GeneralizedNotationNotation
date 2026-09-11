"""Earlier name; implementation moved to
``gnn/utils/arguments/pipeline_arguments.py`` (S2-33 Step 2)."""

import warnings

warnings.warn(
    "gnn.utils.pipeline_arguments is the earlier name; import gnn.utils.arguments.pipeline_arguments instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.arguments.pipeline_arguments import (  # noqa: E402,F401
    DEFAULT_ONTOLOGY_TERMS_FILE,
    PipelineArguments,
)

__all__ = [
    "DEFAULT_ONTOLOGY_TERMS_FILE",
    "PipelineArguments",
]
