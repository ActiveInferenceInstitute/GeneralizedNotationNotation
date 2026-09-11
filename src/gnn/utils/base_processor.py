"""Earlier name; implementation moved to
``gnn/utils/pipeline_orchestration/base_processor.py`` (S2-33 Step 3)."""

import warnings

warnings.warn(
    "gnn.utils.base_processor is the earlier name; import gnn.utils.pipeline_orchestration.base_processor instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.pipeline_orchestration.base_processor import (  # noqa: E402,F401
    BaseProcessor,
    ProcessingResult,
    create_processor,
)

__all__ = [
    "BaseProcessor",
    "ProcessingResult",
    "create_processor",
]
