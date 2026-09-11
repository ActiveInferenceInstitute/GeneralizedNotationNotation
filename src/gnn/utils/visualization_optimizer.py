"""Earlier name; implementation moved to
``gnn/utils/observability/visualization_optimizer.py`` (S2-33 Step 5)."""

import warnings

warnings.warn(
    "gnn.utils.visualization_optimizer is the earlier name; import gnn.utils.observability.visualization_optimizer instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.observability.visualization_optimizer import (  # noqa: E402,F401
    DataSampler,
    ParallelVisualizationProcessor,
    VisualizationCache,
    VisualizationOptimizer,
    get_visualization_optimizer,
    monitor_visualization_performance,
    optimize_visualization_processing,
)

# Historical pass-through re-export (tests/utils/test_shared_helpers.py pins
# the delegation identity to the canonical probe).
from gnn.utils.runtime_safety.resource_manager import (
    get_memory_usage,  # noqa: E402,F401
)

__all__ = [
    "DataSampler",
    "get_memory_usage",
    "ParallelVisualizationProcessor",
    "VisualizationCache",
    "VisualizationOptimizer",
    "get_visualization_optimizer",
    "monitor_visualization_performance",
    "optimize_visualization_processing",
]
