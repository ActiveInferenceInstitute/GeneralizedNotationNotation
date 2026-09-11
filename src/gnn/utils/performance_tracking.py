"""Earlier name; implementation moved to
``gnn/utils/observability/performance_tracking.py`` (S2-33 Step 5).

Naming invariant (I6): the exported object ``performance_tracker`` shares no
module name — this facade's module name ``performance_tracking`` matches the
new leaf's, so any ``import gnn.utils.performance_tracking`` still yields the
module, never shadowing the re-exported object."""

import warnings

warnings.warn(
    "gnn.utils.performance_tracking is the earlier name; import gnn.utils.observability.performance_tracking instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.observability.performance_tracking import (  # noqa: E402,F401
    PSUTIL_AVAILABLE,
    PerformanceTracker,
    generate_performance_report,
    get_performance_metrics,
    get_performance_tracker,
    performance_tracker,
    start_performance_monitoring,
    stop_performance_monitoring,
    track_operation_standalone,
)

__all__ = [
    "PSUTIL_AVAILABLE",
    "PerformanceTracker",
    "generate_performance_report",
    "get_performance_metrics",
    "get_performance_tracker",
    "performance_tracker",
    "start_performance_monitoring",
    "stop_performance_monitoring",
    "track_operation_standalone",
]
