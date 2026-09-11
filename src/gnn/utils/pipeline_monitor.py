"""Earlier name; implementation moved to
``gnn/utils/pipeline_orchestration/pipeline_monitor.py`` (S2-33 Step 3)."""

import warnings

warnings.warn(
    "gnn.utils.pipeline_monitor is the earlier name; import gnn.utils.pipeline_orchestration.pipeline_monitor instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.pipeline_orchestration.pipeline_monitor import (  # noqa: E402,F401
    Alert,
    AlertLevel,
    HealthStatus,
    PipelineHealth,
    PipelineMonitor,
    StepMetrics,
    generate_pipeline_health_report,
    get_pipeline_health_status,
    pipeline_monitor,
    record_step_execution,
    start_pipeline_monitoring,
    stop_pipeline_monitoring,
)

__all__ = [
    "Alert",
    "AlertLevel",
    "HealthStatus",
    "PipelineHealth",
    "PipelineMonitor",
    "StepMetrics",
    "generate_pipeline_health_report",
    "get_pipeline_health_status",
    "pipeline_monitor",
    "record_step_execution",
    "start_pipeline_monitoring",
    "stop_pipeline_monitoring",
]
