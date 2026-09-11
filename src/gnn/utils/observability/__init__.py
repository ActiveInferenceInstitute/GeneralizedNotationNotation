"""Observability concern package (S2-33 Step 5): canonical home of the
modules that emit structured records, track operation timing, and format
visual output — code that used to live in the ``gnn/utils`` top-level
grab-bag. Distinct from ``gnn/utils/logging/``: that package owns the
canonical logger and correlation context; these leaves build on it.

Lazy re-export note (PEP 562): unlike the other concern packages, this
family's ``__init__`` resolves names lazily through ``__getattr__``. The
eager variant created an import cycle: the logging entry
(``gnn/utils/logging_utils.py``) eagerly imports ``performance_tracking``
here, whose sibling ``visualization_optimizer`` imports
``gnn.utils.runtime_safety.resource_manager``, whose eager family
``__init__`` imports ``dependency_validator`` → back into the partial
``logging_utils``. Lazy resolution keeps every public name importable while
leaves only load on first attribute access (same contract the top-level
``gnn.utils`` facade implements; cycle reported by
``gnn.execute``'s import chain, fixed 2026-09-11).

R4 note: the moved leaves carry their module-level logging singletons
(``structured_logging``'s ``LogAggregator`` and the correlation context
threadlocal) with them, so identity and state are unchanged; this
``__init__`` touches no logging configuration.

Leaf inventory:
- performance_tracking: ``PerformanceTracker``/``performance_tracker`` and
  the monitoring entry points (I6: the exported ``performance_tracker``
  object shares no module name)
- visualization_optimizer: ``VisualizationOptimizer`` and the
  sampling/caching/parallel-processing helpers
- visual_logging: ``VisualLogger``/``VisualConfig`` and the accessible
  visual formatting helpers
- structured_logging: structured log emission with correlation context
  (``log_step_*``, ``set_correlation_context``, ``StructuredLogger``)

Cross-family imports go through leaf modules, never through any facade (I5).
The old top-level paths are deprecation facades over this package.
"""

from importlib import import_module
from typing import Any

_LEAF_BY_NAME: dict[str, str] = {
    # performance_tracking
    "PSUTIL_AVAILABLE": "performance_tracking",
    "PerformanceTracker": "performance_tracking",
    "generate_performance_report": "performance_tracking",
    "get_performance_metrics": "performance_tracking",
    "get_performance_tracker": "performance_tracking",
    "performance_tracker": "performance_tracking",
    "start_performance_monitoring": "performance_tracking",
    "stop_performance_monitoring": "performance_tracking",
    "track_operation_standalone": "performance_tracking",
    # structured_logging
    "LogAggregator": "structured_logging",
    "LogContext": "structured_logging",
    "LogFormat": "structured_logging",
    "LogLevel": "structured_logging",
    "PerformanceMetrics": "structured_logging",
    "StructuredFormatter": "structured_logging",
    "StructuredLogger": "structured_logging",
    "get_pipeline_logger": "structured_logging",
    "get_system_info": "structured_logging",
    "log_pipeline_complete": "structured_logging",
    "log_pipeline_start": "structured_logging",
    "log_step_error": "structured_logging",
    "log_step_start": "structured_logging",
    "log_step_success": "structured_logging",
    "log_step_warning": "structured_logging",
    "set_correlation_context": "structured_logging",
    # visual_logging
    "PROGRESS_CHARS": "visual_logging",
    "RICH_AVAILABLE": "visual_logging",
    "STATUS_ICONS": "visual_logging",
    "VisualConfig": "visual_logging",
    "VisualLogger": "visual_logging",
    "create_visual_logger": "visual_logging",
    "ensure_minimum_width": "visual_logging",
    "format_accessible_message": "visual_logging",
    "format_progress_bar": "visual_logging",
    "format_status_message": "visual_logging",
    "format_step_header": "visual_logging",
    "print_completion_summary": "visual_logging",
    "print_pipeline_banner": "visual_logging",
    "print_step_summary": "visual_logging",
    "strip_visual_elements": "visual_logging",
    # visualization_optimizer
    "DataSampler": "visualization_optimizer",
    "ParallelVisualizationProcessor": "visualization_optimizer",
    "VisualizationCache": "visualization_optimizer",
    "VisualizationOptimizer": "visualization_optimizer",
    "get_visualization_optimizer": "visualization_optimizer",
    "monitor_visualization_performance": "visualization_optimizer",
    "optimize_visualization_processing": "visualization_optimizer",
}


def __getattr__(name: str) -> Any:
    """Resolve one public name from its leaf module (lazy; PEP 562)."""
    leaf = _LEAF_BY_NAME.get(name)
    if leaf is not None:  # noqa: F841 — clarity for readers
        pass
    module_name = _LEAF_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"gnn.utils.observability.{module_name}"), name)


def __dir__() -> list[str]:
    return sorted(_LEAF_BY_NAME)
