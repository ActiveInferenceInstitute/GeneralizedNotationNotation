"""Earlier name; implementation moved to
``gnn/utils/observability/visual_logging.py`` (S2-33 Step 5)."""

import warnings

warnings.warn(
    "gnn.utils.visual_logging is the earlier name; import gnn.utils.observability.visual_logging instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.observability.visual_logging import (  # noqa: E402,F401
    PROGRESS_CHARS,
    RICH_AVAILABLE,
    STATUS_ICONS,
    VisualConfig,
    VisualLogger,
    create_visual_logger,
    ensure_minimum_width,
    format_accessible_message,
    format_progress_bar,
    format_status_message,
    format_step_header,
    print_completion_summary,
    print_pipeline_banner,
    print_step_summary,
    strip_visual_elements,
)

__all__ = [
    "PROGRESS_CHARS",
    "RICH_AVAILABLE",
    "STATUS_ICONS",
    "VisualConfig",
    "VisualLogger",
    "create_visual_logger",
    "ensure_minimum_width",
    "format_accessible_message",
    "format_progress_bar",
    "format_status_message",
    "format_step_header",
    "print_completion_summary",
    "print_pipeline_banner",
    "print_step_summary",
    "strip_visual_elements",
]
