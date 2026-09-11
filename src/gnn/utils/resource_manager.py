"""Earlier name; implementation moved to
``gnn/utils/runtime_safety/resource_manager.py`` (S2-33 Step 4)."""

import warnings

warnings.warn(
    "gnn.utils.resource_manager is the earlier name; import gnn.utils.runtime_safety.resource_manager instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.runtime_safety.resource_manager import (  # noqa: E402,F401
    ResourceTracker,
    check_disk_space,
    estimate_resources,
    get_current_memory_usage,
    get_memory_usage,
    get_system_info,
    log_resource_usage,
    performance_tracker,
    track_peak_memory,
    with_resource_limits,
)

__all__ = [
    "ResourceTracker",
    "check_disk_space",
    "estimate_resources",
    "get_current_memory_usage",
    "get_memory_usage",
    "get_system_info",
    "log_resource_usage",
    "performance_tracker",
    "track_peak_memory",
    "with_resource_limits",
]
