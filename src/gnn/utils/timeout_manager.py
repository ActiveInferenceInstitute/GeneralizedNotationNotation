"""Earlier name; implementation moved to
``gnn/utils/runtime_safety/timeout_manager.py`` (S2-33 Step 4)."""

import warnings

warnings.warn(
    "gnn.utils.timeout_manager is the earlier name; import gnn.utils.runtime_safety.timeout_manager instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.runtime_safety.timeout_manager import (  # noqa: E402,F401
    LLMTimeoutManager,
    ProcessTimeoutManager,
    TimeoutConfig,
    TimeoutManager,
    TimeoutResult,
    TimeoutStrategy,
    get_llm_timeout_manager,
    get_process_timeout_manager,
    get_timeout_manager,
    with_async_timeout,
    with_timeout,
)

__all__ = [
    "LLMTimeoutManager",
    "ProcessTimeoutManager",
    "TimeoutConfig",
    "TimeoutManager",
    "TimeoutResult",
    "TimeoutStrategy",
    "get_llm_timeout_manager",
    "get_process_timeout_manager",
    "get_timeout_manager",
    "with_async_timeout",
    "with_timeout",
]
