"""Performance tracking harness for tests (timing, peak memory, resource limits).

Moved verbatim from ``gnn/utils/testing_utils.py`` (S2-33 Step 1).
``_PerformanceTracker`` stays private (design §4.3.4): no family or facade
re-export."""

import time
from contextlib import contextmanager
from typing import Any, cast

from gnn.utils.runtime_safety.resource_manager import get_memory_usage


class _PerformanceTracker:
    """Holds timing and memory metrics for a performance_tracker context."""

    def __init__(self, start_time: float, start_memory: float) -> None:
        """Store starting metrics and initialize derived metric fields."""
        self.start_time = start_time
        self.start_memory = start_memory
        self.end_time: float | None = None
        self.end_memory: float | None = None
        self.duration: float | None = None
        self.memory_delta: float | None = None

    def finalize(self) -> None:
        """Capture end metrics and compute duration and memory deltas."""
        end_time = time.time()
        end_memory = get_memory_usage()
        self.end_time = end_time
        self.end_memory = end_memory
        self.duration = end_time - self.start_time
        self.memory_delta = max(0.0, end_memory - self.start_memory)
        # Use delta for threshold comparisons; still expose peak for reference
        self.peak_memory_mb = max(self.start_memory, end_memory)
        self.max_memory_mb = self.memory_delta


@contextmanager
def performance_tracker() -> Any:
    """Context manager for tracking test performance."""
    start_time = time.time()
    start_memory = get_memory_usage()
    tracker = _PerformanceTracker(start_time, start_memory)

    try:
        yield tracker
    finally:
        tracker.finalize()


def track_peak_memory(func: Any) -> Any:
    """Decorator to track peak memory usage of a function."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Run the wrapped function while sampling process memory."""
        try:
            import psutil

            process = psutil.Process()
            initial_memory = process.memory_info().rss

            result = func(*args, **kwargs)

            final_memory = process.memory_info().rss
            peak_memory = max(initial_memory, final_memory)

            # Store peak memory in function attributes for testing
            wrapper_state = cast(Any, wrapper)
            wrapper_state.peak_memory_mb = peak_memory / 1024 / 1024
            wrapper_state.memory_delta_mb = (
                (final_memory - initial_memory) / 1024 / 1024
            )

            return result

        except ImportError:
            # If psutil not available, just run the function
            return func(*args, **kwargs)

    return wrapper


@contextmanager
def with_resource_limits(
    max_memory_mb: (int) | None = None, max_cpu_percent: (int) | None = None
) -> Any:
    """Context manager for resource limit testing."""
    try:
        import psutil

        process = psutil.Process()

        # Store initial limits
        initial_memory_limit = getattr(process, "memory_limit", None)
        initial_cpu_limit = getattr(process, "cpu_limit", None)

        # Set limits if specified
        if max_memory_mb:
            process.memory_limit = max_memory_mb * 1024 * 1024  # Convert to bytes
        if max_cpu_percent:
            process.cpu_limit = max_cpu_percent

        yield

    except ImportError:
        # If psutil not available, just yield
        yield
    finally:
        # Restore original limits
        if "process" in locals():
            if initial_memory_limit is not None:
                process.memory_limit = initial_memory_limit
            if initial_cpu_limit is not None:
                process.cpu_limit = initial_cpu_limit
