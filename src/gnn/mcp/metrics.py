#!/usr/bin/env python3
"""Performance metrics and lifecycle reporting for the GNN MCP server.

Mechanical extraction from ``gnn.mcp.mcp`` (MAJ-04 sibling-mixin split):
``MCPMetricsMixin`` holds the verbatim enhanced-status/cache/tool-stat
methods and ``MCP`` in ``mcp.py`` inherits from it, so every response
shape and dict key is unchanged.
"""

from __future__ import annotations

import logging
from concurrent.futures import (
    ThreadPoolExecutor,
)
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ._late_binding import _MCPModuleRef
from .models import (
    MCPModuleInfo,
    MCPPerformanceMetrics,
    MCPResource,
    MCPSDKStatus,
    MCPTool,
)

# Configure logging
logger = logging.getLogger("mcp")

# Late-bound clock: attribute access resolves through ``gnn.mcp.mcp`` so the
# controlled-clock swap in tests/mcp/test_registry_internals.py keeps working.
time = _MCPModuleRef("time")

# Global SDK status instance
_MCP_SDK_STATUS = MCPSDKStatus()


class MCPMetricsMixin:
    """Verbatim metrics methods moved from ``MCP``."""

    if TYPE_CHECKING:
        # Shared ``MCP`` state the moved bodies touch (see discovery.py).
        _lock: Any
        _executor: Optional[ThreadPoolExecutor]
        _start_time: float
        _last_activity: float
        _performance_metrics: MCPPerformanceMetrics
        tools: Dict[str, MCPTool]
        resources: Dict[str, MCPResource]
        modules: Dict[str, MCPModuleInfo]
        _tool_execution_times: Dict[str, List[float]]
        _result_cache: Dict[str, Tuple[Any, float]]
        _result_cache_lock: Any
        _active_executions: Dict[str, int]
        _rate_limit_lock: Any
        _rate_limit_timestamps: Dict[str, List[float]]
        _enable_caching: bool
        _enable_rate_limiting: bool
        _strict_validation: bool

        # Facade assembly property (uptime lives on ``MCP``).
        @property
        def uptime(self) -> float: ...

    def get_enhanced_server_status(self) -> Dict[str, Any]:
        """
        Get enhanced server status with detailed metrics and health information.

        Returns:
            Dict containing comprehensive server status
        """
        with self._lock:
            # Get memory usage if available
            memory_usage = None
            try:
                import psutil

                process = psutil.Process()
                memory_usage = process.memory_info().rss
            except ImportError:
                logger.debug("psutil not available, skipping memory usage reporting")

            # Calculate cache statistics
            cache_size = len(self._result_cache)
            cache_memory_estimate = cache_size * 1024  # Rough estimate

            # Get active executions
            active_executions = dict(self._active_executions)

            # Get rate limit status
            rate_limit_status: dict[Any, Any] = {}
            with self._rate_limit_lock:
                for tool_name, timestamps in self._rate_limit_timestamps.items():
                    current_time = time.time()
                    recent_requests = len(
                        [ts for ts in timestamps if current_time - ts < 1.0]
                    )
                    rate_limit_status[tool_name] = {
                        "recent_requests": recent_requests,
                        "total_requests": len(timestamps),
                    }

            return {
                "server_info": {
                    "name": "GNN MCP Server",
                    "version": "2.0.0",
                    "uptime": self.uptime,
                    "start_time": self._start_time,
                    "last_activity": self._last_activity,
                },
                "performance": {
                    "total_requests": self._performance_metrics.total_requests,
                    "successful_requests": self._performance_metrics.successful_requests,
                    "failed_requests": self._performance_metrics.failed_requests,
                    "success_rate": (
                        self._performance_metrics.successful_requests
                        / max(1, self._performance_metrics.total_requests)
                    ),
                    "average_execution_time": self._performance_metrics.average_execution_time,
                    "max_execution_time": self._performance_metrics.max_execution_time,
                    "min_execution_time": self._performance_metrics.min_execution_time
                    if self._performance_metrics.min_execution_time != float("inf")
                    else 0.0,
                    "cache_hit_ratio": self._performance_metrics.cache_hit_ratio,
                    "cache_hits": self._performance_metrics.cache_hits,
                    "cache_misses": self._performance_metrics.cache_misses,
                    "concurrent_requests": self._performance_metrics.concurrent_requests,
                    "max_concurrent_requests": self._performance_metrics.max_concurrent_requests,
                },
                "resources": {
                    "tools_count": len(self.tools),
                    "resources_count": len(self.resources),
                    "modules_count": len(self.modules),
                    "memory_usage_bytes": memory_usage,
                    "cache_size": cache_size,
                    "cache_memory_estimate_bytes": cache_memory_estimate,
                },
                "modules": {
                    module_name: {
                        "status": info.status,
                        "tools_count": info.tools_count,
                        "resources_count": info.resources_count,
                        "load_time": info.load_time,
                        "last_updated": info.last_updated,
                        "error_message": info.error_message,
                    }
                    for module_name, info in self.modules.items()
                },
                "active_executions": active_executions,
                "rate_limit_status": rate_limit_status,
                "sdk_status": _MCP_SDK_STATUS.to_dict(),
                "health": {
                    "status": "healthy"
                    if self._performance_metrics.failed_requests
                    / max(1, self._performance_metrics.total_requests)
                    < 0.1
                    else "degraded",
                    "error_rate": self._performance_metrics.failed_requests
                    / max(1, self._performance_metrics.total_requests),
                    "cache_efficiency": self._performance_metrics.cache_hit_ratio,
                    "concurrent_load": self._performance_metrics.concurrent_requests
                    / 10.0,  # Normalized to max workers
                },
            }

    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear all caches and return statistics.

        Returns:
            Dict containing cache clearing statistics
        """
        with self._result_cache_lock:
            cache_size_before = len(self._result_cache)
            self._result_cache.clear()

            return {
                "result_cache_cleared": cache_size_before,
                "timestamp": time.time(),
            }

    def get_tool_performance_stats(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed performance statistics for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dict containing tool performance statistics or None if tool not found
        """
        if tool_name not in self.tools:
            return None

        execution_times = self._tool_execution_times.get(tool_name, [])
        if not execution_times:
            return {
                "tool_name": tool_name,
                "execution_count": 0,
                "average_execution_time": 0.0,
                "min_execution_time": 0.0,
                "max_execution_time": 0.0,
                "total_execution_time": 0.0,
                "error_count": self._performance_metrics.error_counts.get(tool_name, 0),
                "success_rate": 1.0,
            }

        total_executions = len(execution_times)
        total_time = sum(execution_times)
        avg_time = total_time / total_executions
        min_time = min(execution_times)
        max_time = max(execution_times)
        error_count = self._performance_metrics.error_counts.get(tool_name, 0)
        success_count = self._performance_metrics.tool_usage_stats.get(tool_name, 0)
        total_attempts = success_count + error_count
        success_rate = success_count / max(1, total_attempts)

        return {
            "tool_name": tool_name,
            "execution_count": total_executions,
            "average_execution_time": avg_time,
            "min_execution_time": min_time,
            "max_execution_time": max_time,
            "total_execution_time": total_time,
            "error_count": error_count,
            "success_count": success_count,
            "success_rate": success_rate,
            "recent_executions": execution_times[-10:]
            if len(execution_times) > 10
            else execution_times,
        }

    def shutdown(self) -> Dict[str, Any]:
        """
        Gracefully shutdown the MCP server.

        Returns:
            Dict containing shutdown statistics
        """
        logger.info("Shutting down MCP server...")

        # Shutdown thread pool (may be None if construction failed at init time)
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True)
            except Exception as e:
                logger.warning(f"Thread pool shutdown failed: {e}")

        # Clear caches
        cache_stats = self.clear_cache()

        # Get final statistics
        final_stats: dict[str, Any] = {
            "uptime": self.uptime,
            "total_requests": self._performance_metrics.total_requests,
            "successful_requests": self._performance_metrics.successful_requests,
            "failed_requests": self._performance_metrics.failed_requests,
            "cache_stats": cache_stats,
            "shutdown_time": time.time(),
        }

        logger.info(f"MCP server shutdown complete: {final_stats}")
        return final_stats

    def set_performance_mode(self, mode: str = "low") -> Any:
        """Set performance mode to optimize resource usage."""
        if mode == "low":
            self._enable_caching = False
            self._enable_rate_limiting = False
            self._strict_validation = False
        elif mode == "high":
            self._enable_caching = True
            self._enable_rate_limiting = True
            self._strict_validation = True
