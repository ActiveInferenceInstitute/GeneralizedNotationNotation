#!/usr/bin/env python3
"""Tool execution with caching, timeouts, and rate limiting.

Mechanical extraction from ``gnn.mcp.mcp`` (MAJ-04 sibling-mixin split):
``MCPExecutionMixin`` holds the verbatim execution methods (including the
result-cache key/store/hit path, the bounded timeout pool, and the sliding
window rate limiter) and ``MCP`` in ``mcp.py`` inherits from it, so cache
keys, limiter behavior, and error envelopes are unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from concurrent.futures import (
    ThreadPoolExecutor,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, cast

from ._late_binding import _MCPModuleRef
from .exceptions import (
    MCPInvalidParamsError,
    MCPRateLimitError,
    MCPToolExecutionError,
    MCPToolNotFoundError,
    MCPToolTimeoutError,
    MCPValidationError,
)
from .jsonrpc import tag_non_json_values
from .models import MCPPerformanceMetrics, MCPTool

# Configure logging
logger = logging.getLogger("mcp")

# Late-bound clock: attribute access resolves through ``gnn.mcp.mcp`` so the
# fake-clock swap in tests/mcp/test_registry_internals.py keeps working.
time = _MCPModuleRef("time")

# Bounded pool enforcing per-tool timeouts. A hung tool occupies one worker
# until its (uncancellable) thread finishes; when the pool is exhausted,
# further timed calls surface a timeout instead of hanging the caller — the
# safe failure direction. Un-timed tools never touch this pool.
_TOOL_TIMEOUT_POOL_SIZE = 8


class MCPExecutionMixin:
    """Verbatim execution methods moved from ``MCP``."""

    if TYPE_CHECKING:
        # Shared ``MCP`` state the moved bodies touch (see discovery.py).
        _lock: Any
        tools: Dict[str, MCPTool]
        _request_count: int
        _error_count: int
        _last_activity: float
        _performance_metrics: MCPPerformanceMetrics
        _enable_caching: bool
        _enable_rate_limiting: bool
        _execution_lock: Any
        _active_executions: Dict[str, int]
        _tool_execution_times: Dict[str, List[float]]
        _result_cache: Dict[str, Tuple[Any, float]]
        _result_cache_lock: Any
        _tool_timeout_executor: Optional[ThreadPoolExecutor]
        _rate_limit_lock: Any
        _rate_limit_timestamps: Dict[str, List[float]]

        # Validation hooks owned by the introspection mixin.
        def _validate_params(
            self, schema: Dict[str, Any], params: Dict[str, Any]
        ) -> None: ...
        def _validate_output(self, result: Any) -> Any: ...

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced tool execution with rate limiting, caching, and performance tracking.

        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool

        Returns:
            Dict containing the tool execution result

        Raises:
            MCPToolNotFoundError: If tool is not found
            MCPInvalidParamsError: If parameters are invalid
            MCPToolExecutionError: If tool execution fails
            MCPPerformanceError: If performance thresholds are exceeded
        """
        with self._lock:
            self._request_count += 1
            self._last_activity = time.time()
            self._performance_metrics.total_requests += 1

            # Check if tool exists
            if tool_name not in self.tools:
                available_tools = list(self.tools.keys())
                self._error_count += 1
                self._performance_metrics.failed_requests += 1
                self._performance_metrics.error_counts[tool_name] = (
                    self._performance_metrics.error_counts.get(tool_name, 0) + 1
                )
                raise MCPToolNotFoundError(tool_name, available_tools)

            tool = self.tools[tool_name]

            # Enforce auth gate: tools that require authentication are unavailable
            # in this local-only server (no auth mechanism is configured).
            if tool.requires_auth:
                self._error_count += 1
                self._performance_metrics.failed_requests += 1
                self._performance_metrics.error_counts[tool_name] = (
                    self._performance_metrics.error_counts.get(tool_name, 0) + 1
                )
                raise MCPToolNotFoundError(
                    tool_name, [t for t, v in self.tools.items() if not v.requires_auth]
                )

        params_raw: object = params
        if not isinstance(params_raw, dict):
            with self._lock:
                self._error_count += 1
                self._performance_metrics.failed_requests += 1
                self._performance_metrics.error_counts[tool_name] = (
                    self._performance_metrics.error_counts.get(tool_name, 0) + 1
                )
            raise MCPInvalidParamsError(
                "Tool parameters must be an object",
                tool_name=tool_name,
                schema=tool.schema,
            )

        if tool.input_validation:
            try:
                self._validate_params(tool.schema, params)
            except MCPValidationError as exc:
                with self._lock:
                    self._error_count += 1
                    self._performance_metrics.failed_requests += 1
                    self._performance_metrics.error_counts[tool_name] = (
                        self._performance_metrics.error_counts.get(tool_name, 0) + 1
                    )
                raise MCPInvalidParamsError(
                    str(exc),
                    details=cast("Dict[str, Any]", exc.data),
                    tool_name=tool_name,
                    schema=tool.schema,
                ) from exc

        if self._enable_rate_limiting and tool.rate_limit is not None:
            try:
                self._check_rate_limit(tool_name, tool.rate_limit)
            except MCPRateLimitError:
                with self._lock:
                    self._error_count += 1
                    self._performance_metrics.failed_requests += 1
                    self._performance_metrics.error_counts[tool_name] = (
                        self._performance_metrics.error_counts.get(tool_name, 0) + 1
                    )
                raise

        # Result cache: only for tools that opted in via cache_ttl, and only
        # when caching is enabled for this server instance (performance_mode
        # "low" disables it, so default pipeline runs never cache results).
        cache_key = ""
        if self._enable_caching and tool.cache_ttl is not None:
            cache_key = self._result_cache_key(tool_name, params)
            if cache_key:
                cache_hit, cached_result = self._cache_get(cache_key)
                if cache_hit:
                    with self._lock:
                        self._performance_metrics.successful_requests += 1
                        self._performance_metrics.tool_usage_stats[tool_name] = (
                            self._performance_metrics.tool_usage_stats.get(tool_name, 0)
                            + 1
                        )
                        self._performance_metrics.update_cache_stats(True)
                    logger.debug(f"Tool {tool_name} served from result cache")
                    return cast("dict[str, Any]", cached_result)
                with self._lock:
                    self._performance_metrics.update_cache_stats(False)

        with self._execution_lock:
            self._active_executions[tool_name] += 1
            self._performance_metrics.concurrent_requests += 1
            self._performance_metrics.max_concurrent_requests = max(
                self._performance_metrics.max_concurrent_requests,
                self._performance_metrics.concurrent_requests,
            )

        # Synchronous execution. When the tool registered a timeout, the call
        # runs on the dedicated timeout pool and the CALLER stops waiting at
        # tool.timeout; Python threads cannot be killed, so the worker may
        # keep running — only the caller's wait is bounded.
        start_time = time.time()
        try:
            with self._track_performance(f"tool_execution_{tool_name}"):
                if tool.timeout is not None:
                    result = self._execute_with_timeout(
                        tool.func, tool_name, params, tool.timeout
                    )
                else:
                    result = tool.func(**params)
            if tool.output_validation:
                self._validate_output(result)
            execution_time = time.time() - start_time
            with self._lock:
                self._performance_metrics.successful_requests += 1
                self._performance_metrics.tool_usage_stats[tool_name] = (
                    self._performance_metrics.tool_usage_stats.get(tool_name, 0) + 1
                )
                self._performance_metrics.update_execution_time(execution_time)
                self._tool_execution_times[tool_name].append(execution_time)
                tool.mark_used()
                if cache_key and tool.cache_ttl is not None:
                    with self._result_cache_lock:
                        # Deep-copy on store AND on hit (see _cache_get): a
                        # cache entry must never alias the caller-visible
                        # result, or a caller mutating its returned dict
                        # would poison every subsequent cache hit.
                        self._result_cache[cache_key] = (
                            copy.deepcopy(result),
                            time.time() + tool.cache_ttl,
                        )
            logger.debug(f"Tool {tool_name} executed successfully")
            return cast("dict[str, Any]", result)
        except MCPToolTimeoutError:
            # Timeout already carries its own wire code and metrics context;
            # do not double-wrap into a -32603 execution error.
            execution_time = time.time() - start_time
            with self._lock:
                self._error_count += 1
                self._performance_metrics.failed_requests += 1
                self._performance_metrics.error_counts[tool_name] = (
                    self._performance_metrics.error_counts.get(tool_name, 0) + 1
                )
            raise
        except TypeError as e:
            # A func(**params) signature mismatch (unexpected or missing
            # keyword arguments) is an INVALID_PARAMS wire failure (-32602),
            # not an internal error. TypeErrors raised inside tool bodies
            # keep the -32603 path unless they carry the signature-mismatch
            # shape.
            message = str(e)
            execution_time = time.time() - start_time
            if "unexpected keyword argument" in message or (
                "missing" in message and "required positional argument" in message
            ):
                with self._lock:
                    self._error_count += 1
                    self._performance_metrics.failed_requests += 1
                    self._performance_metrics.error_counts[tool_name] = (
                        self._performance_metrics.error_counts.get(tool_name, 0) + 1
                    )
                raise MCPInvalidParamsError(
                    f"Invalid parameters for tool '{tool_name}': {message}",
                    tool_name=tool_name,
                ) from e
            raise MCPToolExecutionError(tool_name, e, execution_time) from e
        except Exception as e:
            execution_time = time.time() - start_time
            with self._lock:
                self._error_count += 1
                self._performance_metrics.failed_requests += 1
                self._performance_metrics.error_counts[tool_name] = (
                    self._performance_metrics.error_counts.get(tool_name, 0) + 1
                )
                self._performance_metrics.update_execution_time(execution_time)
                self._tool_execution_times[tool_name].append(execution_time)

            # Log detailed error information
            logger.error(
                f"Tool {tool_name} execution failed after {execution_time:.3f}s: {e}"
            )
            logger.debug(f"Tool {tool_name} parameters: {params}")

            raise MCPToolExecutionError(tool_name, e, execution_time) from e

        finally:
            # Clean up execution tracking
            with self._execution_lock:
                self._active_executions[tool_name] = max(
                    0, self._active_executions[tool_name] - 1
                )
                self._performance_metrics.concurrent_requests = max(
                    0, self._performance_metrics.concurrent_requests - 1
                )

    # --- Result-cache and rate-limit helpers (used by execute_tool) ---------

    @staticmethod
    def _result_cache_key(tool_name: str, params: Dict[str, Any]) -> str:
        """Build a deterministic cache key for a tool invocation.

        Non-JSON-native values are type-tagged (see
        ``gnn.mcp.jsonrpc.tag_non_json_values``) so structurally distinct
        params — the set ``{1}`` and the string ``"{1}"`` — can never alias
        to the same key the way ``default=str`` encoding allowed. Returns ""
        when params cannot be reduced to a stable JSON encoding (uncacheable
        call).
        """
        try:
            encoded = json.dumps(
                {"tool": tool_name, "params": tag_non_json_values(params)},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            return ""
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    # --- Tool timeout enforcement (used by execute_tool) --------------------

    def _execute_with_timeout(
        self,
        func: Callable[..., Any],
        tool_name: str,
        params: Dict[str, Any],
        timeout: float,
    ) -> Any:
        """Run ``func(**params)`` bounded by ``timeout`` seconds.

        Uses a dedicated bounded pool so a hung tool cannot starve module
        discovery or un-timed tools (which run inline). The worker thread
        keeps running after the timeout — Python threads are not
        cancellable — so a pool-saturating set of hung tools makes further
        timed calls surface a timeout instead of hanging the caller.
        """
        executor = self._tool_timeout_executor
        if executor is None:  # pragma: no cover - constructor fallback
            executor = ThreadPoolExecutor(
                max_workers=_TOOL_TIMEOUT_POOL_SIZE,
                thread_name_prefix="MCP-ToolTimeout",
            )
            self._tool_timeout_executor = executor
        future = executor.submit(func, **params)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise MCPToolTimeoutError(tool_name, timeout) from exc

    def _cache_get(self, cache_key: str) -> Tuple[bool, Any]:
        """Read a result-cache entry, returning (is_hit, value).

        Values are deep-copied on store (execute_tool) and again on hit so a
        caller mutating a returned result can never poison the cache.
        """
        with self._result_cache_lock:
            entry = self._result_cache.get(cache_key)
            if entry is None:
                return False, None
            result, expires_at = entry
            if expires_at < time.time():
                self._result_cache.pop(cache_key, None)
                return False, None
            return True, copy.deepcopy(result)

    def _check_rate_limit(self, tool_name: str, rate_limit: float) -> None:
        """Enforce a per-tool sliding-window rate limit.

        Raises:
            MCPRateLimitError: When the tool exceeds ``rate_limit`` requests
                per second over the trailing window.
        """
        now = time.time()
        with self._rate_limit_lock:
            timestamps = self._rate_limit_timestamps[tool_name]
            cutoff = now - 1.0
            recent = [ts for ts in timestamps if ts >= cutoff]
            if len(recent) >= int(rate_limit):
                self._rate_limit_timestamps[tool_name] = recent
                raise MCPRateLimitError(
                    tool_name, rate_limit, current_rate=float(len(recent))
                )
            recent.append(now)
            self._rate_limit_timestamps[tool_name] = recent

    @contextmanager
    def _track_performance(self, operation: str) -> Any:
        """Context manager for tracking operation performance."""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            logger.debug(f"Operation '{operation}' completed in {execution_time:.4f}s")
