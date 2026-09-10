"""Registry-internals contract tests (wave-2 MIN-02).

These behaviors shipped unpinned: the ``requires_auth`` gate, registry-level
non-dict params, the in-process result cache (fixtures elsewhere always
disable it), the per-tool sliding-window rate limiter, and parity between
the committed audit surface and the live registered count.
"""
from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.mcp

from gnn.mcp.exceptions import (
    MCPInvalidParamsError,
    MCPRateLimitError,
    MCPToolExecutionError,
    MCPToolNotFoundError,
)
from gnn.mcp.mcp import MCP


def _registry(**kwargs: Any) -> MCP:
    defaults: dict[str, Any] = {"enable_caching": False, "enable_rate_limiting": False}
    defaults.update(kwargs)
    return MCP(**defaults)


class TestRequiresAuthGate:
    """Auth-gated tools are unreachable and report as not-found."""

    @pytest.mark.unit
    def test_requires_auth_tool_is_not_executable(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="secret_tool",
            func=lambda: {"ok": True},
            schema={},
            description="gated",
            requires_auth=True,
        )
        assert "secret_tool" in registry.tools  # registered but gated
        with pytest.raises(MCPToolNotFoundError):
            registry.execute_tool("secret_tool", {})

    @pytest.mark.unit
    def test_auth_gate_counts_as_error(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="secret_tool",
            func=lambda: {"ok": True},
            schema={},
            description="gated",
            requires_auth=True,
        )
        with pytest.raises(MCPToolNotFoundError):
            registry.execute_tool("secret_tool", {})
        assert registry._performance_metrics.error_counts["secret_tool"] == 1


class TestRegistryParamsGate:
    """execute_tool rejects non-dict params before touching the tool."""

    @pytest.mark.unit
    def test_string_params_rejected(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="echo_tool",
            func=lambda: {"ok": True},
            schema={},
            description="echo",
        )
        with pytest.raises(MCPInvalidParamsError) as excinfo:
            registry.execute_tool("echo_tool", "not-a-dict")
        assert excinfo.value.code == -32602

    @pytest.mark.unit
    def test_list_params_rejected(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="echo_tool",
            func=lambda: {"ok": True},
            schema={},
            description="echo",
        )
        with pytest.raises(MCPInvalidParamsError):
            registry.execute_tool("echo_tool", [1, 2])


class TestResultCache:
    """The in-process cache short-circuits repeats and honours TTL."""

    @pytest.mark.unit
    def test_identical_calls_execute_once(self) -> None:
        registry = _registry(enable_caching=True)
        calls: list[int] = []
        registry.register_tool(
            name="counting_tool",
            func=lambda: calls.append(1) or {"n": len(calls)},
            schema={},
            description="counts executions",
            cache_ttl=60.0,
        )
        first = registry.execute_tool("counting_tool", {})
        second = registry.execute_tool("counting_tool", {})
        assert first == {"n": 1}
        assert second == {"n": 1}  # served from cache, not recomputed
        assert len(calls) == 1

    @pytest.mark.unit
    def test_ttl_expiry_recomputes(self) -> None:
        registry = _registry(enable_caching=True)
        calls: list[int] = []
        registry.register_tool(
            name="expiring_tool",
            func=lambda: calls.append(1) or {"n": len(calls)},
            schema={},
            description="counts executions",
            cache_ttl=0.05,
        )
        registry.execute_tool("expiring_tool", {})
        time.sleep(0.12)
        result = registry.execute_tool("expiring_tool", {})
        assert result == {"n": 2}  # cache expired, tool re-executed
        assert len(calls) == 2

    @pytest.mark.unit
    def test_uncacheable_params_do_not_raise(self) -> None:
        registry = _registry(enable_caching=True)
        calls: list[int] = []
        registry.register_tool(
            name="set_param_tool",
            func=lambda values: calls.append(1) or {"n": len(values)},
            schema={
                "type": "object",
                "properties": {"values": {"type": "array"}},
            },
            description="takes a set-shaped param",
            cache_ttl=60.0,
        )
        # A set is not JSON-serializable: the call must still succeed, simply
        # bypassing the cache.
        assert registry.execute_tool("set_param_tool", {"values": {1, 2}}) == {"n": 2}
        assert len(calls) == 1


class TestPerToolRateLimiter:
    """The sliding-window limiter trips at rate_limit requests/second."""

    @pytest.mark.unit
    def test_third_rapid_call_is_rate_limited(self) -> None:
        registry = _registry(enable_rate_limiting=True)
        registry.register_tool(
            name="hot_tool",
            func=lambda: {"ok": True},
            schema={},
            description="hot",
            rate_limit=2,
        )
        assert registry.execute_tool("hot_tool", {}) == {"ok": True}
        assert registry.execute_tool("hot_tool", {}) == {"ok": True}
        with pytest.raises(MCPRateLimitError) as excinfo:
            registry.execute_tool("hot_tool", {})
        assert excinfo.value.code == -32005
        # The rejected call counts as a failed request.
        assert registry._performance_metrics.failed_requests >= 1

    @pytest.mark.unit
    def test_window_recovery_allows_calls_again(self) -> None:
        registry = _registry(enable_rate_limiting=True)
        registry.register_tool(
            name="hot_tool",
            func=lambda: {"ok": True},
            schema={},
            description="hot",
            rate_limit=1,
        )
        assert registry.execute_tool("hot_tool", {}) == {"ok": True}
        with pytest.raises(MCPRateLimitError):
            registry.execute_tool("hot_tool", {})
        # Outside the 1s window the limiter resets.
        time.sleep(1.05)
        assert registry.execute_tool("hot_tool", {}) == {"ok": True}


class TestAuditSurfaceParity:
    """The committed audit JSON must mirror the live registered count."""

    @pytest.mark.unit
    def test_audit_report_tools_total_matches_live_registry(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        audit_path = repo_root / "src" / "gnn" / "mcp" / "audit_report.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))

        from gnn.mcp import initialize

        initialize(halt_on_missing_sdk=False, force_proceed_flag=True)
        from gnn.mcp import mcp_instance

        # Timed-out modules keep registering via background recovery; poll
        # until the count stabilizes (same contract as tests/mcp/test_mcp_audit).
        last = len(mcp_instance.tools)
        stable_since = time.monotonic()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            time.sleep(0.2)
            current = len(mcp_instance.tools)
            if current == last:
                if time.monotonic() - stable_since >= 1.0:
                    break
            else:
                last = current
                stable_since = time.monotonic()

        assert last == audit["tools_total"], (
            "live registry count drifted from audit_report.json — regenerate "
            "the audit (uv run python src/gnn/mcp/validate_tools.py) in the "
            "same PR that adds or removes tools"
        )


class TestExecutionErrorMetrics:
    """Tool failures must be observable through the metrics surface."""

    @pytest.mark.unit
    def test_failing_tool_records_error_metrics(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="broken_tool",
            func=lambda: 1 / 0,
            schema={},
            description="always raises",
        )
        with pytest.raises(MCPToolExecutionError) as excinfo:
            registry.execute_tool("broken_tool", {})
        assert excinfo.value.code == -32603
        assert registry._performance_metrics.error_counts["broken_tool"] == 1


class TestDuplicateRegistration:
    """Duplicate tool registration is deterministic (MIN-01)."""

    @pytest.mark.unit
    def test_same_callable_reregister_is_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = _registry()

        def echo() -> dict[str, Any]:
            return {"ok": True}

        registry.register_tool("dup_tool", echo, {}, "Echo")
        first_func = registry.tools["dup_tool"].func
        with caplog.at_level("WARNING", logger="mcp"):
            registry.register_tool("dup_tool", echo, {}, "Echo again")
        # Same callable: full re-registration falls through silently (the
        # long-standing overwrite contract), so metadata refreshes apply.
        assert registry.tools["dup_tool"].func is first_func
        assert registry.tools["dup_tool"].description == "Echo again"
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    @pytest.mark.unit
    def test_different_callable_overwrite_warns_deterministically(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = _registry()
        registry.register_tool("dup_tool", lambda: {"v": 1}, {}, "V1")

        def replacement() -> dict[str, Any]:
            return {"v": 2}

        with caplog.at_level("WARNING", logger="mcp"):
            registry.register_tool("dup_tool", replacement, {}, "V2")
        assert registry.tools["dup_tool"].func is replacement
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("already registered" in r.getMessage() for r in warnings)

class TestResultCacheImmutability:
    """Cache hits serve pristine snapshots; callers cannot poison them."""

    @pytest.mark.unit
    def test_mutating_first_caller_cannot_poison_future_hits(self) -> None:
        registry = _registry(enable_caching=True)
        registry.register_tool(
            name="mutating_tool",
            func=lambda: {"items": [1, 2], "meta": {"n": 2}},
            schema={},
            description="returns mutable results",
            cache_ttl=60.0,
        )
        first = registry.execute_tool("mutating_tool", {})
        first["items"].append(999)  # hostile mutation of the returned object
        first["meta"]["n"] = 999
        second = registry.execute_tool("mutating_tool", {})
        assert second == {"items": [1, 2], "meta": {"n": 2}}
        third = registry.execute_tool("mutating_tool", {})
        assert third == {"items": [1, 2], "meta": {"n": 2}}

    @pytest.mark.unit
    def test_cache_hit_returns_fresh_copy_each_time(self) -> None:
        registry = _registry(enable_caching=True)
        registry.register_tool(
            name="snapshot_tool",
            func=lambda: {"n": 1},
            schema={},
            description="snapshot identity",
            cache_ttl=60.0,
        )
        a = registry.execute_tool("snapshot_tool", {})
        b = registry.execute_tool("snapshot_tool", {})
        assert a is not b  # fresh copy per hit, shared storage never leaks
        assert a == b


class TestBatchContractAndWireConsistency:
    """Single-request contract and cross-transport wire consistency (MIN-01)."""

    @pytest.mark.unit
    def test_batch_array_rejected_with_documented_contract(self) -> None:
        """A JSON-RPC batch array gets -32600 (single-request contract)."""
        from gnn.mcp.jsonrpc import validate_request

        batch = [{"jsonrpc": "2.0", "method": "tools/list", "id": 1}]
        error = validate_request(batch)
        assert error is not None
        assert error["error"]["code"] == -32600

    @pytest.mark.unit
    def test_stdio_and_http_serialize_with_same_ensure_ascii(self) -> None:
        """Both transports emit identical bytes for non-ASCII results."""
        import sys

        from gnn.mcp import server_stdio

        registry = _registry()
        registry.register_tool(
            name="unicode_tool",
            func=lambda: {"name": "π-model"},
            schema={},
            description="returns non-ascii",
        )
        stdio = server_stdio.StdioServer()
        stream = io.StringIO()
        stdio.running = True
        stdio.response_queue.put(
            {"jsonrpc": "2.0", "id": 1, "result": {"name": "π-model"}}
        )
        try:
            sys.stdout = _WriterCapture(stream)
            writer = threading.Thread(target=stdio._writer_thread, daemon=True)
            writer.start()
            writer.join(timeout=5)
        finally:
            sys.stdout = sys.__stdout__
        line = stream.getvalue().strip()
        assert line, "writer must emit the response"
        assert "\\u03c0" in line, "stdio must escape non-ascii (ensure_ascii=True)"
        # HTTP serializes through serialize_response(ensure_ascii=True);
        # compare the exact payload both transports write.
        from gnn.mcp.jsonrpc import serialize_response

        expected = serialize_response(
            {"jsonrpc": "2.0", "id": 1, "result": {"name": "π-model"}},
            separators=(",", ":"),
            ensure_ascii=True,
        )
        assert line == expected


class _WriterCapture:
    """Minimal stdout stand-in exposing write/flush for the writer thread."""

    def __init__(self, stream: io.StringIO) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()
