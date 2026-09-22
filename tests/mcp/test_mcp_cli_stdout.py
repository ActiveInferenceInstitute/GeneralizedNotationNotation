#!/usr/bin/env python3
"""
Stdout-contract tests for the MCP CLI (`python -m gnn.mcp.cli`).

Verifies the G-12 contract: with ``--format json``, every command writes
exactly one pure JSON machine document to stdout (no human text, emoji, or
log frames; errors as ``{'error': {'operation', 'message'}}`` with exit 1),
while human mode keeps its logger-driven output untouched.
"""

import argparse
import json
import logging
from typing import Any, Optional

import pytest

import gnn.mcp.cli as cli

pytestmark = pytest.mark.mcp

EMOJI_SAMPLES = ("🚀", "✅", "❌", "🔧", "📊")


class _StubTool:
    """Minimal tool-record stand-in for the tool-info fallback path."""

    name = "stub_tool"
    description = "A stub tool"
    schema: dict = {"type": "object", "properties": {}}
    module = "stub"
    category = "test"
    version = "1.0.0"
    experimental = False


class StubMcpInstance:
    """Stub exposing only the mcp_instance methods the CLI handlers call."""

    def __init__(self) -> None:
        self.tools: dict = {"stub_tool": _StubTool()}

    def get_capabilities(self) -> dict:
        """Mirror ``mcp_instance.get_capabilities``."""
        return {
            "server": {"name": "stub-gnn-mcp", "version": "0.0.0"},
            "tools": [],
            "resources": [],
        }

    def execute_tool(self, tool_name: str, params: dict) -> dict:
        """Mirror ``mcp_instance.execute_tool``."""
        if tool_name == "get_mcp_diagnostics":
            return {
                "diagnostics": {
                    "issues": [],
                    "warnings": [],
                    "recommendations": [],
                },
                "overall_health": "healthy",
            }
        return {"tool": tool_name, "params": params, "ok": True}

    def get_resource(self, uri: str) -> dict:
        """Mirror ``mcp_instance.get_resource``."""
        return {"uri": uri, "content": "stub-content"}

    def get_server_status(self) -> dict:
        """Mirror ``mcp_instance.get_server_status``."""
        return {
            "status": "ok",
            "uptime_formatted": "0s",
            "request_count": 0,
            "error_count": 0,
        }

    def get_tool_info(self, tool_name: str) -> Optional[dict]:
        """Mirror ``mcp_instance.get_tool_info``."""
        return {
            "name": tool_name,
            "description": "A stub tool",
            "schema": {"type": "object", "properties": {}},
            "module": "stub",
            "category": "test",
            "version": "1.0.0",
            "experimental": False,
        }

    def get_enhanced_server_status(self) -> dict:
        """Mirror ``mcp_instance.get_enhanced_server_status`` (human flow)."""
        return {
            "health": {"status": "ok", "score": 100},
            "server_info": {"uptime_formatted": "0s"},
            "performance": {
                "success_rate": 1.0,
                "average_execution_time": 0.0,
                "cache_hit_ratio": 0.0,
            },
        }


@pytest.fixture
def stub_mcp(monkeypatch: pytest.MonkeyPatch) -> StubMcpInstance:
    """Route the CLI seams to the stub instance (no real MCP init)."""
    stub = StubMcpInstance()
    monkeypatch.setattr(cli, "_get_mcp", lambda: (stub, RuntimeError))
    return stub


def _json_args(**kw: Any) -> argparse.Namespace:
    """Build argparse.Namespace args with json mode defaults."""
    kw.setdefault("format", "json")
    kw.setdefault("verbose", False)
    return argparse.Namespace(**kw)


def _human_args(**kw: Any) -> argparse.Namespace:
    """Build argparse.Namespace args with human mode defaults."""
    kw.setdefault("format", "human")
    kw.setdefault("verbose", False)
    return argparse.Namespace(**kw)


def _parse_stdout(capsys: pytest.CaptureFixture[str]) -> Any:
    """Parse the whole stdout as exactly one JSON document."""
    return json.loads(capsys.readouterr().out)


@pytest.mark.unit
class TestJsonStdoutContract:
    """json mode: exactly one pure JSON machine document on stdout."""

    def test_list_capabilities(
        self, stub_mcp: StubMcpInstance, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """list emits the capabilities document and nothing else."""
        cli.list_capabilities(_json_args())
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed == stub_mcp.get_capabilities()
        assert not any(emoji in out for emoji in EMOJI_SAMPLES)

    def test_execute_tool(
        self, stub_mcp: StubMcpInstance, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """execute emits the tool result document and nothing else."""
        cli.execute_tool(
            _json_args(tool_name="stub_tool", params='{"k": 1}', validate=False)
        )
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed == {"tool": "stub_tool", "params": {"k": 1}, "ok": True}
        assert not any(emoji in out for emoji in EMOJI_SAMPLES)

    def test_get_resource(
        self, stub_mcp: StubMcpInstance, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """resource emits the resource content document."""
        cli.get_resource(_json_args(uri="gnn://stub"))
        parsed = _parse_stdout(capsys)
        assert parsed == {"uri": "gnn://stub", "content": "stub-content"}

    def test_get_server_status(
        self, stub_mcp: StubMcpInstance, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """status emits the status document."""
        cli.get_server_status(_json_args())
        parsed = _parse_stdout(capsys)
        assert parsed == stub_mcp.get_server_status()

    def test_get_tool_info(
        self, stub_mcp: StubMcpInstance, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """info emits the tool info document."""
        cli.get_tool_info(_json_args(tool_name="stub_tool"))
        parsed = _parse_stdout(capsys)
        assert parsed["name"] == "stub_tool"
        assert parsed["module"] == "stub"

    def test_get_diagnostics(
        self, stub_mcp: StubMcpInstance, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """diagnostics emits the diagnostics document."""
        cli.get_diagnostics(_json_args())
        parsed = _parse_stdout(capsys)
        assert parsed["overall_health"] == "healthy"
        assert "diagnostics" in parsed


@pytest.mark.unit
class TestJsonErrorContract:
    """json mode failures: one error document on stdout with exit 1."""

    def test_tool_info_unknown_tool(
        self, stub_mcp: StubMcpInstance, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Unknown tool in json mode prints the error document and exits 1."""
        with pytest.raises(SystemExit) as exc_info:
            cli.get_tool_info(_json_args(tool_name="missing_tool"))
        assert exc_info.value.code == 1
        parsed = _parse_stdout(capsys)
        assert parsed["error"]["operation"] == "getting tool info"
        assert "missing_tool" in parsed["error"]["message"]

    def test_execute_tool_unknown_tool(
        self, stub_mcp: StubMcpInstance, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Executing an unknown tool in json mode prints the error document."""
        with pytest.raises(SystemExit) as exc_info:
            cli.execute_tool(
                _json_args(tool_name="missing_tool", params=None, validate=False)
            )
        assert exc_info.value.code == 1
        parsed = _parse_stdout(capsys)
        assert parsed["error"]["operation"] == "executing tool"

    def test_cli_error_json_branch(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_cli_error json branch prints the standardized error document."""
        with pytest.raises(SystemExit) as exc_info:
            cli._cli_error("getting tool info", RuntimeError("boom"), _json_args())
        assert exc_info.value.code == 1
        parsed = _parse_stdout(capsys)
        assert parsed == {
            "error": {"operation": "getting tool info", "message": "boom"}
        }


@pytest.mark.unit
class TestHumanModeUnchanged:
    """human mode keeps emitting through the logging channel, not raw prints."""

    def test_list_capabilities_via_logger(
        self,
        stub_mcp: StubMcpInstance,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Human list output goes through the logger channel only."""
        caplog.set_level(logging.INFO)
        cli.list_capabilities(_human_args())
        out = capsys.readouterr().out
        assert out == ""
        messages = [record.getMessage() for record in caplog.records]
        assert any("GNN MCP Server Capabilities" in msg for msg in messages)

    def test_execute_tool_via_logger(
        self,
        stub_mcp: StubMcpInstance,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Human execute output goes through the logger channel only."""
        caplog.set_level(logging.INFO)
        cli.execute_tool(
            _human_args(tool_name="stub_tool", params=None, validate=False)
        )
        out = capsys.readouterr().out
        assert out == ""
        messages = [record.getMessage() for record in caplog.records]
        assert any("Tool executed successfully" in msg for msg in messages)
