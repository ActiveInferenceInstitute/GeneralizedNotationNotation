#!/usr/bin/env python3
"""Tests for the opt-in Step-12 sandbox wrapper (execute.sandbox)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from execute import sandbox


class TestSandbox:
    """Sandbox wrapper behavior (mode resolution, wrapping, execution)."""

    def test_detect_sandbox_returns_spec_or_none(self) -> None:
        spec = sandbox.detect_sandbox()
        assert spec is None or isinstance(spec, sandbox.SandboxSpec)

    def test_wrap_command_prepends_prefix(self) -> None:
        spec = sandbox.SandboxSpec("firejail", ("firejail", "--net=none"))
        wrapped = sandbox.wrap_command(["python3", "x.py"], spec)
        assert wrapped == ["firejail", "--net=none", "python3", "x.py"]

    def test_run_sandboxed_off_executes_unsandboxed(self) -> None:
        result = sandbox.run_sandboxed(["python3", "-c", "print(42)"], mode="off")
        assert result["success"] is True
        assert result["sandboxed"] is False
        assert result["return_code"] == 0

    def test_run_sandboxed_require_blocks_without_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sandbox, "detect_sandbox", lambda: None)
        result = sandbox.run_sandboxed(["python3", "-c", "print(1)"], mode="require")
        assert result["blocked"] is True
        assert result["sandboxed"] is False
        assert result["success"] is False

    def test_run_sandboxed_prefer_falls_back_without_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sandbox, "detect_sandbox", lambda: None)
        result = sandbox.run_sandboxed(["python3", "-c", "print(1)"], mode="prefer")
        assert result["blocked"] is False
        assert result["sandboxed"] is False
        assert result["success"] is True

    def test_run_sandboxed_uses_backend_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = sandbox.SandboxSpec("firejail", ("echo", "SAND"))
        monkeypatch.setattr(sandbox, "detect_sandbox", lambda: spec)
        result = sandbox.run_sandboxed(["python3", "-c", "print(1)"], mode="require")
        assert result["sandboxed"] is True
        assert result["sandbox"] == "firejail"

    def test_unknown_mode_falls_back_to_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GNN_SANDBOX", "bogus")
        assert sandbox._resolve_mode(None) == "off"
