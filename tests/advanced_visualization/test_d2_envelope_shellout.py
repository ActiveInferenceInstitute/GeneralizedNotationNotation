"""Behavior-level tests for the D2 compile shell-out to the subprocess envelope.

Covers:
- ``compile_d2_diagram`` routes the ``d2`` CLI invocation through the shared
  ``run_subprocess_envelope`` with the pinned 30s wall-clock timeout.
- Timeout classification: an over-budget compile yields the verbatim
  ``Timeout compiling to <fmt>`` warning and no output files.
- Non-zero exit classification: the child's stderr is surfaced in the
  failure warning.
- Absent CLI: a fresh visualizer reports ``d2_available`` False and compile
  returns the missing-CLI error result.
- Live CLI: the real ``d2`` binary compiles a minimal diagram through the
  same envelope path (``needs_d2_cli``).
"""

import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if TYPE_CHECKING:
    from gnn.advanced_visualization.d2_visualizer import D2DiagramSpec

FAKE_D2 = """#!/usr/bin/env python3
import os, pathlib, sys, time
args = sys.argv[1:]
mode = os.environ.get("FAKE_D2_MODE", "ok")
out = pathlib.Path(args[-1])
if mode == "ok":
    out.touch()
    sys.exit(0)
if mode == "fail":
    sys.stderr.write("boom")
    sys.exit(1)
if mode == "hang":
    time.sleep(float(os.environ.get("FAKE_D2_SLEEP", "3")))
sys.exit(0)
"""


def _write_fake_d2(tmp_path: Path) -> Path:
    """Materialize a scripted ``d2`` executable (mode via FAKE_D2_MODE)."""
    fake = tmp_path / "d2"
    fake.write_text(FAKE_D2, encoding="utf-8")
    os.chmod(fake, 0o755)
    return fake


def _make_spec() -> "D2DiagramSpec":
    from gnn.advanced_visualization.d2_visualizer import D2DiagramSpec

    return D2DiagramSpec(name="t", description="d", d2_content="a: {shape: circle}")


def _patch_fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from gnn.advanced_visualization import d2_visualizer as d2v_module

    fake = _write_fake_d2(tmp_path)
    monkeypatch.setattr(d2v_module, "_resolve_d2_binary", lambda: str(fake))
    return fake


class TestD2EnvelopeShellout:
    """compile_d2_diagram behavior against a scripted fake ``d2`` CLI."""

    def test_compile_routes_through_shared_envelope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.advanced_visualization import d2_visualizer as d2v_module
        from gnn.execute import subprocess_envelope as envelope_module

        fake = _patch_fake_cli(tmp_path, monkeypatch)
        monkeypatch.setenv("FAKE_D2_MODE", "ok")

        calls: list[tuple[list[str], dict[str, object]]] = []
        orig = envelope_module.run_subprocess_envelope

        def spy(cmd, **kwargs):
            calls.append((list(cmd), dict(kwargs)))
            return orig(cmd, **kwargs)

        monkeypatch.setattr(
            "gnn.execute.subprocess_envelope.run_subprocess_envelope", spy
        )

        visualizer = d2v_module.D2Visualizer()
        result = visualizer.compile_d2_diagram(_make_spec(), tmp_path, formats=["svg"])

        assert result.success is True
        assert result.output_files == [tmp_path / "t.svg"]
        assert result.d2_file == tmp_path / "t.d2"
        assert (tmp_path / "t.svg").exists()

        assert len(calls) == 1
        cmd, kwargs = calls[0]
        assert cmd[0] == str(fake)
        assert "--layout=elk" in cmd
        assert "--theme=1" in cmd
        assert "--pad=20" in cmd
        assert cmd[-2].endswith("t.d2")
        assert cmd[-1].endswith("t.svg")
        assert kwargs["timeout"] == 30

    def test_compile_timeout_classifies_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.advanced_visualization import d2_visualizer as d2v_module

        _patch_fake_cli(tmp_path, monkeypatch)
        monkeypatch.setenv("FAKE_D2_MODE", "hang")
        monkeypatch.setenv("FAKE_D2_SLEEP", "3")
        monkeypatch.setattr(d2v_module, "D2_COMPILE_TIMEOUT_S", 1)

        visualizer = d2v_module.D2Visualizer()
        result = visualizer.compile_d2_diagram(_make_spec(), tmp_path, formats=["svg"])

        assert result.success is False
        assert "Timeout compiling to svg" in result.warnings
        assert result.output_files == []

    def test_compile_nonzero_exit_reports_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.advanced_visualization import d2_visualizer as d2v_module

        _patch_fake_cli(tmp_path, monkeypatch)
        monkeypatch.setenv("FAKE_D2_MODE", "fail")

        visualizer = d2v_module.D2Visualizer()
        result = visualizer.compile_d2_diagram(_make_spec(), tmp_path, formats=["svg"])

        assert result.success is False
        assert "Failed to generate svg: boom" in result.warnings

    def test_compile_absent_cli_graceful_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.advanced_visualization import d2_visualizer as d2v_module

        monkeypatch.setattr(d2v_module, "_resolve_d2_binary", lambda: None)

        visualizer = d2v_module.D2Visualizer()
        assert visualizer.d2_available is False

        result = visualizer.compile_d2_diagram(_make_spec(), tmp_path, formats=["svg"])

        assert result.success is False
        assert result.error_message == d2v_module.D2_MISSING_MESSAGE


class TestD2LiveCliShellout:
    """Real-CLI path, selected only where the ``d2`` binary exists."""

    @pytest.mark.needs_d2_cli
    def test_live_cli_compiles_through_envelope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.advanced_visualization import d2_visualizer as d2v_module
        from gnn.execute import subprocess_envelope as envelope_module

        resolved = shutil.which("d2")
        assert resolved is not None

        calls: list[tuple[list[str], dict[str, object]]] = []
        orig = envelope_module.run_subprocess_envelope

        def spy(cmd, **kwargs):
            calls.append((list(cmd), dict(kwargs)))
            return orig(cmd, **kwargs)

        monkeypatch.setattr(
            "gnn.execute.subprocess_envelope.run_subprocess_envelope", spy
        )

        visualizer = d2v_module.D2Visualizer()
        result = visualizer.compile_d2_diagram(_make_spec(), tmp_path, formats=["svg"])

        assert result.success is True
        assert calls
        assert calls[0][0][0] == resolved
