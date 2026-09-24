"""Degraded-mode visibility for ``_resolve_dirs`` output-dir fallback (M-06).

When per-step output-dir resolution fails, the fallback to the shared
``output_dir`` root must be RECORDED (WARNING naming the step, the
failure, and the fallback root) — never silent.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytest

from gnn.utils.pipeline_orchestration.pipeline_template import _resolve_dirs


@pytest.mark.unit
class TestResolveDirsDegradedMode:
    """The output-dir fallback chain must not degrade silently."""

    def test_fallback_logs_warning_naming_step_and_root(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """Resolution failure -> WARNING names the step, error, and output root."""

        def _boom(script_name: str, base_output_dir: Path) -> Path:
            raise RuntimeError(f"config unavailable for {script_name}")

        monkeypatch.setattr(
            "gnn.utils.pipeline_orchestration.pipeline_template."
            "_get_output_dir_for_script",
            _boom,
        )
        args = argparse.Namespace(
            target_dir=str(tmp_path / "target"), output_dir=str(tmp_path / "out")
        )

        with caplog.at_level(logging.WARNING, logger="root"):
            target_dir, step_output_dir = _resolve_dirs(args, "step_3", None)

        assert target_dir == Path(args.target_dir)
        assert step_output_dir == Path(args.output_dir)
        assert any(
            record.levelno == logging.WARNING
            and "step_3" in record.getMessage()
            and "config unavailable for step_3" in record.getMessage()
            and str(args.output_dir) in record.getMessage()
            for record in caplog.records
        )

    def test_success_path_stays_silent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """Successful resolution emits no degraded-mode warning."""
        monkeypatch.setattr(
            "gnn.utils.pipeline_orchestration.pipeline_template."
            "_get_output_dir_for_script",
            lambda script_name, base_output_dir: base_output_dir / script_name,
        )
        args = argparse.Namespace(
            target_dir=str(tmp_path / "target"), output_dir=str(tmp_path / "out")
        )

        with caplog.at_level(logging.WARNING, logger="root"):
            _target_dir, step_output_dir = _resolve_dirs(args, "step_3", None)

        assert step_output_dir == Path(args.output_dir) / "step_3"
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
