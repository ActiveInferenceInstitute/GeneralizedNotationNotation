#!/usr/bin/env python3
"""Regression tests for the gui_3 pipeline-kwarg contract.

The standardized step runner
(``gnn.utils.pipeline_orchestration.pipeline_template``) forwards every parsed
CLI argument — including ``recursive`` — into ``process_gui(**kwargs)``, which
splats the same kwarg soup into each GUI wrapper. gui_1/gui_2/oxdraw tolerate
the extras; gui_3's wrapper must extract its supported keys before calling
``run_gui`` (whose signature is closed).

A regression that forwards unknown kwargs into ``run_gui`` again must fail
these tests: at the wrapper level, and on the real CLI path
(``python src/gnn/22_gui.py --gui-types gui_3 --headless``) whose exit 1 /
failed summary is what pipeline users saw before the fix.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
EXEMPLAR = REPO_ROOT / "input/gnn_files/discrete/hmm_baseline.md"


def _write_target(target: Path) -> None:
    """Materialize one real exemplar as the step-22 target directory."""
    target.mkdir(parents=True, exist_ok=True)
    target.joinpath(EXEMPLAR.name).write_text(EXEMPLAR.read_text())


class TestGui3PipelineKwargs:
    """gui_3 survives the pipeline kwarg soup and stays green on the CLI."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_wrapper_tolerates_pipeline_kwargs(self, isolated_temp_dir: Any) -> None:
        """The exact kwargs the standardized runner forwards must not crash.

        Regression: the wrapper splatted ``**kwargs`` into the closed-signature
        ``run_gui``, so the pipeline's ``recursive=True`` raised
        ``TypeError: run_gui() got an unexpected keyword argument 'recursive'``
        and gui_3 reported success=False on every pipeline/CLI run.
        """
        from gnn.gui.gui_3 import gui_3

        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)

        result = gui_3(
            target_dir=target,
            output_dir=output,
            logger=logging.getLogger("test_gui3_pipeline_kwargs"),
            recursive=True,
            verbose=False,
            headless=True,
            interactive=False,
            gui_types="gui_3",
            open_browser=False,
        )

        assert result["success"] is True, f"gui_3 failed: {result.get('error')}"
        assert "error" not in result or result["error"] is None
        # Headless artifacts land in the resolved step output root
        # (<output_dir>/22_gui_output when the name is not already the step dir).
        assert (output / "22_gui_output" / "designed_model_gui_3.md").is_file()
        assert (output / "22_gui_output" / "design_analysis.json").is_file()

    @pytest.mark.integration
    def test_headless_cli_exits_zero(self, isolated_temp_dir: Any) -> None:
        """Script-mode ``22_gui.py --gui-types gui_3 --headless`` must exit 0.

        Regression: the real CLI path exited 1 with gui_3's kwarg TypeError
        recorded in gui_processing_summary.json (overall_success false), even
        though gui_1/gui_2/oxdraw succeeded.
        """
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "out" / "22_gui_output"
        _write_target(target)

        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC)
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "src/gnn/22_gui.py"),
                "--target-dir",
                str(target),
                "--output-dir",
                str(output),
                "--gui-types",
                "gui_3",
                "--headless",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        assert proc.returncode == 0, (
            f"22_gui.py exited {proc.returncode}; output tail:\n"
            f"{(proc.stdout + proc.stderr)[-2000:]}"
        )
        summary = json.loads((output / "gui_processing_summary.json").read_text())
        assert summary["overall_success"] is True
        assert summary["results"]["gui_3"]["success"] is True
        # gui_3 writes its headless artifacts to the resolved step output
        # root, the same 22_gui_output directory it was handed.
        assert (output / "designed_model_gui_3.md").is_file()
        assert (output / "design_analysis.json").is_file()
