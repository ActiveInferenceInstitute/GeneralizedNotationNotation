#!/usr/bin/env python3
"""FE#6 regression: gui_1 and gui_2 status artifacts are namespaced.

Both processors record their launch status into the same resolved step
output root. Pre-fix, both wrote a bare ``gui_status.json`` into
``22_gui_output``, so whichever GUI ran second silently clobbered the
first's payload. This regression runs gui_1 then gui_2 headless into one
shared step directory and asserts each GUI keeps its own status file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from gnn.gui.gui_1.processor import run_gui as run_gui_1
from gnn.gui.gui_2.processor import run_gui as run_gui_2

STEP_DIR_NAME = "22_gui_output"


def _write_target(target: Path) -> None:
    """Materialize a minimal target directory for both GUIs."""
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.md").write_text(
        "# Test Model\n\n"
        "components:\n"
        "  - name: example_component\n"
        "    type: observation\n"
        "    states: [s1, s2]\n\n"
    )


class TestGuiStatusNamespacing:
    """gui_1/gui_2 status writes land in distinct, correctly-named files."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gui1_and_gui2_keep_distinct_status_files(
        self, isolated_temp_dir: Any
    ) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)

        logger = logging.getLogger("test_gui_status_namespacing")

        assert run_gui_1(
            target_dir=target, output_dir=output, logger=logger, headless=True
        )
        assert run_gui_2(
            target_dir=target, output_dir=output, logger=logger, headless=True
        )

        step_dir = output / STEP_DIR_NAME

        gui_1_status = json.loads((step_dir / "gui_1_status.json").read_text())
        gui_2_status = json.loads((step_dir / "gui_2_status.json").read_text())

        assert gui_1_status["gui_type"] == "form_based_constructor"
        assert gui_2_status["gui_type"] == "visual_matrix_editor"
        assert gui_1_status["launched"] is False
        assert gui_2_status["launched"] is False

        # Pre-fix regression pair: the bare shared file must not exist, and
        # the first GUI's payload must not be clobbered by the second.
        assert not (step_dir / "gui_status.json").exists()
        assert gui_1_status["gui_type"] != gui_2_status["gui_type"]
