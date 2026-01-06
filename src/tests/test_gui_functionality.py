#!/usr/bin/env python3
import pytest
from pathlib import Path
import logging

from gui import process_gui


class _Logger:
    def __init__(self):
        self.msgs = []
    def info(self, m, *a, **k):
        self.msgs.append(("info", m))
    def warning(self, m, *a, **k):
        self.msgs.append(("warning", m))
    def error(self, m, *a, **k):
        self.msgs.append(("error", m))
    def setLevel(self, *_):
        pass


class TestGUIFunctionality:
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_headless_artifacts(self, isolated_temp_dir):
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test\n")
        logger = _Logger()
        ok = process_gui(target_dir=target, output_dir=output, logger=logger, verbose=True, headless=True)
        assert ok
        status = output / "22_gui_output" / "gui_processing_summary.json"
        # process_gui uses get_output_dir_for_script, which nests under output dir passed to script.
        # In this headless call we passed output directly, so artifacts should exist under output/22_gui_output.
        assert status.exists()


