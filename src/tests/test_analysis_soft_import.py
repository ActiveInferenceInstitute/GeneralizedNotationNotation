#!/usr/bin/env python3
"""Phase 1.2 regression: analysis/processor.py must not crash when analyzer import fails.

Step 16 (analysis) is NOT a hard-import step, so framework-specific analyzer
failures must degrade to ANALYZER_AVAILABLE=False + stub callables rather than
crash at module load. Simulates the failure via ``sys.modules[...] = None`` —
a real ImportError fixture (zero-mock per CLAUDE.md).
"""

import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_analysis_processor_loads_when_analyzer_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "analysis.analyzer", None)
    for name in list(sys.modules):
        if name.startswith("analysis.") and name != "analysis.analyzer":
            monkeypatch.delitem(sys.modules, name, raising=False)
        elif name == "analysis":
            monkeypatch.delitem(sys.modules, name, raising=False)

    proc = importlib.import_module("analysis.processor")
    assert proc.ANALYZER_AVAILABLE is False
    # Stubs must be present and callable.
    assert callable(proc.calculate_complexity_metrics)
    assert callable(proc.perform_statistical_analysis)
    # Stub returns a dict (never raises).
    assert proc.calculate_complexity_metrics(Path("/nonexistent"), False) == {}
    assert proc.generate_matrix_visualizations({}, Path("/tmp"), "x") == []


def test_analysis_processor_loads_normally():
    # Clean slate.
    for name in list(sys.modules):
        if name.startswith("analysis.") or name == "analysis":
            del sys.modules[name]
    proc = importlib.import_module("analysis.processor")
    # When analyzer imports succeed (the normal case), the flag is True.
    assert proc.ANALYZER_AVAILABLE is True
    assert callable(proc.perform_statistical_analysis)
