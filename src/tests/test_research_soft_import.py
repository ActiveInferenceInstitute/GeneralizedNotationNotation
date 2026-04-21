#!/usr/bin/env python3
"""Phase 1.2 regression: research/__init__.py must not crash when processor import fails.

Step 19 (research) is NOT a hard-import step, so missing optional deps must
degrade to a warning (exit-code 2), not crash at module load. This simulates a
missing processor via ``sys.modules[...] = None`` — a real ImportError fixture,
zero-mock compliant per CLAUDE.md.
"""

import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_research_module_loads_when_processor_missing(tmp_path, monkeypatch):
    # sys.modules[name] = None causes any `import name` to raise ImportError —
    # a real interpreter-level mechanism, not a mock library.
    monkeypatch.setitem(sys.modules, "research.processor", None)
    # Remove any cached research module so the import re-runs with the poisoned entry.
    for name in list(sys.modules):
        if name == "research" or name.startswith("research."):
            if name != "research.processor":  # keep our None sentinel
                monkeypatch.delitem(sys.modules, name, raising=False)

    # Reimport research — the except-ImportError branch must engage the stub.
    research = importlib.import_module("research")
    assert hasattr(research, "process_research")
    # FEATURES.basic_processing reflects whether the real processor loaded.
    assert research.FEATURES["basic_processing"] is False

    # Stub must return exit-code 2 without raising.
    result = research.process_research(
        target_dir=tmp_path,
        output_dir=tmp_path / "out",
    )
    assert result == 2


def test_research_module_loads_normally_when_processor_available():
    # Clear sys.modules first so fresh import runs cleanly.
    for name in list(sys.modules):
        if name == "research" or name.startswith("research."):
            del sys.modules[name]
    research = importlib.import_module("research")
    # When imports succeed, basic_processing should be True.
    assert research.FEATURES["basic_processing"] is True
    assert callable(research.process_research)
