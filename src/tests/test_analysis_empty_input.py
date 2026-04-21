#!/usr/bin/env python3
"""Regression test for Phase 1.1 silent-success fix in analysis/processor.py.

Before the fix, `process_analysis` returned True even when target_dir was empty
or missing — the pipeline reported step 16 as successful while silently skipping
all work. After the fix it returns 2 (the "warnings/skipped" exit code), which
the widened pipeline_template contract (Phase 0.1) then surfaces as a warning
log line and a non-zero exit status.
"""

import logging
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def empty_target_dir(tmp_path):
    d = tmp_path / "empty_gnn"
    d.mkdir()
    return d


@pytest.fixture
def missing_target_dir(tmp_path):
    # Not created — process_analysis must detect absence and return 2.
    return tmp_path / "does_not_exist"


def test_process_analysis_returns_2_when_no_gnn_files(empty_target_dir, tmp_path):
    try:
        from analysis.processor import process_analysis
    except ImportError:
        pytest.skip("analysis.processor not importable (expected when optional deps missing)")
    output_dir = tmp_path / "analysis_output"
    result = process_analysis(
        target_dir=empty_target_dir,
        output_dir=output_dir,
        verbose=False,
    )
    assert result == 2, f"expected exit code 2 for empty target_dir, got {result!r}"


def test_process_analysis_returns_2_when_target_dir_missing(missing_target_dir, tmp_path):
    try:
        from analysis.processor import process_analysis
    except ImportError:
        pytest.skip("analysis.processor not importable (expected when optional deps missing)")
    output_dir = tmp_path / "analysis_output"
    result = process_analysis(
        target_dir=missing_target_dir,
        output_dir=output_dir,
        verbose=False,
    )
    assert result == 2, f"expected exit code 2 for missing target_dir, got {result!r}"
