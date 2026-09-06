"""Tests for the slim execution-summary contract of Step 12's aggregate.

``SweepDataCollector`` must keep harvesting timing data when the aggregate
``summaries/execution_summary.json`` omits heavy fields (stdout/stderr/
simulation_data) but retains timing and benchmark keys per detail row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.integration.meta_analysis.collector import SweepDataCollector


def _write_slim_summary(exec_root: Path, detail: dict[str, Any]) -> None:
    summaries = exec_root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    payload = {"execution_summary_format": "slim_v1", "execution_details": [detail]}
    (summaries / "execution_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_slim_summary_preserves_timing_and_benchmark(tmp_path: Path) -> None:
    exec_root = tmp_path / "12_execute_output"
    _write_slim_summary(
        exec_root,
        {
            "model_name": "pymdp_scaling_N8_T50",
            "framework": "pymdp",
            "success": True,
            "skipped": False,
            "execution_time": 1.25,
            "execution_time_std": 0.05,
            "execution_time_mean": 1.3,
            "execution_benchmark_repeats": 4,
            "execution_time_samples": [1.2, 1.25, 1.3, 1.25],
        },
    )
    records = SweepDataCollector(exec_root).collect()
    assert len(records) == 1
    record = records[0]
    assert record.model_name == "pymdp_scaling_N8_T50"
    assert record.num_states == 8
    assert record.num_timesteps == 50
    assert record.execution_time == 1.25
    assert record.execution_time_std == 0.05
    assert record.execution_time_mean == 1.3
    assert record.execution_benchmark_repeats == 4
    assert record.execution_time_samples == [1.2, 1.25, 1.3, 1.25]


def test_slim_summary_marked_skipped_is_not_success(tmp_path: Path) -> None:
    exec_root = tmp_path / "12_execute_output"
    _write_slim_summary(
        exec_root,
        {
            "model_name": "pymdp_scaling_N2_T10",
            "framework": "pymdp",
            "success": True,
            "skipped": True,
        },
    )
    records = SweepDataCollector(exec_root).collect()
    assert len(records) == 1
    assert records[0].success is False


def test_timed_out_error_message_sets_timed_out_flag(tmp_path: Path) -> None:
    exec_root = tmp_path / "12_execute_output"
    _write_slim_summary(
        exec_root,
        {
            "error": "Process timed out after 60s",
        },
    )
    records = SweepDataCollector(exec_root).collect()
    assert records[0].timed_out is True
