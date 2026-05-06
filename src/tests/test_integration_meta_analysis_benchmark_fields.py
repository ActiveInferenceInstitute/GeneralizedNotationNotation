"""Integration meta-analysis collector: benchmark timing fields.

``SweepDataCollector`` reads ``summaries/execution_summary.json`` whether the
aggregate uses ``execution_summary_format: slim_v1`` or legacy rows; timing and
benchmark keys on each detail dict must remain present (see also
``test_collect_slim_execution_summary_preserves_timing_for_collector``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from integration.meta_analysis.collector import SweepDataCollector


def test_collects_execution_benchmark_metadata(tmp_path: Path) -> None:
    exec_root = tmp_path / "12_execute_output"
    summaries = exec_root / "summaries"
    summaries.mkdir(parents=True)
    payload = {
        "execution_details": [
            {
                "model_name": "pymdp_scaling_N4_T10",
                "framework": "pymdp",
                "success": True,
                "skipped": False,
                "execution_time": 2.0,
                "execution_time_mean": 2.1,
                "execution_time_std": 0.2,
                "execution_benchmark_repeats": 3,
                "execution_time_samples": [2.0, 2.2, 2.0],
            }
        ]
    }
    (summaries / "execution_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    collector = SweepDataCollector(exec_root)
    records = collector.collect()
    assert len(records) == 1
    r = records[0]
    assert r.execution_time == pytest.approx(2.0)
    assert r.execution_time_mean == pytest.approx(2.1)
    assert r.execution_time_std == pytest.approx(0.2)
    assert r.execution_benchmark_repeats == 3
    assert r.execution_time_samples == [2.0, 2.2, 2.0]
    assert r.num_states == 4
    assert r.num_timesteps == 10
