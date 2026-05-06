"""Meta-analysis validator, statistics export, and run_meta_analysis wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from integration.meta_analysis import run_meta_analysis
from integration.meta_analysis.collector import SweepDataCollector, SweepRecord
from integration.meta_analysis.statistics import compute_meta_statistics
from integration.meta_analysis.validator import validate_sweep_records


def test_validate_empty_records() -> None:
    payload = validate_sweep_records([])
    assert payload["schema_version"]
    assert payload["issues"] == []
    assert payload["summary"] == {"info": 0, "warning": 0, "error": 0}


def test_validate_full_grid_no_grid_issues() -> None:
    records = [
        SweepRecord("pymdp_scaling_N2_T10", "pymdp", num_states=2, num_timesteps=10, success=True),
        SweepRecord("pymdp_scaling_N2_T20", "pymdp", num_states=2, num_timesteps=20, success=True),
        SweepRecord("pymdp_scaling_N4_T10", "pymdp", num_states=4, num_timesteps=10, success=True),
        SweepRecord("pymdp_scaling_N4_T20", "pymdp", num_states=4, num_timesteps=20, success=True),
    ]
    payload = validate_sweep_records(records)
    assert not any(i["code"] == "GRID_MISSING_CELL" for i in payload["issues"])
    assert payload["grid"]["expected_cells"] == 4
    assert payload["grid"]["record_cells"] == 4


def test_validate_grid_missing_cell() -> None:
    records = [
        SweepRecord("pymdp_scaling_N2_T10", "pymdp", num_states=2, num_timesteps=10, success=True),
        SweepRecord("pymdp_scaling_N4_T10", "pymdp", num_states=4, num_timesteps=10, success=True),
        SweepRecord("pymdp_scaling_N2_T50", "pymdp", num_states=2, num_timesteps=50, success=True),
    ]
    payload = validate_sweep_records(records)
    assert payload["schema_version"]
    codes = {i["code"] for i in payload["issues"]}
    assert "GRID_MISSING_CELL" in codes


def test_validate_timestep_mismatch(tmp_path: Path) -> None:
    sim = tmp_path / "simulation_results.json"
    sim.write_text(json.dumps({"observations": [1, 1, 1, 1, 1], "num_timesteps": 5}), encoding="utf-8")
    records = [
        SweepRecord(
            "pymdp_scaling_N4_T10",
            "pymdp",
            num_states=4,
            num_timesteps=10,
            success=True,
            simulation_results_path=str(sim),
        )
    ]
    payload = validate_sweep_records(records)
    assert any(i["code"] == "TIMESTEP_MISMATCH" for i in payload["issues"])


def test_validate_benchmark_std_missing() -> None:
    r = SweepRecord(
        "pymdp_scaling_N2_T10",
        "pymdp",
        num_states=2,
        num_timesteps=10,
        success=True,
        execution_benchmark_repeats=3,
        execution_time_std=0.0,
        execution_time_samples=[1.0, 2.0, 3.0],
    )
    payload = validate_sweep_records([r])
    assert any(i["code"] == "BENCHMARK_STD_MISSING" for i in payload["issues"])


def test_validate_timing_without_success() -> None:
    r = SweepRecord(
        "pymdp_scaling_N2_T10",
        "pymdp",
        num_states=2,
        num_timesteps=10,
        success=False,
        execution_time=1.2,
    )
    payload = validate_sweep_records([r])
    assert any(i["code"] == "TIMING_WITHOUT_SUCCESS" for i in payload["issues"])


def test_validate_benchmark_samples_short() -> None:
    r = SweepRecord(
        "pymdp_scaling_N2_T10",
        "pymdp",
        num_states=2,
        num_timesteps=10,
        success=True,
        execution_benchmark_repeats=4,
        execution_time_samples=[1.0],
        execution_time_std=0.05,
    )
    payload = validate_sweep_records([r])
    assert any(i["code"] == "BENCHMARK_SAMPLES_SHORT" for i in payload["issues"])


def test_validate_sim_json_unreadable(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    r = SweepRecord(
        "pymdp_scaling_N2_T10",
        "pymdp",
        num_states=2,
        num_timesteps=10,
        success=True,
        simulation_results_path=str(bad),
    )
    payload = validate_sweep_records([r])
    assert any(i["code"] == "SIM_JSON_READ" for i in payload["issues"])


def test_compute_meta_statistics_smoke() -> None:
    pytest.importorskip("numpy")
    records = [
        SweepRecord(
            "p_N2_T10",
            "pymdp",
            num_states=2,
            num_timesteps=10,
            success=True,
            execution_time=1.5,
        ),
        SweepRecord(
            "p_N2_T10",
            "jax",
            num_states=2,
            num_timesteps=10,
            success=True,
            execution_time=2.5,
        ),
    ]
    stats = compute_meta_statistics(records)
    assert stats.get("schema_version")
    assert "error" not in stats
    assert "pymdp" in stats["per_framework"]
    assert "jax" in stats["per_framework"]
    assert stats["per_cell_best_framework"]
    best = stats["per_cell_best_framework"][0]
    assert best["framework"] == "pymdp"
    assert best["execution_time_s"] == 1.5


def test_compute_meta_statistics_loglog_pymdp_by_t() -> None:
    """At least two N values at fixed T yields a slope entry for that T."""
    pytest.importorskip("numpy")
    records = [
        SweepRecord(
            "pymdp_scaling_N2_T100",
            "pymdp",
            num_states=2,
            num_timesteps=100,
            success=True,
            execution_time=0.5,
        ),
        SweepRecord(
            "pymdp_scaling_N4_T100",
            "pymdp",
            num_states=4,
            num_timesteps=100,
            success=True,
            execution_time=1.0,
        ),
        SweepRecord(
            "pymdp_scaling_N8_T100",
            "pymdp",
            num_states=8,
            num_timesteps=100,
            success=True,
            execution_time=2.0,
        ),
    ]
    stats = compute_meta_statistics(records)
    assert "error" not in stats
    by_t = stats.get("loglog_runtime_vs_n_by_T") or {}
    assert "100" in by_t
    assert "slope" in by_t["100"]
    assert by_t["100"]["n_points"] >= 2


def test_run_meta_analysis_writes_artifacts_and_returns_paths(tmp_path: Path) -> None:
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
                "execution_time": 1.25,
                "execution_benchmark_repeats": 1,
            }
        ]
    }
    (summaries / "execution_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "meta_out"
    result = run_meta_analysis(execute_output_dir=exec_root, output_dir=out_dir)
    assert result is not None
    assert result["records"] == 1
    assert Path(result["validation_json"]).is_file()
    assert Path(result["statistics_json"]).is_file()
    assert Path(result["report"]).is_file()
    sweep_val = json.loads(Path(result["validation_json"]).read_text(encoding="utf-8"))
    assert sweep_val["schema_version"]
    meta_stats = json.loads(Path(result["statistics_json"]).read_text(encoding="utf-8"))
    assert meta_stats["schema_version"]
    assert "pymdp" in meta_stats["per_framework"]


def test_run_meta_analysis_returns_none_without_records(tmp_path: Path) -> None:
    exec_root = tmp_path / "12_execute_output"
    summaries = exec_root / "summaries"
    summaries.mkdir(parents=True)
    (summaries / "execution_summary.json").write_text(
        json.dumps({"execution_details": []}), encoding="utf-8"
    )
    out_dir = tmp_path / "meta_empty"
    assert run_meta_analysis(execute_output_dir=exec_root, output_dir=out_dir) is None

def test_collect_slim_execution_summary_preserves_timing_for_collector(tmp_path: Path) -> None:
    """Slim aggregate rows omit stdout but keep fields SweepDataCollector needs."""
    exec_root = tmp_path / "12_execute_output"
    summaries = exec_root / "summaries"
    summaries.mkdir(parents=True)
    detail = {
        "model_name": "pymdp_scaling_N16_T50",
        "framework": "pymdp",
        "success": True,
        "skipped": False,
        "execution_time": 3.3,
        "execution_time_std": 0.1,
        "execution_time_mean": 3.25,
        "execution_benchmark_repeats": 3,
        "execution_time_samples": [3.2, 3.3, 3.4],
        "stdout_length": 42,
    }
    payload = {"execution_summary_format": "slim_v1", "execution_details": [detail]}
    (summaries / "execution_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    collector = SweepDataCollector(exec_root)
    records = collector.collect()
    assert len(records) == 1
    r = records[0]
    assert r.execution_time == pytest.approx(3.3)
    assert r.execution_time_std == pytest.approx(0.1)
    assert r.execution_benchmark_repeats == 3
    assert r.execution_time_samples == [3.2, 3.3, 3.4]
    assert r.num_states == 16
    assert r.num_timesteps == 50
