"""Unit tests for aggregate execution summary slimming (no subprocess)."""

from __future__ import annotations

import pytest

from execute.processor import _slim_execution_detail, _summarize_collected_outputs


def test_slim_execution_detail_strips_bulk_fields_and_keeps_lengths() -> None:
    heavy = {
        "script_path": "/tmp/x.py",
        "script_name": "x.py",
        "framework": "pymdp",
        "model_name": "demo_N2_T5",
        "executor": "/usr/bin/python",
        "success": True,
        "skipped": False,
        "execution_time": 0.42,
        "execution_benchmark_repeats": 2,
        "execution_time_mean": 0.5,
        "execution_time_std": 0.08,
        "execution_time_samples": [0.4, 0.6],
        "stdout": "hello\n" * 100,
        "stderr": "warn\n",
        "simulation_data": {"huge": list(range(10_000))},
        "collected_outputs": {"files": ["/a", "/b", "/c"]},
    }
    slim = _slim_execution_detail(heavy)
    assert "simulation_data" not in slim
    assert "stdout" not in slim
    assert "stderr" not in slim
    assert slim["stdout_length"] == 600
    assert slim["stderr_length"] == 5
    assert slim["execution_time"] == 0.42
    assert slim["execution_time_samples"] == [0.4, 0.6]
    assert slim.get("collected_outputs_summary") == {"files": {"count": 3}}


def test_slim_execution_detail_omits_missing_optional_keys() -> None:
    minimal = {"script_name": "y.py", "framework": "jax", "model_name": "m", "success": True}
    slim = _slim_execution_detail(minimal)
    assert set(slim.keys()) == {"script_name", "framework", "model_name", "success"}


@pytest.mark.parametrize(
    "coll, expected",
    [
        (None, None),
        ([1, 2, 3], {"count": 3}),
        ({"a": [1], "b": {"x": 1}}, {"a": {"count": 1}, "b": {"n_keys": 1}}),
    ],
)
def test_summarize_collected_outputs(coll, expected) -> None:
    assert _summarize_collected_outputs(coll) == expected
