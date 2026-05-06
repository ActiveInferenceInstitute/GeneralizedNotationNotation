"""Benchmark aggregation for Step 12 repeated executions."""

from __future__ import annotations

import pytest

from execute.processor import _aggregate_benchmark_samples


def test_aggregate_single_sample_has_zero_std() -> None:
    agg = _aggregate_benchmark_samples([1.5])
    assert agg["execution_time"] == pytest.approx(1.5)
    assert agg["execution_time_mean"] == pytest.approx(1.5)
    assert agg["execution_time_std"] == 0.0
    assert agg["execution_time_samples"] == [1.5]


def test_aggregate_median_and_population_std() -> None:
    import statistics

    samples = [1.0, 2.0, 10.0]
    agg = _aggregate_benchmark_samples(samples)
    assert agg["execution_time"] == pytest.approx(2.0)
    assert agg["execution_time_mean"] == pytest.approx(statistics.mean(samples))
    assert agg["execution_time_std"] == pytest.approx(statistics.pstdev(samples))
