"""Pins for ``utils/pipeline_monitor.PipelineMonitor`` (previously 34%)."""

from __future__ import annotations

import math

from gnn.utils.pipeline_orchestration.pipeline_monitor import (
    HealthStatus,
    PipelineMonitor,
)


def test_record_step_lifecycle_updates_health() -> None:
    monitor = PipelineMonitor()

    execution_id = monitor.record_step_start("3_gnn")
    assert execution_id.startswith("3_gnn_")

    monitor.record_step_success("3_gnn", execution_id, 1.25)

    health = monitor.get_pipeline_health()
    assert health is not None
    assert health.overall_status != HealthStatus.UNKNOWN
    assert health.total_steps >= 1
    assert health.total_executions >= 1
    assert "3_gnn" in monitor.step_metrics


def test_record_step_failure_records_error() -> None:
    monitor = PipelineMonitor()

    execution_id = monitor.record_step_start("12_execute")
    monitor.record_step_failure(
        "12_execute",
        execution_id,
        0.5,
        error_type="subprocess",
        error_message="subprocess died",
    )

    health = monitor.get_pipeline_health()
    assert health.total_failures >= 1
    metrics = monitor.step_metrics["12_execute"]
    assert metrics.failures == 1
    assert metrics.error_types.get("subprocess") == 1


def test_recent_avg_duration_and_success_rate() -> None:
    monitor = PipelineMonitor()

    execution_id = monitor.record_step_start("3_gnn")
    monitor.record_step_success("3_gnn", execution_id, 2.0)
    execution_id = monitor.record_step_start("3_gnn")
    monitor.record_step_success("3_gnn", execution_id, 4.0)

    metrics = monitor.step_metrics["3_gnn"]
    assert math.isclose(metrics.get_success_rate(), 100.0)


def test_health_percentage_bounds() -> None:
    monitor = PipelineMonitor()

    execution_id = monitor.record_step_start("3_gnn")
    monitor.record_step_success("3_gnn", execution_id, 1.0)

    health = monitor.get_pipeline_health()
    assert 0.0 <= health.get_health_percentage() <= 100.0


def test_start_and_stop_monitoring_toggle() -> None:
    monitor = PipelineMonitor()

    monitor.start_monitoring()
    assert monitor.monitoring_active is True
    monitor.stop_monitoring()
    assert monitor.monitoring_active is False
