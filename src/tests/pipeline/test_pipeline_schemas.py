"""Real component tests for the typed pipeline schemas (``pipeline.schemas``)."""

from __future__ import annotations

from datetime import datetime

from pipeline.schemas import (
    ExecutionResult,
    GNNModelSummary,
    GNNParseOutput,
    PipelineSummary,
    RenderOutput,
    ValidationOutput,
)


def test_gnn_model_summary_defaults() -> None:
    model = GNNModelSummary()
    assert model.name == ""
    assert model.file_path == ""
    assert model.section_count == 0
    assert model.variable_count == 0
    assert model.connection_count == 0


def test_gnn_model_summary_sets_fields() -> None:
    model = GNNModelSummary(
        name="toy",
        file_path="models/toy.md",
        section_count=3,
        variable_count=4,
        connection_count=5,
    )
    assert model.model_dump()["name"] == "toy"
    assert model.connection_count == 5


def test_gnn_parse_output_lists_are_independent() -> None:
    first = GNNParseOutput()
    second = GNNParseOutput()
    first.models.append(GNNModelSummary(name="a"))
    # Defaults are factories, so mutating one instance must not leak to another.
    assert second.models == []
    assert second.parse_errors == []


def test_gnn_parse_output_parse_timestamp_is_iso() -> None:
    out = GNNParseOutput()
    datetime.fromisoformat(out.parse_timestamp)


def test_validation_output_tracks_counts() -> None:
    v = ValidationOutput(valid_count=7, error_count=2, warnings=["w1"])
    assert v.valid_count == 7
    assert v.error_count == 2
    assert v.warnings == ["w1"]


def test_render_output_roundtrip() -> None:
    r = RenderOutput(
        framework="pymdp",
        output_path="/tmp/out.py",
        success=False,
        error="boom",
    )
    assert r.success is False
    assert r.error == "boom"
    dumped = r.model_dump()
    assert dumped["framework"] == "pymdp"
    assert dumped["output_path"] == "/tmp/out.py"


def test_execution_result_defaults() -> None:
    res = ExecutionResult(step_name="3_gnn")
    assert res.step_num == -1
    assert res.status == "PENDING"
    assert res.duration == 0.0
    assert res.artifacts == []
    assert res.errors == []


def test_execution_result_roundtrip_preserves_artifacts() -> None:
    res = ExecutionResult(
        step_name="11_render",
        step_num=11,
        status="SUCCESS",
        duration=1.5,
        artifacts=["a.py", "b.py"],
        errors=["ignored"],
    )
    dumped = res.model_dump()
    assert dumped["artifacts"] == ["a.py", "b.py"]
    assert dumped["errors"] == ["ignored"]


def test_pipeline_summary_aggregates_counts() -> None:
    summary = PipelineSummary(
        model_count=3,
        artifact_count=9,
        steps=[ExecutionResult(step_name="x")],
        errors=["e"],
    )
    assert summary.model_count == 3
    assert summary.artifact_count == 9
    assert summary.success is True
    assert len(summary.steps) == 1
    assert summary.errors == ["e"]


def test_pipeline_summary_timestamp_is_iso() -> None:
    datetime.fromisoformat(PipelineSummary().timestamp)
