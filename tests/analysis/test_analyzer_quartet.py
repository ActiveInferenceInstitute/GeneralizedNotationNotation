"""Direct unit tests for the ``analyzer`` quartet (W2-11).

``perform_statistical_analysis``, ``run_performance_benchmarks``,
``perform_model_comparisons``, and ``generate_analysis_summary`` are called
by ``processor.process_analysis`` (Step 16) but previously had zero direct
coverage — only their MCP wrappers were exercised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gnn.analysis.analyzer import (
    generate_analysis_summary,
    perform_model_comparisons,
    perform_statistical_analysis,
    run_performance_benchmarks,
)

_MODEL = """## GNNSection
ActInfPOMDP

## ModelName
QuartetModel

## StateSpaceBlock
s[2,1,type=float]
o[2,1,type=int]

## Connections
s-s
s-o

## Footer
QuartetModel
"""


@pytest.fixture()
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "quartet.gnn"
    path.write_text(_MODEL, encoding="utf-8")
    return path


def test_perform_statistical_analysis_shape(model_file: Path) -> None:
    result = perform_statistical_analysis(model_file)

    assert result["file_name"] == "quartet.gnn"
    assert result["line_count"] == len(_MODEL.splitlines())
    assert result["file_size"] == model_file.stat().st_size
    # Two state blocks + one connection line are extracted.
    assert len(result["variables"]) >= 2
    assert len(result["connections"]) >= 2
    for key in (
        "variable_statistics",
        "connection_statistics",
        "section_statistics",
        "distributions",
        "correlations",
    ):
        assert key in result
    assert "analysis_timestamp" in result


def test_perform_statistical_analysis_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Failed to analyze"):
        perform_statistical_analysis(tmp_path / "missing.gnn")


def test_run_performance_benchmarks_uses_real_metrics(model_file: Path) -> None:
    result = run_performance_benchmarks(model_file)

    assert result["file_name"] == "quartet.gnn"
    assert result["parse_time"] >= 0.0
    assert result["memory_usage"] > 0
    # Complexity is the real extracted element count, not a simulated value.
    assert result["complexity_score"] >= 4
    assert result["estimated_runtime"] == result["complexity_score"] * 0.01
    assert "benchmark_timestamp" in result


def test_perform_model_comparisons_requires_two_models() -> None:
    result = perform_model_comparisons([{"variables": []}])
    assert result == {"error": "Need at least 2 models for comparison"}


def _analysis(variables: int, connections: int, size: int) -> dict[str, Any]:
    return {
        "file_size": size,
        "variables": [{"name": f"v{i}"} for i in range(variables)],
        "connections": [{"source": "a", "target": "b"} for _ in range(connections)],
        "distributions": {"complexity_metrics": {"total_elements": variables + connections}},
    }


def test_perform_model_comparisons_aggregates() -> None:
    result = perform_model_comparisons([_analysis(2, 1, 100), _analysis(4, 3, 300)])

    assert result["model_count"] == 2
    assert result["complexity_comparison"]["min"] == 3
    assert result["complexity_comparison"]["max"] == 7
    assert result["complexity_comparison"]["mean"] == 5.0
    assert result["size_comparison"]["max"] == 300
    assert result["structure_comparison"]["variable_counts"]["mean"] == 3.0
    assert result["structure_comparison"]["connection_counts"]["mean"] == 2.0


def test_generate_analysis_summary_reports_counts_and_errors() -> None:
    results = {
        "processed_files": 2,
        "success": False,
        "errors": [
            {"file": "a.gnn", "error": "boom"},
            "flat string error",
        ],
        "statistical_analysis": [
            {"variables": [1, 2], "connections": [1]},
            {"variables": [1], "connections": []},
        ],
        "performance_benchmarks": [{}],
    }
    summary = generate_analysis_summary(results)

    assert "**Files Processed**: 2" in summary
    assert "**Errors**: 2" in summary
    assert "a.gnn**: boom" in summary
    assert "flat string error" in summary
    assert "Total variables across all models: 3" in summary
    assert "Total connections across all models: 1" in summary
    assert "Average variables per model: 1.5" in summary


def test_generate_analysis_summary_empty_results() -> None:
    summary = generate_analysis_summary({})
    assert "No errors encountered" in summary
    assert "**Success**: False" in summary
