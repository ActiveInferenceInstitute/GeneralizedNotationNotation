"""Real (no-mock) tests for the type checker analysis and output utility layers.

These modules are pure, deterministic helpers: ``analysis_utils`` computes
variable/connection/complexity statistics and ``output_utils`` renders
per-file and cross-file reports. Assertions pin actual output shapes and
values rather than implementation details.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from type_checker import output_utils as ou
from type_checker.analysis_utils import (
    analyze_connections,
    analyze_variable_types,
    estimate_computational_complexity,
)

# --- analyze_variable_types --------------------------------------------------


def test_analyze_variable_types_basic() -> None:
    variables = [
        {"name": "a", "type": "categorical", "data_type": "float", "dimensions": [2, 1]},
        {"name": "b", "type": "categorical", "data_type": "float", "dimensions": [3]},
    ]
    result = analyze_variable_types(variables)
    assert result["total_variables"] == 2
    assert result["type_distribution"] == {"categorical": 2}
    assert result["data_type_distribution"] == {"float": 2}
    assert result["dimension_analysis"]["max_dimensions"] == 2
    assert result["dimension_analysis"]["avg_dimensions"] == pytest.approx(1.5)
    assert result["dimension_analysis"]["dimension_distribution"] == {"2D": 1, "1D": 1}
    assert result["complexity_metrics"]["total_elements"] == 5
    assert result["complexity_metrics"]["estimated_memory_bytes"] == 40


def test_analyze_variable_types_empty() -> None:
    result = analyze_variable_types([])
    assert result["total_variables"] == 0
    assert result["type_distribution"] == {}
    assert result["dimension_analysis"]["max_dimensions"] == 0
    assert result["dimension_analysis"]["avg_dimensions"] == 0
    assert result["complexity_metrics"]["total_elements"] == 0


def test_analyze_variable_types_defaults_for_missing_keys() -> None:
    result = analyze_variable_types([{"name": "x"}])
    # Missing type/data_type default to "unknown"; missing dimensions to [1].
    assert result["type_distribution"] == {"unknown": 1}
    assert result["data_type_distribution"] == {"unknown": 1}
    assert result["dimension_analysis"]["dimension_distribution"] == {"1D": 1}


def test_analyze_variable_types_elements_multiplication() -> None:
    result = analyze_variable_types(
        [{"name": "t", "type": "tensor", "data_type": "float", "dimensions": [2, 3, 4]}]
    )
    assert result["complexity_metrics"]["total_elements"] == 24
    assert result["dimension_analysis"]["max_dimensions"] == 3
    assert result["dimension_analysis"]["dimension_distribution"] == {"3D": 1}


# -----------------------------------------------------------------------------
# analyze_connections


def test_analyze_connections_basic() -> None:
    connections = [
        {"type": "directed", "source_variables": ["a"], "target_variables": ["b"]},
        {"type": "undirected", "source_variables": ["b"], "target_variables": ["a"]},
    ]
    result = analyze_connections(connections)
    assert result["total_connections"] == 2
    assert result["connection_type_distribution"] == {"directed": 1, "undirected": 1}
    assert result["connectivity_metrics"]["avg_connections_per_variable"] == pytest.approx(1.0)
    assert result["connectivity_metrics"]["max_connections_per_variable"] == 1
    assert result["connectivity_metrics"]["isolated_variables"] == 0


def test_analyze_connections_empty() -> None:
    result = analyze_connections([])
    assert result["total_connections"] == 0
    assert result["connection_type_distribution"] == {}
    assert result["connectivity_metrics"]["avg_connections_per_variable"] == 0
    assert result["connectivity_metrics"]["max_connections_per_variable"] == 0


def test_analyze_connections_multihop_metrics() -> None:
    # A variable appearing as both source and target accumulates degree counts.
    connections = [
        {
            "type": "directed",
            "source_variables": ["a", "b"],
            "target_variables": ["c"],
        },
        {
            "type": "directed",
            "source_variables": ["c"],
            "target_variables": ["a"],
        },
    ]
    result = analyze_connections(connections)
    assert result["total_connections"] == 2
    # a: out=1, in=1; b: out=1, in=0; c: out=1, in=1
    assert result["connectivity_metrics"]["max_connections_per_variable"] == 1
    assert result["connectivity_metrics"]["isolated_variables"] == 0


# -----------------------------------------------------------------------------
# estimate_computational_complexity


def _typed_analysis() -> dict[str, Any]:
    variables = [
        {"name": "a", "type": "cat", "data_type": "float", "dimensions": [10]},
        {"name": "b", "type": "cat", "data_type": "float", "dimensions": [10]},
    ]
    connections = [
        {"type": "directed", "source_variables": ["a"], "target_variables": ["b"]}
    ]
    return {
        "ta": analyze_variable_types(variables),
        "ca": analyze_connections(connections),
    }


def test_complexity_small_model_low_parallelism() -> None:
    d = _typed_analysis()
    result = estimate_computational_complexity(d["ta"], d["ca"])
    assert result["inference_complexity"]["operations_per_step"] == 20  # 20 elements * 1 conn
    assert result["inference_complexity"]["parallelization_potential"] == "low"
    assert result["resource_requirements"]["ram_gb_recommended"] == 1
    assert result["resource_requirements"]["cpu_cores_recommended"] == 1


def test_complexity_medium_parallelism() -> None:
    # total_elements > 100 triggers "medium" parallelization.
    ta = analyze_variable_types(
        [{"name": "v", "type": "cat", "data_type": "float", "dimensions": [12, 12]}]
    )
    ca = analyze_connections(
        [{"type": "directed", "source_variables": ["v"], "target_variables": ["v"]}]
    )
    result = estimate_computational_complexity(ta, ca)
    assert result["inference_complexity"]["parallelization_potential"] == "medium"
    assert result["resource_requirements"]["ram_gb_recommended"] == 1


def test_complexity_high_parallelism_and_ram() -> None:
    # total_elements > 1000 and memory_mb > 100 → "high" + 4 cores + 4 GB RAM.
    ta = analyze_variable_types(
        [{"name": "v", "type": "dim", "data_type": "float", "dimensions": [100, 100]}]
    )
    ca = analyze_connections(
        [{"type": "directed", "source_variables": ["v"], "target_variables": ["v"]}]
    )
    result = estimate_computational_complexity(ta, ca)
    assert result["inference_complexity"]["parallelization_potential"] == "high"
    assert result["resource_requirements"]["cpu_cores_recommended"] == 4
    assert result["resource_requirements"]["ram_gb_recommended"] == 1  # 80KB memory → low RAM


# -----------------------------------------------------------------------------
# output_utils: file writers


def test_write_markdown_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "report.md"
    ou.write_markdown(target, "# Hello\n")
    assert target.read_text(encoding="utf-8") == "# Hello\n"


def test_write_json_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "data.json"
    payload = {"items": [1, 2, 3]}
    ou.write_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_write_csv_with_and_without_header(tmp_path: Path) -> None:
    target = tmp_path / "table.csv"
    ou.write_csv(target, [["a", "b"], ["c", "d"]], header=["x", "y"])
    assert target.read_text(encoding="utf-8") == "x,y\na,b\nc,d\n"

    target2 = tmp_path / "noheader.csv"
    ou.write_csv(target2, [["a", "b"]])
    assert target2.read_text(encoding="utf-8") == "a,b\n"


# -----------------------------------------------------------------------------
# output_utils: per-file and summary renderers


def _sample_result() -> dict[str, Any]:
    return {
        "is_valid": True,
        "file_path": "models/probe.gnn",
        "errors": [],
        "warnings": ["deprecated flag"],
        "model_type": "POMDP",
        "variable_count": 1,
        "connection_count": 1,
        "model_complexity": {
            "overall_complexity": 3.5,
            "variable_complexity": 1,
            "connection_complexity": 1,
            "equation_complexity": 1,
        },
        "sections": {"GNNSection": True, "Footer": False},
        "variables": [
            {"name": "a", "type": "cat", "dimensions": [2], "total_elements": 2}
        ],
        "connections": [
            {"source": "a", "target": "b", "type": "directed", "is_temporal": False}
        ],
        "connection_types": {"directed": 1, "undirected": 0, "temporal": 0},
        "type_distribution": {"categorical": 1},
        "dimension_analysis": {
            "scalar_vars": 0,
            "vector_vars": 1,
            "matrix_vars": 0,
            "tensor_vars": 0,
            "max_dimensions": 1,
        },
        "time_dynamics": {"is_dynamic": False},
    }


def test_per_file_markdown_report_includes_sections() -> None:
    md = ou.per_file_markdown_report("model.gnn", _sample_result())
    assert "Type Check Report: model.gnn" in md
    assert "VALID" in md
    assert "model" in md  # warning text surfaced
    assert "POMDP" in md
    assert "3.50" in md  # overall complexity formatted
    assert "GNNSection" in md
    assert "Footer" in md


def test_per_file_markdown_report_invalid_status() -> None:
    res = _sample_result()
    res["is_valid"] = False
    res["errors"] = ["bad dimension"]
    md = ou.per_file_markdown_report("model.gnn", res)
    assert "INVALID" in md
    assert "bad dimension" in md


def test_per_file_json_report_identity() -> None:
    res = _sample_result()
    assert ou.per_file_json_report("model.gnn", res) is res


def test_summary_markdown_report_statistics() -> None:
    res = _sample_result()
    summary = ou.summary_markdown_report({"a.gnn": res, "b.gnn": res})
    assert "Total Files" in summary
    assert "100.0%" in summary


def test_summary_json_report_counts() -> None:
    res = _sample_result()
    invalid = {**res, "is_valid": False}
    summary = ou.summary_json_report({"a.gnn": res, "b.gnn": invalid})
    assert summary["files_checked"] == 2
    assert summary["valid"] == 1
    assert summary["invalid"] == 1


# -----------------------------------------------------------------------------
# output_utils: CSV table builders


def test_variables_table_csv() -> None:
    res = _sample_result()
    rows = ou.variables_table_csv({"model.gnn": res})
    assert rows == [["model.gnn", "a", "cat", [2]]]


def test_section_presence_matrix_csv() -> None:
    res = _sample_result()
    rows = ou.section_presence_matrix_csv(
        {"model.gnn": res}, ["GNNSection", "Footer"]
    )
    assert rows[0] == ["File", "GNNSection", "Footer"]
    assert rows[1] == ["model.gnn", 1, 0]


def test_connections_table_csv() -> None:
    res = _sample_result()
    rows = ou.connections_table_csv({"model.gnn": res})
    assert rows == [["model.gnn", "a", "b", "directed", "No"]]


def test_complexity_analysis_csv() -> None:
    res = _sample_result()
    rows = ou.complexity_analysis_csv({"model.gnn": res})
    assert rows[0][0] == "model.gnn"
    assert rows[0][1] == 1  # variable_count
    assert rows[0][2] == 1  # connection_count
    assert rows[0][4] == pytest.approx(3.5)


def test_type_distribution_csv() -> None:
    res = _sample_result()
    rows = ou.type_distribution_csv({"model.gnn": res})
    assert rows == [["model.gnn", "categorical", 1]]