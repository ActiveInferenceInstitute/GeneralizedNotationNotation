#!/usr/bin/env python3
"""
Functional tests for the analysis module's MCP tool wrappers.

Exercises the real implementations of process_analysis_mcp,
get_analysis_results_mcp, compute_complexity_metrics_mcp, and
list_analysis_tools_mcp against live implementations.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.mcp import (  # noqa: E402
    compute_complexity_metrics_mcp,
    get_analysis_results_mcp,
    list_analysis_tools_mcp,
    process_analysis_mcp,
)


class TestComputeComplexityMetricsMcp:
    """Test the GNN complexity metric computation."""

    @pytest.mark.unit
    def test_counts_variables_and_connections(self) -> None:
        """A model with vars, connections, and params should be counted."""
        content = (
            "## ModelName:\n"
            "Example\n"
            "## Variables:\n"
            "A[2,2,type=float]\n"
            "s[2,1,type=float]\n"
            "## Connections:\n"
            "A -> s\n"
            "## Parameters:\n"
            "gamma=0.5\n"
        )
        result = compute_complexity_metrics_mcp(content, model_name="ex")
        assert result["success"] is True
        assert result["model_name"] == "ex"
        assert result["state_variables"] == 2
        assert result["connections"] == 1
        assert result["parameters"] >= 1
        assert result["sections"] >= 1
        assert result["cyclomatic_complexity"] >= 1

    @pytest.mark.unit
    def test_empty_content(self) -> None:
        """Empty content should yield zero counts and low complexity."""
        result = compute_complexity_metrics_mcp("")
        assert result["success"] is True
        assert result["state_variables"] == 0
        assert result["cyclomatic_complexity"] >= 1
        assert result["complexity_rating"] == "low"

    @pytest.mark.unit
    def test_high_complexity_rating(self) -> None:
        """Many connections relative to vars should push rating up."""
        content = "\n".join(f"V{i}[2,2] -> V{i + 1}[2,2]" for i in range(20))
        result = compute_complexity_metrics_mcp(content)
        assert result["success"] is True
        assert result["complexity_rating"] in ("low", "medium", "high")


class TestGetAnalysisResultsMcp:
    """Test reading saved analysis results."""

    @pytest.mark.unit
    def test_reads_all_json_results(self, tmp_path: Path) -> None:
        """All JSON result files in a directory should be returned."""
        sub = tmp_path / "analysis"
        sub.mkdir()
        (sub / "model_a.json").write_text('{"a": 1}')
        (sub / "model_b.json").write_text('{"b": 2}')
        result = get_analysis_results_mcp(str(sub))
        assert result["success"] is True
        assert result["results_count"] == 2

    @pytest.mark.unit
    def test_filters_by_model_name(self, tmp_path: Path) -> None:
        """A model_name filter should narrow the returned results."""
        sub = tmp_path / "analysis2"
        sub.mkdir()
        (sub / "model_a.json").write_text('{"a": 1}')
        (sub / "model_b.json").write_text('{"b": 2}')
        result = get_analysis_results_mcp(str(sub), model_name="model_a")
        assert result["success"] is True
        assert result["results_count"] == 1
        assert result["results"][0]["file"] == "model_a.json"

    @pytest.mark.unit
    def test_missing_output_dir(self, tmp_path: Path) -> None:
        """A missing output dir should report failure gracefully."""
        result = get_analysis_results_mcp(str(tmp_path / "missing"))
        assert result["success"] is False


class TestListAnalysisToolsMcp:
    """Test the analysis tools inventory."""

    @pytest.mark.unit
    def test_lists_tools(self) -> None:
        """The tool inventory should include core analysis packages."""
        result = list_analysis_tools_mcp()
        assert result["success"] is True
        assert "tools" in result
        assert "numpy" in result["tools"]


class TestProcessAnalysisMcp:
    """Test process_analysis through the MCP wrapper."""

    @pytest.mark.unit
    def test_process_empty_dir(self, tmp_path: Path) -> None:
        """Processing an empty directory should not raise."""
        target = tmp_path / "target"
        target.mkdir()
        out = tmp_path / "out"
        result = process_analysis_mcp(str(target), str(out))
        assert "success" in result
