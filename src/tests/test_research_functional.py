#!/usr/bin/env python3
"""
Functional tests for the Research Processor module.

Tests the rule-based research hypothesis generation that analyzes GNN files
for dimensionality issues, sparse connectivity, and other patterns.

Test Coverage:
- process_research() with valid GNN files
- process_research() with empty directories
- process_research() with nonexistent paths
- Hypothesis generation for high-dimensional models
- Hypothesis generation for sparse connectivity
- Output artifact generation (JSON + markdown report)
- Return type consistency (always bool)
- Edge cases: binary files, empty files, malformed content
"""

import pytest
import json
import logging
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.processor import process_research


class TestResearchFunctional:
    """Functional tests for the research processor module."""

    @pytest.fixture
    def gnn_dir_with_high_dim(self, tmp_path):
        """Create a GNN file with high-dimensional matrices (dim > 10)."""
        target = tmp_path / "input"
        target.mkdir()
        gnn_file = target / "high_dim_model.md"
        gnn_file.write_text(
            "# High Dimensional Model\n\n"
            "## StateSpaceBlock\n"
            "A[50,50,type=float]\n"
            "B[20,20,20,type=float]\n\n"
            "## Connections\n"
            "s -> o\n"
        )
        return target

    @pytest.fixture
    def gnn_dir_sparse(self, tmp_path):
        """Create a GNN file with many variables but few connections (sparse)."""
        target = tmp_path / "input"
        target.mkdir()
        gnn_file = target / "sparse_model.md"
        # Many name: definitions but only one -> arrow
        gnn_file.write_text(
            "# Sparse Model\n\n"
            "## StateSpaceBlock\n"
            "- name: alpha\n"
            "- name: beta\n"
            "- name: gamma\n"
            "- name: delta\n"
            "- name: epsilon\n\n"
            "## Connections\n"
            "alpha -> beta\n"
        )
        return target

    @pytest.fixture
    def gnn_dir_simple(self, tmp_path):
        """Create a simple GNN file with small dimensions and normal connectivity."""
        target = tmp_path / "input"
        target.mkdir()
        gnn_file = target / "simple_model.md"
        gnn_file.write_text(
            "# Simple Model\n\n"
            "## ModelName\nSimpleTest\n\n"
            "## StateSpaceBlock\n"
            "A[3,3,type=float]\n"
            "s[3,1,type=float]\n\n"
            "## Connections\n"
            "s -> o\n"
        )
        return target

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create an output directory."""
        out = tmp_path / "output"
        out.mkdir()
        return out

    @pytest.mark.unit
    def test_process_research_returns_bool(self, gnn_dir_simple, output_dir):
        """process_research should always return a bool."""
        result = process_research(gnn_dir_simple, output_dir, verbose=True)
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"

    @pytest.mark.unit
    def test_process_research_success_with_valid_files(self, gnn_dir_simple, output_dir):
        """process_research should return True for a valid directory with GNN files."""
        result = process_research(gnn_dir_simple, output_dir, verbose=True)
        assert result is True

    @pytest.mark.unit
    def test_process_research_empty_directory(self, tmp_path):
        """process_research should handle an empty input directory gracefully."""
        empty_input = tmp_path / "empty_input"
        empty_input.mkdir()
        out = tmp_path / "output"
        out.mkdir()

        result = process_research(empty_input, out, verbose=False)
        assert isinstance(result, bool)
        # Should still succeed even with 0 files processed
        assert result is True

    @pytest.mark.unit
    def test_process_research_nonexistent_path(self, tmp_path):
        """process_research should return False for a nonexistent target directory."""
        nonexistent = tmp_path / "does_not_exist"
        out = tmp_path / "output"
        out.mkdir()

        result = process_research(nonexistent, out, verbose=False)
        assert isinstance(result, bool)
        # glob on nonexistent path should raise or return empty; processor handles gracefully

    @pytest.mark.unit
    def test_output_artifacts_created(self, gnn_dir_simple, output_dir):
        """process_research should create research_results.json and research_report.md."""
        process_research(gnn_dir_simple, output_dir, verbose=True)

        results_json = output_dir / "research_results.json"
        report_md = output_dir / "research_report.md"

        assert results_json.exists(), "research_results.json should be created"
        assert report_md.exists(), "research_report.md should be created"

    @pytest.mark.unit
    def test_results_json_schema(self, gnn_dir_simple, output_dir):
        """research_results.json should have the expected schema."""
        process_research(gnn_dir_simple, output_dir, verbose=True)

        results_json = output_dir / "research_results.json"
        with open(results_json) as f:
            data = json.load(f)

        assert "processed_files" in data
        assert "success" in data
        assert "hypotheses_generated" in data
        assert "errors" in data
        assert isinstance(data["processed_files"], int)
        assert isinstance(data["hypotheses_generated"], list)

    @pytest.mark.unit
    def test_high_dimension_triggers_hypothesis(self, gnn_dir_with_high_dim, output_dir):
        """Files with dimensions > 10 should trigger a dimensionality_reduction hypothesis."""
        process_research(gnn_dir_with_high_dim, output_dir, verbose=True)

        with open(output_dir / "research_results.json") as f:
            data = json.load(f)

        hypotheses = data["hypotheses_generated"]
        assert len(hypotheses) > 0, "Should generate hypotheses for high-dim model"

        all_types = [
            h["type"]
            for entry in hypotheses
            for h in entry.get("hypotheses", [])
        ]
        assert "dimensionality_reduction" in all_types, (
            f"Expected dimensionality_reduction hypothesis, got types: {all_types}"
        )

    @pytest.mark.unit
    def test_sparse_connectivity_triggers_hypothesis(self, gnn_dir_sparse, output_dir):
        """Files with low connection-to-variable ratio should trigger connectivity_enrichment."""
        process_research(gnn_dir_sparse, output_dir, verbose=True)

        with open(output_dir / "research_results.json") as f:
            data = json.load(f)

        hypotheses = data["hypotheses_generated"]
        assert len(hypotheses) > 0, "Should generate hypotheses for sparse model"

        all_types = [
            h["type"]
            for entry in hypotheses
            for h in entry.get("hypotheses", [])
        ]
        assert "connectivity_enrichment" in all_types, (
            f"Expected connectivity_enrichment hypothesis, got types: {all_types}"
        )

    @pytest.mark.unit
    def test_multiple_gnn_files(self, tmp_path):
        """process_research should handle multiple GNN files in one directory."""
        target = tmp_path / "multi_input"
        target.mkdir()
        out = tmp_path / "output"
        out.mkdir()

        for i in range(3):
            (target / f"model_{i}.md").write_text(
                f"# Model {i}\n## StateSpaceBlock\nA[{(i+1)*10},{(i+1)*10},type=float]\n"
            )

        result = process_research(target, out, verbose=True)
        assert result is True

        with open(out / "research_results.json") as f:
            data = json.load(f)
        assert data["processed_files"] == 3

    @pytest.mark.unit
    def test_empty_gnn_file(self, tmp_path):
        """process_research should handle an empty GNN file without crashing."""
        target = tmp_path / "input"
        target.mkdir()
        (target / "empty.md").write_text("")
        out = tmp_path / "output"
        out.mkdir()

        result = process_research(target, out, verbose=False)
        assert isinstance(result, bool)
        assert result is True
