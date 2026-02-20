#!/usr/bin/env python3
"""
Functional tests for the Report module.

Tests report generation, multiple output formats (JSON, HTML, Markdown),
handling of missing/partial input data, and report validation.
"""

import pytest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from report.processor import (
    process_report,
    generate_comprehensive_report,
    analyze_gnn_file,
    generate_html_report,
    generate_markdown_report,
)


@pytest.fixture
def sample_gnn_dir(tmp_path):
    """Create a temporary directory with sample GNN files."""
    gnn_dir = tmp_path / "gnn_files"
    gnn_dir.mkdir()
    sample = gnn_dir / "test_model.md"
    sample.write_text(
        "# TestModel\n"
        "## GNNSection\n"
        "ActInfPOMDP\n"
        "## ModelName:\n"
        "Test Model\n"
        "## StateSpaceBlock:\n"
        "A[3,3,type=float]\n"
        "B[3,3,3,type=float]\n"
        "s[3,1,type=float]\n"
        "## GNNVersionAndFlags:\n"
        "v1.0\n"
    )
    return gnn_dir


@pytest.fixture
def empty_gnn_dir(tmp_path):
    """Create an empty directory with no GNN files."""
    d = tmp_path / "empty"
    d.mkdir()
    return d


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    d = tmp_path / "output"
    d.mkdir()
    return d


class TestProcessReport:
    """Test the top-level process_report function."""

    @pytest.mark.unit
    def test_process_report_success(self, sample_gnn_dir, output_dir):
        """process_report should succeed and write results JSON."""
        result = process_report(sample_gnn_dir, output_dir)
        assert result is True
        results_file = output_dir / "report_results.json"
        assert results_file.exists()
        data = json.loads(results_file.read_text())
        assert data["success"] is True
        assert data["processed_files"] == 1

    @pytest.mark.unit
    def test_process_report_empty_dir(self, empty_gnn_dir, output_dir):
        """process_report should still succeed even with no GNN files."""
        result = process_report(empty_gnn_dir, output_dir)
        assert result is True
        data = json.loads((output_dir / "report_results.json").read_text())
        assert data["processed_files"] == 0

    @pytest.mark.unit
    def test_process_report_creates_output_dir(self, sample_gnn_dir, tmp_path):
        """process_report should create the output directory if missing."""
        out = tmp_path / "new_output"
        result = process_report(sample_gnn_dir, out)
        assert result is True
        assert out.exists()

    @pytest.mark.unit
    def test_process_report_multiple_files(self, tmp_path, output_dir):
        """process_report should count multiple GNN files."""
        gnn_dir = tmp_path / "multi"
        gnn_dir.mkdir()
        for i in range(3):
            (gnn_dir / f"model_{i}.md").write_text(f"# Model {i}\n## GNNSection\nActInfPOMDP\n")
        result = process_report(gnn_dir, output_dir)
        assert result is True
        data = json.loads((output_dir / "report_results.json").read_text())
        assert data["processed_files"] == 3


class TestAnalyzeGnnFile:
    """Test file-level GNN analysis for reports."""

    @pytest.mark.unit
    def test_analyze_basic_gnn_file(self, sample_gnn_dir):
        """analyze_gnn_file should extract sections and metadata."""
        gnn_file = list(sample_gnn_dir.glob("*.md"))[0]
        result = analyze_gnn_file(gnn_file)
        assert "file_size" in result
        assert result["file_size"] > 0
        assert "lines" in result
        assert result["lines"] > 0
        assert "sections" in result
        assert isinstance(result["sections"], list)

    @pytest.mark.unit
    def test_analyze_detects_state_space(self, sample_gnn_dir):
        """analyze_gnn_file should detect StateSpaceBlock presence."""
        gnn_file = list(sample_gnn_dir.glob("*.md"))[0]
        result = analyze_gnn_file(gnn_file)
        assert result["has_state_space"] is True

    @pytest.mark.unit
    def test_analyze_detects_model_name(self, sample_gnn_dir):
        """analyze_gnn_file should detect ModelName presence."""
        gnn_file = list(sample_gnn_dir.glob("*.md"))[0]
        result = analyze_gnn_file(gnn_file)
        assert result["has_model_name"] is True

    @pytest.mark.unit
    def test_analyze_nonexistent_file(self, tmp_path):
        """analyze_gnn_file should return error for nonexistent file."""
        result = analyze_gnn_file(tmp_path / "nonexistent.md")
        assert "error" in result

    @pytest.mark.unit
    def test_analyze_empty_file(self, tmp_path):
        """analyze_gnn_file should handle an empty file gracefully."""
        empty = tmp_path / "empty.md"
        empty.write_text("")
        result = analyze_gnn_file(empty)
        assert result["file_size"] == 0
        assert result["lines"] == 1  # empty string split produces ['']


class TestComprehensiveReport:
    """Test generate_comprehensive_report across all formats."""

    @pytest.mark.unit
    def test_json_format(self, sample_gnn_dir, output_dir):
        """Should generate a JSON report file."""
        result = generate_comprehensive_report(sample_gnn_dir, output_dir, format="json")
        assert result["success"] is True
        assert result["format"] == "json"
        report_file = Path(result["report_file"])
        assert report_file.exists()
        data = json.loads(report_file.read_text())
        assert "total_files" in data
        assert data["total_files"] == 1

    @pytest.mark.unit
    def test_html_format(self, sample_gnn_dir, output_dir):
        """Should generate an HTML report file."""
        result = generate_comprehensive_report(sample_gnn_dir, output_dir, format="html")
        assert result["success"] is True
        assert result["format"] == "html"
        report_file = Path(result["report_file"])
        assert report_file.exists()
        content = report_file.read_text()
        assert "<html>" in content
        assert "GNN Comprehensive Report" in content

    @pytest.mark.unit
    def test_markdown_format(self, sample_gnn_dir, output_dir):
        """Should generate a Markdown report file."""
        result = generate_comprehensive_report(sample_gnn_dir, output_dir, format="markdown")
        assert result["success"] is True
        assert result["format"] == "markdown"
        report_file = Path(result["report_file"])
        assert report_file.exists()
        content = report_file.read_text()
        assert "# GNN Comprehensive Report" in content

    @pytest.mark.unit
    def test_empty_dir_report(self, empty_gnn_dir, output_dir):
        """Should generate a report even with no GNN files."""
        result = generate_comprehensive_report(empty_gnn_dir, output_dir, format="json")
        assert result["success"] is True
        assert result["files_analyzed"] == 0


class TestHtmlReport:
    """Test HTML report generation helper."""

    @pytest.mark.unit
    def test_generate_html_report_basic(self):
        """generate_html_report should return valid HTML with summary data."""
        report_data = {
            "timestamp": "2026-01-01",
            "total_files": 2,
            "files_analyzed": [
                {"file": "model_a.md", "info": {}},
                {"file": "model_b.md", "info": {}},
            ],
            "summary": {"success": True, "errors": []},
        }
        html = generate_html_report(report_data)
        assert "<!DOCTYPE html>" in html
        assert "Total files analyzed: 2" in html
        assert "model_a.md" in html
        assert "model_b.md" in html

    @pytest.mark.unit
    def test_generate_html_empty_data(self):
        """generate_html_report should handle empty report data."""
        html = generate_html_report({})
        assert "<html>" in html
        assert "Total files analyzed: 0" in html


class TestMarkdownReport:
    """Test Markdown report generation helper."""

    @pytest.mark.unit
    def test_generate_markdown_report_basic(self):
        """generate_markdown_report should produce Markdown with file list."""
        report_data = {
            "timestamp": "2026-01-01",
            "total_files": 1,
            "files_analyzed": [{"file": "test.md", "info": {}}],
            "summary": {"success": True, "errors": []},
        }
        md = generate_markdown_report(report_data)
        assert "# GNN Comprehensive Report" in md
        assert "test.md" in md
        assert "**Total files analyzed**: 1" in md

    @pytest.mark.unit
    def test_generate_markdown_empty_data(self):
        """generate_markdown_report should handle empty report data."""
        md = generate_markdown_report({})
        assert "# GNN Comprehensive Report" in md
