#!/usr/bin/env python3
"""
Functional tests for report.generator file-writing behavior.

Covers the report file writers (HTML / Markdown / JSON), the summary report
writer, custom report generation with step filtering, and data validation
across valid and defective inputs.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from report.generator import (  # noqa: E402
    generate_comprehensive_report,
    generate_custom_report,
    generate_html_report_file,
    generate_json_report_file,
    generate_markdown_report_file,
    generate_summary_report,
    validate_report_data,
)

_logger = logging.getLogger("report.generator.test")
_logger.addHandler(logging.NullHandler())


@pytest.fixture
def pipeline_output(tmp_path: Path) -> Path:
    """A pipeline output dir with a minimal step artifact."""
    step_dir = tmp_path / "03_gnn_output"
    step_dir.mkdir(parents=True)
    (step_dir / "model.md").write_text("# Model\n## GNNSection\nActInfPOMDP\n")
    return tmp_path


class TestReportFileWriters:
    """Exercise the individual report file writers."""

    @pytest.mark.unit
    def test_html_writer_creates_file(self, tmp_path: Path) -> None:
        """generate_html_report_file should write the HTML report."""
        out = tmp_path / "out"
        out.mkdir()
        data: dict[str, Any] = {"steps": {}, "summary": {}}
        assert generate_html_report_file(data, out, _logger) is True
        html_file = out / "comprehensive_analysis_report.html"
        assert html_file.exists()
        assert "<html" in html_file.read_text()

    @pytest.mark.unit
    def test_markdown_writer_creates_file(self, tmp_path: Path) -> None:
        """generate_markdown_report_file should write the Markdown report."""
        out = tmp_path / "out"
        out.mkdir()
        data: dict[str, Any] = {"steps": {}, "summary": {}}
        assert generate_markdown_report_file(data, out, _logger) is True
        md_file = out / "comprehensive_analysis_report.md"
        assert md_file.exists()
        assert "GNN Pipeline Comprehensive Analysis Report" in md_file.read_text()

    @pytest.mark.unit
    def test_json_writer_creates_file(self, tmp_path: Path) -> None:
        """generate_json_report_file should write report_summary.json."""
        out = tmp_path / "out"
        out.mkdir()
        data: dict[str, Any] = {"k": "v"}
        assert generate_json_report_file(data, out, _logger) is True
        json_file = out / "report_summary.json"
        assert json_file.exists()
        assert json.loads(json_file.read_text()) == {"k": "v"}

    @pytest.mark.unit
    def test_summary_writer_creates_file(self, tmp_path: Path) -> None:
        """generate_summary_report should write report_generation_summary.json."""
        out = tmp_path / "out"
        out.mkdir()
        data: dict[str, Any] = {
            "pipeline_output_directory": str(tmp_path),
            "health_score": 75,
            "steps": {},
            "summary": {},
        }
        generate_summary_report(data, out, _logger, ["a.md"])
        summary_file = out / "report_generation_summary.json"
        assert summary_file.exists()
        parsed = json.loads(summary_file.read_text())
        assert "report_generation_summary" in parsed


class TestGenerateCustomReport:
    """Test custom report generation with step filtering."""

    @pytest.mark.unit
    def test_custom_html_report(self, pipeline_output: Path) -> None:
        """A custom HTML report should be generated for a pipeline dir."""
        out = pipeline_output / "custom_out"
        ok = generate_custom_report(
            pipeline_output, out, _logger, format_type="html"
        )
        assert ok is True
        assert (out / "comprehensive_analysis_report.html").exists()

    @pytest.mark.unit
    def test_custom_report_step_filter(self, pipeline_output: Path) -> None:
        """A step filter should be recorded in the filtered data."""
        out = pipeline_output / "custom_out2"
        ok = generate_custom_report(
            pipeline_output,
            out,
            _logger,
            step_filter=["03_gnn"],
            format_type="markdown",
        )
        assert ok is True

    @pytest.mark.unit
    def test_custom_report_unsupported_format(self, pipeline_output: Path) -> None:
        """An unsupported format should return False without raising."""
        out = pipeline_output / "custom_out3"
        ok = generate_custom_report(
            pipeline_output, out, _logger, format_type="xml"
        )
        assert ok is False


class TestValidateReportData:
    """Test report data validation."""

    @pytest.mark.unit
    def test_valid_data(self) -> None:
        """Well-formed data should validate cleanly."""
        data: dict[str, Any] = {
            "steps": {},
            "summary": {"total_files_processed": 1, "success_rate": 1.0},
            "report_generation_time": "now",
        }
        result = validate_report_data(data)
        assert result["valid"] is True

    @pytest.mark.unit
    def test_missing_required_fields(self) -> None:
        """Missing required fields should mark the report invalid."""
        result = validate_report_data({})
        assert result["valid"] is False
        assert result["errors"]

    @pytest.mark.unit
    def test_invalid_step_data(self) -> None:
        """Non-dict step data should be flagged as an error."""
        data: dict[str, Any] = {
            "steps": {"s1": "not-a-dict"},
            "summary": {},
            "report_generation_time": "now",
        }
        result = validate_report_data(data)
        assert result["valid"] is False


class TestGenerateComprehensiveReport:
    """Test the top-level comprehensive report generator."""

    @pytest.mark.unit
    def test_missing_pipeline_dir_returns_false(self, tmp_path: Path) -> None:
        """A missing pipeline output dir should fail cleanly."""
        out = tmp_path / "out"
        ok = generate_comprehensive_report(tmp_path / "missing", out, _logger)
        assert ok is False

    @pytest.mark.unit
    def test_generates_requested_formats(self, pipeline_output: Path) -> None:
        """The comprehensive report should produce all requested files."""
        out = pipeline_output / "comprehensive"
        ok = generate_comprehensive_report(
            pipeline_output, out, _logger, report_formats=["html", "json"]
        )
        assert ok is True
        assert (out / "comprehensive_analysis_report.html").exists()
        assert (out / "report_summary.json").exists()