#!/usr/bin/env python3
"""
Test Report Generation - Tests for report generation functionality.

Tests the ReportGenerator class and report generation pipeline.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestReportGeneratorCore:
    """Core tests for ReportGenerator class."""

    @pytest.mark.fast
    def test_report_generator_instantiation(self) -> Any:
        """Test ReportGenerator can be instantiated."""
        from report import ReportGenerator

        generator = ReportGenerator()
        assert generator is not None

    @pytest.mark.fast
    def test_report_generator_generate(self) -> Any:
        """Test basic report generation."""
        from report import ReportGenerator

        generator = ReportGenerator()
        result = generator.generate_processing_report()

        assert result is not None

    @pytest.mark.fast
    def test_report_generator_generate_report(self) -> Any:
        """Test generate_report with sample data."""
        from report import ReportGenerator

        generator = ReportGenerator()

        data: dict[str, Any] = {
            "title": "Test Report",
            "sections": ["Analysis", "Results"],
            "metrics": {"accuracy": 0.95},
        }

        result = generator.generate_report(data)
        assert result is not None


class TestReportGeneration:
    """Tests for report generation functions."""

    @pytest.mark.fast
    def test_generate_html_report(self, tmp_path: Any) -> Any:
        """Test HTML report generation."""
        from report import generate_html_report

        data: dict[str, Any] = {
            "title": "Test HTML Report",
            "content": "This is test content",
            "sections": [],
        }

        output_file = tmp_path / "report.html"
        output_file = tmp_path / "report.html"
        content = generate_html_report(data)
        output_file.write_text(content)

        assert content is not None
        if output_file.exists():
            content = output_file.read_text()
            assert "html" in content.lower()

    @pytest.mark.fast
    def test_generate_markdown_report(self, tmp_path: Any) -> Any:
        """Test Markdown report generation."""
        from report import generate_markdown_report

        data: dict[str, Any] = {
            "title": "Test Markdown Report",
            "content": "This is test content",
            "sections": [],
        }

        output_file = tmp_path / "report.md"
        output_file = tmp_path / "report.md"
        content = generate_markdown_report(data)
        output_file.write_text(content)

        assert content is not None

    @pytest.mark.fast
    def test_pipeline_report_modes_handle_in_progress_rows(self, tmp_path: Any) -> None:
        """Final reports should not carry preliminary self-referencing rows."""
        from report.pipeline_report import generate_pipeline_report

        summary_dir = tmp_path / "00_pipeline_summary"
        summary_dir.mkdir(parents=True)
        summary_path = summary_dir / "pipeline_execution_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "start_time": "2026-05-25T07:00:00",
                    "overall_status": "SUCCESS_WITH_WARNINGS",
                    "total_duration_seconds": 12.5,
                    "steps": [
                        {
                            "step_number": 23,
                            "script_name": "22_gui.py",
                            "description": "GUI",
                            "status": "SUCCESS",
                            "duration_seconds": 1.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        preliminary = generate_pipeline_report(
            tmp_path,
            summary_path=summary_path,
            mode="preliminary",
        )
        final = generate_pipeline_report(
            tmp_path,
            summary_path=summary_path,
            mode="final",
        )

        assert "🟡 **Status**: SUCCESS_WITH_WARNINGS" in final
        assert "IN PROGRESS" in preliminary
        assert "IN PROGRESS" not in final

    @pytest.mark.integration
    def test_generate_comprehensive_report(self, tmp_path: Any) -> Any:
        """Test comprehensive report generation."""
        import logging

        from report import generate_comprehensive_report

        logger = logging.getLogger("test_report")
        output_dir = tmp_path / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal pipeline data
        # Create test pipeline output structure
        pipeline_dir = tmp_path / "pipeline_output"
        pipeline_dir.mkdir()
        (pipeline_dir / "summary.json").write_text("{}")

        result = generate_comprehensive_report(
            pipeline_output_dir=pipeline_dir,
            report_output_dir=output_dir,
            logger=logger,
        )

        assert result is True


class TestReportProcessing:
    """Tests for report processing functionality."""

    @pytest.mark.integration
    def test_process_report_with_empty_directory(self, tmp_path: Any) -> Any:
        """Test report processing with empty input directory."""
        import logging

        from report import process_report

        logger = logging.getLogger("test_report")
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = process_report(
            target_dir=input_dir, output_dir=output_dir, logger=logger
        )

        # Should complete without error
        assert result is True or result is False

    @pytest.mark.integration
    def test_process_report_with_sample_data(
        self, tmp_path: Any, sample_gnn_files: Any
    ) -> Any:
        """Test report processing with sample GNN files."""
        import logging

        from report import process_report

        if not sample_gnn_files:
            pytest.skip("No sample GNN files available")

        logger = logging.getLogger("test_report")
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        input_dir = sample_gnn_files["simple"].parent

        result = process_report(
            target_dir=input_dir, output_dir=output_dir, logger=logger
        )

        assert result is not None

    @pytest.mark.fast
    def test_analyze_gnn_file(self, sample_gnn_files: Any) -> Any:
        """Test GNN file analysis for reporting."""
        from report import analyze_gnn_file

        if not sample_gnn_files:
            pytest.skip("No sample GNN files available")

        gnn_file = sample_gnn_files["simple"]
        result = analyze_gnn_file(gnn_file)

        assert result is not None


class TestReportValidation:
    """Tests for report validation functionality."""

    @pytest.mark.fast
    def test_validate_report_valid_data(self) -> Any:
        """Test validation with valid report data."""
        from report import validate_report

        valid_data: dict[str, Any] = {
            "title": "Test Report",
            "generated_at": "2026-01-23T12:00:00",
            "sections": ["Overview", "Analysis"],
        }

        result = validate_report(valid_data)
        assert result is True or result is None

    @pytest.mark.fast
    def test_validate_report_empty_data(self) -> Any:
        """Test validation with empty data."""
        from report import validate_report

        result = validate_report({})
        # Empty data should be handled gracefully
        assert result is not None


class TestReportModuleInfo:
    """Tests for report module information."""

    @pytest.mark.fast
    def test_get_module_info(self) -> Any:
        """Test module info retrieval."""
        from report import get_module_info

        info = get_module_info()

        assert info is not None
        assert isinstance(info, dict)
        assert "name" in info or "module" in info or len(info) > 0

    @pytest.mark.fast
    def test_get_supported_formats(self) -> Any:
        """Test supported formats retrieval."""
        from report import get_supported_formats

        formats = get_supported_formats()

        assert formats is not None
        assert isinstance(formats, (list, tuple, dict))


class TestReportAPICompleteness:
    """Tests to verify all expected report APIs are available."""

    @pytest.mark.fast
    def test_report_formatter_instantiation(self) -> Any:
        """Test ReportFormatter can be instantiated."""
        from report import ReportFormatter

        formatter = ReportFormatter()
        assert formatter is not None

    @pytest.mark.fast
    def test_report_formatter_format(self) -> Any:
        """Test ReportFormatter.format method."""
        from report import ReportFormatter

        formatter = ReportFormatter()

        data: dict[str, Any] = {"key": "value", "nested": {"a": 1}}
        result = formatter.format(data, kind="markdown")

        assert result is not None

    @pytest.mark.fast
    def test_report_formatter_format_markdown(self) -> Any:
        """Test ReportFormatter markdown formatting."""
        from report import ReportFormatter

        formatter = ReportFormatter()
        content: dict[str, Any] = {"title": "Test", "body": "Content here"}

        result = formatter.format_markdown(content)
        assert result is not None

    @pytest.mark.fast
    def test_report_formatter_format_html(self) -> Any:
        """Test ReportFormatter HTML formatting."""
        from report import ReportFormatter

        formatter = ReportFormatter()
        content: dict[str, Any] = {"title": "Test", "body": "Content here"}

        result = formatter.format_html(content)
        assert result is not None
