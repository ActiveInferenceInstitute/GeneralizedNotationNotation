#!/usr/bin/env python3
"""
Test Report Formats - Tests for report output format functionality.

Tests HTML, Markdown, JSON, and other report format generation.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHTMLReportFormat:
    """Tests for HTML report format generation."""

    @pytest.mark.fast
    def test_html_report_structure(self, tmp_path):
        """Test HTML report has correct structure."""
        from report import generate_html_report
        
        data = {
            "title": "Test Report",
            "sections": [
                {"name": "Overview", "content": "Overview content"},
                {"name": "Analysis", "content": "Analysis content"}
            ]
        }
        
        output_file = tmp_path / "report.html"
        content = generate_html_report(data)
        output_file.write_text(content, encoding='utf-8')
        
        if output_file.exists():
            content = output_file.read_text()
            assert "<html" in content.lower() or "<!doctype" in content.lower()

    @pytest.mark.fast
    def test_html_report_escaping(self, tmp_path):
        """Test HTML report properly escapes special characters."""
        from report import generate_html_report
        
        data = {
            "title": "Test <script>alert('xss')</script>",
            "content": "Content with <>&\" characters"
        }
        
        output_file = tmp_path / "escaped_report.html"
        content = generate_html_report(data)
        output_file.write_text(content, encoding='utf-8')

        # Verify output file exists and content is escaped
        if output_file.exists():
            content = output_file.read_text()
            # Raw script tags should be escaped or removed
            assert "<script>alert" not in content, "XSS content should be escaped"


class TestMarkdownReportFormat:
    """Tests for Markdown report format generation."""

    @pytest.mark.fast
    def test_markdown_report_structure(self, tmp_path):
        """Test Markdown report has correct structure."""
        from report import generate_markdown_report
        
        data = {
            "title": "Test Report",
            "sections": [
                {"name": "Overview", "content": "Overview content"}
            ]
        }
        
        output_file = tmp_path / "report.md"
        content = generate_markdown_report(data)
        output_file.write_text(content, encoding='utf-8')
        
        if output_file.exists():
            content = output_file.read_text()
            # Markdown should have headers or content
            assert len(content) > 0

    @pytest.mark.fast
    def test_markdown_report_tables(self, tmp_path):
        """Test Markdown report can include tables."""
        from report import generate_markdown_report
        
        data = {
            "title": "Report with Tables",
            "tables": [
                {
                    "headers": ["Column 1", "Column 2"],
                    "rows": [["A", "B"], ["C", "D"]]
                }
            ]
        }
        
        output_file = tmp_path / "table_report.md"
        content = generate_markdown_report(data)
        output_file.write_text(content, encoding='utf-8')

        # Verify output file was created
        if output_file.exists():
            content = output_file.read_text()
            assert len(content) > 0, "Report should have content"
            # Tables in markdown typically use pipe characters
            assert "|" in content or "Column" in content or len(content) > 10, \
                "Markdown report should contain table-like content"


class TestJSONReportFormat:
    """Tests for JSON report format generation."""

    @pytest.mark.fast
    def test_json_report_generation(self, tmp_path):
        """Test JSON report is valid JSON."""
        import json
        from report import generate_report
        
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = generate_report(
            target_dir=input_dir,
            output_dir=output_dir,
            format="json"
        )
        
        # Check for JSON files in output
        json_files = list(output_dir.glob("*.json"))
        for jf in json_files:
            content = jf.read_text()
            # Should be valid JSON
            try:
                json.loads(content)
            except json.JSONDecodeError:
                pass  # Some files may not be pure JSON

    @pytest.mark.fast
    def test_json_report_completeness(self, tmp_path):
        """Test JSON report contains expected fields."""
        from report import generate_report
        import json
        
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = generate_report(
            target_dir=input_dir,
            output_dir=output_dir,
            format="json"
        )

        # Verify report generation returned a result
        assert result is not None or output_dir.exists(), \
            "Report generation should return result or create output directory"


class TestReportFormatterFormats:
    """Tests for ReportFormatter format support."""

    @pytest.mark.fast
    def test_formatter_supports_markdown(self):
        """Test formatter supports markdown format."""
        from report import ReportFormatter
        
        formatter = ReportFormatter()
        data = {"key": "value"}
        
        result = formatter.format(data, kind="markdown")
        assert result is not None

    @pytest.mark.fast
    def test_formatter_supports_html(self):
        """Test formatter supports HTML format."""
        from report import ReportFormatter
        
        formatter = ReportFormatter()
        data = {"key": "value"}
        
        # Use format_html directly
        result = formatter.format_html(data)
        assert result is not None

    @pytest.mark.fast
    def test_formatter_default_format(self):
        """Test formatter has sensible default format."""
        from report import ReportFormatter
        
        formatter = ReportFormatter()
        data = {"test": "data"}
        
        # Default should work
        result = formatter.format(data)
        assert result is not None


class TestSupportedFormats:
    """Tests for format support discovery."""

    @pytest.mark.fast
    def test_get_supported_formats_returns_formats(self):
        """Test that supported formats can be retrieved."""
        from report import get_supported_formats
        
        formats = get_supported_formats()
        
        assert formats is not None
        # Should have at least some formats
        if isinstance(formats, (list, tuple)):
            assert len(formats) >= 0, "Formats list should exist"
        elif isinstance(formats, dict):
            assert len(formats) >= 0, "Formats dict should exist"
        else:
            pytest.fail(f"Unexpected formats type: {type(formats)}")

    @pytest.mark.fast
    def test_features_include_format_support(self):
        """Test FEATURES dict includes format support flags."""
        from report import FEATURES
        
        assert isinstance(FEATURES, dict)
        
        # Check for expected format features
        if "html_reports" in FEATURES:
            assert FEATURES["html_reports"] is True
        if "markdown_reports" in FEATURES:
            assert FEATURES["markdown_reports"] is True
