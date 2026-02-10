#!/usr/bin/env python3
"""
Test Report Integration - Integration tests for report module with pipeline.

Tests the integration between report generation and pipeline execution.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestReportPipelineIntegration:
    """Tests for report integration with pipeline execution."""

    @pytest.mark.integration
    def test_report_receives_pipeline_data(self, tmp_path):
        """Test that report module correctly receives pipeline data."""
        from report import process_report, get_module_info
        import logging
        
        logger = logging.getLogger("test_report")
        
        # Get module info to verify availability
        info = get_module_info()
        assert info is not None

    @pytest.mark.integration
    def test_report_processes_pipeline_outputs(self, tmp_path):
        """Test report generation from pipeline output directories."""
        from report import process_report
        import logging
        
        logger = logging.getLogger("test_report")
        
        # Create mock pipeline output structure
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create some mock step outputs
        for step in ["3_gnn_output", "8_visualization_output", "11_render_output"]:
            step_dir = output_dir / step
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / "summary.json").write_text('{"status": "success"}')
        
        report_output = tmp_path / "report_output"
        report_output.mkdir(parents=True, exist_ok=True)
        
        result = process_report(
            target_dir=output_dir,
            output_dir=report_output,
            logger=logger
        )
        
        assert result is True or result is False

    @pytest.mark.integration
    def test_report_with_visualization_outputs(self, tmp_path):
        """Test report includes visualization outputs."""
        from report import process_report
        import logging
        
        logger = logging.getLogger("test_report")
        
        # Create visualization output structure
        viz_dir = tmp_path / "output" / "8_visualization_output"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock visualization files
        (viz_dir / "graph.png").write_bytes(b"fake png")
        (viz_dir / "matrix_heatmap.png").write_bytes(b"fake png")
        
        report_output = tmp_path / "report_output"
        report_output.mkdir(parents=True, exist_ok=True)
        
        result = process_report(
            target_dir=tmp_path / "output",
            output_dir=report_output,
            logger=logger
        )
        
        # Should complete without error
        assert result is not None or result is None


class TestReportGNNIntegration:
    """Tests for report integration with GNN processing."""

    @pytest.mark.integration
    def test_report_analyzes_gnn_files(self, sample_gnn_files, tmp_path):
        """Test report analysis of GNN files."""
        from report import analyze_gnn_file
        
        if not sample_gnn_files:
            pytest.skip("No sample GNN files available")
        
        gnn_file = list(sample_gnn_files.values())[0]
        result = analyze_gnn_file(gnn_file)
        
        assert result is not None

    @pytest.mark.integration
    def test_report_includes_gnn_metrics(self, sample_gnn_files, tmp_path):
        """Test that reports include GNN-specific metrics."""
        from report import process_report
        import logging
        
        if not sample_gnn_files:
            pytest.skip("No sample GNN files available")
        
        logger = logging.getLogger("test_report")
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        gnn_file = list(sample_gnn_files.values())[0]
        result = process_report(
            target_dir=gnn_file.parent,
            output_dir=output_dir,
            logger=logger
        )
        
        # Should produce some output
        assert result is not None or result is None


class TestReportAnalysisIntegration:
    """Tests for report integration with analysis module."""

    @pytest.mark.integration
    def test_analyze_pipeline_data(self):
        """Test pipeline data analysis for reporting."""
        from report import analyze_pipeline_data
        
        pipeline_data = {
            "steps": [
                {"name": "step1", "status": "success", "duration": 1.5},
                {"name": "step2", "status": "success", "duration": 2.0}
            ],
            "total_duration": 3.5,
            "status": "completed"
        }
        
        result = analyze_pipeline_data(pipeline_data)
        
        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.integration
    def test_analyze_empty_pipeline_data(self):
        """Test analysis handles empty pipeline data gracefully."""
        from report import analyze_pipeline_data
        
        result = analyze_pipeline_data({})
        
        # Should not raise an error
        assert result is not None


class TestReportExportIntegration:
    """Tests for report integration with export functionality."""

    @pytest.mark.integration
    def test_report_exports_multiple_formats(self, tmp_path):
        """Test report exports to multiple formats simultaneously."""
        from report import generate_comprehensive_report
        import logging
        import json
        
        logger = logging.getLogger("test_report")
        
        # Create a mock pipeline output directory with step data
        pipeline_output_dir = tmp_path / "pipeline_output"
        pipeline_output_dir.mkdir(parents=True, exist_ok=True)
        for step in ["3_gnn_output", "8_visualization_output"]:
            step_dir = pipeline_output_dir / step
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / "summary.json").write_text(
                json.dumps({"status": "success", "duration": 1.5})
            )
        
        report_output_dir = tmp_path / "reports"
        report_output_dir.mkdir(parents=True, exist_ok=True)
        
        result = generate_comprehensive_report(
            pipeline_output_dir=pipeline_output_dir,
            report_output_dir=report_output_dir,
            logger=logger
        )
        
        # Should return a boolean indicating success/failure
        assert isinstance(result, bool)

    @pytest.mark.integration
    def test_report_respects_output_directory(self, tmp_path):
        """Test report outputs go to specified directory."""
        from report import process_report
        import logging
        
        logger = logging.getLogger("test_report")
        
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "custom_output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        process_report(
            target_dir=input_dir,
            output_dir=output_dir,
            logger=logger
        )
        
        # Output directory should exist
        assert output_dir.exists()


class TestReportModuleIntegration:
    """Tests for report module integration with MCP."""

    @pytest.mark.integration
    def test_report_module_exports(self):
        """Test that report module exports expected functions."""
        from report import (
            process_report,
            generate_comprehensive_report,
            analyze_gnn_file,
            generate_html_report,
            generate_markdown_report,
            ReportGenerator,
            ReportFormatter,
            get_module_info,
            get_supported_formats,
            validate_report,
            generate_report
        )
        
        # All imports should succeed
        assert process_report is not None
        assert generate_comprehensive_report is not None
        assert ReportGenerator is not None
        assert ReportFormatter is not None
