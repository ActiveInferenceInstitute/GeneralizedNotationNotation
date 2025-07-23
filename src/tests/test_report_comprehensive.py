#!/usr/bin/env python3
"""
Comprehensive Tests for Report Generation Module

This module provides comprehensive testing for all report generation functionality
including data analysis, formatting, generation, and MCP integration.
"""

import pytest
import json
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from report import (
    # Generator functions
    generate_comprehensive_report,
    generate_html_report_file,
    generate_markdown_report_file,
    generate_json_report_file,
    generate_custom_report,
    validate_report_data,
    
    # Analyzer functions
    collect_pipeline_data,
    analyze_step_directory,
    analyze_file_types_across_steps,
    analyze_step_dependencies,
    analyze_errors,
    get_pipeline_health_score,
    
    # Formatter functions
    generate_html_report,
    generate_markdown_report,
    get_health_color,
    
    # MCP functions
    generate_pipeline_report,
    analyze_pipeline_data as mcp_analyze_pipeline_data,
    get_report_module_info,
    register_tools
)

class TestReportAnalyzer:
    """Test the report analyzer functionality."""
    
    @pytest.fixture
    def sample_pipeline_data(self):
        """Create sample pipeline data for testing."""
        return {
            "report_generation_time": datetime.now().isoformat(),
            "pipeline_output_directory": "/test/output",
            "steps": {
                "setup_artifacts": {
                    "directory": "/test/output/setup_artifacts",
                    "exists": True,
                    "file_count": 5,
                    "total_size_mb": 2.5,
                    "file_types": {".json": {"count": 3, "total_size_mb": 1.5}, ".md": {"count": 2, "total_size_mb": 1.0}},
                    "last_modified": datetime.now().isoformat(),
                    "status": "success"
                },
                "gnn_processing_step": {
                    "directory": "/test/output/gnn_processing_step",
                    "exists": True,
                    "file_count": 3,
                    "total_size_mb": 1.8,
                    "file_types": {".md": {"count": 2, "total_size_mb": 1.2}, ".json": {"count": 1, "total_size_mb": 0.6}},
                    "last_modified": datetime.now().isoformat(),
                    "status": "success"
                },
                "test_reports": {
                    "directory": "/test/output/test_reports",
                    "exists": False,
                    "file_count": 0,
                    "total_size_mb": 0.0,
                    "file_types": {},
                    "last_modified": None,
                    "status": "missing"
                }
            },
            "summary": {
                "total_files_processed": 8,
                "total_size_mb": 4.3,
                "success_rate": 66.7
            }
        }
    
    @pytest.mark.unit
    def test_collect_pipeline_data(self, isolated_temp_dir, mock_logger):
        """Test pipeline data collection."""
        # Create mock pipeline output structure
        pipeline_dir = isolated_temp_dir / "pipeline_output"
        pipeline_dir.mkdir()
        
        # Create some step directories
        (pipeline_dir / "setup_artifacts").mkdir()
        (pipeline_dir / "gnn_processing_step").mkdir()
        
        # Create some files
        (pipeline_dir / "setup_artifacts" / "test.json").write_text('{"test": "data"}')
        (pipeline_dir / "gnn_processing_step" / "report.md").write_text('# Test Report')
        
        # Test data collection
        data = collect_pipeline_data(pipeline_dir, mock_logger)
        
        assert "steps" in data
        assert "summary" in data
        assert "report_generation_time" in data
        assert data["pipeline_output_directory"] == str(pipeline_dir)
        
        # Check step data
        assert "setup_artifacts" in data["steps"]
        assert "gnn_processing_step" in data["steps"]
        assert data["steps"]["setup_artifacts"]["exists"] is True
        assert data["steps"]["setup_artifacts"]["file_count"] > 0
    
    @pytest.mark.unit
    def test_analyze_step_directory(self, isolated_temp_dir, mock_logger):
        """Test step directory analysis."""
        step_dir = isolated_temp_dir / "test_step"
        step_dir.mkdir()
        
        # Create test files
        (step_dir / "test1.json").write_text('{"data": "test"}')
        (step_dir / "test2.md").write_text('# Test')
        (step_dir / "test3.txt").write_text('plain text')
        
        # Test analysis
        result = analyze_step_directory(step_dir, "test_step", mock_logger)
        
        assert result["exists"] is True
        assert result["file_count"] == 3
        assert result["total_size_mb"] > 0
        assert ".json" in result["file_types"]
        assert ".md" in result["file_types"]
        assert ".txt" in result["file_types"]
        assert result["status"] == "success"
    
    @pytest.mark.unit
    def test_analyze_file_types_across_steps(self, sample_pipeline_data, mock_logger):
        """Test file type analysis across steps."""
        result = analyze_file_types_across_steps(sample_pipeline_data["steps"], mock_logger)
        
        assert "total_by_type" in result
        assert ".json" in result["total_by_type"]
        assert ".md" in result["total_by_type"]
        assert result["total_by_type"][".json"]["count"] == 4  # 3 + 1
        assert result["total_by_type"][".md"]["count"] == 4    # 2 + 2
    
    @pytest.mark.unit
    def test_analyze_step_dependencies(self, sample_pipeline_data, mock_logger):
        """Test step dependency analysis."""
        result = analyze_step_dependencies(sample_pipeline_data["steps"], mock_logger)
        
        assert "step_order" in result
        assert "dependency_chain" in result
        assert "missing_prerequisites" in result
        assert len(result["step_order"]) > 0
    
    @pytest.mark.unit
    def test_analyze_errors(self, mock_logger):
        """Test error analysis."""
        errors = [
            {"type": "validation_error", "step": "gnn_processing_step", "severity": "error"},
            {"type": "file_not_found", "step": "test_reports", "severity": "warning"},
            {"type": "validation_error", "step": "gnn_processing_step", "severity": "critical"}
        ]
        
        result = analyze_errors(errors, mock_logger)
        
        assert result["total_errors"] == 3
        assert "validation_error" in result["error_types"]
        assert result["error_types"]["validation_error"] == 2
        assert len(result["critical_errors"]) == 1
        assert len(result["warnings"]) == 1
    
    @pytest.mark.unit
    def test_get_pipeline_health_score(self, sample_pipeline_data):
        """Test health score calculation."""
        score = get_pipeline_health_score(sample_pipeline_data)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    @pytest.mark.unit
    def test_get_pipeline_health_score_perfect(self):
        """Test health score for perfect pipeline."""
        perfect_data = {
            "steps": {
                "step1": {"exists": True, "status": "success"},
                "step2": {"exists": True, "status": "success"},
                "step3": {"exists": True, "status": "success"}
            },
            "summary": {"success_rate": 100},
            "error_analysis": {"total_errors": 0, "critical_errors": []},
            "performance_metrics": {"execution_time": 300}  # 5 minutes
        }
        
        score = get_pipeline_health_score(perfect_data)
        assert score > 90  # Should be very high for perfect pipeline
    
    @pytest.mark.unit
    def test_get_pipeline_health_score_failing(self):
        """Test health score for failing pipeline."""
        failing_data = {
            "steps": {
                "step1": {"exists": False, "status": "missing"},
                "step2": {"exists": False, "status": "missing"},
                "step3": {"exists": False, "status": "missing"}
            },
            "summary": {"success_rate": 0},
            "error_analysis": {"total_errors": 10, "critical_errors": [{"error": "critical"}]},
            "performance_metrics": {"execution_time": 3600}  # 1 hour
        }
        
        score = get_pipeline_health_score(failing_data)
        assert score < 50  # Should be low for failing pipeline

class TestReportFormatters:
    """Test the report formatters functionality."""
    
    @pytest.fixture
    def sample_pipeline_data_for_formatting(self):
        """Create sample pipeline data for formatting tests."""
        return {
            "report_generation_time": "2024-01-01T12:00:00",
            "pipeline_output_directory": "/test/output",
            "steps": {
                "setup_artifacts": {
                    "exists": True,
                    "file_count": 5,
                    "total_size_mb": 2.5,
                    "file_types": {".json": {"count": 3, "total_size_mb": 1.5}},
                    "last_modified": "2024-01-01T12:00:00",
                    "status": "success"
                }
            },
            "summary": {
                "total_files_processed": 5,
                "total_size_mb": 2.5,
                "success_rate": 100.0
            },
            "health_score": 95.0
        }
    
    @pytest.mark.unit
    def test_generate_html_report(self, sample_pipeline_data_for_formatting, mock_logger):
        """Test HTML report generation."""
        html_content = generate_html_report(sample_pipeline_data_for_formatting, mock_logger)
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "<html" in html_content
        assert "GNN Pipeline Comprehensive Analysis Report" in html_content
        assert "Health Score: 95.0/100" in html_content
        assert "setup_artifacts" in html_content
    
    @pytest.mark.unit
    def test_generate_markdown_report(self, sample_pipeline_data_for_formatting, mock_logger):
        """Test Markdown report generation."""
        markdown_content = generate_markdown_report(sample_pipeline_data_for_formatting, mock_logger)
        
        assert isinstance(markdown_content, str)
        assert "# 🎯 GNN Pipeline Comprehensive Analysis Report" in markdown_content
        assert "Health Score: 95.0/100" in markdown_content
        assert "## 📊 Pipeline Overview" in markdown_content
        assert "setup_artifacts" in markdown_content
    
    @pytest.mark.unit
    def test_get_health_color(self):
        """Test health color function."""
        assert get_health_color(95) == "#28a745"  # Green
        assert get_health_color(75) == "#ffc107"  # Yellow
        assert get_health_color(45) == "#fd7e14"  # Orange
        assert get_health_color(25) == "#dc3545"  # Red
    
    @pytest.mark.unit
    def test_html_report_with_errors(self, mock_logger):
        """Test HTML report generation with error data."""
        data_with_errors = {
            "report_generation_time": "2024-01-01T12:00:00",
            "pipeline_output_directory": "/test/output",
            "steps": {},
            "summary": {"total_files_processed": 0, "success_rate": 0},
            "error_analysis": {
                "total_errors": 3,
                "error_types": {"validation_error": 2, "file_not_found": 1},
                "critical_errors": [{"error": "critical"}],
                "warnings": [{"warning": "minor"}]
            },
            "health_score": 30.0
        }
        
        html_content = generate_html_report(data_with_errors, mock_logger)
        assert "Error Analysis" in html_content
        assert "Total Errors: 3" in html_content
        assert "validation_error" in html_content

class TestReportGenerator:
    """Test the report generator functionality."""
    
    @pytest.mark.unit
    def test_generate_comprehensive_report(self, isolated_temp_dir, mock_logger):
        """Test comprehensive report generation."""
        # Create mock pipeline output
        pipeline_dir = isolated_temp_dir / "pipeline_output"
        pipeline_dir.mkdir()
        (pipeline_dir / "setup_artifacts").mkdir()
        (pipeline_dir / "setup_artifacts" / "test.json").write_text('{"test": "data"}')
        
        # Create report output directory
        report_dir = isolated_temp_dir / "report_output"
        
        # Test report generation
        success = generate_comprehensive_report(pipeline_dir, report_dir, mock_logger)
        
        assert success is True
        assert (report_dir / "comprehensive_analysis_report.html").exists()
        assert (report_dir / "comprehensive_analysis_report.md").exists()
        assert (report_dir / "report_summary.json").exists()
        assert (report_dir / "report_generation_summary.json").exists()
    
    @pytest.mark.unit
    def test_generate_html_report_file(self, sample_pipeline_data, isolated_temp_dir, mock_logger):
        """Test HTML report file generation."""
        report_dir = isolated_temp_dir / "report_output"
        report_dir.mkdir()
        
        success = generate_html_report_file(sample_pipeline_data, report_dir, mock_logger)
        
        assert success is True
        assert (report_dir / "comprehensive_analysis_report.html").exists()
    
    @pytest.mark.unit
    def test_generate_markdown_report_file(self, sample_pipeline_data, isolated_temp_dir, mock_logger):
        """Test Markdown report file generation."""
        report_dir = isolated_temp_dir / "report_output"
        report_dir.mkdir()
        
        success = generate_markdown_report_file(sample_pipeline_data, report_dir, mock_logger)
        
        assert success is True
        assert (report_dir / "comprehensive_analysis_report.md").exists()
    
    @pytest.mark.unit
    def test_generate_json_report_file(self, sample_pipeline_data, isolated_temp_dir, mock_logger):
        """Test JSON report file generation."""
        report_dir = isolated_temp_dir / "report_output"
        report_dir.mkdir()
        
        success = generate_json_report_file(sample_pipeline_data, report_dir, mock_logger)
        
        assert success is True
        assert (report_dir / "report_summary.json").exists()
        
        # Verify JSON content
        with open(report_dir / "report_summary.json", 'r') as f:
            data = json.load(f)
        assert "steps" in data
        assert "summary" in data
    
    @pytest.mark.unit
    def test_generate_custom_report(self, isolated_temp_dir, mock_logger):
        """Test custom report generation with filtering."""
        # Create mock pipeline output
        pipeline_dir = isolated_temp_dir / "pipeline_output"
        pipeline_dir.mkdir()
        (pipeline_dir / "setup_artifacts").mkdir()
        (pipeline_dir / "gnn_processing_step").mkdir()
        (pipeline_dir / "setup_artifacts" / "test.json").write_text('{"test": "data"}')
        (pipeline_dir / "gnn_processing_step" / "report.md").write_text('# Test')
        
        # Create report output directory
        report_dir = isolated_temp_dir / "report_output"
        
        # Test with step filtering
        success = generate_custom_report(
            pipeline_dir, 
            report_dir, 
            mock_logger,
            step_filter=["setup_artifacts"],
            format_type="html"
        )
        
        assert success is True
        assert (report_dir / "comprehensive_analysis_report.html").exists()
    
    @pytest.mark.unit
    def test_validate_report_data(self, sample_pipeline_data):
        """Test report data validation."""
        result = validate_report_data(sample_pipeline_data)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    @pytest.mark.unit
    def test_validate_report_data_invalid(self):
        """Test report data validation with invalid data."""
        invalid_data = {
            "steps": "not_a_dict",  # Invalid
            "summary": {}
        }
        
        result = validate_report_data(invalid_data)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
    
    @pytest.mark.unit
    def test_generate_comprehensive_report_nonexistent_pipeline(self, isolated_temp_dir, mock_logger):
        """Test report generation with nonexistent pipeline directory."""
        pipeline_dir = isolated_temp_dir / "nonexistent"
        report_dir = isolated_temp_dir / "report_output"
        
        success = generate_comprehensive_report(pipeline_dir, report_dir, mock_logger)
        
        assert success is False

class TestReportMCP:
    """Test the report MCP integration functionality."""
    
    @pytest.mark.unit
    def test_generate_pipeline_report(self, isolated_temp_dir, mock_logger):
        """Test MCP pipeline report generation."""
        # Create mock pipeline output
        pipeline_dir = isolated_temp_dir / "pipeline_output"
        pipeline_dir.mkdir()
        (pipeline_dir / "setup_artifacts").mkdir()
        (pipeline_dir / "setup_artifacts" / "test.json").write_text('{"test": "data"}')
        
        result = generate_pipeline_report(str(pipeline_dir))
        
        assert result["success"] is True
        assert "pipeline_directory" in result
        assert "report_directory" in result
        assert "generated_files" in result
        assert len(result["generated_files"]) > 0
    
    @pytest.mark.unit
    def test_generate_pipeline_report_nonexistent(self):
        """Test MCP pipeline report generation with nonexistent directory."""
        result = generate_pipeline_report("/nonexistent/directory")
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.unit
    def test_mcp_analyze_pipeline_data(self, isolated_temp_dir):
        """Test MCP pipeline data analysis."""
        # Create mock pipeline output
        pipeline_dir = isolated_temp_dir / "pipeline_output"
        pipeline_dir.mkdir()
        (pipeline_dir / "setup_artifacts").mkdir()
        (pipeline_dir / "setup_artifacts" / "test.json").write_text('{"test": "data"}')
        
        result = mcp_analyze_pipeline_data(str(pipeline_dir))
        
        assert result["success"] is True
        assert "analysis_summary" in result
        assert "step_details" in result
        assert result["analysis_summary"]["total_steps_analyzed"] > 0
    
    @pytest.mark.unit
    def test_get_report_module_info(self):
        """Test report module info retrieval."""
        info = get_report_module_info()
        
        assert info["module_name"] == "report"
        assert "description" in info
        assert "version" in info
        assert "available_functions" in info
        assert "supported_formats" in info
        assert "supported_steps" in info
        assert "dependencies" in info
    
    @pytest.mark.unit
    def test_register_tools(self, mock_logger):
        """Test MCP tool registration."""
        mock_mcp = Mock()
        
        register_tools(mock_mcp)
        
        # Verify tools were registered
        assert mock_mcp.register_tool.call_count >= 4  # At least 4 tools should be registered

class TestReportIntegration:
    """Integration tests for the report module."""
    
    @pytest.mark.integration
    def test_full_report_generation_workflow(self, isolated_temp_dir, mock_logger):
        """Test complete report generation workflow."""
        # Create comprehensive mock pipeline output
        pipeline_dir = isolated_temp_dir / "pipeline_output"
        pipeline_dir.mkdir()
        
        # Create multiple step directories with various file types
        steps = [
            "setup_artifacts",
            "gnn_processing_step",
            "test_reports",
            "type_check",
            "gnn_exports",
            "visualization"
        ]
        
        for step in steps:
            step_dir = pipeline_dir / step
            step_dir.mkdir()
            
            # Create different file types for each step
            if step == "setup_artifacts":
                (step_dir / "installed_packages.json").write_text('{"packages": ["test"]}')
                (step_dir / "directory_structure.json").write_text('{"structure": "test"}')
            elif step == "gnn_processing_step":
                (step_dir / "gnn_discovery_report.json").write_text('{"discovered": 5}')
                (step_dir / "processing_summary.md").write_text('# Processing Summary')
            elif step == "test_reports":
                (step_dir / "pytest_report.xml").write_text('<testsuite></testsuite>')
                (step_dir / "test_summary.json").write_text('{"passed": 10, "failed": 0}')
            elif step == "type_check":
                (step_dir / "type_check_report.md").write_text('# Type Check Report')
            elif step == "gnn_exports":
                (step_dir / "export.json").write_text('{"export": "data"}')
                (step_dir / "export.xml").write_text('<export></export>')
            elif step == "visualization":
                (step_dir / "graph.png").write_bytes(b'fake_png_data')
                (step_dir / "matrix.svg").write_text('<svg></svg>')
        
        # Create pipeline execution summary
        summary = {
            "start_time": "2024-01-01T10:00:00",
            "end_time": "2024-01-01T12:00:00",
            "overall_status": "success",
            "steps": [{"name": step, "status": "success"} for step in steps],
            "performance_metrics": {
                "execution_time": 7200,  # 2 hours
                "memory_usage_mb": 512,
                "cpu_usage_percent": 75
            },
            "errors": []
        }
        
        with open(pipeline_dir / "pipeline_execution_summary.json", 'w') as f:
            json.dump(summary, f)
        
        # Generate comprehensive report
        report_dir = isolated_temp_dir / "report_output"
        success = generate_comprehensive_report(pipeline_dir, report_dir, mock_logger)
        
        assert success is True
        
        # Verify all report files were generated
        expected_files = [
            "comprehensive_analysis_report.html",
            "comprehensive_analysis_report.md",
            "report_summary.json",
            "report_generation_summary.json"
        ]
        
        for expected_file in expected_files:
            assert (report_dir / expected_file).exists()
        
        # Verify HTML report content
        html_content = (report_dir / "comprehensive_analysis_report.html").read_text()
        assert "GNN Pipeline Comprehensive Analysis Report" in html_content
        assert "Health Score" in html_content
        assert "setup_artifacts" in html_content
        assert "gnn_processing_step" in html_content
        
        # Verify JSON report structure
        with open(report_dir / "report_summary.json", 'r') as f:
            json_data = json.load(f)
        
        assert "steps" in json_data
        assert "summary" in json_data
        assert "health_score" in json_data
        assert "performance_metrics" in json_data
        assert len(json_data["steps"]) == len(steps)
        
        # Verify health score is reasonable
        health_score = json_data["health_score"]
        assert isinstance(health_score, (int, float))
        assert 0 <= health_score <= 100
    
    @pytest.mark.integration
    def test_report_with_errors_and_warnings(self, isolated_temp_dir, mock_logger):
        """Test report generation with errors and warnings."""
        pipeline_dir = isolated_temp_dir / "pipeline_output"
        pipeline_dir.mkdir()
        
        # Create pipeline summary with errors
        summary = {
            "start_time": "2024-01-01T10:00:00",
            "end_time": "2024-01-01T12:00:00",
            "overall_status": "partial_success",
            "steps": [
                {"name": "setup_artifacts", "status": "success"},
                {"name": "gnn_processing_step", "status": "error"},
                {"name": "test_reports", "status": "warning"}
            ],
            "performance_metrics": {
                "execution_time": 3600
            },
            "errors": [
                {"type": "validation_error", "step": "gnn_processing_step", "severity": "error"},
                {"type": "file_not_found", "step": "test_reports", "severity": "warning"},
                {"type": "timeout", "step": "gnn_processing_step", "severity": "critical"}
            ]
        }
        
        with open(pipeline_dir / "pipeline_execution_summary.json", 'w') as f:
            json.dump(summary, f)
        
        # Create some step directories
        (pipeline_dir / "setup_artifacts").mkdir()
        (pipeline_dir / "setup_artifacts" / "test.json").write_text('{"test": "data"}')
        
        # Generate report
        report_dir = isolated_temp_dir / "report_output"
        success = generate_comprehensive_report(pipeline_dir, report_dir, mock_logger)
        
        assert success is True
        
        # Verify error analysis is included
        with open(report_dir / "report_summary.json", 'r') as f:
            json_data = json.load(f)
        
        assert "error_analysis" in json_data
        error_analysis = json_data["error_analysis"]
        assert error_analysis["total_errors"] == 3
        assert "validation_error" in error_analysis["error_types"]
        assert len(error_analysis["critical_errors"]) == 1
        assert len(error_analysis["warnings"]) == 1
        
        # Verify health score reflects errors
        health_score = json_data["health_score"]
        assert health_score < 80  # Should be lower due to errors

def test_report_module_completeness():
    """Test that all expected functions are available."""
    import report
    
    expected_functions = [
        'generate_comprehensive_report',
        'generate_html_report_file',
        'generate_markdown_report_file',
        'generate_json_report_file',
        'generate_custom_report',
        'validate_report_data',
        'collect_pipeline_data',
        'analyze_step_directory',
        'analyze_file_types_across_steps',
        'analyze_step_dependencies',
        'analyze_errors',
        'get_pipeline_health_score',
        'generate_html_report',
        'generate_markdown_report',
        'get_health_color',
        'generate_pipeline_report',
        'mcp_analyze_pipeline_data',
        'get_report_module_info',
        'register_tools'
    ]
    
    for func_name in expected_functions:
        assert hasattr(report, func_name), f"Missing function: {func_name}"

@pytest.mark.slow
def test_report_performance_characteristics():
    """Test report generation performance with large datasets."""
    # This test would measure performance characteristics
    # but is marked as slow to avoid running in regular test suites
    assert True, "Performance test placeholder" 