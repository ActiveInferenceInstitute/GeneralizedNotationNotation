#!/usr/bin/env python3
"""
Test Intelligent Analysis Overall Tests

This file contains comprehensive tests for the intelligent_analysis module functionality.
"""

import pytest
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


# Sample pipeline summary data for testing
SAMPLE_PIPELINE_SUMMARY = {
    "start_time": "2024-01-23T10:00:00",
    "end_time": "2024-01-23T10:05:30",
    "total_duration_seconds": 330.5,
    "overall_status": "SUCCESS_WITH_WARNINGS",
    "steps": [
        {
            "step_number": 1,
            "script_name": "0_template.py",
            "description": "Template initialization",
            "status": "SUCCESS",
            "duration_seconds": 2.5,
            "exit_code": 0,
            "peak_memory_mb": 150.0
        },
        {
            "step_number": 2,
            "script_name": "1_setup.py",
            "description": "Environment setup",
            "status": "SUCCESS",
            "duration_seconds": 15.3,
            "exit_code": 0,
            "peak_memory_mb": 200.0
        },
        {
            "step_number": 3,
            "script_name": "2_tests.py",
            "description": "Test execution",
            "status": "SUCCESS_WITH_WARNINGS",
            "duration_seconds": 120.0,
            "exit_code": 0,
            "peak_memory_mb": 512.0,
            "stdout": "WARNING: Some tests skipped"
        },
        {
            "step_number": 4,
            "script_name": "3_gnn.py",
            "description": "GNN processing",
            "status": "SUCCESS",
            "duration_seconds": 45.2,
            "exit_code": 0,
            "peak_memory_mb": 350.0
        }
    ],
    "performance_summary": {
        "peak_memory_mb": 512.0,
        "successful_steps": 4,
        "failed_steps": 0,
        "warnings": 1,
        "total_steps": 4
    },
    "environment_info": {
        "python_version": "3.12.0",
        "platform": "darwin"
    }
}

SAMPLE_FAILED_SUMMARY = {
    "start_time": "2024-01-23T10:00:00",
    "end_time": "2024-01-23T10:02:30",
    "total_duration_seconds": 150.0,
    "overall_status": "FAILED",
    "steps": [
        {
            "step_number": 1,
            "script_name": "0_template.py",
            "status": "SUCCESS",
            "duration_seconds": 2.0,
            "exit_code": 0,
            "peak_memory_mb": 100.0
        },
        {
            "step_number": 2,
            "script_name": "1_setup.py",
            "status": "FAILED",
            "duration_seconds": 30.0,
            "exit_code": 1,
            "peak_memory_mb": 200.0,
            "stderr": "ModuleNotFoundError: No module named 'missing_dependency'"
        },
        {
            "step_number": 3,
            "script_name": "2_tests.py",
            "status": "FAILED",
            "duration_seconds": 5.0,
            "exit_code": 1,
            "peak_memory_mb": 150.0,
            "stderr": "Error: Prerequisite step failed"
        }
    ],
    "performance_summary": {
        "peak_memory_mb": 200.0,
        "successful_steps": 1,
        "failed_steps": 2,
        "warnings": 0,
        "total_steps": 3
    }
}


class TestIntelligentAnalysisModuleComprehensive:
    """Comprehensive tests for the intelligent_analysis module."""

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_module_imports(self):
        """Test that intelligent_analysis module can be imported."""
        try:
            import intelligent_analysis
            assert hasattr(intelligent_analysis, '__version__')
            assert hasattr(intelligent_analysis, 'FEATURES')
            assert hasattr(intelligent_analysis, 'process_intelligent_analysis')
            assert hasattr(intelligent_analysis, 'IntelligentAnalyzer')
        except ImportError:
            pytest.skip("intelligent_analysis module not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_module_version(self):
        """Test module version is set correctly."""
        try:
            import intelligent_analysis
            assert intelligent_analysis.__version__ == "2.0.0"
        except ImportError:
            pytest.skip("intelligent_analysis module not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_features_dict(self):
        """Test FEATURES dictionary contains expected capabilities."""
        try:
            import intelligent_analysis
            expected_features = [
                "pipeline_analysis",
                "failure_root_cause",
                "performance_optimization",
                "llm_powered_insights",
                "executive_reports"
            ]
            for feature in expected_features:
                assert feature in intelligent_analysis.FEATURES
        except ImportError:
            pytest.skip("intelligent_analysis module not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_get_module_info(self):
        """Test module information retrieval."""
        try:
            from intelligent_analysis import get_module_info
            info = get_module_info()
            assert isinstance(info, dict)
            assert 'version' in info
            assert 'description' in info
            assert 'features' in info
            assert 'report_formats' in info
        except ImportError:
            pytest.skip("intelligent_analysis module not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_get_supported_analysis_types(self):
        """Test analysis types retrieval."""
        try:
            from intelligent_analysis import get_supported_analysis_types
            types = get_supported_analysis_types()
            assert isinstance(types, list)
            assert len(types) > 0
            expected_types = ["failure_analysis", "performance_analysis", "executive_summary"]
            for t in expected_types:
                assert t in types
        except ImportError:
            pytest.skip("intelligent_analysis module not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_validate_pipeline_summary(self):
        """Test pipeline summary validation."""
        try:
            from intelligent_analysis import validate_pipeline_summary

            # Valid summary
            assert validate_pipeline_summary(SAMPLE_PIPELINE_SUMMARY) is True

            # Invalid summary (missing required fields)
            invalid_summary = {"foo": "bar"}
            assert validate_pipeline_summary(invalid_summary) is False
        except ImportError:
            pytest.skip("intelligent_analysis module not available")


class TestProcessorFunctions:
    """Tests for processor functions."""

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_analyze_pipeline_summary(self):
        """Test pipeline summary analysis."""
        try:
            from intelligent_analysis.processor import analyze_pipeline_summary

            analysis = analyze_pipeline_summary(SAMPLE_PIPELINE_SUMMARY)
            assert isinstance(analysis, dict)
            assert "overall_status" in analysis
            assert "health_score" in analysis
            assert "failures" in analysis
            assert "warnings" in analysis
            assert analysis["overall_status"] == "SUCCESS_WITH_WARNINGS"
            assert 0 <= analysis["health_score"] <= 100
        except ImportError:
            pytest.skip("intelligent_analysis.processor not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_analyze_failed_summary(self):
        """Test analysis of failed pipeline summary."""
        try:
            from intelligent_analysis.processor import analyze_pipeline_summary

            analysis = analyze_pipeline_summary(SAMPLE_FAILED_SUMMARY)
            assert analysis["overall_status"] == "FAILED"
            assert len(analysis["failures"]) == 2
            assert analysis["health_score"] < 50  # Should be low due to failures
        except ImportError:
            pytest.skip("intelligent_analysis.processor not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_identify_bottlenecks(self):
        """Test bottleneck identification."""
        try:
            from intelligent_analysis.processor import identify_bottlenecks

            bottlenecks = identify_bottlenecks(SAMPLE_PIPELINE_SUMMARY)
            assert isinstance(bottlenecks, list)
            # 2_tests.py with 120s should be identified as a bottleneck
            if bottlenecks:
                assert any(b["step"] == "2_tests.py" for b in bottlenecks)
        except ImportError:
            pytest.skip("intelligent_analysis.processor not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_identify_bottlenecks_custom_threshold(self):
        """Test bottleneck identification with custom threshold."""
        try:
            from intelligent_analysis.processor import identify_bottlenecks

            # Low threshold should catch more bottlenecks
            bottlenecks = identify_bottlenecks(SAMPLE_PIPELINE_SUMMARY, threshold_seconds=10.0)
            assert isinstance(bottlenecks, list)
            assert len(bottlenecks) >= 1
        except ImportError:
            pytest.skip("intelligent_analysis.processor not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_extract_failure_context(self):
        """Test failure context extraction."""
        try:
            from intelligent_analysis.processor import extract_failure_context

            failures = extract_failure_context(SAMPLE_FAILED_SUMMARY)
            assert isinstance(failures, list)
            assert len(failures) == 2

            # Check first failure has expected context
            first_failure = failures[0]
            assert "step_name" in first_failure
            assert "exit_code" in first_failure
            assert "error_output" in first_failure
        except ImportError:
            pytest.skip("intelligent_analysis.processor not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        try:
            from intelligent_analysis.processor import (
                analyze_pipeline_summary,
                analyze_individual_steps,
                identify_bottlenecks,
                generate_recommendations
            )

            analysis = analyze_pipeline_summary(SAMPLE_FAILED_SUMMARY)
            bottlenecks = identify_bottlenecks(SAMPLE_FAILED_SUMMARY)
            step_analyses, flags_by_type = analyze_individual_steps(SAMPLE_FAILED_SUMMARY)
            recommendations = generate_recommendations(analysis, bottlenecks, flags_by_type)

            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
            # Should have failure-related recommendation (red flags or critical)
            assert any("critical" in r.lower() or "red" in r.lower() or "failed" in r.lower() for r in recommendations)
        except ImportError:
            pytest.skip("intelligent_analysis.processor not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_generate_executive_report(self):
        """Test executive report generation."""
        try:
            from intelligent_analysis.processor import (
                analyze_pipeline_summary,
                analyze_individual_steps,
                identify_bottlenecks,
                extract_failure_context,
                generate_recommendations,
                generate_executive_report
            )

            analysis = analyze_pipeline_summary(SAMPLE_PIPELINE_SUMMARY)
            bottlenecks = identify_bottlenecks(SAMPLE_PIPELINE_SUMMARY)
            failures = extract_failure_context(SAMPLE_PIPELINE_SUMMARY)
            step_analyses, flags_by_type = analyze_individual_steps(SAMPLE_PIPELINE_SUMMARY)
            recommendations = generate_recommendations(analysis, bottlenecks, flags_by_type)

            report = generate_executive_report(
                analysis, bottlenecks, failures, recommendations,
                step_analyses, flags_by_type
            )

            assert isinstance(report, str)
            assert "# Pipeline Intelligent Analysis Report" in report
            assert "Quick Overview" in report or "Executive Summary" in report
            assert "Recommendations" in report
            assert "Per-Step" in report
        except ImportError:
            pytest.skip("intelligent_analysis.processor not available")


class TestAnalyzerClass:
    """Tests for the IntelligentAnalyzer class."""

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_analyzer_instantiation(self):
        """Test IntelligentAnalyzer class instantiation."""
        try:
            from intelligent_analysis import IntelligentAnalyzer, AnalysisContext

            analyzer = IntelligentAnalyzer()
            assert analyzer is not None
            assert analyzer.context is None
        except ImportError:
            pytest.skip("IntelligentAnalyzer not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_analyzer_with_context(self):
        """Test analyzer with context."""
        try:
            from intelligent_analysis import IntelligentAnalyzer, AnalysisContext

            context = AnalysisContext(summary_data=SAMPLE_PIPELINE_SUMMARY)
            analyzer = IntelligentAnalyzer(context=context)

            assert analyzer.context is not None
            assert analyzer.context.overall_status == "SUCCESS_WITH_WARNINGS"
        except ImportError:
            pytest.skip("IntelligentAnalyzer not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_analyzer_set_context(self):
        """Test setting context on analyzer."""
        try:
            from intelligent_analysis import IntelligentAnalyzer, AnalysisContext

            analyzer = IntelligentAnalyzer()
            context = AnalysisContext(summary_data=SAMPLE_PIPELINE_SUMMARY)
            analyzer.set_context(context)

            assert analyzer.context is not None
        except ImportError:
            pytest.skip("IntelligentAnalyzer not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_analyzer_analyze(self):
        """Test full analysis."""
        try:
            from intelligent_analysis import IntelligentAnalyzer, AnalysisContext

            context = AnalysisContext(summary_data=SAMPLE_PIPELINE_SUMMARY)
            analyzer = IntelligentAnalyzer(context=context)
            results = analyzer.analyze()

            assert isinstance(results, dict)
            assert "health_score" in results
            assert "failure_analysis" in results
            assert "performance_analysis" in results
            assert "patterns" in results
            assert "optimizations" in results
        except ImportError:
            pytest.skip("IntelligentAnalyzer not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_analyzer_no_context_error(self):
        """Test that analyze raises error without context."""
        try:
            from intelligent_analysis import IntelligentAnalyzer

            analyzer = IntelligentAnalyzer()
            with pytest.raises(ValueError):
                analyzer.analyze()
        except ImportError:
            pytest.skip("IntelligentAnalyzer not available")


class TestAnalyzerFunctions:
    """Tests for analyzer module functions."""

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_calculate_health_score(self):
        """Test health score calculation."""
        try:
            from intelligent_analysis import calculate_pipeline_health_score

            # Healthy pipeline should have high score
            healthy_score = calculate_pipeline_health_score(SAMPLE_PIPELINE_SUMMARY)
            assert 50 <= healthy_score <= 100

            # Failed pipeline should have lower score than healthy one
            failed_score = calculate_pipeline_health_score(SAMPLE_FAILED_SUMMARY)
            assert 0 <= failed_score < healthy_score
        except ImportError:
            pytest.skip("calculate_pipeline_health_score not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_classify_failure_severity(self):
        """Test failure severity classification."""
        try:
            from intelligent_analysis import classify_failure_severity

            # Critical failure
            critical_step = {
                "stderr": "MemoryError: Out of memory",
                "exit_code": 137
            }
            assert classify_failure_severity(critical_step) == "critical"

            # Major failure
            major_step = {
                "stderr": "FileNotFoundError: No such file",
                "exit_code": 1
            }
            assert classify_failure_severity(major_step) == "major"

            # Minor failure
            minor_step = {
                "stderr": "Some warning message",
                "exit_code": 1
            }
            assert classify_failure_severity(minor_step) == "minor"
        except ImportError:
            pytest.skip("classify_failure_severity not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_detect_performance_patterns(self):
        """Test performance pattern detection."""
        try:
            from intelligent_analysis import detect_performance_patterns

            patterns = detect_performance_patterns(SAMPLE_PIPELINE_SUMMARY)
            assert isinstance(patterns, list)

            # Check pattern structure if any detected
            for pattern in patterns:
                assert "type" in pattern
                assert "description" in pattern
                assert "severity" in pattern
        except ImportError:
            pytest.skip("detect_performance_patterns not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_generate_optimization_suggestions(self):
        """Test optimization suggestion generation."""
        try:
            from intelligent_analysis import generate_optimization_suggestions

            suggestions = generate_optimization_suggestions(SAMPLE_PIPELINE_SUMMARY)
            assert isinstance(suggestions, list)

            # Check suggestion structure if any generated
            for suggestion in suggestions:
                assert "type" in suggestion
                assert "impact" in suggestion
                assert "description" in suggestion
        except ImportError:
            pytest.skip("generate_optimization_suggestions not available")


class TestAnalysisContext:
    """Tests for AnalysisContext dataclass."""

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_context_creation(self):
        """Test AnalysisContext creation."""
        try:
            from intelligent_analysis import AnalysisContext

            context = AnalysisContext(summary_data=SAMPLE_PIPELINE_SUMMARY)
            assert context.summary_data == SAMPLE_PIPELINE_SUMMARY
            assert context.overall_status == "SUCCESS_WITH_WARNINGS"
        except ImportError:
            pytest.skip("AnalysisContext not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_context_properties(self):
        """Test AnalysisContext properties."""
        try:
            from intelligent_analysis import AnalysisContext

            context = AnalysisContext(summary_data=SAMPLE_PIPELINE_SUMMARY)

            assert context.total_duration == 330.5
            assert len(context.steps) == 4
            assert isinstance(context.performance_summary, dict)
        except ImportError:
            pytest.skip("AnalysisContext not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_context_get_steps_methods(self):
        """Test step retrieval methods."""
        try:
            from intelligent_analysis import AnalysisContext

            context = AnalysisContext(summary_data=SAMPLE_FAILED_SUMMARY)

            failed_steps = context.get_failed_steps()
            assert len(failed_steps) == 2

            successful_steps = context.get_successful_steps()
            assert len(successful_steps) == 1

            warning_steps = context.get_warning_steps()
            assert len(warning_steps) == 0
        except ImportError:
            pytest.skip("AnalysisContext not available")


class TestIntegration:
    """Integration tests for intelligent_analysis module."""

    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_full_analysis_workflow(self, isolated_temp_dir):
        """Test complete analysis workflow."""
        try:
            from intelligent_analysis.processor import (
                analyze_pipeline_summary,
                analyze_individual_steps,
                identify_bottlenecks,
                extract_failure_context,
                generate_recommendations,
                generate_executive_report
            )

            # Run full analysis workflow
            analysis = analyze_pipeline_summary(SAMPLE_FAILED_SUMMARY)
            step_analyses, flags_by_type = analyze_individual_steps(SAMPLE_FAILED_SUMMARY)
            bottlenecks = identify_bottlenecks(SAMPLE_FAILED_SUMMARY)
            failures = extract_failure_context(SAMPLE_FAILED_SUMMARY)
            recommendations = generate_recommendations(analysis, bottlenecks, flags_by_type)
            report = generate_executive_report(
                analysis, bottlenecks, failures, recommendations,
                step_analyses, flags_by_type
            )

            # Save report to temp dir
            report_path = isolated_temp_dir / "analysis_report.md"
            with open(report_path, 'w') as f:
                f.write(report)

            assert report_path.exists()
            assert report_path.stat().st_size > 0

            # Verify report contains key sections
            assert "Red Flags" in report or "Yellow Flags" in report or "Per-Step" in report
        except ImportError:
            pytest.skip("intelligent_analysis module not available")

    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_check_analysis_tools(self):
        """Test analysis tools check."""
        try:
            from intelligent_analysis import check_analysis_tools

            tools = check_analysis_tools()
            assert isinstance(tools, dict)
            # Should have at least numpy check
            if "numpy" in tools:
                assert "available" in tools["numpy"]
        except ImportError:
            pytest.skip("intelligent_analysis module not available")


def test_module_completeness():
    """Test that intelligent_analysis module has all required components."""
    required_components = [
        '__version__',
        'FEATURES',
        'get_module_info',
        'get_supported_analysis_types',
        'validate_pipeline_summary',
        'process_intelligent_analysis',
        'analyze_pipeline_summary',
        'generate_executive_report',
        'IntelligentAnalyzer',
        'AnalysisContext'
    ]

    try:
        import intelligent_analysis
        for component in required_components:
            assert hasattr(intelligent_analysis, component), f"Missing component: {component}"
    except ImportError:
        pytest.skip("intelligent_analysis module not available")


@pytest.mark.slow
def test_module_performance():
    """Test intelligent_analysis module performance characteristics."""
    try:
        from intelligent_analysis.processor import analyze_pipeline_summary
        import time

        start_time = time.time()

        # Run analysis multiple times
        for _ in range(10):
            analyze_pipeline_summary(SAMPLE_PIPELINE_SUMMARY)

        processing_time = time.time() - start_time
        # 10 analyses should complete within 1 second
        assert processing_time < 1.0

    except ImportError:
        pytest.skip("intelligent_analysis module not available")
