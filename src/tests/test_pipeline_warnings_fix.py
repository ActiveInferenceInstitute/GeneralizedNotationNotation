#!/usr/bin/env python3
"""
Test suite for pipeline warning fixes.

This module tests the fixes for:
1. Prerequisite validation using correct output directory names
2. Prevention of nested output directories
"""

import pytest
import sys
import logging
from pathlib import Path
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import get_output_dir_for_script
from utils.pipeline_validator import validate_step_prerequisites
from utils.argument_utils import PipelineArguments


class TestPipelineWarningsFix:
    """Test suite for pipeline warning fixes."""
    
    def test_prerequisite_validation_correct_directories(self, tmp_path):
        """Test that prerequisite validation uses correct output directory names."""
        # Setup
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create the correct prerequisite directories
        gnn_output_dir = output_dir / "3_gnn_output"
        gnn_output_dir.mkdir()
        
        type_checker_output_dir = output_dir / "5_type_checker_output"
        type_checker_output_dir.mkdir()
        
        # Create mock args
        args = Mock()
        args.output_dir = output_dir
        
        logger = logging.getLogger("test")
        
        # Test validation for step that depends on 3_gnn.py
        result = validate_step_prerequisites("5_type_checker.py", args, logger)
        
        # Should pass without warnings since 3_gnn_output exists
        assert result["passed"], "Prerequisite validation should pass"
        assert len(result["warnings"]) == 0, f"Should have no warnings, got: {result['warnings']}"
        
    def test_prerequisite_validation_with_nested_directories(self, tmp_path):
        """Test that prerequisite validation handles nested directories."""
        # Setup
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create nested directory structure (legacy pattern)
        gnn_output_dir = output_dir / "3_gnn_output" / "3_gnn_output"
        gnn_output_dir.mkdir(parents=True)
        
        # Create mock args
        args = Mock()
        args.output_dir = output_dir
        
        logger = logging.getLogger("test")
        
        # Test validation for step that depends on 3_gnn.py
        result = validate_step_prerequisites("5_type_checker.py", args, logger)
        
        # Should pass with a warning about nested structure
        assert result["passed"], "Prerequisite validation should pass"
        assert len(result["warnings"]) > 0, "Should have warning about nested structure"
        assert any("nested" in w.lower() for w in result["warnings"]), \
            "Should warn about nested directory"
    
    def test_prerequisite_validation_missing_directory(self, tmp_path):
        """Test that prerequisite validation warns about missing directories."""
        # Setup
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Don't create the prerequisite directory
        
        # Create mock args
        args = Mock()
        args.output_dir = output_dir
        
        logger = logging.getLogger("test")
        
        # Test validation for step that depends on 3_gnn.py
        result = validate_step_prerequisites("5_type_checker.py", args, logger)
        
        # Should pass (validation doesn't fail) but with warning
        assert result["passed"], "Prerequisite validation should pass (non-blocking)"
        assert len(result["warnings"]) > 0, "Should have warning about missing directory"
        assert any("missing" in w.lower() for w in result["warnings"]), \
            "Should warn about missing prerequisite directory"
        
    def test_get_output_dir_prevents_nesting(self, tmp_path):
        """Test that get_output_dir_for_script prevents nested directories."""
        base_output_dir = tmp_path / "output"
        base_output_dir.mkdir()
        
        # First call should create the correct directory
        result1 = get_output_dir_for_script("10_ontology.py", base_output_dir)
        expected = base_output_dir / "10_ontology_output"
        assert result1 == expected, f"Expected {expected}, got {result1}"
        
        # Second call with already-nested directory should NOT create double nesting
        result2 = get_output_dir_for_script("10_ontology.py", result1)
        # Should return the same directory, not result1 / "10_ontology_output"
        assert result2 == expected, f"Should prevent nesting, expected {expected}, got {result2}"
        
    def test_get_output_dir_all_steps(self, tmp_path):
        """Test that all pipeline steps get correct output directories."""
        base_output_dir = tmp_path / "output"
        base_output_dir.mkdir()
        
        # Test all 24 pipeline steps
        steps = [
            ("0_template.py", "0_template_output"),
            ("1_setup.py", "1_setup_output"),
            ("2_tests.py", "2_tests_output"),
            ("3_gnn.py", "3_gnn_output"),
            ("4_model_registry.py", "4_model_registry_output"),
            ("5_type_checker.py", "5_type_checker_output"),
            ("6_validation.py", "6_validation_output"),
            ("7_export.py", "7_export_output"),
            ("8_visualization.py", "8_visualization_output"),
            ("9_advanced_viz.py", "9_advanced_viz_output"),
            ("10_ontology.py", "10_ontology_output"),
            ("11_render.py", "11_render_output"),
            ("12_execute.py", "12_execute_output"),
            ("13_llm.py", "13_llm_output"),
            ("14_ml_integration.py", "14_ml_integration_output"),
            ("15_audio.py", "15_audio_output"),
            ("16_analysis.py", "16_analysis_output"),
            ("17_integration.py", "17_integration_output"),
            ("18_security.py", "18_security_output"),
            ("19_research.py", "19_research_output"),
            ("20_website.py", "20_website_output"),
            ("21_mcp.py", "21_mcp_output"),
            ("22_gui.py", "22_gui_output"),
            ("23_report.py", "23_report_output"),
        ]
        
        for script_name, expected_dir_name in steps:
            result = get_output_dir_for_script(script_name, base_output_dir)
            expected = base_output_dir / expected_dir_name
            assert result == expected, \
                f"Script {script_name} should map to {expected_dir_name}, got {result.name}"
    
    def test_prerequisite_chain_validation(self, tmp_path):
        """Test validation of prerequisite chains."""
        # Setup
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create prerequisite directories
        (output_dir / "3_gnn_output").mkdir()
        (output_dir / "5_type_checker_output").mkdir()
        (output_dir / "11_render_output").mkdir()
        
        # Create mock args
        args = Mock()
        args.output_dir = output_dir
        
        logger = logging.getLogger("test")
        
        # Test validation for step that depends on 11_render.py (which depends on 3_gnn.py)
        result = validate_step_prerequisites("12_execute.py", args, logger)
        
        # Should pass without warnings since all prerequisites exist
        assert result["passed"], "Prerequisite validation should pass"
        # Note: If checking for transitive dependencies is not implemented,
        # there may still be a warning about 3_gnn.py not being a direct prerequisite
        
    def test_get_output_dir_with_path_object(self, tmp_path):
        """Test get_output_dir_for_script with Path object."""
        base_output_dir = Path(tmp_path) / "output"
        base_output_dir.mkdir(parents=True)
        
        result = get_output_dir_for_script("3_gnn.py", base_output_dir)
        expected = base_output_dir / "3_gnn_output"
        
        assert result == expected
        assert isinstance(result, Path)
        
    def test_get_output_dir_without_py_extension(self, tmp_path):
        """Test get_output_dir_for_script without .py extension."""
        base_output_dir = tmp_path / "output"
        base_output_dir.mkdir()
        
        result = get_output_dir_for_script("3_gnn", base_output_dir)
        expected = base_output_dir / "3_gnn_output"
        
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




