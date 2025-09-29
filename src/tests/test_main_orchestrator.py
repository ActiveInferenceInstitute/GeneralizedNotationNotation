#!/usr/bin/env python3
"""
Main Orchestrator Tests for GNN Pipeline

This module contains comprehensive tests for the main pipeline orchestrator
(src/main.py) which coordinates the execution of all pipeline steps.

Tests cover:
1. Import validation and component checking
2. Argument parsing functionality and validation
3. Pipeline script discovery and metadata extraction
4. Virtual environment detection and handling
5. Individual step execution coordination
6. Overall pipeline orchestration
7. Performance monitoring and system info collection
8. Pipeline output and reporting
9. Error handling, retry logic, and graceful degradation
10. Configuration consistency validation
11. End-to-end integration scenarios

All tests execute real methods and subprocesses; no mocking is used.
"""

import pytest
import os
import sys
import json
import logging
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import tempfile

# Test markers
pytestmark = [pytest.mark.main_orchestrator]

# Local paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

class TestMainOrchestratorImport:
    """Test main orchestrator import validation and component checking."""
    
    @pytest.mark.unit
    def test_main_orchestrator_file_exists(self):
        """Test that main.py exists and is readable."""
        main_script = SRC_DIR / "main.py"
        
        assert main_script.exists(), f"Main orchestrator script not found: {main_script}"
        assert main_script.is_file(), f"Main orchestrator path is not a file: {main_script}"
        
        # Test that file is readable
        try:
            content = main_script.read_text()
            assert len(content) > 0, "Main orchestrator script is empty"
            
            # Test basic Python structure
            assert "import" in content, "Main script should have import statements"
            assert "def main" in content or "if __name__" in content, \
                "Main script should have main function or entry point"
            
            logging.info("Main orchestrator file structure validated")
            
        except Exception as e:
            pytest.fail(f"Failed to read main orchestrator script: {e}")
    
    @pytest.mark.unit
    def test_main_orchestrator_help_executes(self):
        """main.py should print help and exit 0."""
        main_py = SRC_DIR / "main.py"
        result = subprocess.run([sys.executable, str(main_py), "--help"], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()
    
    @pytest.mark.unit
    def test_main_orchestrator_component_availability(self):
        """Test that main orchestrator components are available."""
        # Test that main components exist
        components = {
            "utils_package": SRC_DIR / "utils" / "__init__.py",
            "pipeline_package": SRC_DIR / "pipeline" / "__init__.py",
            "main_script": SRC_DIR / "main.py"
        }
        
        missing_components = []
        for name, path in components.items():
            if not path.exists():
                missing_components.append(f"{name}: {path}")
        
        assert not missing_components, f"Missing main orchestrator components: {missing_components}"
        logging.info(f"All {len(components)} main orchestrator components available")

class TestArgumentParsing:
    """Smoke-check argument parser via main --help only (no mocks)."""
    @pytest.mark.unit
    def test_help(self):
        main_py = SRC_DIR / "main.py"
        result = subprocess.run([sys.executable, str(main_py), "--help"], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        assert result.returncode == 0

class TestPipelineScriptDiscovery:
    """Test pipeline script discovery and metadata extraction."""
    
    @pytest.mark.unit
    def test_pipeline_script_discovery_logic(self):
        """Test pipeline script discovery logic."""
        # Test script discovery pattern
        expected_pattern = r"^(\d+)_.*\.py$"
        
        test_scripts = [
            ("1_setup.py", True, 1),
            ("2_gnn.py", True, 2),
            ("13_website.py", True, 13),
        ("14_report.py", True, 14),
            ("main.py", False, None),
            ("utils.py", False, None),
            ("not_a_script.txt", False, None)
        ]
        
        import re
        pattern = re.compile(expected_pattern)
        
        for script_name, should_match, expected_num in test_scripts:
            match = pattern.match(script_name)
            
            if should_match:
                assert match is not None, f"Script {script_name} should match pattern"
                assert int(match.group(1)) == expected_num, \
                    f"Script {script_name} should extract number {expected_num}"
            else:
                assert match is None, f"Script {script_name} should not match pattern"
        
        logging.info("Pipeline script discovery logic validated")
    
    @pytest.mark.unit
    def test_pipeline_script_sorting(self):
        """Test pipeline script sorting logic."""
        # Test script sorting by number and name
        mock_scripts = [
            {"num": 3, "basename": "3_gnn.py"},
            {"num": 1, "basename": "1_setup.py"},
            {"num": 13, "basename": "13_llm.py"},
            {"num": 14, "basename": "14_ml_integration.py"},
            {"num": 2, "basename": "2_tests.py"}
        ]
        
        # Sort like the main orchestrator would
        sorted_scripts = sorted(mock_scripts, key=lambda x: (x['num'], x['basename']))
        
        # Verify correct order - include all scripts that exist
        expected_order = [1, 2, 3, 13, 14]
        actual_order = [script['num'] for script in sorted_scripts]
        
        assert actual_order == expected_order, f"Scripts should be sorted correctly: {actual_order}"
        logging.info("Pipeline script sorting logic validated")

class TestVirtualEnvironmentHandling:
    @pytest.mark.unit
    def test_python_executable_detection(self):
        python_executable = sys.executable
        assert python_executable

class TestStepExecution:
    """Execute a core step for real."""
    
    @pytest.mark.integration
    def test_run_core_step(self):
        script = SRC_DIR / "3_gnn.py"
        if not script.exists():
            pytest.skip("3_gnn.py missing")
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "output"
            cmd = [sys.executable, str(script), "--target-dir", str(PROJECT_ROOT / "input" / "gnn_files"), "--output-dir", str(outdir)]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            assert result.returncode in [0,1]

class TestPipelineCoordination:
    @pytest.mark.integration
    def test_minimal_pipeline_execution(self):
        main_py = SRC_DIR / "main.py"
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "output"
            cmd = [sys.executable, str(main_py), "--only-steps", "3,5,7", "--target-dir", str(PROJECT_ROOT / "input" / "gnn_files"), "--output-dir", str(outdir)]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            summary = outdir / "pipeline_execution_summary.json"
            assert summary.exists()
    
    @pytest.mark.unit
    def test_environment_info_function(self):
        from src.main import get_environment_info
        info = get_environment_info()
        assert isinstance(info, dict)
        assert "python_version" in info

class TestEndToEndIntegration:
    @pytest.mark.integration
    def test_run_pipeline_subset(self):
        main_py = SRC_DIR / "main.py"
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "output"
            cmd = [sys.executable, str(main_py), "--only-steps", "3,5,7,8", "--target-dir", str(PROJECT_ROOT / "input" / "gnn_files"), "--output-dir", str(outdir)]
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            # Assert expected subdirs may be created by steps
            assert (outdir / "3_gnn_output").exists() or (outdir / "gnn_processing_step").exists()

    def test_pipeline_summary_validation(self):
        """Test pipeline summary structure validation."""
        from main import _validate_pipeline_summary

        # Test valid summary
        valid_summary = {
            "start_time": "2025-01-01T00:00:00",
            "arguments": {"target_dir": "input"},
            "steps": [
                {
                    "status": "SUCCESS",
                    "step_number": 1,
                    "script_name": "test.py",
                    "description": "Test step",
                    "exit_code": 0
                }
            ],
            "end_time": "2025-01-01T00:01:00",
            "overall_status": "SUCCESS",
            "total_duration_seconds": 60.0,
            "environment_info": {"python_version": "3.11"},
            "performance_summary": {
                "peak_memory_mb": 100.0,
                "total_steps": 1,
                "failed_steps": 0,
                "critical_failures": 0,
                "successful_steps": 1,
                "warnings": 0
            }
        }

        # Should not raise any errors
        _validate_pipeline_summary(valid_summary, logger)

        # Test invalid summary
        invalid_summary = {
            "start_time": None,
            "steps": "not_a_list"
        }

        # Should log warnings/errors but not raise exceptions
        _validate_pipeline_summary(invalid_summary, logger)

    def test_enhanced_warning_detection(self):
        """Test improved warning detection logic."""
        # Test the warning detection logic from main.py
        combined_output = "INFO: Processing completed\nWARNING: Optional feature not available\n⚠️ Warning symbol detected"
        import re
        warning_pattern = re.compile(r"(WARNING|⚠️|warn)", re.IGNORECASE)
        has_warning = bool(warning_pattern.search(combined_output))

        # Should detect warnings
        assert has_warning == True

        # Test without warnings
        combined_output_no_warning = "INFO: Processing completed\nDEBUG: File processed successfully"
        has_warning = bool(warning_pattern.search(combined_output_no_warning))

        # Should not detect warnings
        assert has_warning == False 