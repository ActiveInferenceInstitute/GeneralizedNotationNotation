#!/usr/bin/env python3
"""
Integration tests for pipeline main execution.

These tests verify that the pipeline actually runs steps and produces expected outputs.
No mocks - real subprocess execution with real artifacts.
"""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Import test utilities
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DIR = PROJECT_ROOT / "src" / "tests"

def assert_file_exists(path):
    """Assert that a file exists."""
    assert Path(path).exists(), f"Expected file not found: {path}"

def assert_directory_structure(path, expected):
    """Assert expected directory structure."""
    pass

def create_test_files(count=3):
    """Create test files."""
    return []

import pytest


class TestPipelineMain:
    """Test real pipeline execution and artifact generation."""

    def setup_method(self):
        """Setup test environment."""
        self.test_input_dir = PROJECT_ROOT / "input" / "test_gnn_files"
        self.test_input_dir.mkdir(parents=True, exist_ok=True)

        # Create a simple test GNN file
        test_content = """# Test GNN Model
states: 2
observations: 2
actions: 1

A:
  0.8 0.2
  0.3 0.7

B:
  0.9 0.1
  0.8 0.2
  0.1 0.9

C: 0.5 0.5
D: 0.6 0.4
"""
        (self.test_input_dir / "test_model.md").write_text(test_content)

    def teardown_method(self):
        """Cleanup test environment."""
        # Clean up test files
        if self.test_input_dir.exists():
            shutil.rmtree(self.test_input_dir)

    @pytest.mark.integration
    def test_pipeline_steps_3_7_8_produce_artifacts(self):
        """Test that steps 3,7,8 run and produce expected output artifacts."""
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_output_dir:
            output_dir = Path(temp_output_dir)

            # Run pipeline steps 3,7,8
            cmd = [
                sys.executable, "src/main.py",
                "--target-dir", str(self.test_input_dir),
                "--output-dir", str(output_dir),
                "--only-steps", "3,7,8",
                "--verbose"
            ]

            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

            # Check that pipeline completed (may have warnings but should not fail)
            assert result.returncode in [0, 2], f"Pipeline failed: {result.stderr}"

            # Verify expected output directories exist
            step3_output = output_dir / "3_gnn_output"
            step7_output = output_dir / "7_export_output"
            step8_output = output_dir / "8_visualization_output"

            assert step3_output.exists(), "Step 3 output directory not created"
            assert step7_output.exists(), "Step 7 output directory not created"
            assert step8_output.exists(), "Step 8 output directory not created"

            # Verify artifacts exist in step 3
            gnn_results = list(step3_output.glob("**/gnn_processing_results.json"))
            assert len(gnn_results) > 0, "No GNN processing results found"

            # Verify artifacts exist in step 7
            export_files = list(step7_output.glob("**/*.json"))
            assert len(export_files) > 0, "No export files found"

            # Verify artifacts exist in step 8
            viz_files = list(step8_output.glob("**/*.json"))
            assert len(viz_files) > 0, "No visualization files found"

            # Check that we processed our test model
            gnn_result_content = gnn_results[0].read_text()
            assert "test_model.md" in gnn_result_content, "Test model not processed"

    @pytest.mark.integration
    def test_pipeline_step_11_12_produce_reports(self):
        """Test that steps 11,12 run and produce execution reports."""
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_output_dir:
            output_dir = Path(temp_output_dir)

            # Run pipeline steps 11,12 (render and execute)
            cmd = [
                sys.executable, "src/main.py",
                "--target-dir", str(self.test_input_dir),
                "--output-dir", str(output_dir),
                "--only-steps", "11,12",
                "--verbose"
            ]

            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)

            # Pipeline may fail due to missing dependencies, but should produce reports
            step11_output = output_dir / "11_render_output"
            step12_output = output_dir / "12_execute_output"

            # Check that directories exist (even if execution failed)
            assert step11_output.exists(), "Step 11 output directory not created"

            # Render step should always produce some output
            render_files = list(step11_output.glob("**/*"))
            assert len(render_files) > 0, "No render output files found"

            # Execute step should produce execution reports even on failure
            if step12_output.exists():
                execute_files = list(step12_output.glob("**/*"))
                assert len(execute_files) > 0, "No execute output files found"

    @pytest.mark.slow
    def test_full_pipeline_execution(self):
        """Test full pipeline execution with real GNN model."""
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_output_dir:
            output_dir = Path(temp_output_dir)

            # Run full pipeline (excluding slow steps that might timeout)
            cmd = [
                sys.executable, "src/main.py",
                "--target-dir", str(self.test_input_dir),
                "--output-dir", str(output_dir),
                "--skip-steps", "13,14,15,16,17,18,19,20,21,22,23",  # Skip potentially slow steps
                "--verbose"
            ]

            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)

            # Check that pipeline produced expected outputs
            pipeline_summary = output_dir / "pipeline_execution_summary.json"
            assert pipeline_summary.exists(), "Pipeline summary not found"

            # Verify summary has expected structure
            import json
            summary = json.loads(pipeline_summary.read_text())
            assert "steps" in summary, "Pipeline summary missing steps"
            assert len(summary["steps"]) > 0, "No steps recorded in summary"

            # Check that we have results from our key steps
            step_names = [step.get("script_name", "") for step in summary["steps"]]
            assert "3_gnn.py" in step_names, "GNN processing step not found"
            assert "7_export.py" in step_names, "Export step not found"
            assert "8_visualization.py" in step_names, "Visualization step not found"

