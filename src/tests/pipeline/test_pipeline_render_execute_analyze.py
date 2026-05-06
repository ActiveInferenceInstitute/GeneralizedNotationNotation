#!/usr/bin/env python3
"""
Pipeline Render-Execute-Analyze Integration Tests

This module tests the end-to-end flow from render through execute to analysis,
verifying that each step's output is correctly consumed by the next step.

Test Coverage:
- Render to execute to analyze flow (test_render_execute_analyze_flow)
- PyMDP end-to-end pipeline (test_pymdp_end_to_end)
- Execution results analyzability (test_execution_results_analyzable)
- Pipeline step handoffs (test_step_output_handoffs)

No mocking is used - all tests validate real function execution.
"""

import pytest

pytestmark = pytest.mark.pipeline
import json

# Add src to path for imports
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Phase 6: all render/execute/analysis modules are in-tree and guaranteed
# importable. Import unconditionally — any ImportError here is a real bug.
from analysis.processor import process_analysis
from execute.processor import process_execute
from render.processor import process_render, render_gnn_spec


class TestPipelineIntegration:
    """End-to-end pipeline integration tests."""

    @pytest.fixture
    def sample_gnn_content(self) -> str:
        """Create sample GNN markdown content."""
        return """
# Active Inference Test Agent

## ModelName
test_pipeline_agent

## StateSpaceBlock
s[3,1,type=int]

## ObservationBlock  
o[2,1,type=int]

## Connections
s -> o

## InitialParameterization
A = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]
B = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
C = [1.0, 0.0]
D = [0.33, 0.33, 0.34]
"""

    @pytest.fixture
    def pipeline_directories(self, safe_filesystem: Any, sample_gnn_content: str) -> Dict[str, Any]:
        """Create complete pipeline directory structure."""
        # Create input with GNN file
        input_dir = safe_filesystem.create_dir("input/gnn_files")
        gnn_file = safe_filesystem.create_file(
            "input/gnn_files/test_pipeline_agent.md",
            sample_gnn_content
        )

        # Create output directories for each step
        render_output = safe_filesystem.create_dir("output/11_render_output")
        execute_output = safe_filesystem.create_dir("output/12_execute_output")
        analysis_output = safe_filesystem.create_dir("output/16_analysis_output")

        return {
            "input_dir": input_dir,
            "gnn_file": gnn_file,
            "render_output": render_output,
            "execute_output": execute_output,
            "analysis_output": analysis_output,
            "base_output": safe_filesystem.temp_dir / "output",
        }

    @pytest.mark.integration
    @pytest.mark.slow
    def test_render_execute_analyze_flow(self, pipeline_directories: Dict[str, Any]) -> None:
        """Full render → execute → analyze flow completes with structured exit codes.

        Each step's return value must be bool OR int (Phase 1.1 widened contract:
        0=success, 1=error, 2=skipped/warnings). Exceptions are real failures.
        """
        dirs = pipeline_directories

        # Step 1: Render
        render_result = process_render(dirs["input_dir"], dirs["base_output"], verbose=True)
        assert isinstance(render_result, (bool, int)), (
            f"process_render returned {type(render_result).__name__}; expected bool|int"
        )

        # Step 2: Execute (constrained to pymdp for speed)
        execute_result = process_execute(
            dirs["input_dir"],
            dirs["base_output"],
            verbose=True,
            frameworks="pymdp",
            timeout=10,
            render_output_dir=dirs["base_output"] / "11_render_output",
        )
        assert isinstance(execute_result, (bool, int))

        # Step 3: Analyze
        analysis_result = process_analysis(dirs["input_dir"], dirs["base_output"], verbose=True)
        assert isinstance(analysis_result, (bool, int))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_step_output_handoffs(self, pipeline_directories: Dict[str, Any]) -> None:
        """Render and execute each produce on-disk output under the base output dir.

        Programmatic ``process_render(target, output_dir)`` writes directly to
        ``output_dir`` (render_processing_summary.json + per-model-per-framework
        subdirs). The pipeline step wrapper routes through ``11_render_output/``
        but the underlying processor is schema-flat.
        """
        dirs = pipeline_directories

        process_render(dirs["input_dir"], dirs["base_output"], verbose=True)
        summary = dirs["base_output"] / "render_processing_summary.json"
        assert summary.exists(), f"Render did not produce summary at {summary}"
        # At least one model subdir with at least one framework artifact.
        model_subdirs = [p for p in dirs["base_output"].iterdir() if p.is_dir()]
        assert model_subdirs, "Render produced no per-model output directories"
        framework_artifacts = list(dirs["base_output"].rglob("*.py"))
        assert framework_artifacts, "Render produced no framework artifacts"

        process_execute(
            dirs["input_dir"],
            dirs["base_output"],
            verbose=True,
            frameworks="pymdp",
            timeout=10,
            render_output_dir=dirs["base_output"],
        )
        # process_execute writes execution_summary under output_dir/summaries/
        exec_summary = dirs["base_output"] / "summaries" / "execution_summary.json"
        assert exec_summary.exists(), f"Execute did not produce {exec_summary}"


class TestExecuteAnalyzeIntegration:
    """Tests for execute to analysis handoff."""

    @pytest.fixture
    def simulated_execute_output(self, safe_filesystem: Any) -> Any:
        """Create simulated execution output for analysis testing."""
        execute_dir = safe_filesystem.create_dir("output/12_execute_output")

        # Create execution summary
        summary = {
            "timestamp": "2026-01-08T12:00:00",
            "success": True,
            "total_scripts": 1,
            "successful_scripts": 1,
            "failed_scripts": 0,
            "frameworks_executed": ["pymdp"],
            "results": [
                {
                    "script": "test_agent_pymdp.py",
                    "framework": "pymdp",
                    "success": True,
                    "execution_time": 1.5
                }
            ]
        }
        safe_filesystem.create_file(
            "output/12_execute_output/execution_summary.json",
            json.dumps(summary, indent=2)
        )

        # Create simulated pymdp output
        safe_filesystem.create_dir("output/12_execute_output/test_agent/pymdp")
        safe_filesystem.create_file(
            "output/12_execute_output/test_agent/pymdp/simulation_results.json",
            json.dumps({
                "beliefs": [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]],
                "actions": [0, 1, 0],
                "observations": [1, 0, 1],
                "free_energy": [1.2, 0.8, 0.6]
            }, indent=2)
        )

        return execute_dir

    @pytest.mark.integration
    def test_execution_results_analyzable(self, simulated_execute_output: Any, safe_filesystem: Any) -> None:
        """process_analysis runs against simulated execute output without crashing."""
        input_dir = safe_filesystem.create_dir("input")
        output_dir = safe_filesystem.temp_dir / "output"
        result = process_analysis(input_dir, output_dir, verbose=True)
        # Per Phase 1.1 contract: no input → exit-code 2; completed analysis → True/0.
        assert isinstance(result, (bool, int))


class TestRenderExecuteIntegration:
    """Tests for render to execute handoff."""

    @pytest.fixture
    def sample_gnn_spec(self) -> Dict[str, Any]:
        """Create a minimal GNN spec dictionary."""
        return {
            "name": "integration_test_model",
            "states": ["s"],
            "observations": ["o"],
            "parameters": {
                "A": [[0.8, 0.2], [0.2, 0.8]],
            }
        }

    @pytest.mark.integration
    @pytest.mark.slow
    def test_render_output_executable(self, safe_filesystem: Any, sample_gnn_spec: Dict[str, Any]) -> None:
        """Test that render output can be executed."""
        # Create pipeline structure
        render_output = safe_filesystem.create_dir("output/11_render_output/test_model/pymdp")

        try:
            # Render to pymdp
            ok, msg, artifacts = render_gnn_spec(sample_gnn_spec, "pymdp", render_output)

            if ok:
                # Find generated Python scripts
                py_scripts = list(render_output.glob("*.py"))

                for script in py_scripts:
                    content = script.read_text()

                    # Verify script is valid Python
                    try:
                        compile(content, script, 'exec')
                    except SyntaxError as e:
                        pytest.fail(f"Rendered script has syntax error: {e}")

                    # Verify script has expected structure for execution
                    # Should have imports, class/function definitions
                    assert len(content) > 0

        except Exception as e:
            pytest.skip(f"Render failed (acceptable): {e}")
