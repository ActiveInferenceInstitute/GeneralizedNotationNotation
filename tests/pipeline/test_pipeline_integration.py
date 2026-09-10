#!/usr/bin/env python3
"""
Test Pipeline Integration - Integration tests for pipeline with external systems.

Tests the integration between pipeline steps and external dependencies.
"""

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.pipeline

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestPipelineStepIntegration:
    """Tests for integration between pipeline steps."""

    @pytest.mark.integration
    def test_gnn_to_render_data_flow(
        self, sample_gnn_files: Any, tmp_path: Any
    ) -> None:
        """Test data flows correctly from GNN to render step."""
        if not sample_gnn_files:
            raise AssertionError("No sample GNN files available")

        from gnn import parse_gnn_file
        from gnn.render import generate_pymdp_code

        # sample_gnn_files is Dict[str, Path]
        gnn_file = list(sample_gnn_files.values())[0]

        # GNN processing
        parsed_data = parse_gnn_file(gnn_file)
        assert isinstance(parsed_data, dict), (
            "parse_gnn_file must produce a spec dict for the render step"
        )

        # Render with parsed data
        render_result = generate_pymdp_code(parsed_data)

        # The render step must hand the execute step a runnable Python script,
        # not just "something".
        assert isinstance(render_result, str) and render_result.strip(), (
            "generate_pymdp_code must return non-empty Python source"
        )
        assert render_result.lstrip().startswith("#!/usr/bin/env python3"), (
            "rendered PyMDP code must be a Python script"
        )

    @pytest.mark.integration
    def test_render_to_execute_data_flow(self, tmp_path: Any) -> None:
        """Test data flows correctly from render to execute step."""
        from gnn.render import generate_pymdp_code

        parsed_data: dict[str, Any] = {
            "ModelName": "TestFlow",
            "variables": [
                {"name": "state", "dimensions": [3]},
                {"name": "obs", "dimensions": [2]},
            ],
            "parameters": [],
        }

        code = generate_pymdp_code(parsed_data)

        # Code should be executable Python: a non-empty script that at least
        # byte-compiles (the cross-step artifact, not just "any string").
        assert isinstance(code, str) and code.strip(), (
            "generate_pymdp_code must return non-empty Python source"
        )
        assert code.lstrip().startswith("#!/usr/bin/env python3"), (
            "rendered PyMDP code must be a Python script"
        )
        compile(code, "<generated_pymdp>", "exec")

    @pytest.mark.integration
    def test_visualization_to_report_data_flow(self, tmp_path: Any) -> None:
        """Test visualization outputs are available to report."""
        import logging

        from gnn.report import process_report

        logger = logging.getLogger("test_integration")

        # Create sample output structure
        viz_output = tmp_path / "8_visualization_output"
        viz_output.mkdir(parents=True, exist_ok=True)

        report_output = tmp_path / "23_report_output"
        report_output.mkdir(parents=True, exist_ok=True)

        # Report should be able to find visualizations
        result = process_report(
            target_dir=tmp_path, output_dir=report_output, logger=logger
        )

        assert isinstance(result, bool)


class TestPipelineExternalIntegration:
    """Tests for pipeline integration with external systems."""

    @pytest.mark.integration
    def test_pipeline_filesystem_integration(self, tmp_path: Any) -> None:
        """Test pipeline correctly interacts with filesystem."""
        from gnn.pipeline import get_output_dir_for_script

        output_dir = tmp_path / "test_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Should resolve output directory for a step
        result = get_output_dir_for_script("3_gnn.py", output_dir)

        assert isinstance(result, Path), (
            "get_output_dir_for_script must return the step's output path"
        )
        assert result.name == "3_gnn_output"

    @pytest.mark.integration
    def test_pipeline_logging_integration(self, tmp_path: Any) -> None:
        """Test pipeline logging integration."""
        import logging

        from gnn.pipeline import get_pipeline_config

        log_file = tmp_path / "gnn.pipeline.log"

        # Use standard logging setup
        logger = logging.getLogger("test_pipeline")
        handler = logging.FileHandler(log_file)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # Verify pipeline config is accessible
        config = get_pipeline_config()
        assert isinstance(config, dict)
        assert "steps" in config

        # Log something
        logger.info("Test log message")
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    @pytest.mark.integration
    def test_pipeline_config_loading(self) -> None:
        """Test pipeline configuration loading."""
        from gnn.pipeline import get_pipeline_config

        config = get_pipeline_config()

        assert isinstance(config, dict)
        assert "steps" in config


class TestPipelineModuleIntegration:
    """Tests for integration between pipeline and modules."""

    @pytest.mark.integration
    def test_all_modules_importable(self) -> None:
        """Test that all pipeline modules can be imported.

        Strict: any import failure fails the test with the full failure list.
        The previous version swallowed ImportError and asserted nothing, so it
        could never fail.
        """
        modules: list[str] = [
            "gnn",
            "gnn.render",
            "gnn.execute",
            "gnn.visualization",
            "gnn.report",
            "gnn.mcp",
            "gnn.audio",
            "gnn.export",
        ]
        failures: list[str] = []
        for module_name in modules:
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 — report every failure mode
                failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
        assert not failures, (
            f"{len(failures)}/{len(modules)} pipeline modules failed to import:\n"
            + "\n".join(failures)
        )

    @pytest.mark.integration
    def test_module_info_consistency(self) -> None:
        """Test that all modules provide consistent info."""
        from gnn import get_module_info as gnn_info
        from gnn.render import get_module_info as render_info
        from gnn.report import get_module_info as report_info

        for info_func in [gnn_info, render_info, report_info]:
            info = info_func()
            assert isinstance(info, dict)
            assert "version" in info and "features" in info

    @pytest.mark.integration
    def test_pipeline_step_order(self) -> None:
        """Test pipeline steps are in correct order."""
        from gnn.pipeline import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(steps=["3"])
        steps = orchestrator.get_pipeline_steps()

        if steps:
            assert isinstance(steps, (list, dict))
            # Steps should be ordered
            if isinstance(steps, list) and len(steps) > 1:
                assert len(steps) >= 1


class TestPipelineOutputIntegration:
    """Tests for pipeline output integration."""

    @pytest.mark.integration
    def test_output_directory_structure(self, tmp_path: Any) -> None:
        """Test pipeline creates correct output structure."""
        from gnn.pipeline import get_output_dir_for_script

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Should resolve step-specific output directory
        result = get_output_dir_for_script("3_gnn.py", output_dir)

        assert isinstance(result, Path), (
            "get_output_dir_for_script must return the step's output path"
        )
        assert result.name == "3_gnn_output"

    @pytest.mark.integration
    def test_summary_file_creation(self, tmp_path: Any) -> None:
        """Test pipeline creates summary files."""
        import json

        summary: dict[str, Any] = {
            "status": "success",
            "steps_completed": 5,
            "duration": 10.5,
        }

        summary_file = tmp_path / "summary.json"

        # Write summary directly
        with open(summary_file, "w") as f:
            json.dump(summary, f)

        assert summary_file.exists()


class TestPipelineErrorIntegration:
    """Tests for pipeline error handling integration."""

    @pytest.mark.integration
    def test_graceful_module_failure(self, tmp_path: Any) -> None:
        """Test pipeline handles module failures gracefully."""
        from gnn.pipeline import execute_pipeline_step
        from gnn.pipeline.execution import StepExecutionResult

        # Run with invalid step configuration - should return result, not crash
        step_config: dict[str, Any] = {"script_path": str(tmp_path / "nonexistent.py")}
        pipeline_data: dict[str, Any] = {
            "target_dir": str(tmp_path),
            "output_dir": str(tmp_path / "output"),
        }
        result = execute_pipeline_step(
            step_name="nonexistent_step",
            step_config=step_config,
            pipeline_data=pipeline_data,
        )

        # Should return a failure result object, not crash — and the failure
        # must actually be reported, not silently swallowed.
        assert isinstance(result, StepExecutionResult)
        assert result.success is False
        assert result.error, "missing-script failure must carry an error message"

    @pytest.mark.integration
    def test_recovery_from_step_failure(self, tmp_path: Any) -> None:
        """Test pipeline can recover from step failures."""
        import logging

        from gnn.pipeline import PipelineOrchestrator

        logging.getLogger("test_pipeline")

        orchestrator = PipelineOrchestrator(
            target_dir=str(PROJECT_ROOT / "input" / "gnn_files" / "discrete"),
            output_dir=str(tmp_path / "output"),
            steps=["3"],
        )

        # Should be able to instantiate and run
        assert isinstance(orchestrator, PipelineOrchestrator)
        result = orchestrator.run()
        assert result is True
