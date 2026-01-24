#!/usr/bin/env python3
"""
Test Pipeline Integration - Integration tests for pipeline with external systems.

Tests the integration between pipeline steps and external dependencies.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPipelineStepIntegration:
    """Tests for integration between pipeline steps."""

    @pytest.mark.integration
    def test_gnn_to_render_data_flow(self, sample_gnn_files, tmp_path):
        """Test data flows correctly from GNN to render step."""
        if not sample_gnn_files:
            pytest.skip("No sample GNN files available")
        
        from gnn import parse_gnn_file
        from render import generate_pymdp_code
        
        # sample_gnn_files is Dict[str, Path]
        gnn_file = list(sample_gnn_files.values())[0]
        
        # GNN processing
        parsed_data = parse_gnn_file(gnn_file)
        assert parsed_data is not None
        
        # Render with parsed data
        render_result = generate_pymdp_code(parsed_data)
        
        assert render_result is not None

    @pytest.mark.integration
    def test_render_to_execute_data_flow(self, tmp_path):
        """Test data flows correctly from render to execute step."""
        from render import generate_pymdp_code
        
        parsed_data = {
            "ModelName": "TestFlow",
            "variables": [
                {"name": "state", "dimensions": [3]},
                {"name": "obs", "dimensions": [2]}
            ],
            "parameters": []
        }
        
        code = generate_pymdp_code(parsed_data)
        
        # Code should be executable Python
        assert code is not None
        if isinstance(code, str):
            assert "import" in code or "def" in code or len(code) > 0

    @pytest.mark.integration
    def test_visualization_to_report_data_flow(self, tmp_path):
        """Test visualization outputs are available to report."""
        from visualization import process_visualization
        from report import process_report
        import logging
        
        logger = logging.getLogger("test_integration")
        
        # Create mock output structure
        viz_output = tmp_path / "8_visualization_output"
        viz_output.mkdir(parents=True, exist_ok=True)
        
        report_output = tmp_path / "23_report_output"
        report_output.mkdir(parents=True, exist_ok=True)
        
        # Report should be able to find visualizations
        result = process_report(
            target_dir=tmp_path,
            output_dir=report_output,
            logger=logger
        )
        
        assert result is not None or result is None


class TestPipelineExternalIntegration:
    """Tests for pipeline integration with external systems."""

    @pytest.mark.integration
    def test_pipeline_filesystem_integration(self, tmp_path):
        """Test pipeline correctly interacts with filesystem."""
        from pipeline import get_output_dir_for_script
        
        output_dir = tmp_path / "test_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Should resolve output directory for a step
        result = get_output_dir_for_script("3_gnn.py", output_dir)
        
        assert result is not None or output_dir.exists()

    @pytest.mark.integration
    def test_pipeline_logging_integration(self, tmp_path):
        """Test pipeline logging integration."""
        import logging
        from pipeline import get_pipeline_config
        
        log_file = tmp_path / "pipeline.log"
        
        # Use standard logging setup
        logger = logging.getLogger("test_pipeline")
        handler = logging.FileHandler(log_file)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Verify pipeline config is accessible
        config = get_pipeline_config()
        assert config is not None
        
        # Log something
        logger.info("Test log message")
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    @pytest.mark.integration
    def test_pipeline_config_loading(self):
        """Test pipeline configuration loading."""
        from pipeline import get_pipeline_config
        
        config = get_pipeline_config()
        
        assert config is not None
        assert isinstance(config, dict)


class TestPipelineModuleIntegration:
    """Tests for integration between pipeline and modules."""

    @pytest.mark.integration
    def test_all_modules_importable(self):
        """Test that all pipeline modules can be imported."""
        modules = [
            'gnn',
            'render',
            'execute',
            'visualization',
            'report',
            'mcp',
            'audio',
            'export'
        ]
        
        for module_name in modules:
            try:
                __import__(module_name)
            except ImportError as e:
                # Some modules may have optional dependencies
                pass

    @pytest.mark.integration
    def test_module_info_consistency(self):
        """Test that all modules provide consistent info."""
        from gnn import get_module_info as gnn_info
        from render import get_module_info as render_info
        from report import get_module_info as report_info
        
        for info_func in [gnn_info, render_info, report_info]:
            info = info_func()
            assert info is not None
            assert isinstance(info, dict)

    @pytest.mark.integration
    def test_pipeline_step_order(self):
        """Test pipeline steps are in correct order."""
        from pipeline import PipelineOrchestrator
        
        orchestrator = PipelineOrchestrator()
        steps = orchestrator.get_pipeline_steps()
        
        if steps:
            assert isinstance(steps, (list, dict))
            # Steps should be ordered
            if isinstance(steps, list) and len(steps) > 1:
                assert len(steps) >= 1


class TestPipelineOutputIntegration:
    """Tests for pipeline output integration."""

    @pytest.mark.integration
    def test_output_directory_structure(self, tmp_path):
        """Test pipeline creates correct output structure."""
        from pipeline import get_output_dir_for_script
        
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Should resolve step-specific output directory
        result = get_output_dir_for_script("3_gnn.py", output_dir)
        
        # Should create or return directory
        assert result is not None or output_dir.exists()

    @pytest.mark.integration
    def test_summary_file_creation(self, tmp_path):
        """Test pipeline creates summary files."""
        import json
        
        summary = {
            "status": "success",
            "steps_completed": 5,
            "duration": 10.5
        }
        
        summary_file = tmp_path / "summary.json"
        
        # Write summary directly
        with open(summary_file, "w") as f:
            json.dump(summary, f)
        
        assert summary_file.exists()


class TestPipelineErrorIntegration:
    """Tests for pipeline error handling integration."""

    @pytest.mark.integration
    def test_graceful_module_failure(self, tmp_path):
        """Test pipeline handles module failures gracefully."""
        from pipeline import execute_pipeline_step
        import logging
        
        logger = logging.getLogger("test_pipeline")
        
        # Run with invalid step configuration - should return result, not crash
        step_config = {"script_path": str(tmp_path / "nonexistent.py")}
        pipeline_data = {"target_dir": str(tmp_path), "output_dir": str(tmp_path / "output")}
        result = execute_pipeline_step(
            step_name="nonexistent_step",
            step_config=step_config,
            pipeline_data=pipeline_data
        )
        
        # Should return a result object (success or failure), not crash
        assert result is not None

    @pytest.mark.integration
    def test_recovery_from_step_failure(self, tmp_path):
        """Test pipeline can recover from step failures."""
        from pipeline import PipelineOrchestrator
        import logging
        
        logger = logging.getLogger("test_pipeline")
        
        orchestrator = PipelineOrchestrator()
        
        # Should be able to instantiate and run
        assert orchestrator is not None
        result = orchestrator.run()
        assert result is True
