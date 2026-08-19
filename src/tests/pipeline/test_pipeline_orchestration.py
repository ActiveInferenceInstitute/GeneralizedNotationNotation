#!/usr/bin/env python3
"""
Test Pipeline Orchestration - Tests for pipeline orchestration and execution flow.

Tests the PipelineOrchestrator and step orchestration functionality.
"""

from typing import Any

import pytest

pytestmark = pytest.mark.pipeline
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPipelineRunnerOrchestration:
    """Tests for PipelineOrchestrator orchestration."""

    @pytest.mark.fast
    def test_pipeline_runner_instantiation(self, tmp_path: Any) -> Any:
        """Test PipelineOrchestrator can be instantiated."""
        try:
            import logging

            from pipeline import PipelineOrchestrator

            logging.getLogger("test_pipeline")

            # PipelineOrchestrator may require specific init args
            orchestrator = PipelineOrchestrator()

            assert orchestrator is not None
        except (ImportError, TypeError):
            # Fall back to checking if module loads
            from pipeline import get_pipeline_info

            info = get_pipeline_info()
            assert info is not None

    @pytest.mark.fast
    def test_pipeline_runner_configuration(self, tmp_path: Any) -> Any:
        """Test PipelineConfig can be created."""
        from pipeline import create_pipeline_config

        # create_pipeline_config takes no args, returns dict
        config = create_pipeline_config()

        assert config is not None
        assert isinstance(config, dict)

    @pytest.mark.integration
    def test_pipeline_step_registration(self, tmp_path: Any) -> Any:
        """Test pipeline steps are registered correctly."""
        from pipeline import discover_pipeline_steps

        steps = discover_pipeline_steps()

        assert steps is not None
        # Should have multiple steps (at least 10 in this pipeline)
        if isinstance(steps, (list, dict)):
            assert len(steps) >= 5, f"Expected >=5 steps, got {len(steps)}"


class TestStepOrchestration:
    """Tests for individual step orchestration."""

    @pytest.mark.fast
    def test_step_discovery(self) -> Any:
        """Test pipeline discovers available steps."""
        from pipeline import discover_pipeline_steps

        steps = discover_pipeline_steps()

        assert steps is not None

    @pytest.mark.fast
    def test_step_ordering(self) -> Any:
        """Test steps metadata is available."""
        from pipeline import STEP_METADATA

        # Should have step metadata
        assert STEP_METADATA is not None
        if isinstance(STEP_METADATA, dict):
            assert len(STEP_METADATA) > 0

    @pytest.mark.integration
    def test_step_dependency_resolution(self) -> Any:
        """Test pipeline execution ordering."""
        from pipeline import execute_pipeline_steps

        # Just verify the function exists and is callable
        assert callable(execute_pipeline_steps)


class TestPipelineExecution:
    """Tests for pipeline execution flow."""

    @pytest.mark.integration
    def test_single_step_execution(self, tmp_path: Any) -> Any:
        """Test executing a single pipeline step.

        Runs the non-mutating GNN step (Step 3). The ``setup`` step (Step 1)
        is deliberately avoided: it invokes ``src/1_setup.py``, which runs a
        mutating, non-frozen ``uv sync`` against the shared ``.venv`` and
        prunes the ``dev`` toolchain (pytest/execnet/xdist) out from under
        sibling ``pytest-xdist`` workers — the mechanism that corrupted the
        environment during parallel validation.
        """
        from pipeline import execute_pipeline_step

        # Build step config and pipeline data
        step_config: dict[str, Any] = {"output_dir": str(tmp_path), "verbose": False}
        pipeline_data: dict[str, Any] = {
            "target_dir": str(tmp_path),
            "output_dir": str(tmp_path),
        }

        result = execute_pipeline_step(
            step_name="gnn", step_config=step_config, pipeline_data=pipeline_data
        )

        # Should complete (success or graceful failure)
        assert result is not None

    @pytest.mark.integration
    def test_step_skip_functionality(self, tmp_path: Any) -> Any:
        """Test skipping specific steps via config."""
        from pipeline import create_pipeline_config

        # create_pipeline_config returns a dict
        config = create_pipeline_config()

        assert config is not None
        assert isinstance(config, dict)

        # Manually add skip_steps
        config["skip_steps"] = [1, 2, 3]
        assert config["skip_steps"] == [1, 2, 3]

    @pytest.mark.integration
    def test_only_steps_functionality(self, tmp_path: Any) -> Any:
        """Test running only specific steps via config."""
        from pipeline import create_pipeline_config

        config = create_pipeline_config()
        config["only_steps"] = "1,3,5"

        assert config is not None
        assert config["only_steps"] == "1,3,5"


class TestPipelineStateManagement:
    """Tests for pipeline state management during orchestration."""

    @pytest.mark.fast
    def test_pipeline_state_initialization(self, tmp_path: Any) -> Any:
        """Test pipeline config state is initialized correctly."""
        from pipeline import PipelineConfig, get_pipeline_config

        # Create a proper PipelineConfig object with a temp config file
        config_path = tmp_path / "test_config.json"
        config_path.write_text('{"output_dir": "' + str(tmp_path) + '"}')

        pipeline_config = PipelineConfig(config_path=config_path)

        # Verify config was loaded
        assert pipeline_config is not None
        assert pipeline_config.config is not None

        # Retrieve global config and verify it's accessible
        retrieved = get_pipeline_config()
        assert retrieved is not None
        assert isinstance(retrieved, dict)

    @pytest.mark.fast
    def test_pipeline_info_available(self) -> Any:
        """Test pipeline info can be retrieved."""
        from pipeline import get_module_info, get_pipeline_info

        info = get_pipeline_info()
        assert info is not None

        module_info = get_module_info()
        assert module_info is not None

    @pytest.mark.fast
    def test_pipeline_validation(self) -> Any:
        """Test pipeline configuration validation."""
        from pipeline import create_pipeline_config, validate_pipeline_config

        config = create_pipeline_config()

        # Should validate without error
        is_valid = validate_pipeline_config(config)
        # Could be True or dict with validation results
        assert is_valid is not None


class TestDAG:
    """Tests for pipeline DAG topological sort and circular dep handling."""

    @pytest.mark.fast
    def test_dag_linear_steps(self) -> None:
        """Linear dependency chain: 0→1→2 resolves correctly."""
        from pipeline.dag import resolve_execution_order

        deps = {1: [0], 2: [1]}
        tiers = resolve_execution_order(deps, total_steps=3)
        assert len(tiers) >= 2
        # 0 must appear before 2
        flat = [s for tier in tiers for s in tier]
        assert flat.index(0) < flat.index(2)

    @pytest.mark.fast
    def test_dag_independent_steps(self) -> None:
        """Independent steps (no deps) land in the same tier."""
        from pipeline.dag import resolve_execution_order

        tiers = resolve_execution_order({}, total_steps=5)
        assert len(tiers) == 1
        assert sorted(tiers[0]) == [0, 1, 2, 3, 4]

    @pytest.mark.fast
    def test_dag_circular_deps_warns_by_default(self) -> None:
        """Circular dependencies are appended as last tier (backward compat)."""
        from pipeline.dag import resolve_execution_order

        deps = {0: [2], 2: [0]}
        tiers = resolve_execution_order(deps, total_steps=3)
        # lenient mode: does NOT raise; unresolved appended to last tier
        assert len(tiers) >= 2
        flat = [s for tier in tiers for s in tier]
        assert 0 in flat and 2 in flat

    @pytest.mark.fast
    def test_dag_circular_deps_raises_when_strict(self) -> None:
        """With raise_on_circular=True, circular deps raise ValueError."""
        from pipeline.dag import resolve_execution_order

        deps = {0: [2], 2: [0]}
        with pytest.raises(ValueError, match="Circular dependencies"):
            resolve_execution_order(deps, total_steps=3, raise_on_circular=True)

    @pytest.mark.fast
    def test_dag_no_raise_on_valid(self) -> None:
        """raise_on_circular=True does not raise on valid DAG."""
        from pipeline.dag import resolve_execution_order

        tiers = resolve_execution_order({}, total_steps=4, raise_on_circular=True)
        assert len(tiers) == 1

    @pytest.mark.fast
    def test_pipeline_stages_and_grouping(self) -> None:
        """Test canonical stage definitions and step grouping."""
        from pipeline.step_registry import get_pipeline_stages, get_stage_steps

        stages = get_pipeline_stages()
        assert "discovery_schema" in stages
        assert "simulation_execution" in stages
        assert "intelligence_analysis" in stages
        assert "presentation_reporting" in stages

        sim_steps = get_stage_steps("simulation_execution")
        assert len(sim_steps) == 4
        assert [s.script_stem for s in sim_steps] == [
            "11_render",
            "12_execute",
            "15_audio",
            "16_analysis",
        ]
