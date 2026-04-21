"""
Test Pipeline Overall Tests

This file contains comprehensive tests for the pipeline module functionality.
"""
import pytest
pytestmark = pytest.mark.pipeline
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
pipeline = pytest.importorskip('pipeline')

class TestPipelineModuleComprehensive:
    """Comprehensive tests for the pipeline module."""

    @pytest.mark.unit
    def test_pipeline_module_imports(self):
        """Test that pipeline module can be imported."""
        import pipeline
        assert hasattr(pipeline, '__version__')
        assert hasattr(pipeline, 'PipelineOrchestrator')
        assert hasattr(pipeline, 'PipelineStep')
        assert hasattr(pipeline, 'get_pipeline_config')

    @pytest.mark.unit
    def test_pipeline_orchestrator_instantiation(self):
        """Test PipelineOrchestrator class instantiation."""
        from pipeline import PipelineOrchestrator
        orchestrator = PipelineOrchestrator()
        assert orchestrator is not None
        assert hasattr(orchestrator, 'execute_pipeline')
        assert hasattr(orchestrator, 'get_pipeline_steps')

    @pytest.mark.unit
    def test_pipeline_step_instantiation(self):
        """Test PipelineStep class instantiation."""
        from pipeline import PipelineStep
        step = PipelineStep('test_step')
        assert step is not None
        assert hasattr(step, 'execute')
        assert hasattr(step, 'validate')

    @pytest.mark.unit
    def test_pipeline_module_info(self):
        """Test pipeline module information retrieval."""
        from pipeline import get_module_info
        info = get_module_info()
        assert isinstance(info, dict)
        assert 'version' in info
        assert 'description' in info
        assert 'pipeline_steps' in info

    @pytest.mark.unit
    def test_pipeline_config(self):
        """Test pipeline configuration retrieval."""
        from pipeline import get_pipeline_config
        config = get_pipeline_config()
        assert isinstance(config, dict)
        assert 'steps' in config
        assert 'timeout' in config
        assert 'parallel' in config

class TestPipelineFunctionality:
    """Tests for pipeline functionality."""

    @pytest.mark.unit
    def test_pipeline_execution(self, comprehensive_test_data):
        """Test pipeline execution functionality."""
        from pipeline import PipelineOrchestrator
        orchestrator = PipelineOrchestrator()
        pipeline_data = comprehensive_test_data.get('pipeline_data', {})
        result = orchestrator.execute_pipeline(pipeline_data)
        assert result is not None

    @pytest.mark.unit
    def test_pipeline_step_validation(self):
        """Test pipeline step validation."""
        from pipeline import validate_pipeline_step
        result = validate_pipeline_step('test_step')
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_pipeline_discovery(self):
        """Test pipeline step discovery."""
        from pipeline import discover_pipeline_steps
        steps = discover_pipeline_steps()
        assert isinstance(steps, list)
        assert len(steps) > 0

class TestPipelineIntegration:
    """Integration tests for pipeline module."""

    @pytest.mark.integration
    def test_pipeline_module_integration(self, sample_gnn_files, isolated_temp_dir):
        """Test pipeline module integration with other modules."""
        from pipeline import PipelineOrchestrator
        orchestrator = PipelineOrchestrator()
        result = orchestrator.get_pipeline_steps()
        assert result is not None
        assert isinstance(result, list)

    @pytest.mark.integration
    def test_pipeline_mcp_integration(self):
        """Test pipeline MCP integration."""
        from pipeline.mcp import register_tools
        assert callable(register_tools)

def test_pipeline_module_completeness():
    """Test that pipeline module has all required components."""
    required_components = ['PipelineOrchestrator', 'PipelineStep', 'get_module_info', 'get_pipeline_config', 'validate_pipeline_step', 'discover_pipeline_steps']
    try:
        import pipeline
        for component in required_components:
            assert hasattr(pipeline, component), f'Missing component: {component}'
    except ImportError:
        pytest.skip('Pipeline module not available')

@pytest.mark.slow
def test_pipeline_module_performance():
    """Test pipeline module performance characteristics."""
    import time
    from pipeline import PipelineOrchestrator
    orchestrator = PipelineOrchestrator()
    start_time = time.time()
    orchestrator.get_pipeline_steps()
    processing_time = time.time() - start_time
    assert processing_time < 5.0