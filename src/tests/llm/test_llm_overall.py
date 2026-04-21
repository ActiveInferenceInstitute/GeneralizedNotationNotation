"""
Test LLM Overall Tests

This file contains comprehensive tests for the LLM module functionality.
"""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestLLMModuleComprehensive:
    """Comprehensive tests for the LLM module."""

    @pytest.mark.unit
    def test_llm_module_imports(self):
        """Test that LLM module can be imported."""
        import llm
        assert hasattr(llm, '__version__')
        assert hasattr(llm, 'LLMProcessor')
        assert hasattr(llm, 'LLMAnalyzer')
        assert hasattr(llm, 'get_module_info')

    @pytest.mark.unit
    def test_llm_processor_instantiation(self):
        """Test LLMProcessor class instantiation."""
        from llm import LLMProcessor
        processor = LLMProcessor()
        assert processor is not None
        assert hasattr(processor, 'analyze_model')
        assert hasattr(processor, 'generate_description')

    @pytest.mark.unit
    def test_llm_analyzer_instantiation(self):
        """Test LLMAnalyzer class instantiation."""
        from llm import LLMAnalyzer
        analyzer = LLMAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze_content')
        assert hasattr(analyzer, 'extract_insights')

    @pytest.mark.unit
    def test_llm_module_info(self):
        """Test LLM module information retrieval."""
        from llm import get_module_info
        info = get_module_info()
        assert isinstance(info, dict)
        assert 'version' in info
        assert 'description' in info
        assert 'providers' in info

    @pytest.mark.unit
    def test_llm_providers(self):
        """Test LLM providers retrieval."""
        from llm import get_available_providers
        providers = get_available_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0

class TestLLMFunctionality:
    """Tests for LLM functionality."""

    @pytest.mark.unit
    def test_model_analysis(self, comprehensive_test_data):
        """Test model analysis functionality."""
        from llm import LLMProcessor
        processor = LLMProcessor()
        model_data = comprehensive_test_data.get('model_data', {})
        result = processor.analyze_model(model_data)
        assert result is not None

    @pytest.mark.unit
    def test_description_generation(self):
        """Test description generation functionality."""
        from llm import LLMProcessor
        processor = LLMProcessor()
        content = 'Test model content'
        result = processor.generate_description(content)
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_content_analysis(self):
        """Test content analysis functionality."""
        from llm import LLMAnalyzer
        analyzer = LLMAnalyzer()
        content = 'Test content for analysis'
        result = analyzer.analyze_content(content)
        assert result is not None

class TestLLMIntegration:
    """Integration tests for LLM module."""

    @pytest.mark.integration
    def test_llm_pipeline_integration(self, sample_gnn_files, isolated_temp_dir):
        """Test LLM module integration with pipeline."""
        from llm import LLMProcessor
        processor = LLMProcessor()
        gnn_file = list(sample_gnn_files.values())[0]
        with open(gnn_file, 'r') as f:
            gnn_content = f.read()
        result = processor.analyze_model({'content': gnn_content})
        assert result is not None

    @pytest.mark.integration
    def test_llm_mcp_integration(self):
        """Test LLM MCP integration."""
        from llm.mcp import register_tools
        assert callable(register_tools)

def test_llm_module_completeness():
    """Test that LLM module has all required components."""
    required_components = ['LLMProcessor', 'LLMAnalyzer', 'get_module_info', 'get_available_providers']
    try:
        import llm
        for component in required_components:
            assert hasattr(llm, component), f'Missing component: {component}'
    except ImportError:
        pytest.skip('LLM module not available')

@pytest.mark.slow
def test_llm_module_performance():
    """Test LLM module performance characteristics."""
    import time
    from llm import LLMProcessor
    processor = LLMProcessor()
    start_time = time.time()
    processor.analyze_model({'test': 'data'})
    processing_time = time.time() - start_time
    assert processing_time < 30.0