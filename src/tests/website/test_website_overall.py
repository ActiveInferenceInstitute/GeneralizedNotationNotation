"""
Test Website Overall Tests

This file contains comprehensive tests for the website module functionality.
"""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestWebsiteModuleComprehensive:
    """Comprehensive tests for the website module."""

    @pytest.mark.unit
    def test_website_module_imports(self):
        """Test that website module can be imported."""
        import website
        assert hasattr(website, '__version__')
        assert hasattr(website, 'WebsiteGenerator')
        assert hasattr(website, 'WebsiteRenderer')
        assert hasattr(website, 'get_module_info')

    @pytest.mark.unit
    def test_website_generator_instantiation(self):
        """Test WebsiteGenerator class instantiation."""
        from website import WebsiteGenerator
        generator = WebsiteGenerator()
        assert generator is not None
        assert hasattr(generator, 'generate_website')
        assert hasattr(generator, 'create_pages')

    @pytest.mark.unit
    def test_website_renderer_instantiation(self):
        """Test WebsiteRenderer class instantiation."""
        from website import WebsiteRenderer
        renderer = WebsiteRenderer()
        assert renderer is not None
        assert hasattr(renderer, 'render_html')
        assert hasattr(renderer, 'render_css')

    @pytest.mark.unit
    def test_website_module_info(self):
        """Test website module information retrieval."""
        from website import get_module_info
        info = get_module_info()
        assert isinstance(info, dict)
        assert 'version' in info
        assert 'description' in info
        assert 'supported_file_types' in info

    @pytest.mark.unit
    def test_supported_file_types(self):
        """Test supported file types retrieval."""
        from website import get_supported_file_types
        file_types = get_supported_file_types()
        assert isinstance(file_types, list)
        assert len(file_types) > 0
        expected_types = ['html', 'css', 'js', 'json']
        for file_type in expected_types:
            assert file_type in file_types

class TestWebsiteFunctionality:
    """Tests for website functionality."""

    @pytest.mark.unit
    def test_website_generation(self, comprehensive_test_data):
        """Test website generation functionality."""
        from website import WebsiteGenerator
        generator = WebsiteGenerator()
        website_data = comprehensive_test_data.get('website_data', {})
        result = generator.generate_website(website_data)
        assert result is not None

    @pytest.mark.unit
    def test_html_rendering(self):
        """Test HTML rendering functionality."""
        from website import WebsiteRenderer
        renderer = WebsiteRenderer()
        content = 'Test website content'
        result = renderer.render_html(content)
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_website_validation(self):
        """Test website validation functionality."""
        from website import validate_website_config
        result = validate_website_config('test config')
        assert isinstance(result, bool)

class TestWebsiteIntegration:
    """Integration tests for website module."""

    @pytest.mark.integration
    def test_website_pipeline_integration(self, sample_gnn_files, isolated_temp_dir):
        """Test website module integration with pipeline."""
        from website import WebsiteGenerator
        generator = WebsiteGenerator()
        gnn_file = list(sample_gnn_files.values())[0]
        with open(gnn_file, 'r') as f:
            gnn_content = f.read()
        result = generator.generate_website({'content': gnn_content})
        assert result is not None

    @pytest.mark.integration
    def test_website_mcp_integration(self):
        """Test website MCP integration."""
        from website.mcp import register_tools
        assert callable(register_tools)

def test_website_module_completeness():
    """Test that website module has all required components."""
    required_components = ['WebsiteGenerator', 'WebsiteRenderer', 'get_module_info', 'get_supported_file_types', 'validate_website_config']
    try:
        import website
        for component in required_components:
            assert hasattr(website, component), f'Missing component: {component}'
    except ImportError:
        pytest.skip('Website module not available')

@pytest.mark.slow
def test_website_module_performance():
    """Test website module performance characteristics."""
    import time
    from website import WebsiteGenerator
    generator = WebsiteGenerator()
    start_time = time.time()
    generator.generate_website({'test': 'data'})
    processing_time = time.time() - start_time
    assert processing_time < 10.0

class TestWebsiteMCP:
    """Smoke tests for website.mcp sub-module."""

    def test_module_importable(self):
        from website import mcp

    def test_get_website_module_info_mcp(self):
        from website.mcp import get_website_module_info_mcp
        result = get_website_module_info_mcp()
        assert isinstance(result, dict)

    def test_get_website_status_mcp_nonexistent(self, tmp_path):
        from website.mcp import get_website_status_mcp
        result = get_website_status_mcp(str(tmp_path / 'nonexistent'))
        assert isinstance(result, dict)

    def test_list_generated_pages_mcp_empty(self, tmp_path):
        from website.mcp import list_generated_pages_mcp
        result = list_generated_pages_mcp(str(tmp_path))
        assert isinstance(result, dict)