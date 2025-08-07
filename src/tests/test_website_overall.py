#!/usr/bin/env python3
"""
Test Website Overall Tests

This file contains comprehensive tests for the website module functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


class TestWebsiteModuleComprehensive:
    """Comprehensive tests for the website module."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_website_module_imports(self):
        """Test that website module can be imported."""
        try:
            import website
            assert hasattr(website, '__version__')
            assert hasattr(website, 'WebsiteGenerator')
            assert hasattr(website, 'WebsiteRenderer')
            assert hasattr(website, 'get_module_info')
        except ImportError:
            pytest.skip("Website module not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_website_generator_instantiation(self):
        """Test WebsiteGenerator class instantiation."""
        try:
            from website import WebsiteGenerator
            generator = WebsiteGenerator()
            assert generator is not None
            assert hasattr(generator, 'generate_website')
            assert hasattr(generator, 'create_pages')
        except ImportError:
            pytest.skip("WebsiteGenerator not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_website_renderer_instantiation(self):
        """Test WebsiteRenderer class instantiation."""
        try:
            from website import WebsiteRenderer
            renderer = WebsiteRenderer()
            assert renderer is not None
            assert hasattr(renderer, 'render_html')
            assert hasattr(renderer, 'render_css')
        except ImportError:
            pytest.skip("WebsiteRenderer not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_website_module_info(self):
        """Test website module information retrieval."""
        try:
            from website import get_module_info
            info = get_module_info()
            assert isinstance(info, dict)
            assert 'version' in info
            assert 'description' in info
            assert 'supported_file_types' in info
        except ImportError:
            pytest.skip("Website module info not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_supported_file_types(self):
        """Test supported file types retrieval."""
        try:
            from website import get_supported_file_types
            file_types = get_supported_file_types()
            assert isinstance(file_types, list)
            assert len(file_types) > 0
            # Check for common file types
            expected_types = ['html', 'css', 'js', 'json']
            for file_type in expected_types:
                assert file_type in file_types
        except ImportError:
            pytest.skip("Website file types not available")


class TestWebsiteFunctionality:
    """Tests for website functionality."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_website_generation(self, comprehensive_test_data):
        """Test website generation functionality."""
        try:
            from website import WebsiteGenerator
            generator = WebsiteGenerator()
            
            # Test website generation with sample data
            website_data = comprehensive_test_data.get('website_data', {})
            result = generator.generate_website(website_data)
            assert result is not None
        except ImportError:
            pytest.skip("WebsiteGenerator not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_html_rendering(self):
        """Test HTML rendering functionality."""
        try:
            from website import WebsiteRenderer
            renderer = WebsiteRenderer()
            
            # Test HTML rendering
            content = "Test website content"
            result = renderer.render_html(content)
            assert result is not None
            assert isinstance(result, str)
        except ImportError:
            pytest.skip("WebsiteRenderer not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_website_validation(self):
        """Test website validation functionality."""
        try:
            from website import validate_website_config
            result = validate_website_config("test config")
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Website validation not available")


class TestWebsiteIntegration:
    """Integration tests for website module."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_website_pipeline_integration(self, sample_gnn_files, isolated_temp_dir):
        """Test website module integration with pipeline."""
        try:
            from website import WebsiteGenerator
            generator = WebsiteGenerator()
            
            # Test end-to-end website generation
            gnn_file = list(sample_gnn_files.values())[0]
            with open(gnn_file, 'r') as f:
                gnn_content = f.read()
            
            result = generator.generate_website({'content': gnn_content})
            assert result is not None
            
        except ImportError:
            pytest.skip("Website module not available")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_website_mcp_integration(self):
        """Test website MCP integration."""
        try:
            from website.mcp import register_tools
            # Test that MCP tools can be registered
            assert callable(register_tools)
        except ImportError:
            pytest.skip("Website MCP not available")


def test_website_module_completeness():
    """Test that website module has all required components."""
    required_components = [
        'WebsiteGenerator',
        'WebsiteRenderer',
        'get_module_info',
        'get_supported_file_types',
        'validate_website_config'
    ]
    
    try:
        import website
        for component in required_components:
            assert hasattr(website, component), f"Missing component: {component}"
    except ImportError:
        pytest.skip("Website module not available")


@pytest.mark.slow
def test_website_module_performance():
    """Test website module performance characteristics."""
    try:
        from website import WebsiteGenerator
        import time
        
        generator = WebsiteGenerator()
        start_time = time.time()
        
        # Test generation performance
        result = generator.generate_website({'test': 'data'})
        
        processing_time = time.time() - start_time
        assert processing_time < 10.0  # Should complete within 10 seconds
        
    except ImportError:
        pytest.skip("Website module not available") 