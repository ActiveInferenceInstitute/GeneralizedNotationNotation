#!/usr/bin/env python3
"""
Test LLM Overall Tests

This file contains comprehensive tests for the LLM module functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


class TestLLMModuleComprehensive:
    """Comprehensive tests for the LLM module."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_llm_module_imports(self):
        """Test that LLM module can be imported."""
        try:
            import llm
            assert hasattr(llm, '__version__')
            assert hasattr(llm, 'LLMProcessor')
            assert hasattr(llm, 'LLMAnalyzer')
            assert hasattr(llm, 'get_module_info')
        except ImportError:
            pytest.skip("LLM module not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_llm_processor_instantiation(self):
        """Test LLMProcessor class instantiation."""
        try:
            from llm import LLMProcessor
            processor = LLMProcessor()
            assert processor is not None
            assert hasattr(processor, 'analyze_model')
            assert hasattr(processor, 'generate_description')
        except ImportError:
            pytest.skip("LLMProcessor not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_llm_analyzer_instantiation(self):
        """Test LLMAnalyzer class instantiation."""
        try:
            from llm import LLMAnalyzer
            analyzer = LLMAnalyzer()
            assert analyzer is not None
            assert hasattr(analyzer, 'analyze_content')
            assert hasattr(analyzer, 'extract_insights')
        except ImportError:
            pytest.skip("LLMAnalyzer not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_llm_module_info(self):
        """Test LLM module information retrieval."""
        try:
            from llm import get_module_info
            info = get_module_info()
            assert isinstance(info, dict)
            assert 'version' in info
            assert 'description' in info
            assert 'providers' in info
        except ImportError:
            pytest.skip("LLM module info not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_llm_providers(self):
        """Test LLM providers retrieval."""
        try:
            from llm import get_available_providers
            providers = get_available_providers()
            assert isinstance(providers, list)
            assert len(providers) > 0
        except ImportError:
            pytest.skip("LLM providers not available")


class TestLLMFunctionality:
    """Tests for LLM functionality."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_model_analysis(self, comprehensive_test_data):
        """Test model analysis functionality."""
        try:
            from llm import LLMProcessor
            processor = LLMProcessor()
            
            # Test model analysis with sample data
            model_data = comprehensive_test_data.get('model_data', {})
            result = processor.analyze_model(model_data)
            assert result is not None
        except ImportError:
            pytest.skip("LLMProcessor not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_description_generation(self):
        """Test description generation functionality."""
        try:
            from llm import LLMProcessor
            processor = LLMProcessor()
            
            # Test description generation
            content = "Test model content"
            result = processor.generate_description(content)
            assert result is not None
            assert isinstance(result, str)
        except ImportError:
            pytest.skip("LLMProcessor not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_content_analysis(self):
        """Test content analysis functionality."""
        try:
            from llm import LLMAnalyzer
            analyzer = LLMAnalyzer()
            
            # Test content analysis
            content = "Test content for analysis"
            result = analyzer.analyze_content(content)
            assert result is not None
        except ImportError:
            pytest.skip("LLMAnalyzer not available")


class TestLLMIntegration:
    """Integration tests for LLM module."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_llm_pipeline_integration(self, sample_gnn_files, isolated_temp_dir):
        """Test LLM module integration with pipeline."""
        try:
            from llm import LLMProcessor
            processor = LLMProcessor()
            
            # Test end-to-end LLM analysis
            gnn_file = list(sample_gnn_files.values())[0]
            with open(gnn_file, 'r') as f:
                gnn_content = f.read()
            
            result = processor.analyze_model({'content': gnn_content})
            assert result is not None
            
        except ImportError:
            pytest.skip("LLM module not available")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_llm_mcp_integration(self):
        """Test LLM MCP integration."""
        try:
            from llm.mcp import register_tools
            # Test that MCP tools can be registered
            assert callable(register_tools)
        except ImportError:
            pytest.skip("LLM MCP not available")


def test_llm_module_completeness():
    """Test that LLM module has all required components."""
    required_components = [
        'LLMProcessor',
        'LLMAnalyzer',
        'get_module_info',
        'get_available_providers'
    ]
    
    try:
        import llm
        for component in required_components:
            assert hasattr(llm, component), f"Missing component: {component}"
    except ImportError:
        pytest.skip("LLM module not available")


@pytest.mark.slow
def test_llm_module_performance():
    """Test LLM module performance characteristics."""
    try:
        from llm import LLMProcessor
        import time
        
        processor = LLMProcessor()
        start_time = time.time()
        
        # Test analysis performance
        result = processor.analyze_model({'test': 'data'})
        
        processing_time = time.time() - start_time
        assert processing_time < 30.0  # Should complete within 30 seconds
        
    except ImportError:
        pytest.skip("LLM module not available") 