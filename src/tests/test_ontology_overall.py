#!/usr/bin/env python3
"""
Test Ontology Overall Tests

This file contains comprehensive tests for the ontology module functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


class TestOntologyModuleComprehensive:
    """Comprehensive tests for the ontology module."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_ontology_module_imports(self):
        """Test that ontology module can be imported."""
        try:
            import ontology
            assert hasattr(ontology, '__version__')
            assert hasattr(ontology, 'OntologyProcessor')
            assert hasattr(ontology, 'OntologyValidator')
            assert hasattr(ontology, 'get_module_info')
        except ImportError:
            pytest.skip("Ontology module not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_ontology_processor_instantiation(self):
        """Test OntologyProcessor class instantiation."""
        try:
            from ontology import OntologyProcessor
            processor = OntologyProcessor()
            assert processor is not None
            assert hasattr(processor, 'process_ontology')
            assert hasattr(processor, 'validate_terms')
        except ImportError:
            pytest.skip("OntologyProcessor not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_ontology_validator_instantiation(self):
        """Test OntologyValidator class instantiation."""
        try:
            from ontology import OntologyValidator
            validator = OntologyValidator()
            assert validator is not None
            assert hasattr(validator, 'validate_ontology')
            assert hasattr(validator, 'check_consistency')
        except ImportError:
            pytest.skip("OntologyValidator not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_ontology_module_info(self):
        """Test ontology module information retrieval."""
        try:
            from ontology import get_module_info
            info = get_module_info()
            assert isinstance(info, dict)
            assert 'version' in info
            assert 'description' in info
            assert 'ontology_types' in info
        except ImportError:
            pytest.skip("Ontology module info not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_ontology_processing_options(self):
        """Test ontology processing options retrieval."""
        try:
            from ontology import get_ontology_processing_options
            options = get_ontology_processing_options()
            assert isinstance(options, dict)
            assert 'validation_levels' in options
            assert 'output_formats' in options
        except ImportError:
            pytest.skip("Ontology processing options not available")


class TestOntologyFunctionality:
    """Tests for ontology functionality."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_ontology_processing(self, comprehensive_test_data):
        """Test ontology processing functionality."""
        try:
            from ontology import OntologyProcessor
            processor = OntologyProcessor()
            
            # Test ontology processing with sample data
            ontology_data = comprehensive_test_data.get('ontology_data', {})
            result = processor.process_ontology(ontology_data)
            assert result is not None
        except ImportError:
            pytest.skip("OntologyProcessor not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_ontology_validation(self):
        """Test ontology validation functionality."""
        try:
            from ontology import OntologyValidator
            validator = OntologyValidator()
            
            # Test ontology validation
            ontology_content = "test ontology content"
            result = validator.validate_ontology(ontology_content)
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("OntologyValidator not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_term_validation(self):
        """Test term validation functionality."""
        try:
            from ontology import validate_ontology_terms
            result = validate_ontology_terms("test term")
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Ontology term validation not available")


class TestOntologyIntegration:
    """Integration tests for ontology module."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_ontology_pipeline_integration(self, sample_gnn_files, isolated_temp_dir):
        """Test ontology module integration with pipeline."""
        try:
            from ontology import OntologyProcessor
            processor = OntologyProcessor()
            
            # Test end-to-end ontology processing
            gnn_file = list(sample_gnn_files.values())[0]
            with open(gnn_file, 'r') as f:
                gnn_content = f.read()
            
            result = processor.process_ontology({'content': gnn_content})
            assert result is not None
            
        except ImportError:
            pytest.skip("Ontology module not available")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_ontology_mcp_integration(self):
        """Test ontology MCP integration."""
        try:
            from ontology.mcp import register_tools
            # Test that MCP tools can be registered
            assert callable(register_tools)
        except ImportError:
            pytest.skip("Ontology MCP not available")


def test_ontology_module_completeness():
    """Test that ontology module has all required components."""
    required_components = [
        'OntologyProcessor',
        'OntologyValidator',
        'get_module_info',
        'get_ontology_processing_options',
        'validate_ontology_terms'
    ]
    
    try:
        import ontology
        for component in required_components:
            assert hasattr(ontology, component), f"Missing component: {component}"
    except ImportError:
        pytest.skip("Ontology module not available")


@pytest.mark.slow
def test_ontology_module_performance():
    """Test ontology module performance characteristics."""
    try:
        from ontology import OntologyProcessor
        import time
        
        processor = OntologyProcessor()
        start_time = time.time()
        
        # Test processing performance
        result = processor.process_ontology({'test': 'data'})
        
        processing_time = time.time() - start_time
        assert processing_time < 10.0  # Should complete within 10 seconds
        
    except ImportError:
        pytest.skip("Ontology module not available") 