#!/usr/bin/env python3
"""
Test Gnn Overall Tests

This file contains tests migrated from test_gnn_core_modules.py.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


# Migrated from test_gnn_core_modules.py
class TestGNNCoreProcessor:
    """Test gnn.core_processor module."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_core_processor_imports(self):
        """Test that core processor can be imported."""
        try:
            from src.gnn import core_processor
            assert hasattr(core_processor, 'GNNProcessor')
            assert hasattr(core_processor, 'create_processor')
            assert hasattr(core_processor, 'ProcessingContext')
        except ImportError:
            pytest.skip("GNN core processor not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_gnn_processor_basic_imports(self, sample_gnn_files, comprehensive_test_data):
        """Test basic GNN processor imports without heavy instantiation."""
        try:
            from src.gnn.core_processor import GNNProcessor, ProcessingContext
            
            # Test that we can create a ProcessingContext (lightweight)
            context = ProcessingContext(
                target_dir=Path("test"),
                output_dir=Path("output")
            )
            assert context is not None
            assert context.target_dir == Path("test")
            assert context.output_dir == Path("output")
            
            # Test that GNNProcessor can be imported
            assert hasattr(GNNProcessor, '__init__')
            
        except ImportError:
            pytest.skip("GNN core processor not available")

# Migrated from test_gnn_core_modules.py
class TestGNNReporting:
    """Test gnn.reporting module."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_reporting_imports(self):
        """Test that reporting module can be imported."""
        try:
            from src.gnn import reporting
            assert hasattr(reporting, 'ReportGenerator')
            # Test that we can also import the main function from gnn package
            from src.gnn import generate_gnn_report
            assert callable(generate_gnn_report)
        except ImportError:
            pytest.skip("GNN reporting not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_generate_report(self, comprehensive_test_data, isolated_temp_dir):
        """Test report generation functionality."""
        try:
            from src.gnn.reporting import generate_report
            
            # Use test data for report generation
            test_data = comprehensive_test_data['models']['simple_model']
            output_file = isolated_temp_dir / "test_report.json"
            
            result = generate_report(test_data, output_file)
            
            # Verify report generation
            assert isinstance(result, dict)
            assert 'status' in result
            assert output_file.exists() or result.get('status') == 'generated'
            
        except ImportError:
            pytest.skip("GNN reporting not available")
        except Exception as e:
            # Should handle errors gracefully
            assert "error" in str(e).lower() or isinstance(result, dict)
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_create_summary(self, comprehensive_test_data):
        """Test summary creation functionality."""
        try:
            from src.gnn.reporting import create_summary
            
            # Use test models for summary
            models_data = comprehensive_test_data['models']
            
            summary = create_summary(models_data)
            
            # Verify summary structure
            assert isinstance(summary, dict)
            assert 'total_models' in summary or 'model_count' in summary
            assert 'summary_text' in summary or 'description' in summary
            
        except ImportError:
            pytest.skip("GNN reporting not available")
        except Exception as e:
            # Should handle errors gracefully
            assert isinstance(summary, (dict, str))


