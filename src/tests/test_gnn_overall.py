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
    def test_core_processor_imports(self):
        """Test that core processor can be imported."""
        from gnn import core_processor
        assert hasattr(core_processor, 'GNNProcessor')
    
    @pytest.mark.unit
    def test_gnn_processor_basic_imports(self, sample_gnn_files, comprehensive_test_data):
        """Test basic GNN processor imports without heavy instantiation."""
        from gnn.core_processor import GNNProcessor
        
        # Test that GNNProcessor can be imported
        assert hasattr(GNNProcessor, '__init__')


# Migrated from test_gnn_core_modules.py
class TestGNNReporting:
    """Test gnn.reporting module."""
    
    @pytest.mark.unit
    def test_reporting_imports(self):
        """Test that reporting module can be imported."""
        from gnn import reporting
        from gnn import generate_gnn_report
        assert callable(generate_gnn_report)
    
    @pytest.mark.unit
    def test_generate_report(self, comprehensive_test_data, isolated_temp_dir):
        """Test report generation functionality."""
        from gnn import generate_gnn_report
        
        # generate_gnn_report expects a processing_results dict, not a path
        processing_results = {
            'target_directory': str(isolated_temp_dir),
            'files_found': 0,
            'files_processed': 0,
            'success': True,
            'errors': [],
            'parsed_files': []
        }
        
        result = generate_gnn_report(processing_results)
        
        # Verify report generation returns a string
        assert isinstance(result, str)
        assert 'GNN Processing Report' in result


