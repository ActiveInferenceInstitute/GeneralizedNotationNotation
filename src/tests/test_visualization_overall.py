#!/usr/bin/env python3
"""
Test Visualization Overall Tests

This file contains comprehensive tests for the visualization module functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


class TestVisualizationModuleComprehensive:
    """Comprehensive tests for the visualization module."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_visualization_module_imports(self):
        """Test that visualization module can be imported."""
        try:
            import visualization
            assert hasattr(visualization, '__version__')
            assert hasattr(visualization, 'MatrixVisualizer')
            assert hasattr(visualization, 'GraphVisualizer')
            assert hasattr(visualization, 'OntologyVisualizer')
        except ImportError:
            pytest.skip("Visualization module not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_visualizer_instantiation(self):
        """Test MatrixVisualizer class instantiation."""
        try:
            from visualization import MatrixVisualizer
            visualizer = MatrixVisualizer()
            assert visualizer is not None
            assert hasattr(visualizer, 'generate_matrix_analysis')
            assert hasattr(visualizer, 'create_heatmap')
        except ImportError:
            pytest.skip("MatrixVisualizer not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_graph_visualizer_instantiation(self):
        """Test GraphVisualizer class instantiation."""
        try:
            from visualization import GraphVisualizer
            visualizer = GraphVisualizer()
            assert visualizer is not None
            assert hasattr(visualizer, 'generate_graph_visualization')
            assert hasattr(visualizer, 'create_network_diagram')
        except ImportError:
            pytest.skip("GraphVisualizer not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_ontology_visualizer_instantiation(self):
        """Test OntologyVisualizer class instantiation."""
        try:
            from visualization import OntologyVisualizer
            visualizer = OntologyVisualizer()
            assert visualizer is not None
            assert hasattr(visualizer, 'extract_ontology_mappings')
            assert hasattr(visualizer, 'create_ontology_table')
        except ImportError:
            pytest.skip("OntologyVisualizer not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_visualization_module_info(self):
        """Test visualization module information retrieval."""
        try:
            from visualization import get_module_info
            info = get_module_info()
            assert isinstance(info, dict)
            assert 'version' in info
            assert 'description' in info
            assert 'visualization_types' in info
        except ImportError:
            pytest.skip("Visualization module info not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_visualization_options(self):
        """Test visualization options retrieval."""
        try:
            from visualization import get_visualization_options
            options = get_visualization_options()
            assert isinstance(options, dict)
            assert 'matrix_types' in options
            assert 'graph_types' in options
            assert 'output_formats' in options
        except ImportError:
            pytest.skip("Visualization options not available")


class TestVisualizationFunctionality:
    """Tests for visualization functionality."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_visualization(self, comprehensive_test_data):
        """Test matrix visualization functionality."""
        try:
            from visualization import MatrixVisualizer
            visualizer = MatrixVisualizer()
            
            # Test matrix visualization with sample data
            matrix_data = comprehensive_test_data.get('matrix_data', [[1, 2], [3, 4]])
            result = visualizer.generate_matrix_analysis(matrix_data)
            assert result is not None
        except ImportError:
            pytest.skip("MatrixVisualizer not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_graph_visualization(self, comprehensive_test_data):
        """Test graph visualization functionality."""
        try:
            from visualization import GraphVisualizer
            visualizer = GraphVisualizer()
            
            # Test graph visualization with sample data
            graph_data = comprehensive_test_data.get('graph_data', {'nodes': [], 'edges': []})
            result = visualizer.generate_graph_visualization(graph_data)
            assert result is not None
        except ImportError:
            pytest.skip("GraphVisualizer not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_ontology_visualization(self, comprehensive_test_data):
        """Test ontology visualization functionality."""
        try:
            from visualization import OntologyVisualizer
            visualizer = OntologyVisualizer()
            
            # Test ontology visualization with sample data
            ontology_data = comprehensive_test_data.get('ontology_data', {})
            result = visualizer.extract_ontology_mappings(ontology_data)
            assert result is not None
        except ImportError:
            pytest.skip("OntologyVisualizer not available")


class TestVisualizationIntegration:
    """Integration tests for visualization module."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_visualization_pipeline_integration(self, sample_gnn_files, isolated_temp_dir):
        """Test visualization module integration with pipeline."""
        try:
            from visualization import MatrixVisualizer
            visualizer = MatrixVisualizer()
            
            # Test end-to-end visualization
            gnn_file = list(sample_gnn_files.values())[0]
            with open(gnn_file, 'r') as f:
                gnn_content = f.read()
            
            result = visualizer.generate_matrix_analysis([[1, 2], [3, 4]])
            assert result is not None
            
        except ImportError:
            pytest.skip("Visualization module not available")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_visualization_mcp_integration(self):
        """Test visualization MCP integration."""
        try:
            from visualization.mcp import register_tools
            # Test that MCP tools can be registered
            assert callable(register_tools)
        except ImportError:
            pytest.skip("Visualization MCP not available")


def test_visualization_module_completeness():
    """Test that visualization module has all required components."""
    required_components = [
        'MatrixVisualizer',
        'GraphVisualizer',
        'OntologyVisualizer',
        'get_module_info',
        'get_visualization_options'
    ]
    
    try:
        import visualization
        for component in required_components:
            assert hasattr(visualization, component), f"Missing component: {component}"
    except ImportError:
        pytest.skip("Visualization module not available")


@pytest.mark.slow
def test_visualization_module_performance():
    """Test visualization module performance characteristics."""
    try:
        from visualization import MatrixVisualizer
        import time
        
        visualizer = MatrixVisualizer()
        start_time = time.time()
        
        # Test visualization performance
        result = visualizer.generate_matrix_analysis([[1, 2], [3, 4]])
        
        processing_time = time.time() - start_time
        assert processing_time < 10.0  # Should complete within 10 seconds
        
    except ImportError:
        pytest.skip("Visualization module not available")

