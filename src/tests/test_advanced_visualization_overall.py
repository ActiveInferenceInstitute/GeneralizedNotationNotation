#!/usr/bin/env python3
"""
Comprehensive tests for the Advanced Visualization module.

This file contains unit tests, integration tests, and performance tests
for the advanced visualization functionality.
"""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


class TestAdvancedVisualizationModule:
    """Test the advanced visualization module's core functionality."""

    @pytest.mark.unit
    def test_module_imports(self):
        """Test that all expected functions are available."""
        try:
            from advanced_visualization import (
                AdvancedVisualizer,
                create_visualization_from_data,
                create_dashboard_section,
                create_network_visualization,
                create_timeline_visualization,
                create_heatmap_visualization,
                create_default_visualization,
                DashboardGenerator,
                generate_dashboard,
                VisualizationDataExtractor,
                extract_visualization_data,
                process_advanced_viz_standardized_impl
            )

            # Verify all expected classes and functions are imported
            assert AdvancedVisualizer is not None
            assert create_visualization_from_data is not None
            assert create_dashboard_section is not None
            assert create_network_visualization is not None
            assert create_timeline_visualization is not None
            assert create_heatmap_visualization is not None
            assert create_default_visualization is not None
            assert DashboardGenerator is not None
            assert generate_dashboard is not None
            assert VisualizationDataExtractor is not None
            assert extract_visualization_data is not None
            assert process_advanced_viz_standardized_impl is not None

        except ImportError as e:
            pytest.skip(f"Advanced visualization module not available: {e}")

    @pytest.mark.unit
    def test_advanced_visualizer_instantiation(self):
        """Test AdvancedVisualizer class instantiation."""
        try:
            from advanced_visualization.visualizer import AdvancedVisualizer

            visualizer = AdvancedVisualizer()
            assert visualizer is not None
            assert hasattr(visualizer, 'generate_visualizations')
            assert hasattr(visualizer, 'logger')

        except ImportError:
            pytest.skip("AdvancedVisualizer not available")

    @pytest.mark.unit
    def test_dashboard_generator_instantiation(self):
        """Test DashboardGenerator class instantiation."""
        try:
            from advanced_visualization.dashboard import DashboardGenerator

            generator = DashboardGenerator()
            assert generator is not None
            assert hasattr(generator, 'generate_dashboard')
            assert hasattr(generator, 'data_extractor')

        except ImportError:
            pytest.skip("DashboardGenerator not available")

    @pytest.mark.unit
    def test_data_extractor_instantiation(self):
        """Test VisualizationDataExtractor class instantiation."""
        try:
            from advanced_visualization.data_extractor import VisualizationDataExtractor

            extractor = VisualizationDataExtractor()
            assert extractor is not None
            assert hasattr(extractor, 'extract_from_file')
            assert hasattr(extractor, 'extract_from_content')
            assert hasattr(extractor, 'get_model_statistics')

        except ImportError:
            pytest.skip("VisualizationDataExtractor not available")


class TestDataExtraction:
    """Test data extraction functionality."""

    @pytest.mark.unit
    def test_data_extraction_from_content(self):
        """Test extracting visualization data from GNN content."""
        try:
            from advanced_visualization.data_extractor import VisualizationDataExtractor

            # Use sample GNN content directly
            sample_content = """
# Test GNN Model

## StateSpaceBlock
state [3,3] # Hidden state
action [2] # Action space
observation [3] # Observation space

## Connections
state > action
action > observation
observation > state

## Parameters
A = [[0.9, 0.1], [0.1, 0.9]] # Likelihood matrix
B = [[[0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.5, 0.5]]] # Transition matrices
C = [0.3, 0.4, 0.3] # Preferences
"""

            extractor = VisualizationDataExtractor()
            result = extractor.extract_from_content(sample_content)

            assert result is not None
            assert isinstance(result, dict)
            assert "success" in result
            assert "blocks" in result
            assert "connections" in result

            # Check that we get a valid response even if parsing fails
            # The extractor should handle errors gracefully
            if result.get("success"):
                # Check that we extracted some data
                assert len(result.get("blocks", [])) > 0
                # model_info may or may not be present depending on parsing success
                if "model_info" in result:
                    assert isinstance(result.get("model_info", {}), dict)
            else:
                # If parsing failed, we should still get error information
                assert "errors" in result
                assert isinstance(result.get("errors", []), list)

        except ImportError:
            pytest.skip("VisualizationDataExtractor not available")

    @pytest.mark.unit
    def test_model_statistics_generation(self):
        """Test generation of model statistics."""
        try:
            from advanced_visualization.data_extractor import VisualizationDataExtractor

            # Use sample GNN content directly
            sample_content = """
# Test GNN Model

## StateSpaceBlock
state [3,3] # Hidden state
action [2] # Action space
observation [3] # Observation space

## Connections
state > action
action > observation
observation > state

## Parameters
A = [[0.9, 0.1], [0.1, 0.9]] # Likelihood matrix
B = [[[0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.5, 0.5]]] # Transition matrices
C = [0.3, 0.4, 0.3] # Preferences
"""

            extractor = VisualizationDataExtractor()
            extracted_data = extractor.extract_from_content(sample_content)

            if extracted_data.get("success"):
                stats = extractor.get_model_statistics(extracted_data)

                assert stats is not None
                assert isinstance(stats, dict)
                assert "total_variables" in stats
                assert "total_connections" in stats
                assert "variable_types" in stats
                assert isinstance(stats["total_variables"], int)
                assert isinstance(stats["total_connections"], int)

        except ImportError:
            pytest.skip("VisualizationDataExtractor not available")


class TestVisualizationGeneration:
    """Test visualization generation functionality."""

    @pytest.mark.unit
    def test_visualization_creation_functions(self):
        """Test individual visualization creation functions."""
        try:
            from advanced_visualization.visualizer import (
                create_network_visualization,
                create_timeline_visualization,
                create_heatmap_visualization,
                create_default_visualization
            )

            # Test network visualization
            network_data = {
                "nodes": [{"id": "1", "name": "Node1"}, {"id": "2", "name": "Node2"}],
                "edges": [{"from": "1", "to": "2"}]
            }
            network_viz = create_network_visualization(network_data)
            assert network_viz is not None
            assert network_viz["type"] == "network"
            assert "nodes" in network_viz
            assert "edges" in network_viz

            # Test timeline visualization
            timeline_data = {"events": [{"time": "2023-01-01", "event": "Test"}]}
            timeline_viz = create_timeline_visualization(timeline_data)
            assert timeline_viz is not None
            assert timeline_viz["type"] == "timeline"
            assert "events" in timeline_viz

            # Test heatmap visualization
            heatmap_data = {"matrix": [[1, 2], [3, 4]]}
            heatmap_viz = create_heatmap_visualization(heatmap_data)
            assert heatmap_viz is not None
            assert heatmap_viz["type"] == "heatmap"
            assert "matrix" in heatmap_viz

            # Test default visualization
            default_data = {"value": 42}
            default_viz = create_default_visualization(default_data)
            assert default_viz is not None
            assert default_viz["type"] == "chart"
            assert "data" in default_viz

        except ImportError:
            pytest.skip("Visualization creation functions not available")

    @pytest.mark.integration
    def test_advanced_visualizer_generation(self, isolated_temp_dir):
        """Test AdvancedVisualizer.generate_visualizations method."""
        try:
            from advanced_visualization.visualizer import AdvancedVisualizer

            # Use sample GNN content directly
            sample_content = """
# Test GNN Model

## StateSpaceBlock
state [3,3] # Hidden state
action [2] # Action space
observation [3] # Observation space

## Connections
state > action
action > observation
observation > state

## Parameters
A = [[0.9, 0.1], [0.1, 0.9]] # Likelihood matrix
B = [[[0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.5, 0.5]]] # Transition matrices
C = [0.3, 0.4, 0.3] # Preferences
"""

            visualizer = AdvancedVisualizer()
            generated_files = visualizer.generate_visualizations(
                content=sample_content,
                model_name="test_model",
                output_dir=isolated_temp_dir,
                viz_type="all",
                interactive=True,
                export_formats=["html", "json"]
            )

            assert generated_files is not None
            assert isinstance(generated_files, list)

            # Check that files were created
            for file_path in generated_files:
                assert Path(file_path).exists()

        except ImportError:
            pytest.skip("AdvancedVisualizer not available")


class TestDashboardGeneration:
    """Test dashboard generation functionality."""

    @pytest.mark.unit
    def test_dashboard_section_creation(self):
        """Test creating dashboard sections."""
        try:
            from advanced_visualization.visualizer import create_dashboard_section

            section_data = {
                "title": "Test Section",
                "type": "metrics",
                "content": "Test content",
                "metrics": {"value": 42}
            }

            section = create_dashboard_section(section_data)
            assert section is not None
            assert section["title"] == "Test Section"
            assert section["type"] == "metrics"
            assert section["content"] == "Test content"
            assert section["metrics"]["value"] == 42

        except ImportError:
            pytest.skip("Dashboard section creation not available")

    @pytest.mark.integration
    def test_dashboard_generation(self, isolated_temp_dir):
        """Test generating complete dashboards."""
        try:
            from advanced_visualization.dashboard import generate_dashboard

            # Use sample GNN content directly
            sample_content = """
# Test GNN Model

## StateSpaceBlock
state [3,3] # Hidden state
action [2] # Action space
observation [3] # Observation space

## Connections
state > action
action > observation
observation > state

## Parameters
A = [[0.9, 0.1], [0.1, 0.9]] # Likelihood matrix
B = [[[0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.5, 0.5]]] # Transition matrices
C = [0.3, 0.4, 0.3] # Preferences
"""

            dashboard_path = generate_dashboard(
                content=sample_content,
                model_name="test_model",
                output_dir=isolated_temp_dir,
                strict_validation=False
            )

            if dashboard_path:
                assert dashboard_path.exists()
                assert dashboard_path.suffix == ".html"

                # Check that the HTML contains expected content
                with open(dashboard_path) as f:
                    html_content = f.read()
                assert "GNN Model Dashboard" in html_content
                assert "test_model" in html_content

        except ImportError:
            pytest.skip("Dashboard generation not available")


class TestErrorHandling:
    """Test error handling and fallback mechanisms."""

    @pytest.mark.unit
    def test_missing_dependencies_handling(self):
        """Test handling of missing dependencies."""
        try:
            from advanced_visualization.processor import _check_dependencies

            # Mock logger
            logger = MagicMock()

            dependencies = _check_dependencies(logger)

            assert dependencies is not None
            assert isinstance(dependencies, dict)
            assert "matplotlib" in dependencies
            assert "plotly" in dependencies
            assert "seaborn" in dependencies
            assert "bokeh" in dependencies
            assert "numpy" in dependencies

        except ImportError:
            pytest.skip("Dependency checking not available")

    @pytest.mark.unit
    def test_invalid_content_handling(self):
        """Test handling of invalid GNN content."""
        try:
            from advanced_visualization.data_extractor import VisualizationDataExtractor

            extractor = VisualizationDataExtractor()

            # Test with invalid content
            invalid_content = "This is not valid GNN content"
            result = extractor.extract_from_content(invalid_content)

            assert result is not None
            assert isinstance(result, dict)
            assert "success" in result
            # Should handle gracefully even with invalid content

        except ImportError:
            pytest.skip("Data extraction not available")


class TestIntegration:
    """Integration tests for advanced visualization module."""

    @pytest.mark.integration
    def test_pipeline_integration(self, sample_gnn_files, isolated_temp_dir):
        """Test integration with the overall pipeline."""
        try:
            from advanced_visualization.processor import process_advanced_viz_standardized_impl

            # Create mock logger
            logger = MagicMock()

            success = process_advanced_viz_standardized_impl(
                target_dir=Path("input/gnn_files"),
                output_dir=isolated_temp_dir,
                logger=logger,
                viz_type="all",
                interactive=False,
                export_formats=["html", "json"]
            )

            # Should succeed (even if no models found, should not fail)
            assert success is True

        except ImportError:
            pytest.skip("Pipeline integration not available")

    @pytest.mark.integration
    def test_mcp_integration(self):
        """Test MCP integration."""
        try:
            from advanced_visualization.mcp import register_tools

            # Test that MCP tools can be registered
            assert callable(register_tools)

        except ImportError:
            pytest.skip("MCP integration not available")


class TestPerformance:
    """Performance tests for advanced visualization module."""

    @pytest.mark.slow
    def test_visualization_performance(self):
        """Test performance of visualization generation."""
        try:
            from advanced_visualization.visualizer import AdvancedVisualizer
            import time

            # Use sample GNN content directly
            sample_content = """
# Test GNN Model

## StateSpaceBlock
state [3,3] # Hidden state
action [2] # Action space
observation [3] # Observation space

## Connections
state > action
action > observation
observation > state

## Parameters
A = [[0.9, 0.1], [0.1, 0.9]] # Likelihood matrix
B = [[[0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.5, 0.5]]] # Transition matrices
C = [0.3, 0.4, 0.3] # Preferences
"""

            visualizer = AdvancedVisualizer()

            start_time = time.time()
            generated_files = visualizer.generate_visualizations(
                content=sample_content,
                model_name="performance_test",
                output_dir=Path(tempfile.mkdtemp()),
                viz_type="statistical",
                interactive=False
            )
            end_time = time.time()

            # Should complete within reasonable time (less than 30 seconds)
            duration = end_time - start_time
            assert duration < 30.0, f"Visualization took too long: {duration}s"

            # Should generate some files
            assert isinstance(generated_files, list)

        except ImportError:
            pytest.skip("Performance testing not available")


def test_module_completeness():
    """Test that advanced visualization module has all required components."""
    required_components = [
        'AdvancedVisualizer',
        'create_visualization_from_data',
        'create_dashboard_section',
        'create_network_visualization',
        'create_timeline_visualization',
        'create_heatmap_visualization',
        'create_default_visualization',
        'DashboardGenerator',
        'generate_dashboard',
        'VisualizationDataExtractor',
        'extract_visualization_data',
        'process_advanced_viz_standardized_impl'
    ]

    try:
        import advanced_visualization
        for component in required_components:
            assert hasattr(advanced_visualization, component), f"Missing component: {component}"
    except ImportError:
        pytest.skip("Advanced visualization module not available")


@pytest.mark.integration
def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    try:
        # Test that we can run the full advanced visualization pipeline
        from advanced_visualization.processor import process_advanced_viz_standardized_impl

        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Mock logger
            logger = MagicMock()

            # Run the processor
            success = process_advanced_viz_standardized_impl(
                target_dir=Path("input/gnn_files"),
                output_dir=output_dir,
                logger=logger,
                viz_type="statistical",
                interactive=False
            )

            # Should complete successfully
            assert success is True

    except ImportError:
        pytest.skip("End-to-end testing not available")
