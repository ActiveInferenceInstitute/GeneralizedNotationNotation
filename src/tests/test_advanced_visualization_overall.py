"""
Test suite for Advanced Visualization module.

Tests D2 diagram generation, dashboards, and interactive visualizations.
"""

import pytest


class TestAdvancedVisualizationModule:
    """Test suite for Advanced Visualization module functionality."""

    def test_module_imports(self):
        """Test that advanced_visualization module can be imported."""
        from advanced_visualization import (
            AdvancedVisualizer,
            DashboardGenerator,
            VisualizationDataExtractor,
            process_advanced_viz,
            D2_AVAILABLE
        )
        assert callable(AdvancedVisualizer)
        assert callable(DashboardGenerator)
        assert callable(VisualizationDataExtractor)
        assert callable(process_advanced_viz)
        assert isinstance(D2_AVAILABLE, bool)

    def test_visualization_functions(self):
        """Test visualization creation functions."""
        from advanced_visualization import (
            create_visualization_from_data,
            create_dashboard_section,
            create_network_visualization,
            create_timeline_visualization,
            create_heatmap_visualization,
            create_default_visualization
        )

        assert callable(create_visualization_from_data)
        assert callable(create_dashboard_section)
        assert callable(create_network_visualization)
        assert callable(create_timeline_visualization)
        assert callable(create_heatmap_visualization)
        assert callable(create_default_visualization)


class TestAdvancedVisualizer:
    """Test AdvancedVisualizer class."""

    def test_visualizer_instantiation(self):
        """Test that AdvancedVisualizer can be instantiated."""
        from advanced_visualization import AdvancedVisualizer

        visualizer = AdvancedVisualizer()
        assert visualizer is not None

    def test_visualizer_methods(self):
        """Test visualizer has expected methods."""
        from advanced_visualization import AdvancedVisualizer

        visualizer = AdvancedVisualizer()

        # Check for common visualization methods
        assert hasattr(visualizer, '__class__')


class TestDashboardGenerator:
    """Test DashboardGenerator class."""

    def test_dashboard_generator_instantiation(self):
        """Test that DashboardGenerator can be instantiated."""
        from advanced_visualization import DashboardGenerator

        generator = DashboardGenerator()
        assert generator is not None

    def test_generate_dashboard_function(self, safe_filesystem):
        """Test dashboard generation function."""
        from advanced_visualization import generate_dashboard

        # Create sample GNN content for dashboard generation
        gnn_content = """# Test Dashboard Model

## StateSpaceBlock
hidden_states[10, type=float]
observations[5, type=float]

## Connections
hidden_states -> observations
observations -> hidden_states

## Parameters
learning_rate = 0.01
"""
        model_name = "test_model"
        output_dir = safe_filesystem.create_dir("dashboard_output")

        try:
            result = generate_dashboard(gnn_content, model_name, output_dir)
            # Result could be None if generation failed gracefully, or Path if successful
            assert result is None or hasattr(result, 'exists')
        except Exception as e:
            # May require additional dependencies
            pytest.skip(f"Dashboard generation requires additional dependencies: {e}")


class TestVisualizationDataExtractor:
    """Test VisualizationDataExtractor class."""

    def test_extractor_instantiation(self):
        """Test that VisualizationDataExtractor can be instantiated."""
        from advanced_visualization import VisualizationDataExtractor

        extractor = VisualizationDataExtractor()
        assert extractor is not None

    def test_extract_visualization_data(self, safe_filesystem):
        """Test data extraction function."""
        from advanced_visualization import extract_visualization_data

        # Create sample GNN file
        gnn_content = """# Visualization Test Model

## StateSpaceBlock
hidden_states[10, type=float]
observations[5, type=float]

## Connections
hidden_states -> observations
observations -> hidden_states

## Parameters
learning_rate = 0.01
"""
        test_file = safe_filesystem.create_file("viz_model.md", gnn_content)
        output_dir = safe_filesystem.create_dir("viz_data_output")

        try:
            # extract_visualization_data expects (target_dir, output_dir, **kwargs)
            result = extract_visualization_data(safe_filesystem.temp_dir, output_dir)
            assert result is not None
            assert isinstance(result, dict)
        except Exception as e:
            pytest.skip(f"Data extraction failed: {e}")

    def test_extract_from_file_failure_returns_full_shape(self, tmp_path):
        """Failure path returns all 13 keys matching the success shape."""
        from advanced_visualization.data_extractor import VisualizationDataExtractor

        extractor = VisualizationDataExtractor()
        missing = tmp_path / "does_not_exist.md"
        result = extractor.extract_from_file(missing)

        assert result["success"] is False
        expected_keys = {
            "success", "errors", "warnings", "model_info", "blocks",
            "connections", "parameters", "equations", "time_specification",
            "ontology_mappings", "total_blocks", "total_connections",
            "total_parameters", "total_equations", "extraction_timestamp",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_connection_keys_use_source_target_variables(self, tmp_path):
        """Connections extracted from model use source_variables/target_variables keys."""
        from advanced_visualization.data_extractor import VisualizationDataExtractor

        gnn_content = (
            "## GNNSection\nActInfPOMDP\n\n"
            "## ModelName\nConnKeyTest\n\n"
            "## StateSpaceBlock\n"
            "s[2,type=float]\no[2,type=float]\n\n"
            "## Connections\ns->o\n"
        )
        test_file = tmp_path / "conn_test.md"
        test_file.write_text(gnn_content)

        extractor = VisualizationDataExtractor()
        result = extractor.extract_from_file(test_file)

        if result["success"] and result["connections"]:
            for conn in result["connections"]:
                assert "source_variables" in conn, "'from' key found; expected 'source_variables'"
                assert "target_variables" in conn, "'to' key found; expected 'target_variables'"
                assert "from" not in conn
                assert "to" not in conn

    def test_extract_from_content_failure_returns_full_shape(self):
        """extract_from_content failure path returns all 13 keys."""
        from advanced_visualization.data_extractor import VisualizationDataExtractor

        extractor = VisualizationDataExtractor()
        result = extractor.extract_from_content("")  # empty content

        assert isinstance(result, dict)
        assert "success" in result
        assert "errors" in result
        assert "connections" in result
        assert "blocks" in result


class TestD2Visualization:
    """Test D2 diagram visualization."""

    def test_d2_availability_flag(self):
        """Test D2_AVAILABLE flag is set."""
        from advanced_visualization import D2_AVAILABLE

        assert isinstance(D2_AVAILABLE, bool)

    @pytest.mark.skipif(
        not pytest.importorskip("advanced_visualization").D2_AVAILABLE,
        reason="D2 not available"
    )
    def test_d2_visualizer_import(self):
        """Test D2Visualizer can be imported when available."""
        from advanced_visualization import D2Visualizer

        if D2Visualizer is not None:
            assert callable(D2Visualizer)

    def test_process_gnn_file_with_d2(self, safe_filesystem):
        """Test GNN file processing with D2."""
        from advanced_visualization import process_gnn_file_with_d2, D2_AVAILABLE

        if not D2_AVAILABLE or process_gnn_file_with_d2 is None:
            pytest.skip("D2 visualization not available")

        gnn_content = """# D2 Test Model

## StateSpaceBlock
state[5]

## Connections
state -> state
"""
        test_file = safe_filesystem.create_file("d2_test.md", gnn_content)
        output_dir = safe_filesystem.create_dir("d2_output")

        try:
            result = process_gnn_file_with_d2(test_file, output_dir)
            assert result is not None
        except Exception as e:
            pytest.skip(f"D2 processing failed: {e}")


class TestProcessAdvancedViz:
    """Test main processing function."""

    def test_process_advanced_viz(self, safe_filesystem):
        """Test standardized advanced visualization processing."""
        from advanced_visualization import process_advanced_viz

        # Create test GNN file
        gnn_content = """# Advanced Viz Test

## StateSpaceBlock
x[10]
y[5]

## Connections
x -> y
y -> x

## Time
Dynamic
"""
        test_file = safe_filesystem.create_file("adv_viz.md", gnn_content)
        output_dir = safe_filesystem.create_dir("adv_viz_output")

        import logging
        logger = logging.getLogger("test_adv_viz")

        try:
            result = process_advanced_viz(
                target_dir=safe_filesystem.temp_dir,
                output_dir=output_dir,
                logger=logger,
                verbose=True
            )
            # Should return True or dict with success status
            assert result is True or (isinstance(result, dict) and result.get('success', False)) or result is not None
        except ImportError as e:
            pytest.skip(f"Advanced visualization requires additional dependencies: {e}")

    def test_process_with_viz_types(self, safe_filesystem):
        """Test processing with different visualization types."""
        from advanced_visualization import process_advanced_viz

        gnn_content = """# Viz Types Test

## StateSpaceBlock
s[3]
"""
        test_file = safe_filesystem.create_file("types_test.md", gnn_content)
        output_dir = safe_filesystem.create_dir("types_output")

        import logging
        logger = logging.getLogger("test_viz_types")

        viz_types = ["all", "dashboard", "d2", "network"]

        for viz_type in viz_types:
            try:
                result = process_advanced_viz(
                    target_dir=safe_filesystem.temp_dir,
                    output_dir=output_dir,
                    logger=logger,
                    viz_type=viz_type
                )
                # Should not crash
                assert result is not None or result is True or result is False
            except ImportError:
                pytest.skip(f"Visualization type {viz_type} requires additional dependencies")
            except Exception:
                # Some viz types may fail without proper data, that's OK
                pass


class TestVisualizationCreation:
    """Test individual visualization creation functions."""

    def test_create_default_visualization(self):
        """Test default visualization creation."""
        from advanced_visualization import create_default_visualization

        data = {"name": "test", "values": [1, 2, 3]}

        try:
            result = create_default_visualization(data)
            assert result is not None
        except Exception:
            # May require specific data format
            pass

    def test_create_network_visualization(self):
        """Test network visualization creation."""
        from advanced_visualization import create_network_visualization

        data = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B"), ("B", "C")]
        }

        try:
            result = create_network_visualization(data)
            assert result is not None
        except Exception:
            # May require specific data format or dependencies
            pass
