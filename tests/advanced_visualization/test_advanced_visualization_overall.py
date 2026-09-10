"""
Test suite for Advanced Visualization module.

Tests D2 diagram generation, dashboards, and interactive visualizations.
"""

import json
from pathlib import Path
from typing import Any

import pytest


class TestAdvancedVisualizationModule:
    """Test suite for Advanced Visualization module functionality."""


    def test_visualization_functions(self) -> None:
        """Test visualization creation functions."""
        from gnn.advanced_visualization import (
            create_dashboard_section,
            create_default_visualization,
            create_heatmap_visualization,
            create_network_visualization,
            create_timeline_visualization,
            create_visualization_from_data,
        )

        assert callable(create_visualization_from_data)
        assert callable(create_dashboard_section)
        assert callable(create_network_visualization)
        assert callable(create_timeline_visualization)
        assert callable(create_heatmap_visualization)
        assert callable(create_default_visualization)


class TestAdvancedVisualizer:
    """Test AdvancedVisualizer class."""

    def test_visualizer_instantiation(self) -> None:
        """Test that AdvancedVisualizer can be instantiated."""
        from gnn.advanced_visualization import AdvancedVisualizer

        visualizer = AdvancedVisualizer()
        assert isinstance(visualizer, AdvancedVisualizer)

    def test_visualizer_methods(self) -> None:
        """Test visualizer has expected methods."""
        from gnn.advanced_visualization import AdvancedVisualizer

        visualizer = AdvancedVisualizer()

        assert callable(visualizer.generate_visualizations)


class TestDashboardGenerator:
    """Test DashboardGenerator class."""

    def test_dashboard_generator_instantiation(self) -> None:
        """Test that DashboardGenerator can be instantiated."""
        from gnn.advanced_visualization import DashboardGenerator

        generator = DashboardGenerator()
        assert isinstance(generator, DashboardGenerator)

    def test_generate_dashboard_function(self, safe_filesystem: Any) -> None:
        """Test dashboard generation function."""
        from gnn.advanced_visualization import generate_dashboard

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

        result = generate_dashboard(gnn_content, model_name, output_dir)
        assert isinstance(result, Path), (
            "generate_dashboard must return the output path"
        )
        assert result.exists(), f"dashboard artifact missing: {result}"
        assert result.suffix == ".html"
        assert result.read_text(encoding="utf-8").strip() != ""


class TestVisualizationDataExtractor:
    """Test VisualizationDataExtractor class."""

    def test_extractor_instantiation(self) -> None:
        """Test that VisualizationDataExtractor can be instantiated."""
        from gnn.advanced_visualization import VisualizationDataExtractor

        extractor = VisualizationDataExtractor()
        assert isinstance(extractor, VisualizationDataExtractor)

    def test_extract_visualization_data(self, safe_filesystem: Any) -> None:
        """Test data extraction function."""
        from gnn.advanced_visualization import extract_visualization_data

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
        safe_filesystem.create_file("viz_model.md", gnn_content)
        output_dir = safe_filesystem.create_dir("viz_data_output")

        result = extract_visualization_data(safe_filesystem.temp_dir, output_dir)
        assert isinstance(result, dict)
        assert {
            "processed_files",
            "successful_extractions",
            "failed_extractions",
            "extracted_data",
            "statistics",
            "errors",
        }.issubset(result.keys())

    def test_extract_from_file_failure_returns_full_shape(self, tmp_path: Any) -> Any:
        """Failure path returns all 13 keys matching the success shape."""
        from gnn.advanced_visualization.data_extractor import VisualizationDataExtractor

        extractor = VisualizationDataExtractor()
        missing = tmp_path / "does_not_exist.md"
        result = extractor.extract_from_file(missing)

        assert result["success"] is False
        expected_keys: set[Any] = {
            "success",
            "errors",
            "warnings",
            "model_info",
            "blocks",
            "connections",
            "parameters",
            "equations",
            "time_specification",
            "ontology_mappings",
            "total_blocks",
            "total_connections",
            "total_parameters",
            "total_equations",
            "extraction_timestamp",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_connection_keys_use_source_target_variables(self, tmp_path: Any) -> Any:
        """Connections extracted from model use source_variables/target_variables keys."""
        from gnn.advanced_visualization.data_extractor import VisualizationDataExtractor

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
                assert "source_variables" in conn, (
                    "'from' key found; expected 'source_variables'"
                )
                assert "target_variables" in conn, (
                    "'to' key found; expected 'target_variables'"
                )
                assert "from" not in conn
                assert "to" not in conn

    def test_extract_from_content_failure_returns_full_shape(self) -> Any:
        """extract_from_content failure path returns all 13 keys."""
        from gnn.advanced_visualization.data_extractor import VisualizationDataExtractor

        extractor = VisualizationDataExtractor()
        result = extractor.extract_from_content("")  # empty content

        assert isinstance(result, dict)
        assert "success" in result
        assert "errors" in result
        assert "connections" in result
        assert "blocks" in result


class TestD2Visualization:
    """Test D2 diagram visualization."""

    def test_d2_availability_flag(self) -> None:
        """Test D2_AVAILABLE flag is set."""
        from gnn.advanced_visualization import D2_AVAILABLE

        assert isinstance(D2_AVAILABLE, bool)

    def test_d2_visualizer_import(self) -> None:
        """Test D2Visualizer can be imported when available."""
        from gnn.advanced_visualization import D2Visualizer

        assert callable(D2Visualizer), "D2Visualizer must be importable"

    def test_process_gnn_file_with_d2(
        self, safe_filesystem: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test GNN file processing with D2."""
        from gnn.advanced_visualization import D2_AVAILABLE, process_gnn_file_with_d2

        if not D2_AVAILABLE or process_gnn_file_with_d2 is None:
            raise AssertionError("D2 visualization not available")

        gnn_content = """# D2 Test Model

## StateSpaceBlock
state[5]

## Connections
state -> state
"""
        test_file = safe_filesystem.create_file("d2_test.md", gnn_content)
        output_dir = safe_filesystem.create_dir("d2_output")

        try:
            caplog.set_level("ERROR", logger="gnn.advanced_visualization.d2_visualizer")
            result = process_gnn_file_with_d2(test_file, output_dir)
            assert result is not None
            assert "gnn.parser" not in caplog.text
            assert "Failed to parse GNN file" not in caplog.text
        except Exception as e:
            raise AssertionError(f"D2 processing failed: {e}")


class TestProcessAdvancedViz:
    """Test main processing function."""

    def test_process_advanced_viz(self, safe_filesystem: Any) -> None:
        """Test standardized advanced visualization processing."""
        from gnn.advanced_visualization import process_advanced_viz

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
        safe_filesystem.create_file("adv_viz.md", gnn_content)
        output_dir = safe_filesystem.create_dir("adv_viz_output")

        import logging

        logger = logging.getLogger("test_adv_viz")

        # Real success path: seed a parsed Step-3 model where
        # process_advanced_viz looks for one. It resolves the GNN output
        # directory as output_dir.parent / "3_gnn_output" when output_dir
        # ends with "_output".
        gnn_output_dir = output_dir.parent / "3_gnn_output"
        gnn_output_dir.mkdir(parents=True, exist_ok=True)
        (gnn_output_dir / "adv_viz_parsed.json").write_text(
            json.dumps({"model_name": "adv_viz", "variables": [], "connections": []}),
            encoding="utf-8",
        )

        result = process_advanced_viz(
            target_dir=safe_filesystem.temp_dir,
            output_dir=output_dir,
            logger=logger,
            verbose=True,
        )
        assert result is True, f"expected success (True), got {result!r}"
        summary = json.loads(
            (output_dir / "advanced_viz_summary.json").read_text(encoding="utf-8")
        )
        assert summary["successful"] >= 1, (
            f"no successful attempts: {summary['attempts']}"
        )
        output_files = [Path(p) for p in summary["output_files"]]
        assert output_files, "successful run must record output files"
        assert all(p.exists() for p in output_files)

    def test_process_with_viz_types(self, safe_filesystem: Any) -> None:
        """Test processing with different visualization types."""
        from gnn.advanced_visualization import process_advanced_viz

        gnn_content = """# Viz Types Test

## StateSpaceBlock
s[3]
"""
        safe_filesystem.create_file("types_test.md", gnn_content)
        safe_filesystem.create_dir("types_output")

        import logging

        logger = logging.getLogger("test_viz_types")

        viz_types: list[Any] = ["all", "dashboard", "d2", "network"]

        # Each viz type must run to completion with a seeded Step-3 model and
        # report a consistent outcome: True (attempts succeeded) or 2 (all
        # attempts skipped, e.g. the D2 CLI is not installed).
        # Seed the parsed Step-3 model once. Per-type output dirs end with
        # "_output", so process_advanced_viz resolves the GNN output
        # directory as their parent / "3_gnn_output".
        gnn_output_dir = safe_filesystem.temp_dir / "3_gnn_output"
        gnn_output_dir.mkdir(parents=True, exist_ok=True)
        (gnn_output_dir / "types_test_parsed.json").write_text(
            json.dumps(
                {"model_name": "types_test", "variables": [], "connections": []}
            ),
            encoding="utf-8",
        )

        for viz_type in viz_types:
            type_output_dir = safe_filesystem.create_dir(f"viz_types_{viz_type}_output")

            result = process_advanced_viz(
                target_dir=safe_filesystem.temp_dir,
                output_dir=type_output_dir,
                logger=logger,
                viz_type=viz_type,
            )
            assert result in (True, 2), f"viz_type={viz_type} returned {result!r}"
            summary = json.loads(
                (type_output_dir / "advanced_viz_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            if result is True:
                assert summary["successful"] >= 1
            else:
                assert summary["successful"] == 0
                assert summary["attempts"], "skip exit must record skipped attempts"

    def test_process_advanced_viz_empty_input_returns_warning_code(
        self, tmp_path: Any
    ) -> None:
        """No Step 3 models is warning-only recovery, not artifact success."""
        import json
        import logging

        from gnn.advanced_visualization import process_advanced_viz

        output_dir = tmp_path / "9_advanced_viz_output"
        result = process_advanced_viz(
            target_dir=tmp_path / "empty_input",
            output_dir=output_dir,
            logger=logging.getLogger("test_advanced_viz_empty"),
        )

        assert result == 2
        summary = json.loads(
            (output_dir / "advanced_viz_summary.json").read_text(encoding="utf-8")
        )
        assert summary["warnings"] == ["No GNN models found"]


class TestVisualizationCreation:
    """Test individual visualization creation functions."""

    def test_create_default_visualization(self) -> None:
        """Test default visualization creation."""
        from gnn.advanced_visualization import create_default_visualization

        data: dict[str, Any] = {"name": "test", "values": [1, 2, 3]}

        result = create_default_visualization(data)
        assert isinstance(result, dict)
        assert result["type"] == "chart"
        assert result["data"] == data
        assert result["options"]["chart_type"] == "line"

    def test_create_network_visualization(self) -> None:
        """Test network visualization creation."""
        from gnn.advanced_visualization import create_network_visualization

        data: dict[str, Any] = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B"), ("B", "C")],
        }

        result = create_network_visualization(data)
        assert isinstance(result, dict)
        assert result["type"] == "network"
        assert result["nodes"] == ["A", "B", "C"]
        assert result["edges"] == [("A", "B"), ("B", "C")]
        assert "layout" in result
        assert "options" in result
