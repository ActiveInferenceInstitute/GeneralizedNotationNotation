"""Tests for advanced_visualization module's public API surface not covered by existing tests.

Covers: get_module_info, FEATURES, __version__, create_dashboard_section,
create_visualization_from_data, create_heatmap_visualization,
create_timeline_visualization, D2DiagramSpec, D2GenerationResult.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestAdvancedVizConstants:
    """Test module-level constants."""

    def test_features_dict(self) -> None:
        import advanced_visualization

        assert hasattr(advanced_visualization, "FEATURES")
        assert isinstance(advanced_visualization.FEATURES, dict)
        for key in (
            "d2_diagrams",
            "interactive_dashboards",
            "network_visualization",
            "timeline_visualization",
            "heatmap_visualization",
            "data_extraction",
            "mcp_integration",
        ):
            assert key in advanced_visualization.FEATURES

    def test_version(self) -> None:
        import advanced_visualization

        assert hasattr(advanced_visualization, "__version__")
        assert isinstance(advanced_visualization.__version__, str)

    def test_get_module_info(self) -> None:
        from advanced_visualization import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert info["name"] == "advanced_visualization"
        assert "version" in info
        assert "description" in info
        assert "features" in info


class TestVisualizationCreationFunctions:
    """Test create_* visualization functions."""

    def test_create_dashboard_section(self) -> None:
        from advanced_visualization import create_dashboard_section

        data: dict[str, Any] = {"name": "test", "value": 42}
        result = create_dashboard_section(data)
        assert result is not None

    def test_create_visualization_from_data(self) -> None:
        from advanced_visualization import create_visualization_from_data

        data: dict[str, Any] = {"name": "viz", "values": [1, 2, 3]}
        result = create_visualization_from_data(data)
        assert result is not None

    def test_create_heatmap_visualization(self) -> None:
        from advanced_visualization import create_heatmap_visualization

        data: dict[str, Any] = {"matrix": [[1, 2], [3, 4]]}
        result = create_heatmap_visualization(data)
        assert result is not None

    def test_create_timeline_visualization(self) -> None:
        from advanced_visualization import create_timeline_visualization

        data: dict[str, Any] = {"events": [{"time": 1, "value": "a"}]}
        result = create_timeline_visualization(data)
        assert result is not None

    def test_create_default_visualization_with_dict(self) -> None:
        from advanced_visualization import create_default_visualization

        result = create_default_visualization({"key": "value"})
        assert result is not None

    def test_create_network_visualization_with_graph_data(self) -> None:
        from advanced_visualization import create_network_visualization

        data: dict[str, Any] = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
            "graph": {"nodes": [{"id": "A"}, {"id": "B"}], "edges": []},
        }
        result = create_network_visualization(data)
        assert result is not None


class TestD2Types:
    """Test D2-related types."""

    def test_d2_diagram_spec_instantiable(self) -> None:
        from advanced_visualization import D2DiagramSpec

        spec = D2DiagramSpec("test", "desc", "graph { a -> b }")
        assert spec is not None
        assert spec.name == "test"

    def test_d2_generation_result_instantiable(self) -> None:
        from advanced_visualization import D2GenerationResult

        result = D2GenerationResult(True, "test_diagram")
        assert result is not None
        assert result.success is True
        assert result.diagram_name == "test_diagram"

    def test_d2_visualizer_class(self) -> None:
        from advanced_visualization import D2Visualizer

        if D2Visualizer is not None:
            instance = D2Visualizer()
            assert instance is not None

    def test_process_gnn_file_with_d2_available(self) -> None:
        import advanced_visualization

        assert hasattr(advanced_visualization, "process_gnn_file_with_d2")
