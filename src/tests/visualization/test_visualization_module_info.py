"""Tests for visualization/__init__.py public helpers.

Covers get_module_info, get_visualization_options, and the network-statistics
helper, and verifies the exported public surface resolves.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestModuleInfo:
    def test_get_module_info(self) -> None:
        import visualization

        info = visualization.get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert info["version"] == visualization.__version__
        assert "features" in info
        assert {"matrix", "graph", "ontology"} <= set(info["visualization_types"])

    def test_get_visualization_options(self) -> None:
        import visualization

        opts = visualization.get_visualization_options()
        assert isinstance(opts, dict)
        assert "matrix_types" in opts
        assert "graph_types" in opts
        assert "output_formats" in opts

    def test_features_dict(self) -> None:
        import visualization

        assert isinstance(visualization.FEATURES, dict)
        for key in ("matrix_visualization", "network_graphs", "combined_analysis"):
            assert key in visualization.FEATURES


class TestNetworkStatistics:
    def _fn(self) -> Any:
        import visualization

        return visualization._generate_network_statistics

    def test_with_connections_counts_degrees(self) -> None:
        fn = self._fn()
        variables: dict[str, dict[str, Any]] = {"s1": {}, "s2": {}, "o1": {}}
        connections = [
            {"source": "s1", "target": "o1"},
            {"source": "s2", "target": "o1"},
        ]
        stats = fn(variables, connections)
        assert stats["total_nodes"] == 3
        assert stats["total_connections"] == 2
        assert stats["max_degree"] == 2
        assert "o1" in stats["node_degree_distribution"]
        # s1 and s2 each have degree 1; o1 has degree 2 → no node has degree > 2.
        assert stats["hub_nodes"] == []

    def test_isolated_nodes(self) -> None:
        fn = self._fn()
        variables: dict[str, dict[str, Any]] = {"lonely": {}}
        stats = fn(variables, [])
        assert stats["total_connections"] == 0
        assert stats["average_degree"] == 0
        assert stats["isolated_nodes"] == 1


class TestPublicSurface:
    def test_all_exports_resolve(self) -> None:
        import visualization

        for name in visualization.__all__:
            assert hasattr(visualization, name), f"missing export: {name}"

    def test_version_is_string(self) -> None:
        import visualization

        assert isinstance(visualization.__version__, str)
