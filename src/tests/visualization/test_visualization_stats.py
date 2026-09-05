"""Tests for visualization.graph.stats.compute_connection_statistics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import visualization
from visualization.graph.stats import compute_connection_statistics


class TestComputeConnectionStatistics:
    def test_degree_counts_and_hubs(self) -> None:
        variables: dict[str, dict[str, Any]] = {"s1": {}, "s2": {}, "o1": {}}
        connections = [
            {"source": "s1", "target": "o1"},
            {"source": "s2", "target": "o1"},
            {"source": "o1", "target": "s1"},
        ]
        stats = compute_connection_statistics(variables, connections)
        assert stats["total_nodes"] == 3
        assert stats["total_connections"] == 3
        assert stats["max_degree"] == 3  # o1: 2 in + 1 out
        assert stats["node_degree_distribution"]["o1"] == 3
        assert stats["hub_nodes"] == ["o1"]  # degree > 2
        assert stats["isolated_nodes"] == 0
        assert abs(stats["average_degree"] - 2.0) < 1e-9

    def test_empty_connections(self) -> None:
        stats = compute_connection_statistics({"lonely": {}}, [])
        assert stats["total_connections"] == 0
        assert stats["average_degree"] == 0
        assert stats["isolated_nodes"] == 1
        assert stats["hub_nodes"] == []
        assert stats["node_degree_distribution"] == {}

    def test_missing_endpoints_fall_back_to_unknown(self) -> None:
        stats = compute_connection_statistics({}, [{}])
        assert stats["node_degree_distribution"] == {"unknown": 2}

    def test_package_root_alias_is_same_function(self) -> None:
        # The pinned package-root name must stay the same callable.
        assert (
            visualization._generate_network_statistics is compute_connection_statistics
        )
