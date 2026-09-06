#!/usr/bin/env python3
"""
Regression tests for gnn.dep_graph — model dependency graph rendering.

Pins ``build_dependency_graph`` (shared-variable dependency inference) and
``render_graph_from_file`` (mermaid/text rendering from a real GNN file, with
graceful handling of missing/malformed input instead of crashing).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.dep_graph import build_dependency_graph, render_graph_from_file  # noqa: E402


def _multi_model_gnn_file(tmp_path: Path) -> Path:
    """A two-model GNN file sharing variable ``x`` so an edge is inferred."""
    content = (
        "# Model A\n"
        "## StateSpaceBlock\n"
        "x[2]\n"
        "o[1]\n"
        "## Connections\n"
        "x>o\n"
        "---\n"
        "# Model B\n"
        "## StateSpaceBlock\n"
        "x[2]\n"
        "z[1]\n"
        "## Connections\n"
        "x>z\n"
    )
    path = tmp_path / "multi.gnn"
    path.write_text(content)
    return path


class TestBuildDependencyGraph:
    """Coverage of shared-variable edge inference."""

    @pytest.mark.unit
    def test_nodes_and_variable_counts(self) -> None:
        models: List[Dict[str, Any]] = [
            {"name": "A", "variables": [{"name": "x"}, {"name": "y"}]},
            {"name": "B", "variables": [{"name": "x"}, {"name": "z"}]},
        ]
        graph = build_dependency_graph(models)
        assert [n.name for n in graph.nodes] == ["A", "B"]
        assert [n.variable_count for n in graph.nodes] == [2, 2]

    @pytest.mark.unit
    def test_shared_vars_produce_edge(self) -> None:
        models: List[Dict[str, Any]] = [
            {"name": "A", "variables": [{"name": "x"}, {"name": "y"}]},
            {"name": "B", "variables": [{"name": "x"}, {"name": "z"}]},
        ]
        graph = build_dependency_graph(models)
        assert len(graph.edges) == 1
        edge = graph.edges[0]
        assert edge.source_model == "A"
        assert edge.target_model == "B"
        assert edge.shared_variables == ["x"]

    @pytest.mark.unit
    def test_no_shared_vars_no_edge(self) -> None:
        models: List[Dict[str, Any]] = [
            {"name": "A", "variables": [{"name": "x"}]},
            {"name": "B", "variables": [{"name": "z"}]},
        ]
        assert len(build_dependency_graph(models).edges) == 0

    @pytest.mark.unit
    def test_empty_models_builds_empty_graph(self) -> None:
        graph = build_dependency_graph([])
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    @pytest.mark.unit
    def test_mermaid_renders_nodes_and_edge(self) -> None:
        models: List[Dict[str, Any]] = [
            {"name": "A", "variables": [{"name": "x"}]},
            {"name": "B", "variables": [{"name": "x"}, {"name": "z"}]},
        ]
        out = build_dependency_graph(models).to_mermaid()
        assert out.startswith("graph TD")
        assert "A[" in out and "B[" in out
        assert "shared: x" in out

    @pytest.mark.unit
    def test_adjacency_list_renders(self) -> None:
        models: List[Dict[str, Any]] = [
            {"name": "A", "variables": [{"name": "x"}]},
            {"name": "B", "variables": [{"name": "x"}]},
        ]
        out = build_dependency_graph(models).to_adjacency_list()
        assert out.startswith("Dependency Graph:")
        assert "A" in out


class TestRenderGraphFromFile:
    """Coverage for the public file-rendering entry point."""

    @pytest.mark.unit
    def test_mermaid_from_real_file(self, tmp_path: Path) -> None:
        path = _multi_model_gnn_file(tmp_path)
        out = render_graph_from_file(str(path), output_format="mermaid")
        assert out.startswith("graph TD")
        assert "shared: x" in out

    @pytest.mark.unit
    def test_text_from_real_file(self, tmp_path: Path) -> None:
        path = _multi_model_gnn_file(tmp_path)
        out = render_graph_from_file(str(path), output_format="text")
        assert out.startswith("Dependency Graph:")

    @pytest.mark.unit
    def test_default_format_is_mermaid(self, tmp_path: Path) -> None:
        path = _multi_model_gnn_file(tmp_path)
        assert render_graph_from_file(str(path)).startswith("graph TD")

    @pytest.mark.unit
    def test_missing_file_does_not_crash(self, tmp_path: Path) -> None:
        out = render_graph_from_file(str(tmp_path / "nope.gnn"))
        assert isinstance(out, str)
        assert out.startswith("graph TD")

    @pytest.mark.unit
    def test_missing_file_text_does_not_crash(self, tmp_path: Path) -> None:
        out = render_graph_from_file(str(tmp_path / "nope.gnn"), output_format="text")
        assert isinstance(out, str)

    @pytest.mark.unit
    def test_malformed_content_does_not_crash(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.gnn"
        bad.write_text("@@@ not actually a GNN file @@@\n")
        out = render_graph_from_file(str(bad))
        assert isinstance(out, str)
        assert out.startswith("graph TD")

    @pytest.mark.unit
    def test_empty_file_does_not_crash(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.gnn"
        empty.write_text("")
        out = render_graph_from_file(str(empty))
        assert isinstance(out, str)
